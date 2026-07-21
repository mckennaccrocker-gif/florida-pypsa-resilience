"""Create an explicit voltage-layer transformer network variant.

The current PyPSA network stores bus voltage and line voltage, but some lines
connect directly to buses with a different voltage level. This script converts
those mismatched endpoints into explicit voltage-layer buses plus PyPSA
transformers:

    original bus -- transformer -- auxiliary bus at line voltage -- line

The original bus keeps loads/generators. The auxiliary bus only hosts the line
endpoint. This preserves existing topology while making voltage interfaces
explicit for PyPSA.
"""

from __future__ import annotations

import math
import shutil
from pathlib import Path

import pandas as pd


PROJECT_DIR = Path(__file__).resolve().parents[2]
BASE_NETWORK = PROJECT_DIR / "data" / "Electricity" / "pypsa_florida_network_official_current"
OUTPUT_NETWORK = PROJECT_DIR / "data" / "Electricity" / "pypsa_florida_network_voltage_transformers"

TRANSFORMER_R_PU = 0.01
TRANSFORMER_X_PU = 0.10
TRANSFORMER_CAPACITY_MULTIPLIER = 3.0
VOLTAGE_TOLERANCE_KV = 1e-6


def normalized_voltage(value) -> float:
    voltage = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    if pd.isna(voltage):
        return math.nan
    return float(voltage)


def voltage_label(value: float) -> str:
    if math.isnan(value):
        return "unknown"
    if abs(value - round(value)) < 1e-6:
        return str(int(round(value)))
    return str(value).replace(".", "p")


def copy_network() -> None:
    if OUTPUT_NETWORK.exists():
        shutil.rmtree(OUTPUT_NETWORK)
    OUTPUT_NETWORK.mkdir(parents=True)
    for path in BASE_NETWORK.iterdir():
        if path.is_file():
            shutil.copy2(path, OUTPUT_NETWORK / path.name)


def add_auxiliary_assignment_rows(filename: str, created_buses: pd.DataFrame) -> None:
    path = OUTPUT_NETWORK / filename
    if not path.exists() or created_buses.empty:
        return
    df = pd.read_csv(path)
    if "bus" in df.columns:
        key = "bus"
    elif "name" in df.columns:
        key = "name"
    else:
        return

    additions = []
    for row in created_buses.itertuples(index=False):
        original = df[df[key].astype(str).eq(row.original_bus)]
        if original.empty:
            continue
        copied = original.copy()
        copied[key] = row.name
        additions.append(copied)

    if additions:
        pd.concat([df, *additions], ignore_index=True).to_csv(path, index=False)


def main() -> None:
    copy_network()

    buses = pd.read_csv(OUTPUT_NETWORK / "buses.csv")
    lines = pd.read_csv(OUTPUT_NETWORK / "lines.csv")
    buses_by_name = buses.set_index("name")

    if "active" not in lines.columns:
        lines["active"] = True

    created_bus_rows = []
    transformer_rows = []
    endpoint_rows = []
    transformer_capacity = {}

    for line_idx, line in lines.iterrows():
        line_voltage = normalized_voltage(line.get("v_nom"))
        if math.isnan(line_voltage):
            continue

        for side in ["bus0", "bus1"]:
            original_bus = str(line[side])
            if original_bus not in buses_by_name.index:
                continue
            bus_voltage = normalized_voltage(buses_by_name.loc[original_bus, "v_nom"])
            if math.isnan(bus_voltage) or abs(bus_voltage - line_voltage) <= VOLTAGE_TOLERANCE_KV:
                continue

            aux_bus = f"{original_bus}_v{voltage_label(line_voltage)}"
            transformer = f"trafo_{original_bus}_to_v{voltage_label(line_voltage)}"

            if aux_bus not in buses_by_name.index and aux_bus not in {r["name"] for r in created_bus_rows}:
                original = buses_by_name.loc[original_bus]
                created_bus_rows.append(
                    {
                        "name": aux_bus,
                        "v_nom": line_voltage,
                        "carrier": original.get("carrier", "AC"),
                        "x": original["x"],
                        "y": original["y"],
                        "substation_name": original.get("substation_name", ""),
                        "source_node_id": original.get("source_node_id", ""),
                        "is_step_up_down": True,
                        "connected_edge_count": 0,
                        "original_bus": original_bus,
                        "voltage_layer_bus": True,
                    }
                )

            lines.at[line_idx, side] = aux_bus
            transformer_capacity[transformer] = transformer_capacity.get(transformer, 0.0) + float(
                pd.to_numeric(pd.Series([line.get("s_nom")]), errors="coerce").fillna(0.0).iloc[0]
            )
            endpoint_rows.append(
                {
                    "line": line["name"],
                    "endpoint": side,
                    "original_bus": original_bus,
                    "aux_bus": aux_bus,
                    "original_bus_v_nom": bus_voltage,
                    "line_v_nom": line_voltage,
                    "transformer": transformer,
                }
            )

    created_buses = pd.DataFrame(created_bus_rows)
    if not created_buses.empty:
        buses = pd.concat(
            [buses, created_buses[buses.columns.intersection(created_buses.columns)]],
            ignore_index=True,
        )

    for transformer, base_capacity in sorted(transformer_capacity.items()):
        original_bus = transformer.removeprefix("trafo_").split("_to_v")[0]
        voltage = float(transformer.rsplit("_to_v", 1)[1].replace("p", "."))
        aux_bus = f"{original_bus}_v{voltage_label(voltage)}"
        transformer_rows.append(
            {
                "name": transformer,
                "bus0": original_bus,
                "bus1": aux_bus,
                "model": "t",
                "x": TRANSFORMER_X_PU,
                "r": TRANSFORMER_R_PU,
                "g": 0.0,
                "b": 0.0,
                "s_nom": base_capacity * TRANSFORMER_CAPACITY_MULTIPLIER,
                "s_max_pu": 1.0,
                "tap_ratio": 1.0,
                "phase_shift": 0.0,
                "active": True,
                "transformer_parameter_source": (
                    "voltage_layer_interface_assumption; "
                    f"s_nom=sum_mismatched_line_s_nom_x{TRANSFORMER_CAPACITY_MULTIPLIER:g}; "
                    f"r={TRANSFORMER_R_PU:g}_pu; x={TRANSFORMER_X_PU:g}_pu"
                ),
            }
        )

    transformers = pd.DataFrame(transformer_rows)
    endpoint_changes = pd.DataFrame(endpoint_rows)

    buses.to_csv(OUTPUT_NETWORK / "buses.csv", index=False)
    lines.to_csv(OUTPUT_NETWORK / "lines.csv", index=False)
    transformers.to_csv(OUTPUT_NETWORK / "transformers.csv", index=False)
    endpoint_changes.to_csv(OUTPUT_NETWORK / "voltage_layer_endpoint_reassignments.csv", index=False)
    created_buses.to_csv(OUTPUT_NETWORK / "voltage_layer_auxiliary_buses.csv", index=False)

    for filename in [
        "bus_flood_curve_assignments_f21_f23.csv",
        "bus_tc_wind_curve_assignments_w23.csv",
        "bus_county_population_load_weights.csv",
    ]:
        add_auxiliary_assignment_rows(filename, created_buses)

    readme = f"""# Voltage-Layer Transformer Network

This network is copied from:

`{BASE_NETWORK}`

It adds explicit PyPSA transformer components for line endpoints where the line
voltage does not match the bus voltage.

Method:

1. For each mismatched line endpoint, create an auxiliary voltage-layer bus at
   the original bus coordinates.
2. Reconnect the line endpoint to that auxiliary bus.
3. Add a transformer from the original bus to the auxiliary bus.

Transformer assumptions:

- `r = {TRANSFORMER_R_PU}` pu
- `x = {TRANSFORMER_X_PU}` pu
- `s_nom = {TRANSFORMER_CAPACITY_MULTIPLIER} * sum(original mismatched line s_nom)`

The transformer capacity multiplier is intentionally high enough to avoid
turning missing transformer ratings into a new artificial bottleneck during
the first validation pass.

Outputs:

- `transformers.csv`
- `voltage_layer_endpoint_reassignments.csv`
- `voltage_layer_auxiliary_buses.csv`
"""
    (OUTPUT_NETWORK / "VOLTAGE_LAYER_TRANSFORMER_README.md").write_text(readme, encoding="utf-8")

    print("Saved voltage-layer transformer network:", OUTPUT_NETWORK)
    print("Auxiliary buses:", len(created_buses))
    print("Transformers:", len(transformers))
    print("Line endpoint reassignments:", len(endpoint_changes))
    if not transformers.empty:
        print(transformers["s_nom"].describe().to_string())


if __name__ == "__main__":
    main()
