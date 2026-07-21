"""Create a targeted cleanup network for remaining summer peak artifacts.

Starting point:
    pypsa_florida_network_county_load_local_adjusted

The x3 summer peak diagnosis showed that the remaining no-hazard load shedding
is concentrated at bus_751, bus_918, and bus_311. This script creates a new
network variant that keeps the improved county load allocation and generator
overrides, then applies only targeted corrections:

1. Move all remaining load assigned to bus_311, a 69 kV tap, to nearby stronger
   Marion County buses.
2. Deactivate duplicate low-voltage 69 kV bus_918-bus_9 paths where a 230 kV
   path between the same buses exists.
3. Deactivate the bus_311 69 kV tap lines after its load is moved.

This does not change the official line multiplier. It is a targeted topology
and load-allocation diagnostic network.
"""

from __future__ import annotations

import shutil
from pathlib import Path

import pandas as pd


PROJECT_DIR = Path(__file__).resolve().parents[2]
BASE_NETWORK = PROJECT_DIR / "data" / "Electricity" / "pypsa_florida_network_county_load_local_adjusted"
OUTPUT_NETWORK = PROJECT_DIR / "data" / "Electricity" / "pypsa_florida_network_targeted_topology_cleanup"

SOURCE_BUS = "bus_311"
RECIPIENTS = {
    "bus_672": 0.40,   # 230 kV, strong Marion County bus
    "bus_2097": 0.25,  # nearby 230 kV TAP151445
    "bus_557": 0.20,   # 230 kV, strong Marion County bus
    "bus_612": 0.15,   # 230 kV Ross Prairie
}

DEACTIVATED_LINES = {
    "line_1502": "duplicate 69 kV bus_918-bus_9 path; 230 kV line_2499 also connects these buses",
    "line_2705": "duplicate 69 kV bus_9-bus_918 path; 230 kV line_2499 also connects these buses",
    "line_469": "69 kV bus_311 tap line after bus_311 load moved",
    "line_2792": "69 kV bus_751-bus_311 tap line after bus_311 load moved",
}


def copy_network() -> None:
    if OUTPUT_NETWORK.exists():
        shutil.rmtree(OUTPUT_NETWORK)
    OUTPUT_NETWORK.mkdir(parents=True)
    for path in BASE_NETWORK.iterdir():
        if path.is_file():
            shutil.copy2(path, OUTPUT_NETWORK / path.name)


def move_bus_311_load() -> pd.DataFrame:
    loads = pd.read_csv(OUTPUT_NETWORK / "loads.csv")
    loads_p = pd.read_csv(OUTPUT_NETWORK / "loads-p_set.csv")
    load_by_bus = loads.set_index("bus")["name"].to_dict()

    source_load = load_by_bus[SOURCE_BUS]
    source_profile = loads_p[source_load].copy()
    loads_p[source_load] = 0.0

    rows = []
    for recipient_bus, weight in RECIPIENTS.items():
        recipient_load = load_by_bus[recipient_bus]
        transfer = source_profile * weight
        loads_p[recipient_load] = loads_p[recipient_load] + transfer
        rows.append(
            {
                "source_bus": SOURCE_BUS,
                "source_load": source_load,
                "recipient_bus": recipient_bus,
                "recipient_load": recipient_load,
                "recipient_weight": weight,
                "transferred_peak_mw": float(transfer.max()),
                "transferred_total_mwh_2025": float(transfer.sum()),
            }
        )

    loads_p.to_csv(OUTPUT_NETWORK / "loads-p_set.csv", index=False)
    return pd.DataFrame(rows)


def deactivate_target_lines() -> pd.DataFrame:
    lines = pd.read_csv(OUTPUT_NETWORK / "lines.csv")
    if "active" not in lines.columns:
        lines["active"] = True
    lines["active"] = lines["active"].astype(bool)

    rows = []
    for line_name, reason in DEACTIVATED_LINES.items():
        mask = lines["name"].eq(line_name)
        if not mask.any():
            rows.append({"line": line_name, "found": False, "reason": reason})
            continue
        line = lines.loc[mask].iloc[0]
        lines.loc[mask, "active"] = False
        rows.append(
            {
                "line": line_name,
                "found": True,
                "bus0": line["bus0"],
                "bus1": line["bus1"],
                "v_nom": line["v_nom"],
                "s_nom": line["s_nom"],
                "reason": reason,
            }
        )

    lines.to_csv(OUTPUT_NETWORK / "lines.csv", index=False)
    return pd.DataFrame(rows)


def main() -> None:
    copy_network()
    load_adjustments = move_bus_311_load()
    line_changes = deactivate_target_lines()

    load_adjustments.to_csv(OUTPUT_NETWORK / "targeted_bus_311_load_move.csv", index=False)
    line_changes.to_csv(OUTPUT_NETWORK / "targeted_low_voltage_line_cleanup.csv", index=False)

    readme = f"""# Targeted Topology Cleanup Network

This network is copied from:

`{BASE_NETWORK}`

It preserves the county-population load allocation, generator-bus overrides,
boundary import setup, and previous local load redistribution.

Additional targeted changes:

1. All remaining load at `{SOURCE_BUS}` is moved to nearby stronger Marion County buses:
   {', '.join(f'{bus} ({weight:.0%})' for bus, weight in RECIPIENTS.items())}.
2. Duplicate low-voltage paths around `bus_918`/`bus_9` are deactivated where a
   230 kV path already exists.
3. The now-loadless `bus_311` 69 kV tap path is deactivated.

This is a diagnostic cleanup network. It does not change the official global
line-capacity multiplier.

Files:

- `targeted_bus_311_load_move.csv`
- `targeted_low_voltage_line_cleanup.csv`
- updated `loads-p_set.csv`
- updated `lines.csv`
"""
    (OUTPUT_NETWORK / "TARGETED_TOPOLOGY_CLEANUP_README.md").write_text(readme, encoding="utf-8")

    print("Saved targeted cleanup network:", OUTPUT_NETWORK)
    print("Load moved:")
    print(load_adjustments.to_string(index=False))
    print("Lines deactivated:")
    print(line_changes.to_string(index=False))


if __name__ == "__main__":
    main()
