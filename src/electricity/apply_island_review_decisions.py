"""
Apply manual QGIS/literature review decisions for non-main Florida grid islands.

The script creates a new PyPSA CSV network folder from the reviewed base network:
  - components marked "include" are retained as they are, even when they remain
    separate islands;
  - components marked "disregard" are removed with their island buses, incident
    lines, loads, generators, and matching time-series columns.

The input decisions were mapped from the 2026-07-20 notes PDF source HIFLD line
IDs to PyPSA connected component IDs. Do not assume PDF note order is the same
as component order.
"""

from __future__ import annotations

import math
import shutil
from pathlib import Path

import geopandas as gpd
import networkx as nx
import numpy as np
import pandas as pd
import pypsa
from shapely.geometry import Point


PROJECT_DIR = Path(r"C:\oxford_tc_project")
ELECTRICITY_DIR = PROJECT_DIR / "data" / "Electricity"
INPUT_NETWORK = ELECTRICITY_DIR / "pypsa_florida_network_bus311_load_only_cleanup"
OUTPUT_NETWORK = ELECTRICITY_DIR / "pypsa_florida_network_island_reviewed_no_connectors"
DIAGNOSTICS_DIR = INPUT_NETWORK / "topology_diagnostics"
ISLAND_BUSES = DIAGNOSTICS_DIR / "island_buses_for_qgis.csv"
PROJECTED_CRS = "EPSG:3086"
AC_FREQUENCY_HZ = 60.0

S_NOM_BY_VOLTAGE_MVA = {
    69: 100,
    100: 150,
    115: 200,
    138: 250,
    161: 300,
    230: 600,
    345: 1200,
    500: 2500,
    765: 4000,
}

LINE_PARAMETER_ASSUMPTIONS = {
    69: {"r_ohm_per_km": 0.249, "x_ohm_per_km": 0.400, "c_nf_per_km": 9.0},
    100: {"r_ohm_per_km": 0.200, "x_ohm_per_km": 0.390, "c_nf_per_km": 9.5},
    115: {"r_ohm_per_km": 0.153, "x_ohm_per_km": 0.380, "c_nf_per_km": 10.0},
    138: {"r_ohm_per_km": 0.120, "x_ohm_per_km": 0.370, "c_nf_per_km": 10.5},
    161: {"r_ohm_per_km": 0.100, "x_ohm_per_km": 0.360, "c_nf_per_km": 11.0},
    230: {"r_ohm_per_km": 0.080, "x_ohm_per_km": 0.330, "c_nf_per_km": 12.0},
    345: {"r_ohm_per_km": 0.050, "x_ohm_per_km": 0.300, "c_nf_per_km": 13.0},
    500: {"r_ohm_per_km": 0.030, "x_ohm_per_km": 0.280, "c_nf_per_km": 14.0},
    765: {"r_ohm_per_km": 0.020, "x_ohm_per_km": 0.270, "c_nf_per_km": 15.0},
}


ISLAND_DECISIONS = {
    1: {
        "decision": "include",
        "note": "PDF include-together IDs 141747, 124233, 148904, 142664, 113118, 111325, 143155, 102188, 156193, 118622, 151074, 144236, 133503. Gulf Power / appears connected / part of electrical grid.",
    },
    2: {
        "decision": "include",
        "note": "PDF include-together IDs 170779, 167487, 170780, 170781, 128479. Duke Energy Florida service area; include.",
    },
    3: {
        "decision": "disregard",
        "note": "PDF disregard IDs 123386, 112011, 112424. Disregard this isolated P L Bartow/private-looking plant island per review.",
    },
    4: {
        "decision": "include",
        "note": "PDF include-together IDs 140712, 131923, 138954. Include Duke Energy Florida island.",
    },
    5: {
        "decision": "include",
        "note": "PDF include-together IDs 170557, 170556. Include Duke Energy Florida island.",
    },
    6: {
        "decision": "disregard",
        "note": "PDF ID 120309. Disregard after QGIS review; do not add the long Kaley/Lake Gum connector.",
    },
    7: {
        "decision": "include",
        "note": "PDF include ID 124727. Include Ocala/airport 100 kV Duke Energy Florida island.",
    },
    8: {
        "decision": "disregard",
        "note": "PDF disregard ID 131867. Completely isolated; appears only for Swift Creek chemical company.",
    },
    9: {
        "decision": "include",
        "note": "PDF include ID 144991. Include Georgia Power 69 kV island.",
    },
    10: {
        "decision": "disregard",
        "note": "PDF disregard ID 149097. Not connected to rest of grid visually.",
    },
    11: {
        "decision": "include",
        "note": "PDF include ID 170772. Include JEA/Baptist Medical Center island.",
    },
    12: {
        "decision": "disregard",
        "note": "PDF disregard ID 129742. This HIFLD source edge was dropped as a self-loop; remove associated one-bus W E SWOOPE island.",
    },
}


def standard_voltage(voltage_kv: float) -> int:
    if pd.isna(voltage_kv) or voltage_kv <= 0:
        return 230
    classes = np.array(sorted(S_NOM_BY_VOLTAGE_MVA.keys()), dtype=float)
    return int(classes[np.argmin(np.abs(classes - float(voltage_kv)))])


def line_electrical_parameters(voltage: int, length_km: float) -> dict[str, float | int | str]:
    params = LINE_PARAMETER_ASSUMPTIONS[voltage]
    capacitance_f = params["c_nf_per_km"] * 1e-9 * length_km
    return {
        "r": params["r_ohm_per_km"] * length_km,
        "x": params["x_ohm_per_km"] * length_km,
        "b": 2 * math.pi * AC_FREQUENCY_HZ * capacitance_f,
        "g": 0.0,
        "num_parallel": 1,
        "r_ohm_per_km_assumed": params["r_ohm_per_km"],
        "x_ohm_per_km_assumed": params["x_ohm_per_km"],
        "c_nf_per_km_assumed": params["c_nf_per_km"],
        "line_parameter_source": "island_review_voltage_class_overhead_assumption",
    }


def graph_components(buses: pd.DataFrame, lines: pd.DataFrame) -> dict[str, int]:
    graph = nx.Graph()
    graph.add_nodes_from(buses["name"].astype(str))
    graph.add_edges_from(zip(lines["bus0"].astype(str), lines["bus1"].astype(str)))
    components = sorted(nx.connected_components(graph), key=len, reverse=True)
    mapping = {}
    for component_id, component in enumerate(components):
        for bus in component:
            mapping[bus] = component_id
    return mapping


def nearest_connector(
    island_id: int,
    island_buses: pd.DataFrame,
    main_buses: pd.DataFrame,
) -> dict:
    island_geo = gpd.GeoDataFrame(
        island_buses.copy(),
        geometry=[Point(xy) for xy in zip(island_buses["x"], island_buses["y"])],
        crs="EPSG:4326",
    ).to_crs(PROJECTED_CRS)
    main_geo_all = gpd.GeoDataFrame(
        main_buses.copy(),
        geometry=[Point(xy) for xy in zip(main_buses["x"], main_buses["y"])],
        crs="EPSG:4326",
    ).to_crs(PROJECTED_CRS)

    best = None
    for island_row in island_geo.itertuples():
        same_voltage = main_geo_all[
            np.isclose(
                pd.to_numeric(main_geo_all["v_nom"], errors="coerce"),
                float(island_row.v_nom),
                equal_nan=False,
            )
        ]
        candidates = same_voltage if not same_voltage.empty else main_geo_all
        distances = candidates.geometry.distance(island_row.geometry)
        idx = distances.idxmin()
        main_row = candidates.loc[idx]
        distance_m = float(distances.loc[idx])
        if best is None or distance_m < best["length_m"]:
            best = {
                "review_component_id": island_id,
                "bus0": str(island_row.name),
                "bus1": str(main_row["name"]),
                "bus0_v_nom": float(island_row.v_nom),
                "bus1_v_nom": float(main_row["v_nom"]),
                "bus0_substation_name": getattr(island_row, "substation_name", ""),
                "bus1_substation_name": main_row.get("substation_name", ""),
                "length_m": distance_m,
                "length_km": distance_m / 1000.0,
                "same_voltage_preferred": not same_voltage.empty,
            }
    if best is None:
        raise ValueError(f"No connector found for island component {island_id}")
    voltage = standard_voltage(max(best["bus0_v_nom"], best["bus1_v_nom"]))
    best["v_nom"] = voltage
    best["s_nom"] = float(S_NOM_BY_VOLTAGE_MVA[voltage])
    best.update(line_electrical_parameters(voltage, best["length_km"]))
    return best


def drop_timeseries_columns(path: Path, columns_to_drop: set[str]) -> None:
    if not path.exists() or not columns_to_drop:
        return
    df = pd.read_csv(path)
    keep = [column for column in df.columns if column not in columns_to_drop]
    df[keep].to_csv(path, index=False)


def filter_csv_rows(path: Path, removed_buses: set[str], removed_lines: set[str], removed_generators: set[str]) -> None:
    if not path.exists():
        return
    df = pd.read_csv(path)
    mask = pd.Series(True, index=df.index)
    for column in ["name", "generator"]:
        if column in df.columns:
            mask &= ~df[column].astype(str).isin(removed_generators)
    for column in ["bus", "assigned_bus", "nearest_bus"]:
        if column in df.columns:
            mask &= ~df[column].astype(str).isin(removed_buses)
    for column in ["line", "name"]:
        if column in df.columns:
            mask &= ~df[column].astype(str).isin(removed_lines)
    df.loc[mask].to_csv(path, index=False)


def write_snapshots_from_load_timeseries(network_dir: Path) -> None:
    loads_p_set_path = network_dir / "loads-p_set.csv"
    if not loads_p_set_path.exists():
        return
    snapshots = pd.read_csv(loads_p_set_path, usecols=["snapshot"])
    snapshots.rename(columns={"snapshot": "name"}).to_csv(network_dir / "snapshots.csv", index=False)


def export_checked_netcdf(network_dir: Path) -> None:
    output_path = network_dir / "florida_network.nc"
    output_path.unlink(missing_ok=True)
    network = pypsa.Network()
    network.import_from_csv_folder(str(network_dir))

    # NetCDF cannot infer mixed object dtypes in custom metadata columns.
    for df in [network.buses, network.lines, network.generators, network.loads, network.carriers]:
        for column in df.columns:
            if df[column].dtype == object:
                df[column] = df[column].where(pd.notna(df[column]), "").astype(str)

    network.export_to_netcdf(output_path)
    check = pypsa.Network(str(output_path))
    if len(check.buses) != len(network.buses) or len(check.lines) != len(network.lines):
        raise RuntimeError(f"NetCDF export check failed for {output_path}")


def main() -> None:
    input_abs = INPUT_NETWORK.resolve()
    output_abs = OUTPUT_NETWORK.resolve()
    project_abs = PROJECT_DIR.resolve()
    if project_abs not in output_abs.parents:
        raise ValueError(f"Refusing to write outside project: {output_abs}")

    if OUTPUT_NETWORK.exists():
        shutil.rmtree(OUTPUT_NETWORK)
    shutil.copytree(INPUT_NETWORK, OUTPUT_NETWORK)

    buses = pd.read_csv(OUTPUT_NETWORK / "buses.csv")
    lines = pd.read_csv(OUTPUT_NETWORK / "lines.csv")
    generators = pd.read_csv(OUTPUT_NETWORK / "generators.csv")
    loads = pd.read_csv(OUTPUT_NETWORK / "loads.csv")
    island_buses = pd.read_csv(ISLAND_BUSES)

    component_by_bus = graph_components(buses, lines)
    buses["base_component_id"] = buses["name"].map(component_by_bus)
    main_component = int(buses["base_component_id"].value_counts().idxmax())
    main_buses = buses[buses["base_component_id"].eq(main_component)].copy()

    review_component_by_bus = island_buses.set_index("bus")["component_id"].astype(int).to_dict()
    buses["review_island_component_id"] = buses["name"].map(review_component_by_bus)

    remove_components = {
        component_id
        for component_id, decision in ISLAND_DECISIONS.items()
        if decision["decision"] == "disregard"
    }
    include_components = {
        component_id
        for component_id, decision in ISLAND_DECISIONS.items()
        if decision["decision"] == "include"
    }

    removed_buses = set(
        buses.loc[buses["review_island_component_id"].isin(remove_components), "name"].astype(str)
    )
    removed_lines = set(
        lines.loc[
            lines["bus0"].astype(str).isin(removed_buses)
            | lines["bus1"].astype(str).isin(removed_buses),
            "name",
        ].astype(str)
    )
    removed_loads = set(loads.loc[loads["bus"].astype(str).isin(removed_buses), "name"].astype(str))
    removed_generators = set(
        generators.loc[generators["bus"].astype(str).isin(removed_buses), "name"].astype(str)
    )

    connector_rows = []

    buses_out = buses[~buses["name"].astype(str).isin(removed_buses)].drop(
        columns=["base_component_id", "review_island_component_id"],
        errors="ignore",
    )
    lines_out = lines[~lines["name"].astype(str).isin(removed_lines)].copy()
    generators_out = generators[~generators["name"].astype(str).isin(removed_generators)]
    loads_out = loads[~loads["name"].astype(str).isin(removed_loads)]

    buses_out.to_csv(OUTPUT_NETWORK / "buses.csv", index=False)
    lines_out.to_csv(OUTPUT_NETWORK / "lines.csv", index=False)
    generators_out.to_csv(OUTPUT_NETWORK / "generators.csv", index=False)
    loads_out.to_csv(OUTPUT_NETWORK / "loads.csv", index=False)

    # Keep common time-series files consistent with removed static components.
    drop_timeseries_columns(OUTPUT_NETWORK / "loads-p_set.csv", removed_loads)
    drop_timeseries_columns(OUTPUT_NETWORK / "generators-p_max_pu.csv", removed_generators)
    for aux_file in [
        "generators_with_component_marginal_costs.csv",
        "generators_with_final_marginal_costs.csv",
        "generator_bus_matches_review.csv",
        "applied_generator_bus_overrides.csv",
        "generator_flood_curve_assignments_f11_f14.csv",
        "generator_tc_wind_curve_assignments_w110_w114.csv",
        "bus_flood_curve_assignments_f21_f23.csv",
        "bus_tc_wind_curve_assignments_w23.csv",
        "line_flood_curve_applicability_f51_f61_f62.csv",
        "line_tc_wind_curve_assignments_w63.csv",
    ]:
        filter_csv_rows(OUTPUT_NETWORK / aux_file, removed_buses, removed_lines, removed_generators)
    write_snapshots_from_load_timeseries(OUTPUT_NETWORK)

    pd.DataFrame(connector_rows).to_csv(OUTPUT_NETWORK / "island_review_added_connectors.csv", index=False)
    removal_report = pd.DataFrame(
        [
            {
                "review_component_id": component_id,
                "decision": decision["decision"],
                "note": decision["note"],
                "buses_removed": int(
                    buses["review_island_component_id"].eq(component_id)
                    .where(buses["name"].isin(removed_buses), False)
                    .sum()
                ),
                "buses_in_component": int(buses["review_island_component_id"].eq(component_id).sum()),
            }
            for component_id, decision in sorted(ISLAND_DECISIONS.items())
        ]
    )
    removal_report.to_csv(OUTPUT_NETWORK / "island_review_decision_report.csv", index=False)

    after_components = graph_components(buses_out, lines_out)
    after_counts = pd.Series(after_components).value_counts().sort_values(ascending=False)
    summary = pd.DataFrame(
        [
            {
                "input_network": str(input_abs),
                "output_network": str(output_abs),
                "decision_mapping": "PDF HIFLD/source line IDs explicitly mapped to QGIS/PyPSA island component IDs",
                "removed_components": ";".join(map(str, sorted(remove_components))),
                "included_retained_components": ";".join(map(str, sorted(include_components))),
                "buses_before": len(buses),
                "buses_after": len(buses_out),
                "lines_before": len(lines),
                "lines_after": len(lines_out),
                "generators_before": len(generators),
                "generators_after": len(generators_out),
                "loads_before": len(loads),
                "loads_after": len(loads_out),
                "connectors_added": 0,
                "connected_components_after": int(len(after_counts)),
                "largest_component_buses_after": int(after_counts.iloc[0]),
            }
        ]
    )
    summary.to_csv(OUTPUT_NETWORK / "island_review_network_summary.csv", index=False)

    readme = f"""# Island-Reviewed Florida PyPSA Network

Created by `data/Electricity/apply_island_review_decisions.py`.

Input network:
`{input_abs}`

Output network:
`{output_abs}`

Manual island decisions came from the 2026-07-20 notes PDF. The PDF listed
HIFLD/source line IDs, which were explicitly mapped to PyPSA island component
IDs before this network was created.

Removed/disregarded components:
`{'; '.join(map(str, sorted(remove_components)))}`

Included components retained without artificial connector lines:
`{'; '.join(map(str, sorted(include_components)))}`

Review outputs:
- `island_review_decision_report.csv`
- `island_review_network_summary.csv`

Connector method:
No new manual connector lines were added. Included islands remain in the final
dataset with their original topology, even if they are separate components.
"""
    (OUTPUT_NETWORK / "ISLAND_REVIEWED_NETWORK_README.md").write_text(readme, encoding="utf-8")
    export_checked_netcdf(OUTPUT_NETWORK)

    print(summary.to_string(index=False))
    print("Added connectors:", 0)
    print("Removed buses:", len(removed_buses))
    print("Removed lines:", len(removed_lines))
    print("Output:", OUTPUT_NETWORK)


if __name__ == "__main__":
    main()
