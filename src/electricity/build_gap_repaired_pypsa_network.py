"""
Build a conservative gap-repaired copy of the Florida PyPSA network.

Unlike the crossing-split repair, this script only targets small disconnected
components that sit very close to the main network. It adds short documented
synthetic connector lines from those components to the nearest main-grid bus.

Purpose:
    Test whether obvious endpoint/substation near-miss gaps reduce artificial
    islands without making the aggressive assumption that all overhead line
    crossings are electrically connected.
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import geopandas as gpd
import networkx as nx
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

from convert_florida_assets_to_pypsa import (
    S_NOM_BY_VOLTAGE_MVA,
    line_electrical_parameters,
    standard_voltage,
)


PROJECT_DIR = Path(r"C:\oxford_tc_project")
ELECTRICITY_DIR = PROJECT_DIR / "data" / "Electricity"
BASE_NETWORK_DIR = ELECTRICITY_DIR / "pypsa_florida_network"
DEFAULT_OUTPUT_DIR = ELECTRICITY_DIR / "pypsa_florida_network_gap_repaired"
PROJECTED_CRS = "EPSG:3086"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create conservative gap-repaired PyPSA CSVs.")
    parser.add_argument("--base-network-dir", type=Path, default=BASE_NETWORK_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--max-gap-m",
        type=float,
        default=1500.0,
        help="Only connect disconnected components whose nearest bus is within this distance.",
    )
    parser.add_argument(
        "--same-voltage-only",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="If true, only bridge components to a main bus with the same nominal voltage.",
    )
    return parser.parse_args()


def copy_network_folder(base_dir: Path, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for path in base_dir.glob("*.csv"):
        shutil.copy2(path, output_dir / path.name)


def component_map(buses: pd.DataFrame, lines: pd.DataFrame) -> tuple[list[set[str]], dict[str, int]]:
    graph = nx.Graph()
    graph.add_nodes_from(buses["name"].astype(str))
    graph.add_edges_from(zip(lines["bus0"].astype(str), lines["bus1"].astype(str)))
    components = sorted(nx.connected_components(graph), key=len, reverse=True)
    bus_to_component = {bus: idx for idx, component in enumerate(components) for bus in component}
    return components, bus_to_component


def build_connectors(
    buses: pd.DataFrame,
    lines: pd.DataFrame,
    max_gap_m: float,
    same_voltage_only: bool,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    components, bus_to_component = component_map(buses, lines)
    buses_geo = gpd.GeoDataFrame(
        buses.copy(),
        geometry=gpd.points_from_xy(buses["x"], buses["y"]),
        crs="EPSG:4326",
    ).to_crs(PROJECTED_CRS)
    buses_geo["component_id"] = buses_geo["name"].map(bus_to_component)
    main = buses_geo[buses_geo["component_id"] == 0].copy()

    connector_rows = []
    report_rows = []

    for component_id, component in enumerate(components[1:], start=1):
        island = buses_geo[buses_geo["component_id"] == component_id].copy()
        if island.empty:
            continue

        candidate_main = main
        if same_voltage_only:
            island_voltages = set(pd.to_numeric(island["v_nom"], errors="coerce").dropna())
            candidate_main = main[pd.to_numeric(main["v_nom"], errors="coerce").isin(island_voltages)]
            if candidate_main.empty:
                continue

        tree = cKDTree(np.c_[candidate_main.geometry.x, candidate_main.geometry.y])
        distances, indices = tree.query(np.c_[island.geometry.x, island.geometry.y], k=1)
        best_pos = int(np.argmin(distances))
        distance_m = float(distances[best_pos])
        if distance_m > max_gap_m:
            continue

        island_bus = island.iloc[best_pos]
        main_bus = candidate_main.iloc[int(indices[best_pos])]
        voltage = standard_voltage(max(float(island_bus["v_nom"]), float(main_bus["v_nom"])))
        length_km = max(distance_m / 1000.0, 0.001)
        params = line_electrical_parameters(voltage, length_km)
        s_nom = float(S_NOM_BY_VOLTAGE_MVA.get(voltage, S_NOM_BY_VOLTAGE_MVA[230]))
        line_name = f"topology_gap_connector_component_{component_id}"

        connector_rows.append(
            {
                "name": line_name,
                "bus0": str(island_bus["name"]),
                "bus1": str(main_bus["name"]),
                "carrier": "AC",
                "length": length_km,
                "v_nom": voltage,
                "s_nom": s_nom,
                "source_edge_id": -1,
                "slr_amps": pd.NA,
                "capacity_source": "topology_gap_repair_voltage_placeholder",
                "owner": "synthetic_topology_repair",
                "status": "synthetic",
                **params,
            }
        )
        report_rows.append(
            {
                "component_id": component_id,
                "component_bus_count": len(component),
                "island_bus": island_bus["name"],
                "main_bus": main_bus["name"],
                "distance_m": distance_m,
                "island_bus_v_nom": island_bus["v_nom"],
                "main_bus_v_nom": main_bus["v_nom"],
                "connector_v_nom": voltage,
                "connector_s_nom_mva": s_nom,
                "connector_line": line_name,
            }
        )

    return pd.DataFrame(connector_rows), pd.DataFrame(report_rows)


def summarize_components(buses: pd.DataFrame, lines: pd.DataFrame) -> pd.DataFrame:
    components, _bus_to_component = component_map(buses, lines)
    return pd.DataFrame(
        [{"component_id": idx, "bus_count": len(component)} for idx, component in enumerate(components)]
    )


def main() -> None:
    args = parse_args()
    copy_network_folder(args.base_network_dir, args.output_dir)

    buses = pd.read_csv(args.base_network_dir / "buses.csv")
    lines = pd.read_csv(args.base_network_dir / "lines.csv")
    before = summarize_components(buses, lines)

    connectors, report = build_connectors(
        buses,
        lines,
        max_gap_m=args.max_gap_m,
        same_voltage_only=args.same_voltage_only,
    )
    repaired_lines = pd.concat([lines, connectors], ignore_index=True)
    after = summarize_components(buses, repaired_lines)

    repaired_lines.to_csv(args.output_dir / "lines.csv", index=False)
    before.to_csv(args.output_dir / "topology_components_before_gap_repair.csv", index=False)
    after.to_csv(args.output_dir / "topology_components_after_gap_repair.csv", index=False)
    report.to_csv(args.output_dir / "topology_gap_connectors.csv", index=False)

    summary = pd.DataFrame(
        [
            {
                "base_buses": len(buses),
                "base_lines": len(lines),
                "connectors_added": len(connectors),
                "repaired_lines": len(repaired_lines),
                "components_before": len(before),
                "components_after": len(after),
                "largest_component_before": int(before["bus_count"].max()),
                "largest_component_after": int(after["bus_count"].max()),
                "max_gap_m": args.max_gap_m,
                "same_voltage_only": args.same_voltage_only,
            }
        ]
    )
    summary.to_csv(args.output_dir / "topology_gap_repair_summary.csv", index=False)
    print(summary.to_string(index=False))
    print("Saved gap-repaired network:", args.output_dir)


if __name__ == "__main__":
    main()
