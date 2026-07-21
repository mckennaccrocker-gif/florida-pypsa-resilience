"""
Diagnose topology and islanding in the Florida PyPSA network.

This script helps calibrate the baseline before hazard scenarios by reporting
connected components, load/generation balance by island, import-slack access,
and where baseline load shedding occurs.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import networkx as nx
import pandas as pd
import pypsa


PROJECT_DIR = Path(r"C:\oxford_tc_project")
PYPSA_DIR = PROJECT_DIR / "data" / "Electricity" / "pypsa_florida_network"
DEFAULT_NETWORK_FILE = PYPSA_DIR / "florida_network.nc"
DEFAULT_RESULTS_DIR = PYPSA_DIR / "load_shedding_results_import10"
DEFAULT_OUTPUT_DIR = PYPSA_DIR / "topology_diagnostics"
DISPATCH_TOLERANCE = 1e-6


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Diagnose connected components and baseline load shedding."
    )
    parser.add_argument("--network", type=Path, default=DEFAULT_NETWORK_FILE)
    parser.add_argument("--dispatch-results-dir", type=Path, default=DEFAULT_RESULTS_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def connected_components(network: pypsa.Network) -> tuple[pd.Series, pd.DataFrame]:
    graph = nx.Graph()
    graph.add_nodes_from(network.buses.index)

    active_lines = network.lines[network.lines["active"].astype(bool)].copy()
    for line in active_lines.itertuples():
        graph.add_edge(line.bus0, line.bus1, line=line.Index)

    components = sorted(nx.connected_components(graph), key=len, reverse=True)
    bus_to_component = {}
    records = []
    for component_id, buses in enumerate(components):
        for bus in buses:
            bus_to_component[bus] = component_id
        records.append(
            {
                "component_id": component_id,
                "bus_count": len(buses),
            }
        )
    return pd.Series(bus_to_component, name="component_id"), pd.DataFrame(records)


def bus_peak_load(network: pypsa.Network) -> pd.Series:
    load_bus = network.loads["bus"]
    if network.loads_t.p_set.empty:
        load_p_set = pd.DataFrame([network.loads["p_set"]], columns=network.loads.index)
    else:
        load_p_set = network.loads_t.p_set.reindex(columns=network.loads.index, fill_value=0.0)
    return load_p_set.T.groupby(load_bus).sum().T.max(axis=0)


def read_optional_csv(path: Path) -> pd.DataFrame:
    if path.exists():
        return pd.read_csv(path)
    return pd.DataFrame()


def build_bus_diagnostics(
    network: pypsa.Network,
    bus_components: pd.Series,
    results_dir: Path,
) -> pd.DataFrame:
    buses = network.buses.copy()
    buses["component_id"] = buses.index.map(bus_components)
    buses["peak_load_mw"] = buses.index.map(bus_peak_load(network)).fillna(0.0)
    buses["generator_p_nom_mw"] = (
        network.generators.groupby("bus")["p_nom"].sum().reindex(buses.index).fillna(0.0)
    )
    buses["generator_count"] = (
        network.generators.groupby("bus").size().reindex(buses.index).fillna(0).astype(int)
    )

    shedding = read_optional_csv(results_dir / "load_shedding_by_bus.csv")
    if not shedding.empty and "bus" in shedding.columns:
        buses["baseline_load_shed_mwh"] = (
            shedding.set_index("bus")["total_load_shed_mwh"].reindex(buses.index).fillna(0.0)
        )
        buses["baseline_max_hourly_load_shed_mw"] = (
            shedding.set_index("bus")["max_hourly_load_shed_mw"].reindex(buses.index).fillna(0.0)
        )
    else:
        buses["baseline_load_shed_mwh"] = 0.0
        buses["baseline_max_hourly_load_shed_mw"] = 0.0

    imports = read_optional_csv(results_dir / "import_slack_by_bus.csv")
    buses["has_import_slack"] = False
    buses["baseline_import_slack_mwh"] = 0.0
    if not imports.empty and "bus" in imports.columns:
        buses.loc[buses.index.intersection(imports["bus"]), "has_import_slack"] = True
        buses["baseline_import_slack_mwh"] = (
            imports.set_index("bus")["total_import_slack_mwh"].reindex(buses.index).fillna(0.0)
        )

    return buses.reset_index(names="bus")


def build_component_diagnostics(
    network: pypsa.Network,
    bus_diagnostics: pd.DataFrame,
) -> pd.DataFrame:
    line_components = network.lines.copy()
    bus_component = bus_diagnostics.set_index("bus")["component_id"]
    line_components["component_id"] = line_components["bus0"].map(bus_component)

    gen_by_component = (
        network.generators.assign(component_id=network.generators["bus"].map(bus_component))
        .groupby("component_id")
        .agg(
            generator_count=("p_nom", "size"),
            generator_p_nom_mw=("p_nom", "sum"),
        )
    )
    carrier_p_nom = (
        network.generators.assign(component_id=network.generators["bus"].map(bus_component))
        .pivot_table(
            index="component_id",
            columns="carrier",
            values="p_nom",
            aggfunc="sum",
            fill_value=0.0,
        )
        .add_prefix("p_nom_")
    )

    component = (
        bus_diagnostics.groupby("component_id")
        .agg(
            bus_count=("bus", "size"),
            peak_load_mw=("peak_load_mw", "sum"),
            buses_with_load=("peak_load_mw", lambda s: int((s > DISPATCH_TOLERANCE).sum())),
            buses_with_generation=("generator_p_nom_mw", lambda s: int((s > DISPATCH_TOLERANCE).sum())),
            has_import_slack=("has_import_slack", "any"),
            baseline_import_slack_mwh=("baseline_import_slack_mwh", "sum"),
            baseline_load_shed_mwh=("baseline_load_shed_mwh", "sum"),
            buses_with_load_shedding=("baseline_load_shed_mwh", lambda s: int((s > DISPATCH_TOLERANCE).sum())),
        )
        .join(gen_by_component, how="left")
        .join(carrier_p_nom, how="left")
    )
    component["generator_count"] = component["generator_count"].fillna(0).astype(int)
    component["generator_p_nom_mw"] = component["generator_p_nom_mw"].fillna(0.0)
    component["line_count"] = line_components.groupby("component_id").size().reindex(component.index).fillna(0).astype(int)
    component["load_minus_generation_mw"] = component["peak_load_mw"] - component["generator_p_nom_mw"]
    component["has_load_no_generation"] = (
        (component["peak_load_mw"] > DISPATCH_TOLERANCE)
        & (component["generator_p_nom_mw"] <= DISPATCH_TOLERANCE)
        & (~component["has_import_slack"])
    )
    component["has_baseline_shedding"] = component["baseline_load_shed_mwh"] > DISPATCH_TOLERANCE
    return component.reset_index().sort_values(
        ["has_baseline_shedding", "baseline_load_shed_mwh", "peak_load_mw"],
        ascending=[False, False, False],
    )


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    network = pypsa.Network(args.network)
    bus_components, component_counts = connected_components(network)
    bus_diag = build_bus_diagnostics(network, bus_components, args.dispatch_results_dir)
    component_diag = build_component_diagnostics(network, bus_diag)

    bus_diag.to_csv(args.output_dir / "bus_topology_diagnostics.csv", index=False)
    component_diag.to_csv(args.output_dir / "component_topology_diagnostics.csv", index=False)
    component_counts.to_csv(args.output_dir / "connected_component_counts.csv", index=False)

    shedding_components = component_diag[component_diag["has_baseline_shedding"]]
    load_no_gen_components = component_diag[component_diag["has_load_no_generation"]]

    print("Network:", args.network)
    print("Dispatch results:", args.dispatch_results_dir)
    print("Output directory:", args.output_dir)
    print("Buses:", len(network.buses))
    print("Lines:", len(network.lines))
    print("Connected components:", len(component_diag))
    print("Largest component buses:", int(component_diag["bus_count"].max()))
    print("Components with load but no generation/import:", len(load_no_gen_components))
    print("Components with baseline load shedding:", len(shedding_components))
    print("Total baseline load shed MWh:", round(component_diag["baseline_load_shed_mwh"].sum(), 6))

    print("\nTop components by baseline load shedding:")
    cols = [
        "component_id",
        "bus_count",
        "line_count",
        "peak_load_mw",
        "generator_p_nom_mw",
        "has_import_slack",
        "baseline_import_slack_mwh",
        "baseline_load_shed_mwh",
        "buses_with_load_shedding",
    ]
    print(component_diag[cols].head(12).to_string(index=False))

    print("\nSaved:")
    print(args.output_dir / "component_topology_diagnostics.csv")
    print(args.output_dir / "bus_topology_diagnostics.csv")
    print(args.output_dir / "connected_component_counts.csv")


if __name__ == "__main__":
    main()
