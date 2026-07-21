"""
Diagnose no-hazard baseline load shedding in the Florida PyPSA model.

The diagnostic uses the Jan 1, 2025 baseline benchmark by default, then runs a
counterfactual with emergency import slack at every bus. If load shedding
disappears with import-at-every-bus, the remaining baseline load shedding is a
network deliverability/topology problem rather than a statewide capacity problem.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd

from run_florida_pypsa_baseline_validation import (
    DEFAULT_OUTPUT_DIR,
    PYPSA_DIR,
    dispatch_by_carrier,
    load_latest_network,
    save_import_slack,
    save_load_shedding,
    select_snapshots,
    solve_dispatch_in_chunks,
    snapshot_weights_hours,
)
from run_florida_pypsa_load_shedding_dispatch import (
    DISPATCH_TOLERANCE_MW,
    IMPORT_SLACK_MARGINAL_COST_USD_PER_MWH,
    add_import_slack_generators,
    add_standard_load_shedding,
    infer_import_buses,
)


PROJECT_DIR = Path(r"C:\oxford_tc_project")
BASELINE_RESULTS_DIR = PYPSA_DIR / "baseline_validation_ipm_benchmark"
DEFAULT_DIAGNOSTIC_DIR = PYPSA_DIR / "baseline_load_shedding_diagnostics"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Diagnose Florida baseline load shedding.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_DIAGNOSTIC_DIR)
    parser.add_argument("--baseline-results-dir", type=Path, default=BASELINE_RESULTS_DIR)
    parser.add_argument("--start", default="2025-01-01 00:00:00")
    parser.add_argument("--periods", type=int, default=24)
    parser.add_argument("--solver", default="highs")
    parser.add_argument("--highs-method", default="ipm")
    parser.add_argument("--chunk-size", type=int, default=24)
    parser.add_argument("--num-import-buses", type=int, default=10)
    parser.add_argument(
        "--skip-import-every-bus-solve",
        action="store_true",
        help="Only write static diagnostics and skip the import-at-every-bus optimization.",
    )
    return parser.parse_args()


def build_component_table(network) -> tuple[pd.DataFrame, dict[str, int]]:
    graph = nx.Graph()
    graph.add_nodes_from(network.buses.index)
    for line in network.lines[network.lines["active"].astype(bool)].itertuples():
        graph.add_edge(line.bus0, line.bus1)

    components = sorted(nx.connected_components(graph), key=len, reverse=True)
    bus_to_component = {
        bus: component_id
        for component_id, buses in enumerate(components)
        for bus in buses
    }
    component_rows = []
    for component_id, buses in enumerate(components):
        subgraph = graph.subgraph(buses)
        component_rows.append(
            {
                "component_id": component_id,
                "bus_count": len(buses),
                "line_count": subgraph.number_of_edges(),
            }
        )
    return pd.DataFrame(component_rows), bus_to_component


def bus_load(network, snapshots: pd.Index) -> pd.DataFrame:
    loads_t = network.loads_t.p_set.reindex(snapshots)
    weights = snapshot_weights_hours(network, snapshots)
    loads_t_by_bus = loads_t.copy()
    loads_t_by_bus.columns = network.loads.loc[loads_t_by_bus.columns, "bus"].to_numpy()
    loads_t_by_bus = loads_t_by_bus.T.groupby(level=0).sum().T
    return pd.DataFrame(
        {
            "bus": loads_t_by_bus.columns,
            "total_load_mwh": loads_t_by_bus.multiply(weights, axis=0).sum(axis=0).to_numpy(),
            "peak_load_mw": loads_t_by_bus.max(axis=0).to_numpy(),
        }
    )


def generator_availability(network, snapshots: pd.Index) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    original_generators = network.generators.index[
        ~network.generators["carrier"].isin(["import_slack", "load_shedding"])
    ]
    generators = network.generators.loc[original_generators].copy()
    p_nom_by_carrier = (
        generators.groupby("carrier")["p_nom"]
        .sum()
        .rename("p_nom_mw")
        .reset_index()
        .sort_values("p_nom_mw", ascending=False)
    )

    p_max_pu = pd.DataFrame(1.0, index=snapshots, columns=original_generators)
    profile_cols = network.generators_t.p_max_pu.columns.intersection(original_generators)
    if len(profile_cols) > 0:
        p_max_pu.loc[:, profile_cols] = network.generators_t.p_max_pu.reindex(index=snapshots, columns=profile_cols)
    available = p_max_pu.multiply(generators["p_nom"], axis=1)
    available_by_carrier = available.T.groupby(generators["carrier"]).sum().T
    pmax_pu_by_carrier = available_by_carrier.divide(
        generators.groupby("carrier")["p_nom"].sum(),
        axis=1,
    )
    pmax_pu_by_carrier.insert(0, "snapshot", snapshots)
    available_by_carrier.insert(0, "snapshot", snapshots)

    total_demand = network.loads_t.p_set.reindex(snapshots).sum(axis=1)
    capacity_vs_demand = pd.DataFrame(
        {
            "snapshot": snapshots,
            "demand_mw": total_demand.to_numpy(),
            "available_generation_mw_excluding_import_slack": available.drop(columns=[], errors="ignore").sum(axis=1).to_numpy(),
        }
    )
    capacity_vs_demand["available_minus_demand_mw"] = (
        capacity_vs_demand["available_generation_mw_excluding_import_slack"]
        - capacity_vs_demand["demand_mw"]
    )
    return p_nom_by_carrier, pmax_pu_by_carrier, capacity_vs_demand


def haversine_km(lon1, lat1, lon2, lat2) -> np.ndarray:
    radius_km = 6371.0
    lon1 = np.deg2rad(lon1)
    lat1 = np.deg2rad(lat1)
    lon2 = np.deg2rad(lon2)
    lat2 = np.deg2rad(lat2)
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    return 2 * radius_km * np.arcsin(np.sqrt(a))


def nearest_generator_distance(network, buses: pd.Index) -> pd.DataFrame:
    generator_buses = network.generators.loc[
        ~network.generators["carrier"].isin(["import_slack", "load_shedding"])
        & (network.generators["p_nom"] > 0),
        "bus",
    ].dropna().unique()
    gen_coords = network.buses.loc[generator_buses, ["x", "y"]].dropna()
    rows = []
    for bus in buses:
        bus_lon = float(network.buses.at[bus, "x"])
        bus_lat = float(network.buses.at[bus, "y"])
        distances = haversine_km(
            bus_lon,
            bus_lat,
            gen_coords["x"].to_numpy(dtype=float),
            gen_coords["y"].to_numpy(dtype=float),
        )
        nearest_idx = int(np.nanargmin(distances))
        rows.append(
            {
                "bus": bus,
                "nearest_generator_bus": gen_coords.index[nearest_idx],
                "nearest_generator_distance_km": float(distances[nearest_idx]),
            }
        )
    return pd.DataFrame(rows)


def island_summary(
    network,
    snapshots: pd.Index,
    component_table: pd.DataFrame,
    bus_to_component: dict[str, int],
    import_buses: pd.Index,
) -> pd.DataFrame:
    loads = bus_load(network, snapshots)
    loads["component_id"] = loads["bus"].map(bus_to_component)
    load_by_component = loads.groupby("component_id", as_index=False).agg(
        total_load_mwh=("total_load_mwh", "sum"),
        peak_load_mw=("peak_load_mw", "sum"),
    )

    generators = network.generators.loc[
        ~network.generators["carrier"].isin(["import_slack", "load_shedding"])
    ].copy()
    generators["component_id"] = generators["bus"].map(bus_to_component)
    gen_by_component = generators.groupby("component_id", as_index=False).agg(
        generator_count=("p_nom", "size"),
        total_p_nom_mw=("p_nom", "sum"),
    )

    imports = pd.DataFrame({"bus": import_buses})
    imports["component_id"] = imports["bus"].map(bus_to_component)
    import_by_component = imports.groupby("component_id", as_index=False).agg(
        import_slack_buses=("bus", "size")
    )

    summary = component_table.merge(load_by_component, on="component_id", how="left")
    summary = summary.merge(gen_by_component, on="component_id", how="left")
    summary = summary.merge(import_by_component, on="component_id", how="left")
    for col in ["total_load_mwh", "peak_load_mw", "generator_count", "total_p_nom_mw", "import_slack_buses"]:
        summary[col] = summary[col].fillna(0)
    summary["has_import_slack"] = summary["import_slack_buses"] > 0
    summary["p_nom_minus_peak_load_mw"] = summary["total_p_nom_mw"] - summary["peak_load_mw"]
    return summary.sort_values("total_load_mwh", ascending=False)


def top_load_shed_bus_diagnostics(
    network,
    snapshots: pd.Index,
    baseline_results_dir: Path,
    bus_to_component: dict[str, int],
    import_buses: pd.Index,
    output_dir: Path,
) -> pd.DataFrame:
    load_shed_path = baseline_results_dir / "baseline_load_shedding_by_bus.csv"
    if not load_shed_path.exists():
        raise FileNotFoundError(load_shed_path)
    shed = pd.read_csv(load_shed_path).sort_values("total_load_shed_mwh", ascending=False)
    top = shed.head(20).copy()

    loads = bus_load(network, snapshots)
    line_counts = pd.concat([network.lines["bus0"], network.lines["bus1"]]).value_counts().rename("connected_lines")
    nearest = nearest_generator_distance(network, pd.Index(top["bus"]))
    import_components = {bus_to_component[bus] for bus in import_buses if bus in bus_to_component}

    top = top.merge(loads, on="bus", how="left")
    top["component_id"] = top["bus"].map(bus_to_component)
    top["connected_lines"] = top["bus"].map(line_counts).fillna(0).astype(int)
    top = top.merge(nearest, on="bus", how="left")
    top["import_slack_exists_in_island"] = top["component_id"].isin(import_components)
    top.to_csv(output_dir / "top20_load_shedding_bus_diagnostics.csv", index=False)
    return top


def run_import_every_bus_test(
    snapshots: pd.Index,
    args: argparse.Namespace,
    output_dir: Path,
) -> pd.DataFrame:
    network = load_latest_network(PYPSA_DIR)
    import_buses = pd.Index(network.buses.index)
    import_generators = add_import_slack_generators(network, import_buses)
    load_shedding_generators = add_standard_load_shedding(network)
    status, condition = solve_dispatch_in_chunks(
        network,
        snapshots,
        args.solver,
        args.highs_method,
        args.chunk_size,
    )
    dispatch_by_carrier(network, snapshots, output_dir / "import_every_bus")
    load_shedding_hourly, load_shedding_by_bus = save_load_shedding(
        network,
        load_shedding_generators,
        snapshots,
        output_dir / "import_every_bus",
    )
    import_hourly, import_by_bus = save_import_slack(
        network,
        import_generators,
        snapshots,
        output_dir / "import_every_bus",
    )
    summary = pd.DataFrame(
        [
            {
                "solver_status": status,
                "solver_condition": condition,
                "import_slack_buses": len(import_buses),
                "total_import_slack_mwh": float(import_hourly["import_slack_mwh"].sum()),
                "hours_with_import_slack": int((import_hourly["import_slack_mw"] > DISPATCH_TOLERANCE_MW).sum()),
                "total_load_shed_mwh": float(load_shedding_hourly["load_shed_mwh"].sum()),
                "buses_with_load_shedding": int(len(load_shedding_by_bus)),
                "load_shedding_disappears": bool(load_shedding_hourly["load_shed_mwh"].sum() <= DISPATCH_TOLERANCE_MW),
            }
        ]
    )
    summary.to_csv(output_dir / "import_every_bus_test_summary.csv", index=False)
    import_by_bus.to_csv(output_dir / "import_every_bus_by_bus.csv", index=False)
    return summary


def make_plots(
    top_buses: pd.DataFrame,
    islands: pd.DataFrame,
    p_nom_by_carrier: pd.DataFrame,
    capacity_vs_demand: pd.DataFrame,
    output_dir: Path,
) -> None:
    plot_dir = output_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(11, 6))
    top_plot = top_buses.sort_values("total_load_shed_mwh")
    ax.barh(top_plot["bus"], top_plot["total_load_shed_mwh"], color="#c44e52")
    ax.set_xlabel("Load shed on Jan 1 baseline (MWh)")
    ax.set_ylabel("Bus")
    ax.set_title("Top baseline load-shedding buses")
    fig.tight_layout()
    fig.savefig(plot_dir / "top20_load_shedding_buses.png", dpi=220)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(11, 6))
    islands_plot = islands.sort_values("total_load_mwh", ascending=False).head(20)
    x = np.arange(len(islands_plot))
    width = 0.38
    ax.bar(x - width / 2, islands_plot["peak_load_mw"], width, label="Peak load MW", color="#4c78a8")
    ax.bar(x + width / 2, islands_plot["total_p_nom_mw"], width, label="Generator p_nom MW", color="#59a14f")
    ax.set_xticks(x)
    ax.set_xticklabels(islands_plot["component_id"].astype(str), rotation=45, ha="right")
    ax.set_xlabel("Connected component ID")
    ax.set_ylabel("MW")
    ax.set_title("Largest-load islands: load vs installed generation")
    ax.legend()
    fig.tight_layout()
    fig.savefig(plot_dir / "island_load_vs_generation.png", dpi=220)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(11, 6))
    pnom_plot = p_nom_by_carrier.sort_values("p_nom_mw")
    ax.barh(pnom_plot["carrier"], pnom_plot["p_nom_mw"], color="#f2c12e")
    ax.set_xlabel("Installed p_nom (MW)")
    ax.set_title("Generator p_nom by carrier")
    fig.tight_layout()
    fig.savefig(plot_dir / "p_nom_by_carrier.png", dpi=220)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(11, 6))
    ax.plot(capacity_vs_demand["snapshot"], capacity_vs_demand["demand_mw"], label="Demand", color="#4c78a8", linewidth=2)
    ax.plot(
        capacity_vs_demand["snapshot"],
        capacity_vs_demand["available_generation_mw_excluding_import_slack"],
        label="Available generation excl. imports",
        color="#59a14f",
        linewidth=2,
    )
    ax.set_ylabel("MW")
    ax.set_title("Statewide available generation vs demand")
    ax.legend()
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(plot_dir / "available_capacity_vs_demand.png", dpi=220)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "import_every_bus").mkdir(parents=True, exist_ok=True)

    network = load_latest_network(PYPSA_DIR)
    snapshots = select_snapshots(network, args.start, args.periods, all_snapshots=False)
    import_buses = infer_import_buses(network, None, args.num_import_buses)

    component_table, bus_to_component = build_component_table(network)
    components = island_summary(network, snapshots, component_table, bus_to_component, import_buses)
    components.to_csv(args.output_dir / "connected_component_island_summary.csv", index=False)

    top_buses = top_load_shed_bus_diagnostics(
        network,
        snapshots,
        args.baseline_results_dir,
        bus_to_component,
        import_buses,
        args.output_dir,
    )

    p_nom_by_carrier, pmax_pu_by_carrier, capacity_vs_demand = generator_availability(network, snapshots)
    p_nom_by_carrier.to_csv(args.output_dir / "p_nom_by_carrier.csv", index=False)
    pmax_pu_by_carrier.to_csv(args.output_dir / "p_max_pu_by_carrier_jan1.csv", index=False)
    capacity_vs_demand.to_csv(args.output_dir / "available_capacity_vs_demand_jan1.csv", index=False)

    if args.skip_import_every_bus_solve:
        import_every_bus_summary = pd.DataFrame()
    else:
        import_every_bus_summary = run_import_every_bus_test(snapshots, args, args.output_dir)

    make_plots(top_buses, components, p_nom_by_carrier, capacity_vs_demand, args.output_dir)

    print("\nBaseline load-shedding diagnostics saved to:", args.output_dir)
    print("\nConnected components:")
    print(f"Number of islands: {len(components)}")
    print("Largest islands by load:")
    print(
        components[
            [
                "component_id",
                "bus_count",
                "line_count",
                "total_load_mwh",
                "peak_load_mw",
                "total_p_nom_mw",
                "import_slack_buses",
                "has_import_slack",
            ]
        ]
        .head(10)
        .to_string(index=False, float_format=lambda x: f"{x:,.3f}")
    )

    print("\nTop load-shedding buses:")
    print(
        top_buses[
            [
                "bus",
                "total_load_shed_mwh",
                "total_load_mwh",
                "component_id",
                "connected_lines",
                "nearest_generator_distance_km",
                "import_slack_exists_in_island",
            ]
        ]
        .head(20)
        .to_string(index=False, float_format=lambda x: f"{x:,.3f}")
    )

    print("\nGenerator p_nom by carrier:")
    print(p_nom_by_carrier.to_string(index=False, float_format=lambda x: f"{x:,.3f}"))

    if not import_every_bus_summary.empty:
        print("\nImport slack at every bus test:")
        print(import_every_bus_summary.to_string(index=False, float_format=lambda x: f"{x:,.6f}"))


if __name__ == "__main__":
    main()
