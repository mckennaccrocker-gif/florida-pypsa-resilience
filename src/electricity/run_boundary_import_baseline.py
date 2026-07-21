"""
Run a no-hazard baseline with physically motivated boundary import buses.

Imports are placed at selected northern Florida buses near the Georgia/Alabama
boundary. Optional local imports are added only to disconnected load-bearing
islands that do not already contain a boundary import bus, and are labelled as
topology-artifact support.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import networkx as nx
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from calibrate_florida_pypsa_baseline import (  # noqa: E402
    component_load_generation,
    load_by_bus,
    top_congested_corridors,
    top_main_island_load_shed_buses,
)
from run_florida_pypsa_baseline_validation import (  # noqa: E402
    PYPSA_DIR,
    dispatch_by_carrier,
    load_latest_network,
    save_import_slack,
    save_line_loading,
    save_transformer_loading,
    save_load_shedding,
    select_snapshots,
    solve_dispatch_in_chunks,
    write_baseline_summary,
)
from run_florida_pypsa_load_shedding_dispatch import (  # noqa: E402
    DISPATCH_TOLERANCE_MW,
    add_import_slack_generators,
    add_standard_load_shedding,
    cap_load_shedding_by_bus_load,
    snapshot_weights_hours,
)


DEFAULT_BOUNDARY_BUSES = PYPSA_DIR / "boundary_import_buses" / "selected_boundary_import_buses.csv"
DEFAULT_OUTPUT_DIR = PYPSA_DIR / "baseline_boundary_imports"


def system_cost_from_dispatch(network, snapshots: pd.Index) -> float:
    dispatch = network.generators_t.p.reindex(snapshots).clip(lower=0.0)
    costs = network.generators["marginal_cost"].reindex(dispatch.columns).fillna(0.0)
    weights = snapshot_weights_hours(network, snapshots)
    return float(dispatch.multiply(costs, axis=1).multiply(weights, axis=0).sum().sum())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run boundary-import baseline calibration.")
    parser.add_argument("--network-dir", type=Path, default=PYPSA_DIR)
    parser.add_argument("--boundary-buses", type=Path, default=DEFAULT_BOUNDARY_BUSES)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--start", default="2025-01-01 00:00:00")
    parser.add_argument("--periods", type=int, default=24)
    parser.add_argument("--solver", default="highs")
    parser.add_argument("--highs-method", default="ipm")
    parser.add_argument("--chunk-size", type=int, default=24)
    parser.add_argument("--line-limit-multiplier", type=float, default=1.75)
    parser.add_argument("--add-artifact-island-imports", action="store_true")
    parser.add_argument(
        "--boundary-import-cap-fraction",
        type=float,
        default=1.0,
        help=(
            "Boundary import p_nom is this fraction of the sum of calibrated "
            "s_nom on lines incident to each selected boundary bus."
        ),
    )
    parser.add_argument(
        "--artifact-import-peak-load-margin",
        type=float,
        default=1.10,
        help=(
            "Topology-artifact island import p_nom is peak island load times this margin."
        ),
    )
    return parser.parse_args()


def connected_components(network) -> tuple[list[set[str]], dict[str, int]]:
    graph = nx.Graph()
    graph.add_nodes_from(network.buses.index)
    active_lines = network.lines[network.lines["active"].astype(bool)]
    graph.add_edges_from(zip(active_lines["bus0"], active_lines["bus1"]))
    if hasattr(network, "transformers") and not network.transformers.empty:
        active_transformers = network.transformers[network.transformers["active"].astype(bool)]
        graph.add_edges_from(zip(active_transformers["bus0"], active_transformers["bus1"]))
    components = sorted(nx.connected_components(graph), key=len, reverse=True)
    bus_to_component = {bus: idx for idx, component in enumerate(components) for bus in component}
    return components, bus_to_component


def artifact_import_buses(network, snapshots: pd.Index, boundary_import_buses: pd.Index) -> pd.DataFrame:
    components, bus_to_component = connected_components(network)
    boundary_component_ids = {
        bus_to_component[bus] for bus in boundary_import_buses if bus in bus_to_component
    }
    loads = load_by_bus(network, snapshots)
    loads["component_id"] = loads["bus"].map(bus_to_component)

    rows = []
    for component_id, _component in enumerate(components):
        if component_id in boundary_component_ids:
            continue
        component_loads = loads[
            (loads["component_id"] == component_id)
            & (loads["total_load_mwh"] > DISPATCH_TOLERANCE_MW)
        ].copy()
        if component_loads.empty:
            continue
        chosen = component_loads.sort_values("peak_load_mw", ascending=False).iloc[0]
        rows.append(
            {
                "import_bus": chosen["bus"],
                "import_type": "topology_artifact_island_import",
                "component_id": component_id,
                "component_total_load_mwh": float(component_loads["total_load_mwh"].sum()),
                "chosen_bus_peak_load_mw": float(chosen["peak_load_mw"]),
                "component_peak_load_mw": float(component_loads["peak_load_mw"].sum()),
            }
        )
    return pd.DataFrame(rows)


def incident_line_capacity_by_bus(network) -> pd.Series:
    """Return calibrated incident line capacity at each bus."""
    active = network.lines[network.lines["active"].astype(bool)].copy()
    active["s_nom"] = pd.to_numeric(active["s_nom"], errors="coerce").fillna(0.0)
    bus0 = active.groupby("bus0")["s_nom"].sum()
    bus1 = active.groupby("bus1")["s_nom"].sum()
    return bus0.add(bus1, fill_value=0.0)


def add_import_capacity_caps(
    network,
    import_generators: pd.Index,
    import_report: pd.DataFrame,
) -> pd.DataFrame:
    """Set per-import p_nom caps and save the cap assigned to each import."""
    report = import_report.copy()
    report["import_p_nom_mw"] = pd.to_numeric(report["import_p_nom_mw"], errors="coerce")
    report["import_capacity_source"] = report["import_capacity_source"].fillna("unknown")
    for _, row in report.iterrows():
        bus = str(row["import_bus"])
        generator = f"import_slack_{bus}"
        if generator in import_generators:
            network.generators.loc[generator, "p_nom"] = float(row["import_p_nom_mw"])
    return report


def run_case(args: argparse.Namespace, case_name: str, add_artifact_imports: bool) -> pd.Series:
    output_dir = args.output_dir / case_name
    output_dir.mkdir(parents=True, exist_ok=True)

    network = load_latest_network(args.network_dir)
    snapshots = select_snapshots(network, args.start, args.periods, all_snapshots=False)

    boundary = pd.read_csv(args.boundary_buses)
    boundary_buses = pd.Index(boundary["name"].astype(str))
    missing = sorted(set(boundary_buses).difference(network.buses.index))
    if missing:
        raise ValueError(f"Boundary import buses missing from network: {missing}")

    network.lines["s_nom"] = network.lines["s_nom"].astype(float) * args.line_limit_multiplier
    incident_capacity = incident_line_capacity_by_bus(network)

    boundary_report = boundary.copy()
    boundary_report["import_type"] = "northern_boundary_intertie_import"
    boundary_report["component_id"] = boundary_report["name"].map(connected_components(network)[1])
    boundary_report = boundary_report.rename(columns={"name": "import_bus"})
    boundary_report["incident_line_capacity_mva_after_multiplier"] = boundary_report["import_bus"].map(
        incident_capacity
    )
    boundary_report["import_p_nom_mw"] = (
        pd.to_numeric(
            boundary_report["incident_line_capacity_mva_after_multiplier"],
            errors="coerce",
        ).fillna(0.0)
        * args.boundary_import_cap_fraction
    )
    boundary_report["import_capacity_source"] = (
        "incident_line_s_nom_sum_after_multiplier"
        f"_x{args.boundary_import_cap_fraction:g}"
    )

    artifact_report = (
        artifact_import_buses(network, snapshots, boundary_buses)
        if add_artifact_imports
        else pd.DataFrame(
            columns=[
                "import_bus",
                "import_type",
                "component_id",
                "component_total_load_mwh",
                "chosen_bus_peak_load_mw",
            ]
        )
    )
    import_report = pd.concat(
        [
            boundary_report[
                [
                    "import_bus",
                    "import_type",
                    "component_id",
                    "v_nom",
                    "connected_edge_count",
                    "distance_to_ga_al_boundary_km",
                    "incident_line_capacity_mva_after_multiplier",
                    "import_p_nom_mw",
                    "import_capacity_source",
                ]
            ],
            artifact_report,
        ],
        ignore_index=True,
    )
    artifact_mask = import_report["import_type"].eq("topology_artifact_island_import")
    if artifact_mask.any():
        import_report.loc[artifact_mask, "import_p_nom_mw"] = (
            pd.to_numeric(
                import_report.loc[artifact_mask, "component_peak_load_mw"],
                errors="coerce",
            ).fillna(0.0)
            * args.artifact_import_peak_load_margin
        )
        import_report.loc[artifact_mask, "import_capacity_source"] = (
            "component_peak_load_with_margin"
            f"_x{args.artifact_import_peak_load_margin:g}"
        )
    import_report.to_csv(output_dir / "import_bus_selection.csv", index=False)

    import_buses = pd.Index(import_report["import_bus"].dropna().astype(str).unique())
    import_generators = add_import_slack_generators(network, import_buses)
    import_report = add_import_capacity_caps(network, import_generators, import_report)
    import_report.to_csv(output_dir / "import_bus_selection.csv", index=False)
    load_shedding_generators = add_standard_load_shedding(network)
    cap_load_shedding_by_bus_load(network, load_shedding_generators).to_csv(
        output_dir / "load_shedding_bus_load_caps.csv",
        index=False,
    )

    component_load_generation(network, snapshots).to_csv(
        output_dir / "component_load_generation.csv", index=False
    )
    pd.DataFrame(
        [
            {
                "line_limit_multiplier": args.line_limit_multiplier,
                "boundary_import_bus_count": len(boundary_buses),
                "artifact_import_bus_count": len(artifact_report),
                "total_import_bus_count": len(import_buses),
                "boundary_import_cap_fraction": args.boundary_import_cap_fraction,
                "artifact_import_peak_load_margin": args.artifact_import_peak_load_margin,
                "total_import_p_nom_mw": float(import_report["import_p_nom_mw"].sum()),
                "boundary_import_p_nom_mw": float(
                    import_report.loc[
                        import_report["import_type"].eq("northern_boundary_intertie_import"),
                        "import_p_nom_mw",
                    ].sum()
                ),
                "artifact_import_p_nom_mw": float(
                    import_report.loc[
                        import_report["import_type"].eq("topology_artifact_island_import"),
                        "import_p_nom_mw",
                    ].sum()
                ),
            }
        ]
    ).to_csv(output_dir / "case_assumptions.csv", index=False)

    status, condition = solve_dispatch_in_chunks(
        network,
        snapshots,
        args.solver,
        args.highs_method,
        args.chunk_size,
    )
    dispatch_hourly, generation_by_carrier = dispatch_by_carrier(network, snapshots, output_dir)
    load_shedding_hourly, load_shedding_by_bus = save_load_shedding(
        network,
        load_shedding_generators,
        snapshots,
        output_dir,
    )
    import_hourly, _import_by_bus = save_import_slack(
        network,
        import_generators,
        snapshots,
        output_dir,
    )
    line_loading = save_line_loading(network, snapshots, output_dir)
    save_transformer_loading(network, snapshots, output_dir)
    summary = write_baseline_summary(
        network,
        snapshots,
        status,
        condition,
        generation_by_carrier,
        load_shedding_hourly,
        load_shedding_by_bus,
        import_hourly,
        line_loading,
        output_dir,
    )
    total_system_cost = system_cost_from_dispatch(network, snapshots)
    summary["total_system_cost_usd"] = total_system_cost
    summary.to_csv(output_dir / "baseline_summary.csv", index=False)
    top_main_island_load_shed_buses(network, snapshots, load_shedding_generators, output_dir)
    top_congested_corridors(line_loading, output_dir)
    return summary.iloc[0].rename(case_name)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    summaries = [
        run_case(args, "boundary_imports_only", add_artifact_imports=False),
        run_case(
            args,
            "boundary_imports_plus_artifact_islands",
            add_artifact_imports=args.add_artifact_island_imports,
        ),
    ]
    table = pd.DataFrame(summaries)
    table.insert(0, "case", table.index)
    table.to_csv(args.output_dir / "boundary_import_baseline_summary.csv", index=False)

    keep = [
        "case",
        "solver_status",
        "solver_condition",
        "total_demand_mwh",
        "total_demand_served_mwh",
        "total_import_slack_mwh",
        "total_load_shed_mwh",
        "buses_experiencing_load_shedding",
        "max_line_loading_pu",
    ]
    print(table[keep].to_string(index=False))
    print("Saved:", args.output_dir)


if __name__ == "__main__":
    main()
