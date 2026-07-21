"""
Calibrate the no-hazard Florida PyPSA baseline.

This script runs targeted 24-hour calibration experiments to explain and reduce
baseline load shedding before hazard scenarios are interpreted:

1. Add emergency import slack to every connected island with load.
2. Assign load only to the largest connected component as a comparison case.
3. Relax line limits after island imports to test whether remaining shedding is
   caused by deliverability/congestion rather than statewide generation.

The script does not overwrite the source network CSVs. Each experiment writes
its changed assumptions and dispatch outputs to a separate folder.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import networkx as nx
import pandas as pd

from run_florida_pypsa_baseline_validation import (
    PYPSA_DIR,
    dispatch_by_carrier,
    load_latest_network,
    save_import_slack,
    save_line_loading,
    save_load_shedding,
    select_snapshots,
    solve_dispatch_in_chunks,
    write_baseline_summary,
)
from run_florida_pypsa_load_shedding_dispatch import (
    DISPATCH_TOLERANCE_MW,
    add_import_slack_generators,
    add_standard_load_shedding,
    infer_import_buses,
    snapshot_weights_hours,
)


DEFAULT_OUTPUT_DIR = PYPSA_DIR / "baseline_calibration"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Florida PyPSA baseline calibration cases.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--start", default="2025-01-01 00:00:00")
    parser.add_argument("--periods", type=int, default=24)
    parser.add_argument("--solver", default="highs")
    parser.add_argument("--highs-method", default="ipm")
    parser.add_argument("--chunk-size", type=int, default=24)
    parser.add_argument("--num-import-buses", type=int, default=10)
    parser.add_argument(
        "--line-limit-multiplier",
        type=float,
        default=2.0,
        help="Multiplier for the line-limit diagnostic case.",
    )
    return parser.parse_args()


def connected_components(network) -> tuple[list[set[str]], dict[str, int]]:
    graph = nx.Graph()
    graph.add_nodes_from(network.buses.index)
    active_lines = network.lines[network.lines["active"].astype(bool)]
    for line in active_lines.itertuples():
        graph.add_edge(line.bus0, line.bus1)
    components = sorted(nx.connected_components(graph), key=len, reverse=True)
    bus_to_component = {
        bus: component_id for component_id, buses in enumerate(components) for bus in buses
    }
    return components, bus_to_component


def load_by_bus(network, snapshots: pd.Index) -> pd.DataFrame:
    loads_t = network.loads_t.p_set.reindex(snapshots)
    weights = snapshot_weights_hours(network, snapshots)
    by_bus = loads_t.copy()
    by_bus.columns = network.loads.loc[by_bus.columns, "bus"].to_numpy()
    by_bus = by_bus.T.groupby(level=0).sum().T
    return pd.DataFrame(
        {
            "bus": by_bus.columns,
            "total_load_mwh": by_bus.multiply(weights, axis=0).sum(axis=0).to_numpy(),
            "peak_load_mw": by_bus.max(axis=0).to_numpy(),
        }
    )


def component_load_generation(network, snapshots: pd.Index) -> pd.DataFrame:
    components, bus_to_component = connected_components(network)
    loads = load_by_bus(network, snapshots)
    loads["component_id"] = loads["bus"].map(bus_to_component)
    load_summary = loads.groupby("component_id", as_index=False).agg(
        bus_load_count=("bus", "size"),
        total_load_mwh=("total_load_mwh", "sum"),
        peak_load_mw=("peak_load_mw", "sum"),
    )

    generators = network.generators.loc[
        ~network.generators["carrier"].isin(["import_slack", "load_shedding"])
    ].copy()
    generators["component_id"] = generators["bus"].map(bus_to_component)
    gen_summary = generators.groupby("component_id", as_index=False).agg(
        generator_count=("p_nom", "size"),
        total_p_nom_mw=("p_nom", "sum"),
    )

    rows = []
    for component_id, buses in enumerate(components):
        sub = network.lines[
            network.lines["bus0"].isin(buses) & network.lines["bus1"].isin(buses)
        ]
        rows.append(
            {
                "component_id": component_id,
                "bus_count": len(buses),
                "line_count": len(sub),
            }
        )
    summary = pd.DataFrame(rows).merge(load_summary, on="component_id", how="left")
    summary = summary.merge(gen_summary, on="component_id", how="left")
    for col in ["bus_load_count", "total_load_mwh", "peak_load_mw", "generator_count", "total_p_nom_mw"]:
        summary[col] = summary[col].fillna(0)
    summary["p_nom_minus_peak_load_mw"] = summary["total_p_nom_mw"] - summary["peak_load_mw"]
    return summary.sort_values("total_load_mwh", ascending=False)


def choose_island_import_buses(network, snapshots: pd.Index, original_import_buses: pd.Index) -> pd.DataFrame:
    components, bus_to_component = connected_components(network)
    loads = load_by_bus(network, snapshots)
    loads["component_id"] = loads["bus"].map(bus_to_component)

    rows = []
    original_component_ids = {
        bus_to_component[bus] for bus in original_import_buses if bus in bus_to_component
    }
    for component_id, buses in enumerate(components):
        component_loads = loads[
            (loads["component_id"] == component_id)
            & (loads["total_load_mwh"] > DISPATCH_TOLERANCE_MW)
        ]
        if component_loads.empty:
            continue
        chosen_bus = component_loads.sort_values("peak_load_mw", ascending=False).iloc[0]["bus"]
        rows.append(
            {
                "component_id": component_id,
                "chosen_import_bus": chosen_bus,
                "component_has_original_import": component_id in original_component_ids,
                "component_bus_count": len(buses),
                "component_total_load_mwh": float(component_loads["total_load_mwh"].sum()),
                "chosen_bus_peak_load_mw": float(component_loads["peak_load_mw"].max()),
            }
        )
    return pd.DataFrame(rows)


def zero_loads_outside_largest_component(network, snapshots: pd.Index, output_dir: Path) -> pd.DataFrame:
    components, bus_to_component = connected_components(network)
    largest = set(components[0])
    loads_outside = network.loads.index[~network.loads["bus"].isin(largest)]
    original = network.loads_t.p_set.reindex(index=snapshots, columns=loads_outside, fill_value=0.0)
    removed_by_load = (
        original.sum(axis=0)
        .rename_axis("load")
        .reset_index(name="removed_load_mwh")
    )
    removed_by_load["bus"] = network.loads.loc[removed_by_load["load"], "bus"].to_numpy()
    removed_by_load["component_id"] = removed_by_load["bus"].map(bus_to_component)
    removed_by_load.to_csv(output_dir / "loads_zeroed_outside_largest_component.csv", index=False)
    network.loads_t.p_set.loc[snapshots, loads_outside] = 0.0
    return removed_by_load


def top_main_island_load_shed_buses(
    network,
    snapshots: pd.Index,
    load_shedding_generators: pd.Index,
    output_dir: Path,
) -> pd.DataFrame:
    components, bus_to_component = connected_components(network)
    dispatch = network.generators_t.p.reindex(
        index=snapshots,
        columns=load_shedding_generators,
        fill_value=0.0,
    ).clip(lower=0.0)
    dispatch.columns = network.generators.loc[dispatch.columns, "bus"].to_numpy()
    by_bus = dispatch.T.groupby(level=0).sum().T
    weights = snapshot_weights_hours(network, snapshots)
    summary = pd.DataFrame(
        {
            "bus": by_bus.columns,
            "total_load_shed_mwh": by_bus.multiply(weights, axis=0).sum(axis=0).to_numpy(),
            "max_hourly_load_shed_mw": by_bus.max(axis=0).to_numpy(),
        }
    )
    summary["component_id"] = summary["bus"].map(bus_to_component)
    summary = summary[
        (summary["component_id"] == 0)
        & (summary["total_load_shed_mwh"] > DISPATCH_TOLERANCE_MW)
    ].sort_values("total_load_shed_mwh", ascending=False)
    summary.to_csv(output_dir / "top_main_island_load_shedding_buses.csv", index=False)
    return summary


def top_congested_corridors(line_loading: pd.DataFrame, output_dir: Path) -> pd.DataFrame:
    top = line_loading.head(50).copy()
    top.to_csv(output_dir / "top_congested_corridors.csv", index=False)
    return top


def run_case(
    case_name: str,
    args: argparse.Namespace,
    import_buses: pd.Index,
    load_largest_component_only: bool = False,
    line_limit_multiplier: float | None = None,
) -> pd.Series:
    output_dir = args.output_dir / case_name
    output_dir.mkdir(parents=True, exist_ok=True)

    network = load_latest_network(PYPSA_DIR)
    snapshots = select_snapshots(network, args.start, args.periods, all_snapshots=False)
    changes = []

    if load_largest_component_only:
        removed = zero_loads_outside_largest_component(network, snapshots, output_dir)
        changes.append(
            {
                "change": "zero_loads_outside_largest_component",
                "affected_records": len(removed),
                "mwh_affected": float(removed["removed_load_mwh"].sum()),
            }
        )

    if line_limit_multiplier is not None:
        network.lines["s_nom"] = network.lines["s_nom"].astype(float) * line_limit_multiplier
        changes.append(
            {
                "change": "multiply_all_line_s_nom",
                "affected_records": len(network.lines),
                "mwh_affected": pd.NA,
                "multiplier": line_limit_multiplier,
            }
        )

    pd.DataFrame(changes).to_csv(output_dir / "calibration_changes.csv", index=False)
    pd.DataFrame({"import_bus": import_buses}).to_csv(output_dir / "import_buses.csv", index=False)

    import_generators = add_import_slack_generators(network, import_buses)
    load_shedding_generators = add_standard_load_shedding(network)

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
    top_main_island_load_shed_buses(network, snapshots, load_shedding_generators, output_dir)
    top_congested_corridors(line_loading, output_dir)
    return summary.iloc[0].rename(case_name)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    base_network = load_latest_network(PYPSA_DIR)
    snapshots = select_snapshots(base_network, args.start, args.periods, all_snapshots=False)
    original_import_buses = infer_import_buses(base_network, None, args.num_import_buses)

    components = component_load_generation(base_network, snapshots)
    components.to_csv(args.output_dir / "pre_calibration_component_load_generation.csv", index=False)

    island_imports = choose_island_import_buses(base_network, snapshots, original_import_buses)
    island_imports.to_csv(args.output_dir / "island_import_bus_selection.csv", index=False)
    island_import_buses = pd.Index(
        sorted(set(original_import_buses).union(set(island_imports["chosen_import_bus"])))
    )

    summaries = []
    summaries.append(
        run_case(
            "case_1_import_slack_each_load_island",
            args,
            island_import_buses,
        )
    )
    summaries.append(
        run_case(
            "case_2_load_only_largest_component",
            args,
            original_import_buses,
            load_largest_component_only=True,
        )
    )
    summaries.append(
        run_case(
            f"case_3_island_imports_line_limits_x{args.line_limit_multiplier:g}",
            args,
            island_import_buses,
            line_limit_multiplier=args.line_limit_multiplier,
        )
    )

    summary_table = pd.DataFrame(summaries)
    summary_table.insert(0, "case", summary_table.index)
    summary_table.to_csv(args.output_dir / "calibration_case_summary.csv", index=False)

    print("\nCalibration case summary:")
    keep = [
        "case",
        "solver_status",
        "solver_condition",
        "total_demand_mwh",
        "total_demand_served_mwh",
        "total_import_slack_mwh",
        "total_load_shed_mwh",
        "buses_experiencing_load_shedding",
        "number_overloaded_lines",
        "max_line_loading_pu",
    ]
    print(summary_table[keep].to_string(index=False, float_format=lambda x: f"{x:,.3f}"))
    print("\nSaved calibration outputs:", args.output_dir)


if __name__ == "__main__":
    main()
