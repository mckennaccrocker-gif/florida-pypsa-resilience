"""
Build the no-hazard baseline used by IBTrACS cascading/operational scenarios.

This is intentionally compatible with run_florida_pypsa_calibrated_hazard_scenarios.py:
it writes baseline_summary.csv and import_bus_selection.csv into the selected
baseline directory.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from run_florida_pypsa_baseline_validation import (
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
    add_import_slack_generators,
    add_standard_load_shedding,
    cap_load_shedding_by_bus_load,
)


PROJECT_DIR = Path(r"C:\oxford_tc_project")
NETWORK_DIR = PROJECT_DIR / "data" / "Electricity" / "pypsa_florida_network_extended_tie_lines"
DEFAULT_OUTPUT_DIR = NETWORK_DIR / "baseline_calibrated_no_hazard"
LINE_CAPACITY_MULTIPLIER = 2.0


def system_cost_from_dispatch(network, snapshots: pd.Index) -> float:
    dispatch = network.generators_t.p.reindex(snapshots).clip(lower=0.0)
    costs = network.generators["marginal_cost"].reindex(dispatch.columns).fillna(0.0)
    weights = pd.Series(1.0, index=snapshots)
    if hasattr(network, "snapshot_weightings") and "generators" in network.snapshot_weightings:
        weights = network.snapshot_weightings["generators"].reindex(snapshots).fillna(1.0)
    return float(dispatch.multiply(costs, axis=1).multiply(weights, axis=0).sum().sum())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run extended-tie-grid no-hazard baseline.")
    parser.add_argument("--network-dir", type=Path, default=NETWORK_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--line-capacity-multiplier", type=float, default=LINE_CAPACITY_MULTIPLIER)
    parser.add_argument("--solver", default="highs")
    parser.add_argument("--highs-method", default="ipm")
    parser.add_argument("--start", default="2025-01-01 00:00:00")
    parser.add_argument("--periods", type=int, default=24)
    parser.add_argument("--chunk-size", type=int, default=24)
    return parser.parse_args()


def external_import_buses(network) -> pd.Index:
    if "is_external_tie_import" in network.buses.columns:
        flagged = network.buses.index[
            network.buses["is_external_tie_import"].astype(str).str.lower().isin({"true", "1", "yes"})
        ]
        if len(flagged) > 0:
            return pd.Index(flagged)
    candidates = network.buses.index[network.buses.index.astype(str).str.startswith("external_tie_bus_")]
    if len(candidates) > 0:
        return pd.Index(candidates)
    raise ValueError("Could not identify external tie import buses in the extended network.")


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    network = load_latest_network(args.network_dir)
    network.lines["s_nom"] = network.lines["s_nom"].astype(float) * args.line_capacity_multiplier
    snapshots = select_snapshots(network, args.start, args.periods, all_snapshots=False)

    import_buses = external_import_buses(network)
    import_generators = add_import_slack_generators(network, import_buses)
    load_shedding_generators = add_standard_load_shedding(network)
    cap_load_shedding_by_bus_load(network, load_shedding_generators).to_csv(
        args.output_dir / "load_shedding_bus_load_caps.csv",
        index=False,
    )

    import_selection = pd.DataFrame(
        {
            "import_bus": import_buses,
            "import_p_nom_mw": pd.NA,
            "selection_reason": "external_tie_import_bus_from_extended_grid",
        }
    )
    import_selection.to_csv(args.output_dir / "import_bus_selection.csv", index=False)

    status, condition = solve_dispatch_in_chunks(
        network,
        snapshots,
        args.solver,
        args.highs_method,
        args.chunk_size,
    )
    dispatch_hourly, generation_by_carrier = dispatch_by_carrier(network, snapshots, args.output_dir)
    load_shedding_hourly, load_shedding_by_bus = save_load_shedding(
        network,
        load_shedding_generators,
        snapshots,
        args.output_dir,
    )
    import_hourly, _import_by_bus = save_import_slack(
        network,
        import_generators,
        snapshots,
        args.output_dir,
    )
    line_loading = save_line_loading(network, snapshots, args.output_dir)
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
        args.output_dir,
    )
    summary["line_capacity_multiplier"] = args.line_capacity_multiplier
    summary["total_system_cost_usd"] = system_cost_from_dispatch(network, snapshots)
    summary.to_csv(args.output_dir / "baseline_summary.csv", index=False)
    print(summary.T.to_string(header=False))
    print("Saved baseline:", args.output_dir)


if __name__ == "__main__":
    main()
