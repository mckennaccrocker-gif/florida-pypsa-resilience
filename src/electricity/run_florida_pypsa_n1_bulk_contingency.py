"""
Run a practical N-1 bulk-line contingency screen for the reviewed Florida grid.

This is a DC/linear PyPSA screen: it checks post-contingency thermal overloads,
emergency import use, and load shedding after removing one bulk transmission
line at a time. It does not claim AC voltage-limit violations.

Default quick run:
    python data/Electricity/run_florida_pypsa_n1_bulk_contingency.py --max-contingencies 25

Full bulk screen:
    python data/Electricity/run_florida_pypsa_n1_bulk_contingency.py --all-contingencies
"""

from __future__ import annotations

import argparse
import copy
from pathlib import Path

import numpy as np
import pandas as pd
import pypsa

from run_florida_pypsa_load_shedding_dispatch import (
    DISPATCH_TOLERANCE_MW,
    IMPORT_SLACK_CARRIER,
    LOAD_SHEDDING_CARRIER,
    add_import_slack_generators,
    add_standard_load_shedding,
    infer_import_buses,
    largest_connected_component_buses,
    restrict_to_buses,
    select_snapshots,
    solve_dispatch,
    snapshot_weights_hours,
)


PROJECT_DIR = Path(r"C:\oxford_tc_project")
ELECTRICITY_DIR = PROJECT_DIR / "data" / "Electricity"
DEFAULT_NETWORK = (
    ELECTRICITY_DIR
    / "pypsa_florida_network_island_reviewed_no_connectors"
    / "florida_network.nc"
)
DEFAULT_OUTPUT_DIR = (
    ELECTRICITY_DIR
    / "pypsa_florida_network_island_reviewed_no_connectors"
    / "n1_bulk_contingency_results"
)

BULK_VOLTAGE_MIN_KV = 100.0
DEFAULT_LINE_CAPACITY_MULTIPLIER = 2.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run N-1 bulk-line contingency screen.")
    parser.add_argument("--network", type=Path, default=DEFAULT_NETWORK)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--solver", default="highs")
    parser.add_argument("--start", default=None)
    parser.add_argument("--periods", type=int, default=1)
    parser.add_argument("--all-snapshots", action="store_true")
    parser.add_argument("--bulk-voltage-min-kv", type=float, default=BULK_VOLTAGE_MIN_KV)
    parser.add_argument(
        "--line-capacity-multiplier",
        type=float,
        default=DEFAULT_LINE_CAPACITY_MULTIPLIER,
        help="Use 2.0 to match the calibrated Florida PyPSA hazard baseline.",
    )
    parser.add_argument(
        "--include-islands",
        action="store_true",
        help="Also screen retained non-main islands. Default screens largest/main component only.",
    )
    parser.add_argument("--import-bus", action="append", default=None)
    parser.add_argument("--num-import-buses", type=int, default=1)
    parser.add_argument(
        "--all-contingencies",
        action="store_true",
        help="Run every bulk line. Without this, --max-contingencies is used.",
    )
    parser.add_argument(
        "--max-contingencies",
        type=int,
        default=25,
        help="Quick-test cap unless --all-contingencies is set.",
    )
    return parser.parse_args()


def apply_line_capacity_multiplier(network: pypsa.Network, multiplier: float) -> None:
    if multiplier <= 0:
        raise ValueError("--line-capacity-multiplier must be positive.")
    network.lines["s_nom"] = pd.to_numeric(network.lines["s_nom"], errors="coerce").fillna(0.0) * multiplier


def line_loading_summary(network: pypsa.Network, snapshots: pd.Index) -> pd.DataFrame:
    line_p0 = network.lines_t.p0.reindex(index=snapshots, columns=network.lines.index).abs()
    s_nom = pd.to_numeric(network.lines["s_nom"], errors="coerce").replace(0, np.nan)
    loading = line_p0.divide(s_nom, axis=1)
    return pd.DataFrame(
        {
            "line": loading.columns,
            "bus0": network.lines.loc[loading.columns, "bus0"].to_numpy(),
            "bus1": network.lines.loc[loading.columns, "bus1"].to_numpy(),
            "v_nom_kv": network.lines.loc[loading.columns, "v_nom"].to_numpy(),
            "s_nom_mva": network.lines.loc[loading.columns, "s_nom"].to_numpy(),
            "source_edge_id": network.lines.loc[loading.columns, "source_edge_id"].to_numpy(),
            "owner": network.lines.loc[loading.columns, "owner"].to_numpy(),
            "max_abs_p0_mw": line_p0.max(axis=0).to_numpy(),
            "max_loading_pu": loading.max(axis=0).to_numpy(),
            "hours_overloaded": (loading > 1.0 + DISPATCH_TOLERANCE_MW).sum(axis=0).to_numpy(),
        }
    ).sort_values("max_loading_pu", ascending=False)


def dispatch_metrics(
    network: pypsa.Network,
    snapshots: pd.Index,
    load_shedding_generators: pd.Index,
) -> dict[str, float]:
    weights = snapshot_weights_hours(network, snapshots)
    generator_p = network.generators_t.p.reindex(index=snapshots, columns=network.generators.index, fill_value=0.0)
    generator_p = generator_p.clip(lower=0.0)
    costs = pd.to_numeric(network.generators["marginal_cost"], errors="coerce").reindex(generator_p.columns).fillna(0.0)

    load_shed = generator_p.reindex(columns=load_shedding_generators, fill_value=0.0)
    import_generators = network.generators.index[network.generators["carrier"].eq(IMPORT_SLACK_CARRIER)]
    imports = generator_p.reindex(columns=import_generators, fill_value=0.0)

    return {
        "system_cost_usd": float(generator_p.multiply(costs, axis=1).multiply(weights, axis=0).sum().sum()),
        "total_load_shed_mwh": float(load_shed.multiply(weights, axis=0).sum().sum()),
        "max_hourly_load_shed_mw": float(load_shed.sum(axis=1).max()) if len(load_shed) else 0.0,
        "total_import_slack_mwh": float(imports.multiply(weights, axis=0).sum().sum()),
        "max_hourly_import_slack_mw": float(imports.sum(axis=1).max()) if len(imports) else 0.0,
    }


def bulk_contingency_lines(network: pypsa.Network, minimum_voltage_kv: float) -> pd.Index:
    v_nom = pd.to_numeric(network.lines["v_nom"], errors="coerce")
    s_nom = pd.to_numeric(network.lines["s_nom"], errors="coerce")
    active = network.lines["active"].astype(bool) if "active" in network.lines.columns else True
    candidates = network.lines.index[v_nom.ge(minimum_voltage_kv) & s_nom.gt(0) & active]
    return pd.Index(candidates)


def possible_northern_interface_flag(network: pypsa.Network, lines: pd.Index) -> pd.Series:
    buses = network.buses[["x", "y"]]
    bus0 = network.lines.loc[lines, "bus0"].map(buses["y"])
    bus1 = network.lines.loc[lines, "bus1"].map(buses["y"])
    return pd.Series(np.maximum(bus0, bus1) >= 30.0, index=lines)


def solve_case(
    network: pypsa.Network,
    snapshots: pd.Index,
    solver: str,
    load_shedding_generators: pd.Index,
    output_dir: Path | None = None,
) -> tuple[dict[str, float | str], pd.DataFrame]:
    status, condition = solve_dispatch(network, snapshots, solver)
    loading = line_loading_summary(network, snapshots)
    metrics = dispatch_metrics(network, snapshots, load_shedding_generators)
    metrics.update(
        {
            "optimization_status": status,
            "optimization_condition": condition,
            "max_line_loading_pu": float(loading["max_loading_pu"].max()),
            "overloaded_line_count": int((loading["max_loading_pu"] > 1.0 + DISPATCH_TOLERANCE_MW).sum()),
            "voltage_limits_evaluated": "no_dc_linear_screen_only",
        }
    )
    if output_dir is not None:
        loading.to_csv(output_dir / "base_case_line_loading.csv", index=False)
    return metrics, loading


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    network = pypsa.Network(args.network)
    if not args.include_islands:
        keep_buses = largest_connected_component_buses(network)
        restrict_to_buses(network, keep_buses)

    snapshots = select_snapshots(network, args.start, args.periods, args.all_snapshots)
    apply_line_capacity_multiplier(network, args.line_capacity_multiplier)

    import_buses = infer_import_buses(network, args.import_bus, args.num_import_buses)
    add_import_slack_generators(network, import_buses)
    load_shedding_generators = add_standard_load_shedding(network)

    base_metrics, _base_loading = solve_case(
        network,
        snapshots,
        args.solver,
        load_shedding_generators,
        output_dir=args.output_dir,
    )
    if hasattr(network, "model") and hasattr(network.model, "solver_model"):
        network.model.solver_model = None

    candidates = bulk_contingency_lines(network, args.bulk_voltage_min_kv)
    northern_interface = possible_northern_interface_flag(network, candidates)
    candidates = pd.Index(
        network.lines.loc[candidates]
        .assign(_north=northern_interface.reindex(candidates).fillna(False).to_numpy())
        .sort_values(["v_nom", "_north", "s_nom"], ascending=[False, False, False])
        .index
    )
    if not args.all_contingencies:
        candidates = candidates[: args.max_contingencies]

    base_cost = float(base_metrics["system_cost_usd"])
    base_load_shed = float(base_metrics["total_load_shed_mwh"])
    base_import = float(base_metrics["total_import_slack_mwh"])
    summary_rows = []
    overload_rows = []

    for i, outage_line in enumerate(candidates, start=1):
        print(f"[{i}/{len(candidates)}] N-1 outage: {outage_line}")
        case = network.copy()
        case.lines.loc[outage_line, "s_nom"] = 0.0
        case.lines.loc[outage_line, "active"] = False

        metrics, loading = solve_case(case, snapshots, args.solver, load_shedding_generators)
        overloaded = loading[
            (loading["line"] != outage_line)
            & (loading["max_loading_pu"] > 1.0 + DISPATCH_TOLERANCE_MW)
        ].copy()
        overloaded["outaged_line"] = outage_line
        overload_rows.append(overloaded)

        line = network.lines.loc[outage_line]
        summary_rows.append(
            {
                "outaged_line": outage_line,
                "source_edge_id": line.get("source_edge_id", pd.NA),
                "bus0": line["bus0"],
                "bus1": line["bus1"],
                "v_nom_kv": line["v_nom"],
                "s_nom_mva": line["s_nom"],
                "owner": line.get("owner", ""),
                "possible_northern_interface": bool(northern_interface.get(outage_line, False)),
                **metrics,
                "incremental_system_cost_usd": float(metrics["system_cost_usd"]) - base_cost,
                "incremental_load_shed_mwh": float(metrics["total_load_shed_mwh"]) - base_load_shed,
                "incremental_import_slack_mwh": float(metrics["total_import_slack_mwh"]) - base_import,
            }
        )

    base_df = pd.DataFrame([base_metrics])
    base_df.insert(0, "case", "base")
    base_df["network"] = str(args.network.resolve())
    base_df["snapshots"] = len(snapshots)
    base_df["first_snapshot"] = snapshots[0]
    base_df["last_snapshot"] = snapshots[-1]
    base_df["bulk_voltage_min_kv"] = args.bulk_voltage_min_kv
    base_df["line_capacity_multiplier"] = args.line_capacity_multiplier
    base_df["largest_component_only"] = not args.include_islands
    base_df.to_csv(args.output_dir / "n1_base_case_summary.csv", index=False)

    summary = pd.DataFrame(summary_rows).sort_values(
        ["incremental_load_shed_mwh", "overloaded_line_count", "max_line_loading_pu"],
        ascending=[False, False, False],
    )
    summary.to_csv(args.output_dir / "n1_bulk_contingency_summary.csv", index=False)

    if overload_rows:
        overloads = pd.concat(overload_rows, ignore_index=True)
    else:
        overloads = pd.DataFrame()
    overloads.to_csv(args.output_dir / "n1_bulk_overloaded_lines.csv", index=False)

    candidate_doc = network.lines.loc[candidates].copy()
    candidate_doc["possible_northern_interface"] = northern_interface.reindex(candidates).fillna(False).to_numpy()
    candidate_doc.to_csv(args.output_dir / "n1_bulk_contingency_list.csv")

    print("Base case:", base_df.to_string(index=False))
    print("Contingencies evaluated:", len(summary))
    print("Saved:", args.output_dir)


if __name__ == "__main__":
    main()
