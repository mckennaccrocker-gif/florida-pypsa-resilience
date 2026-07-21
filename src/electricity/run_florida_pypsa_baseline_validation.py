"""
Run a no-hazard Florida PyPSA baseline validation dispatch for 2025.

This script builds the network from the latest CSV tables, applies final
generator marginal costs, imports solar p_max_pu time series, adds emergency
import slack and standard load shedding, then solves a baseline dispatch.

Outputs are written to:
    data/Electricity/pypsa_florida_network/baseline_validation_2025/
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import pypsa

from run_florida_pypsa_load_shedding_dispatch import (
    DISPATCH_TOLERANCE_MW,
    IMPORT_SLACK_CARRIER,
    IMPORT_SLACK_MARGINAL_COST_USD_PER_MWH,
    LOAD_SHEDDING_CARRIER,
    VALUE_OF_LOST_LOAD_USD_PER_MWH,
    add_import_slack_generators,
    add_standard_load_shedding,
    cap_load_shedding_by_bus_load,
    infer_import_buses,
    restrict_to_buses,
    largest_connected_component_buses,
    select_snapshots,
    snapshot_weights_hours,
)


PROJECT_DIR = Path(r"C:\oxford_tc_project")
PYPSA_DIR = PROJECT_DIR / "data" / "Electricity" / "pypsa_florida_network"
DEFAULT_OUTPUT_DIR = PYPSA_DIR / "baseline_validation_2025"

FINAL_GENERATORS = PYPSA_DIR / "generators_with_final_marginal_costs.csv"
LOADS_P_SET = PYPSA_DIR / "loads-p_set.csv"
GENERATORS_P_MAX_PU = PYPSA_DIR / "generators-p_max_pu.csv"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run no-hazard Florida PyPSA baseline validation dispatch."
    )
    parser.add_argument("--network-dir", type=Path, default=PYPSA_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--solver", default="highs")
    parser.add_argument(
        "--highs-method",
        default="ipm",
        choices=["choose", "simplex", "ipm", "pdlp"],
        help="HiGHS LP method to use when --solver highs.",
    )
    parser.add_argument("--import-bus", action="append", default=None)
    parser.add_argument("--num-import-buses", type=int, default=10)
    parser.add_argument("--largest-component-only", action="store_true")
    parser.add_argument("--start", default=None)
    parser.add_argument("--periods", type=int, default=8760)
    parser.add_argument("--all-snapshots", action="store_true")
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=168,
        help=(
            "Solve snapshots in independent chunks. This is equivalent here "
            "because the baseline model has no intertemporal operating constraints."
        ),
    )
    parser.add_argument("--export-solved-network", action="store_true")
    return parser.parse_args()


def load_latest_network(network_dir: Path) -> pypsa.Network:
    network = pypsa.Network()
    network.import_from_csv_folder(str(network_dir), skip_time=True)

    final_generators_path = network_dir / "generators_with_final_marginal_costs.csv"
    loads_p_set_path = network_dir / "loads-p_set.csv"
    generators_p_max_pu_path = network_dir / "generators-p_max_pu.csv"

    if not final_generators_path.exists():
        raise FileNotFoundError(final_generators_path)
    final_generators = pd.read_csv(final_generators_path).set_index("name")
    missing = sorted(set(final_generators.index).difference(network.generators.index))
    if missing:
        raise ValueError(
            "Final marginal-cost table contains generators not in the network: "
            f"{missing[:10]}"
        )
    network.generators.loc[final_generators.index, "marginal_cost"] = final_generators["marginal_cost"]

    if not loads_p_set_path.exists():
        raise FileNotFoundError(loads_p_set_path)
    loads_p_set = pd.read_csv(loads_p_set_path)
    snapshots = pd.to_datetime(loads_p_set.pop("snapshot"), errors="raise")
    missing_loads = sorted(set(loads_p_set.columns).difference(network.loads.index))
    if missing_loads:
        raise ValueError(f"loads-p_set.csv contains unknown loads: {missing_loads[:10]}")
    network.set_snapshots(pd.DatetimeIndex(snapshots))
    loads_p_set.index = network.snapshots
    network.loads_t.p_set = loads_p_set.reindex(columns=network.loads.index, fill_value=0.0)

    if not generators_p_max_pu_path.exists():
        raise FileNotFoundError(generators_p_max_pu_path)
    generators_p_max_pu = pd.read_csv(generators_p_max_pu_path)
    profile_snapshots = pd.to_datetime(generators_p_max_pu.pop("snapshot"), errors="raise")
    missing_generators = sorted(set(generators_p_max_pu.columns).difference(network.generators.index))
    if missing_generators:
        raise ValueError(
            "generators-p_max_pu.csv contains unknown generators: "
            f"{missing_generators[:10]}"
        )
    if not network.snapshots.equals(pd.DatetimeIndex(profile_snapshots)):
        raise ValueError("generators-p_max_pu.csv snapshots do not match loads-p_set.csv.")
    generators_p_max_pu.index = network.snapshots
    network.generators_t.p_max_pu = generators_p_max_pu

    return network


def dispatch_by_carrier(
    network: pypsa.Network,
    snapshots: pd.Index,
    output_dir: Path,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    dispatch = network.generators_t.p.reindex(snapshots).clip(lower=0.0)
    carriers = network.generators["carrier"].reindex(dispatch.columns).fillna("unknown")
    hourly_mw = dispatch.T.groupby(carriers).sum().T
    weights = snapshot_weights_hours(network, snapshots)
    hourly_mwh = hourly_mw.multiply(weights, axis=0)
    hourly_mwh.insert(0, "snapshot", snapshots)
    hourly_mwh.to_csv(output_dir / "baseline_dispatch_by_carrier.csv", index=False)

    summary = (
        hourly_mwh.drop(columns=["snapshot"])
        .sum()
        .rename("generation_mwh")
        .reset_index()
        .rename(columns={"index": "carrier"})
        .sort_values("generation_mwh", ascending=False)
    )
    summary.to_csv(output_dir / "baseline_generation_by_carrier_summary.csv", index=False)
    return hourly_mwh, summary


def save_load_shedding(
    network: pypsa.Network,
    load_shedding_generators: pd.Index,
    snapshots: pd.Index,
    output_dir: Path,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    dispatch = network.generators_t.p.reindex(index=snapshots, columns=load_shedding_generators, fill_value=0.0)
    dispatch = dispatch.clip(lower=0.0)
    weights = snapshot_weights_hours(network, snapshots)
    hourly = pd.DataFrame(
        {
            "snapshot": snapshots,
            "load_shed_mw": dispatch.sum(axis=1).to_numpy(),
            "load_shed_mwh": dispatch.multiply(weights, axis=0).sum(axis=1).to_numpy(),
        }
    )
    hourly["load_shedding_cost_usd"] = hourly["load_shed_mwh"] * VALUE_OF_LOST_LOAD_USD_PER_MWH
    hourly.to_csv(output_dir / "baseline_load_shedding.csv", index=False)

    by_bus = dispatch.copy()
    by_bus.columns = network.generators.loc[by_bus.columns, "bus"].to_numpy()
    by_bus = by_bus.T.groupby(level=0).sum().T
    by_bus_energy = by_bus.multiply(weights, axis=0)
    bus_summary = pd.DataFrame(
        {
            "bus": by_bus_energy.columns,
            "total_load_shed_mwh": by_bus_energy.sum(axis=0).to_numpy(),
            "max_hourly_load_shed_mw": by_bus.max(axis=0).to_numpy(),
        }
    )
    bus_summary = bus_summary[bus_summary["total_load_shed_mwh"] > DISPATCH_TOLERANCE_MW]
    bus_summary.to_csv(output_dir / "baseline_load_shedding_by_bus.csv", index=False)
    return hourly, bus_summary


def save_import_slack(
    network: pypsa.Network,
    import_slack_generators: pd.Index,
    snapshots: pd.Index,
    output_dir: Path,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    dispatch = network.generators_t.p.reindex(index=snapshots, columns=import_slack_generators, fill_value=0.0)
    dispatch = dispatch.clip(lower=0.0)
    weights = snapshot_weights_hours(network, snapshots)
    hourly = pd.DataFrame(
        {
            "snapshot": snapshots,
            "import_slack_mw": dispatch.sum(axis=1).to_numpy(),
            "import_slack_mwh": dispatch.multiply(weights, axis=0).sum(axis=1).to_numpy(),
        }
    )
    hourly["import_slack_cost_usd"] = hourly["import_slack_mwh"] * IMPORT_SLACK_MARGINAL_COST_USD_PER_MWH
    hourly.to_csv(output_dir / "baseline_import_slack.csv", index=False)

    by_bus = dispatch.copy()
    by_bus.columns = network.generators.loc[by_bus.columns, "bus"].to_numpy()
    by_bus = by_bus.T.groupby(level=0).sum().T
    by_bus_energy = by_bus.multiply(weights, axis=0)
    bus_summary = pd.DataFrame(
        {
            "bus": by_bus_energy.columns,
            "total_import_slack_mwh": by_bus_energy.sum(axis=0).to_numpy(),
            "max_hourly_import_slack_mw": by_bus.max(axis=0).to_numpy(),
        }
    )
    bus_summary = bus_summary[bus_summary["total_import_slack_mwh"] > DISPATCH_TOLERANCE_MW]
    bus_summary.to_csv(output_dir / "baseline_import_slack_by_bus.csv", index=False)
    return hourly, bus_summary


def save_line_loading(network: pypsa.Network, snapshots: pd.Index, output_dir: Path) -> pd.DataFrame:
    line_p0 = network.lines_t.p0.reindex(snapshots).abs()
    s_nom = pd.to_numeric(network.lines["s_nom"], errors="coerce").replace(0, np.nan)
    loading = line_p0.divide(s_nom, axis=1)
    line_summary = pd.DataFrame(
        {
            "line": loading.columns,
            "bus0": network.lines.loc[loading.columns, "bus0"].to_numpy(),
            "bus1": network.lines.loc[loading.columns, "bus1"].to_numpy(),
            "s_nom_mva": network.lines.loc[loading.columns, "s_nom"].to_numpy(),
            "max_abs_p0_mw": line_p0.max(axis=0).to_numpy(),
            "max_loading_pu": loading.max(axis=0).to_numpy(),
            "hours_overloaded": (loading > 1.0 + DISPATCH_TOLERANCE_MW).sum(axis=0).to_numpy(),
        }
    )
    line_summary = line_summary.sort_values("max_loading_pu", ascending=False)
    line_summary.to_csv(output_dir / "baseline_line_loading.csv", index=False)
    return line_summary


def save_transformer_loading(network: pypsa.Network, snapshots: pd.Index, output_dir: Path) -> pd.DataFrame:
    if network.transformers.empty or not hasattr(network, "transformers_t") or network.transformers_t.p0.empty:
        transformer_summary = pd.DataFrame(
            columns=[
                "transformer",
                "bus0",
                "bus1",
                "s_nom_mva",
                "max_abs_p0_mw",
                "max_loading_pu",
                "hours_overloaded",
            ]
        )
        transformer_summary.to_csv(output_dir / "baseline_transformer_loading.csv", index=False)
        return transformer_summary

    transformer_p0 = network.transformers_t.p0.reindex(snapshots).abs()
    s_nom = pd.to_numeric(network.transformers["s_nom"], errors="coerce").replace(0, np.nan)
    loading = transformer_p0.divide(s_nom, axis=1)
    transformer_summary = pd.DataFrame(
        {
            "transformer": loading.columns,
            "bus0": network.transformers.loc[loading.columns, "bus0"].to_numpy(),
            "bus1": network.transformers.loc[loading.columns, "bus1"].to_numpy(),
            "s_nom_mva": network.transformers.loc[loading.columns, "s_nom"].to_numpy(),
            "max_abs_p0_mw": transformer_p0.max(axis=0).to_numpy(),
            "max_loading_pu": loading.max(axis=0).to_numpy(),
            "hours_overloaded": (loading > 1.0 + DISPATCH_TOLERANCE_MW).sum(axis=0).to_numpy(),
        }
    )
    transformer_summary = transformer_summary.sort_values("max_loading_pu", ascending=False)
    transformer_summary.to_csv(output_dir / "baseline_transformer_loading.csv", index=False)
    return transformer_summary


def solar_sanity(network: pypsa.Network, snapshots: pd.Index) -> dict[str, float | int]:
    solar_generators = network.generators.index[
        network.generators["carrier"].astype(str).str.lower().eq("solar")
    ]
    if len(solar_generators) == 0:
        return {
            "solar_generators": 0,
            "solar_generation_mwh": 0.0,
            "solar_average_p_max_pu": 0.0,
            "solar_max_p_max_pu": 0.0,
            "night_hours_with_solar_generation": 0,
        }
    weights = snapshot_weights_hours(network, snapshots)
    solar_dispatch = network.generators_t.p.reindex(index=snapshots, columns=solar_generators, fill_value=0.0).clip(lower=0.0)
    solar_profiles = network.generators_t.p_max_pu.reindex(index=snapshots, columns=solar_generators, fill_value=0.0)
    night = pd.DatetimeIndex(snapshots).hour.isin([0, 1, 2, 3, 4])
    return {
        "solar_generators": int(len(solar_generators)),
        "solar_generation_mwh": float(solar_dispatch.multiply(weights, axis=0).sum().sum()),
        "solar_average_p_max_pu": float(solar_profiles.mean().mean()),
        "solar_max_p_max_pu": float(solar_profiles.max().max()),
        "night_hours_with_solar_generation": int((solar_dispatch.loc[night].sum(axis=1) > DISPATCH_TOLERANCE_MW).sum()),
    }


def solve_dispatch_chunk(
    network: pypsa.Network,
    snapshots: pd.Index,
    solver: str,
    highs_method: str,
) -> tuple[str, str]:
    print(f"Solving {len(snapshots)} snapshots with solver={solver}")
    solver_options = {}
    if solver == "highs" and highs_method != "choose":
        solver_options["solver"] = highs_method
    result = network.optimize(
        snapshots=snapshots,
        solver_name=solver,
        solver_options=solver_options,
        include_objective_constant=False,
    )
    if isinstance(result, tuple) and len(result) >= 2:
        return str(result[0]), str(result[1])
    return "unknown", str(result)


def solve_dispatch_in_chunks(
    network: pypsa.Network,
    snapshots: pd.Index,
    solver: str,
    highs_method: str,
    chunk_size: int,
) -> tuple[str, str]:
    if chunk_size <= 0:
        raise ValueError("--chunk-size must be positive.")

    generator_dispatch = []
    line_p0 = []
    transformer_p0 = []
    statuses = []
    conditions = []

    for start in range(0, len(snapshots), chunk_size):
        chunk = snapshots[start : start + chunk_size]
        print(
            f"\nSolving baseline chunk {start // chunk_size + 1} "
            f"({chunk[0]} to {chunk[-1]}, {len(chunk)} snapshots)"
        )
        status, condition = solve_dispatch_chunk(network, chunk, solver, highs_method)
        statuses.append(status)
        conditions.append(condition)
        if status.lower() not in {"ok", "warning"} or condition.lower() != "optimal":
            raise RuntimeError(f"Chunk solve failed: {status}, {condition}")
        generator_dispatch.append(network.generators_t.p.reindex(chunk).copy())
        line_p0.append(network.lines_t.p0.reindex(chunk).copy())
        if not network.transformers.empty and hasattr(network, "transformers_t"):
            transformer_p0.append(network.transformers_t.p0.reindex(chunk).copy())

    network.generators_t.p = pd.concat(generator_dispatch).sort_index()
    network.lines_t.p0 = pd.concat(line_p0).sort_index()
    if transformer_p0:
        network.transformers_t.p0 = pd.concat(transformer_p0).sort_index()

    status = "ok" if all(item.lower() in {"ok", "warning"} for item in statuses) else "failed"
    condition = "optimal" if all(item.lower() == "optimal" for item in conditions) else ";".join(sorted(set(conditions)))
    return status, condition


def write_baseline_summary(
    network: pypsa.Network,
    snapshots: pd.Index,
    status: str,
    condition: str,
    dispatch_summary: pd.DataFrame,
    load_shedding_hourly: pd.DataFrame,
    load_shedding_by_bus: pd.DataFrame,
    import_hourly: pd.DataFrame,
    line_loading: pd.DataFrame,
    output_dir: Path,
) -> pd.DataFrame:
    weights = snapshot_weights_hours(network, snapshots)
    total_demand_mwh = float(network.loads_t.p_set.reindex(snapshots).multiply(weights, axis=0).sum().sum())
    total_load_shed_mwh = float(load_shedding_hourly["load_shed_mwh"].sum())
    total_import_mwh = float(import_hourly["import_slack_mwh"].sum())
    total_served_mwh = total_demand_mwh - total_load_shed_mwh
    overloaded_lines = int((line_loading["max_loading_pu"] > 1.0 + DISPATCH_TOLERANCE_MW).sum())
    solar = solar_sanity(network, snapshots)

    summary = pd.DataFrame(
        [
            {
                "snapshots": len(snapshots),
                "first_snapshot": snapshots[0],
                "last_snapshot": snapshots[-1],
                "solver_status": status,
                "solver_condition": condition,
                "total_demand_mwh": total_demand_mwh,
                "total_demand_served_mwh": total_served_mwh,
                "total_generation_mwh_excluding_load_shedding": float(
                    dispatch_summary.loc[
                        ~dispatch_summary["carrier"].isin([LOAD_SHEDDING_CARRIER]),
                        "generation_mwh",
                    ].sum()
                ),
                "total_import_slack_mwh": total_import_mwh,
                "hours_with_import_slack": int((import_hourly["import_slack_mw"] > DISPATCH_TOLERANCE_MW).sum()),
                "total_load_shed_mwh": total_load_shed_mwh,
                "maximum_hourly_load_shed_mw": float(load_shedding_hourly["load_shed_mw"].max()),
                "buses_experiencing_load_shedding": int(len(load_shedding_by_bus)),
                "number_overloaded_lines": overloaded_lines,
                "max_line_loading_pu": float(line_loading["max_loading_pu"].max()),
                "import_slack_marginal_cost_usd_per_mwh": IMPORT_SLACK_MARGINAL_COST_USD_PER_MWH,
                "value_of_lost_load_usd_per_mwh": VALUE_OF_LOST_LOAD_USD_PER_MWH,
                **solar,
            }
        ]
    )
    summary.to_csv(output_dir / "baseline_summary.csv", index=False)
    return summary


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    network = load_latest_network(args.network_dir)
    if args.largest_component_only:
        keep_buses = largest_connected_component_buses(network)
        removed = restrict_to_buses(network, keep_buses)
        print("Restricted to largest connected component.")
        print(removed)

    snapshots = select_snapshots(
        network,
        start=args.start,
        periods=args.periods,
        all_snapshots=args.all_snapshots,
    )
    import_buses = infer_import_buses(network, args.import_bus, args.num_import_buses)
    import_slack_generators = add_import_slack_generators(network, import_buses)
    load_shedding_generators = add_standard_load_shedding(network)
    cap_load_shedding_by_bus_load(network, load_shedding_generators).to_csv(
        args.output_dir / "load_shedding_bus_load_caps.csv",
        index=False,
    )

    print("Baseline network directory:", args.network_dir)
    print("Output directory:", args.output_dir)
    print("Snapshots:", len(snapshots), snapshots[0], "to", snapshots[-1])
    print("Buses:", len(network.buses))
    print("Lines:", len(network.lines))
    print("Generators before solve:", len(network.generators))
    print("Import slack buses:", ", ".join(import_buses))
    print("Load-shedding generators:", len(load_shedding_generators))

    status, condition = solve_dispatch_in_chunks(
        network, snapshots, args.solver, args.highs_method, args.chunk_size
    )
    print("Optimization status:", status)
    print("Optimization condition:", condition)

    dispatch_hourly, generation_by_carrier = dispatch_by_carrier(network, snapshots, args.output_dir)
    load_shedding_hourly, load_shedding_by_bus = save_load_shedding(
        network, load_shedding_generators, snapshots, args.output_dir
    )
    import_hourly, _import_by_bus = save_import_slack(
        network, import_slack_generators, snapshots, args.output_dir
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

    if args.export_solved_network:
        solved_path = args.output_dir / "baseline_solved_network.nc"
        network.export_to_netcdf(solved_path)
        print("Saved solved network:", solved_path)

    print("\nBaseline validation summary:")
    print(summary.T.to_string(header=False))
    print("\nGeneration by carrier:")
    print(generation_by_carrier.to_string(index=False, float_format=lambda x: f"{x:,.3f}"))
    print("\nSaved outputs:")
    for name in [
        "baseline_dispatch_by_carrier.csv",
        "baseline_load_shedding.csv",
        "baseline_import_slack.csv",
        "baseline_line_loading.csv",
        "baseline_summary.csv",
    ]:
        print(args.output_dir / name)


if __name__ == "__main__":
    main()
