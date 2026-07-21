"""
Run a PyPSA dispatch with standard load shedding generators.

The script loads the Florida PyPSA NetCDF network, adds one high-cost
emergency import generator, adds one high-cost load-shedding generator at every
bus, solves a linear dispatch/network optimization, and writes summary metrics
for emergency imports and unserved energy.

Default usage solves the first 24 hours to keep test runs quick:

    python data/Electricity/run_florida_pypsa_load_shedding_dispatch.py

Run the full available year explicitly:

    python data/Electricity/run_florida_pypsa_load_shedding_dispatch.py --all-snapshots
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
DEFAULT_OUTPUT_DIR = PYPSA_DIR / "load_shedding_results"

LOAD_SHEDDING_CARRIER = "load_shedding"
LOAD_SHEDDING_SUFFIX = "_load_shedding"
VALUE_OF_LOST_LOAD_USD_PER_MWH = 10_000.0
LOAD_SHEDDING_P_NOM_MW = 1_000_000.0
IMPORT_SLACK_CARRIER = "import_slack"
IMPORT_SLACK_PREFIX = "import_slack"
IMPORT_SLACK_MARGINAL_COST_USD_PER_MWH = 5_000.0
IMPORT_SLACK_P_NOM_MW = 1_000_000.0
DISPATCH_TOLERANCE_MW = 1e-6
LOAD_SHEDDING_MIN_P_NOM_MW = 1.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run Florida PyPSA dispatch with load shedding."
    )
    parser.add_argument("--network", type=Path, default=DEFAULT_NETWORK_FILE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--solver", default="highs")
    parser.add_argument(
        "--import-bus",
        action="append",
        default=None,
        help=(
            "Bus for emergency imports. Can be repeated. If omitted, the script "
            "uses boundary/interconnection flags if present, otherwise the most "
            "connected bus."
        ),
    )
    parser.add_argument(
        "--num-import-buses",
        type=int,
        default=1,
        help="Number of fallback import buses to use when --import-bus is omitted.",
    )
    parser.add_argument(
        "--largest-component-only",
        action="store_true",
        help=(
            "Restrict the scenario to the largest connected component before "
            "adding imports and load shedding."
        ),
    )
    parser.add_argument(
        "--start",
        default=None,
        help="Optional first snapshot, e.g. 2025-01-01 00:00:00.",
    )
    parser.add_argument(
        "--periods",
        type=int,
        default=24,
        help="Number of hourly snapshots to solve unless --all-snapshots is used.",
    )
    parser.add_argument(
        "--all-snapshots",
        action="store_true",
        help="Solve all snapshots in the network.",
    )
    parser.add_argument(
        "--export-solved-network",
        action="store_true",
        help="Write solved_network_with_load_shedding.nc to the output directory.",
    )
    return parser.parse_args()


def largest_connected_component_buses(network: pypsa.Network) -> pd.Index:
    graph = nx.Graph()
    graph.add_nodes_from(network.buses.index)

    active_lines = network.lines[network.lines["active"].astype(bool)]
    for line in active_lines.itertuples():
        graph.add_edge(line.bus0, line.bus1)

    components = sorted(nx.connected_components(graph), key=len, reverse=True)
    if not components:
        raise ValueError("Network has no connected components.")
    return pd.Index(sorted(components[0]))


def restrict_to_buses(network: pypsa.Network, keep_buses: pd.Index) -> dict[str, int]:
    """Remove components outside a selected bus set."""
    keep_buses = pd.Index(keep_buses)
    keep_bus_set = set(keep_buses)

    remove_lines = network.lines.index[
        ~network.lines["bus0"].isin(keep_bus_set) | ~network.lines["bus1"].isin(keep_bus_set)
    ]
    remove_generators = network.generators.index[~network.generators["bus"].isin(keep_bus_set)]
    remove_loads = network.loads.index[~network.loads["bus"].isin(keep_bus_set)]
    remove_buses = network.buses.index.difference(keep_buses)

    if len(remove_lines) > 0:
        network.remove("Line", remove_lines)
    if len(remove_generators) > 0:
        network.remove("Generator", remove_generators)
    if len(remove_loads) > 0:
        network.remove("Load", remove_loads)
    if len(remove_buses) > 0:
        network.remove("Bus", remove_buses)

    return {
        "removed_buses": len(remove_buses),
        "removed_lines": len(remove_lines),
        "removed_generators": len(remove_generators),
        "removed_loads": len(remove_loads),
    }


def select_snapshots(
    network: pypsa.Network,
    start: str | None,
    periods: int,
    all_snapshots: bool,
) -> pd.Index:
    if all_snapshots:
        return network.snapshots

    if periods <= 0:
        raise ValueError("--periods must be positive.")

    snapshots = network.snapshots
    if start is None:
        return snapshots[:periods]

    start_ts = pd.Timestamp(start)
    if start_ts not in snapshots:
        raise ValueError(f"Start snapshot {start_ts} is not in the network.")
    start_pos = snapshots.get_loc(start_ts)
    return snapshots[start_pos : start_pos + periods]


def infer_import_buses(network: pypsa.Network, explicit_buses: list[str] | None, count: int) -> pd.Index:
    """Choose boundary/interconnection buses if tagged, else most connected buses."""
    if explicit_buses:
        missing = sorted(set(explicit_buses).difference(network.buses.index))
        if missing:
            raise ValueError(f"Import buses are not in the network: {missing}")
        return pd.Index(explicit_buses)

    if count <= 0:
        raise ValueError("--num-import-buses must be positive.")

    candidate_columns = [
        col
        for col in network.buses.columns
        if any(token in col.lower() for token in ["boundary", "interconnection", "tie"])
    ]
    for col in candidate_columns:
        tagged = network.buses.index[network.buses[col].astype(str).str.lower().isin({"1", "true", "yes"})]
        if len(tagged) > 0:
            return tagged[:count]

    sort_columns = [col for col in ["connected_edge_count", "v_nom"] if col in network.buses.columns]
    if sort_columns:
        buses = network.buses.sort_values(sort_columns, ascending=False).index
    else:
        buses = network.buses.index
    return buses[:count]


def add_import_slack_generators(
    network: pypsa.Network,
    import_buses: pd.Index,
    marginal_cost: float = IMPORT_SLACK_MARGINAL_COST_USD_PER_MWH,
    p_nom: float = IMPORT_SLACK_P_NOM_MW,
) -> pd.Index:
    """Add high-cost emergency import generators at selected buses."""
    if IMPORT_SLACK_CARRIER not in network.carriers.index:
        network.add("Carrier", IMPORT_SLACK_CARRIER, co2_emissions=0.0)

    existing = network.generators.index[
        network.generators.index.str.startswith(f"{IMPORT_SLACK_PREFIX}_")
    ]
    if len(existing) > 0:
        network.remove("Generator", existing)

    added = []
    for bus in import_buses:
        name = f"{IMPORT_SLACK_PREFIX}_{bus}"
        network.add(
            "Generator",
            name,
            bus=bus,
            carrier=IMPORT_SLACK_CARRIER,
            p_nom=p_nom,
            p_nom_extendable=False,
            marginal_cost=marginal_cost,
            efficiency=1.0,
        )
        added.append(name)
    return pd.Index(added)


def add_standard_load_shedding(
    network: pypsa.Network,
    marginal_cost: float = VALUE_OF_LOST_LOAD_USD_PER_MWH,
    p_nom: float = LOAD_SHEDDING_P_NOM_MW,
    cap_by_bus_load: bool = True,
) -> pd.Index:
    """Add one VOLL-priced load-shedding generator at every bus.

    When cap_by_bus_load is true, each load-shedding generator receives an
    hourly p_max_pu profile so it can dispatch no more than the load assigned
    to that same bus in each hour. This keeps load shedding from acting like a
    high-cost local generator that can export power to other buses.
    """
    if LOAD_SHEDDING_CARRIER not in network.carriers.index:
        network.add("Carrier", LOAD_SHEDDING_CARRIER, co2_emissions=0.0)

    existing = network.generators.index[
        network.generators.index.str.endswith(LOAD_SHEDDING_SUFFIX)
    ]
    if len(existing) > 0:
        network.remove("Generator", existing)

    before = set(network.generators.index)
    network.optimize.add_load_shedding(
        suffix=LOAD_SHEDDING_SUFFIX,
        buses=network.buses.index,
        sign=1.0,
        marginal_cost=marginal_cost,
        p_nom=p_nom,
    )
    added = pd.Index([name for name in network.generators.index if name not in before])
    network.generators.loc[added, "carrier"] = LOAD_SHEDDING_CARRIER
    network.generators.loc[added, "efficiency"] = 1.0
    network.generators.loc[added, "marginal_cost"] = marginal_cost
    network.generators.loc[added, "p_nom"] = p_nom
    network.generators.loc[added, "p_nom_extendable"] = False

    if cap_by_bus_load:
        cap_load_shedding_by_bus_load(network, added)
    return added


def cap_load_shedding_by_bus_load(
    network: pypsa.Network,
    load_shedding_generators: pd.Index,
) -> pd.DataFrame:
    """Cap load-shedding generator availability by each bus load profile.

    Returns a table documenting the cap assigned to each load-shedding
    generator. The cap is represented in PyPSA as p_nom times p_max_pu.
    """
    if len(load_shedding_generators) == 0:
        return pd.DataFrame(
            columns=[
                "generator",
                "bus",
                "p_nom_mw",
                "max_bus_load_mw",
                "total_bus_load_mwh",
                "cap_source",
            ]
        )

    if network.loads_t.p_set.empty:
        raise ValueError(
            "Cannot cap load shedding by bus load because network.loads_t.p_set is empty."
        )

    load_profiles = network.loads_t.p_set.reindex(columns=network.loads.index, fill_value=0.0)
    load_buses = network.loads["bus"].reindex(load_profiles.columns)
    bus_load_profiles = load_profiles.T.groupby(load_buses).sum().T
    bus_load_profiles = bus_load_profiles.reindex(columns=network.buses.index, fill_value=0.0).clip(lower=0.0)

    generator_buses = network.generators.loc[load_shedding_generators, "bus"]
    max_bus_load = bus_load_profiles.max(axis=0).reindex(generator_buses.to_numpy()).fillna(0.0)
    p_nom_by_generator = max_bus_load.copy()
    p_nom_by_generator.index = load_shedding_generators
    p_nom_by_generator = p_nom_by_generator.where(
        p_nom_by_generator > DISPATCH_TOLERANCE_MW,
        LOAD_SHEDDING_MIN_P_NOM_MW,
    )
    network.generators.loc[load_shedding_generators, "p_nom"] = p_nom_by_generator

    load_shed_p_max_pu = pd.DataFrame(
        index=network.snapshots,
        columns=load_shedding_generators,
        dtype=float,
    )
    for generator, bus in generator_buses.items():
        cap = float(network.generators.at[generator, "p_nom"])
        if cap <= 0:
            load_shed_p_max_pu[generator] = 0.0
        else:
            load_shed_p_max_pu[generator] = bus_load_profiles[bus] / cap

    load_shed_p_max_pu = load_shed_p_max_pu.clip(lower=0.0, upper=1.0).fillna(0.0)
    if network.generators_t.p_max_pu.empty:
        network.generators_t.p_max_pu = pd.DataFrame(index=network.snapshots)
    existing_p_max_pu = network.generators_t.p_max_pu.reindex(index=network.snapshots).drop(
        columns=load_shedding_generators.intersection(network.generators_t.p_max_pu.columns),
        errors="ignore",
    )
    network.generators_t.p_max_pu = pd.concat(
        [existing_p_max_pu, load_shed_p_max_pu],
        axis=1,
    )

    weights = pd.Series(1.0, index=network.snapshots)
    if not network.snapshot_weightings.empty and "objective" in network.snapshot_weightings:
        weights = network.snapshot_weightings["objective"].reindex(network.snapshots).fillna(1.0)
    documentation = pd.DataFrame(
        {
            "generator": load_shedding_generators,
            "bus": generator_buses.to_numpy(),
            "p_nom_mw": p_nom_by_generator.reindex(load_shedding_generators).to_numpy(),
            "max_bus_load_mw": max_bus_load.to_numpy(),
            "total_bus_load_mwh": bus_load_profiles[generator_buses.to_numpy()]
            .multiply(weights, axis=0)
            .sum(axis=0)
            .to_numpy(),
            "cap_source": "hourly_bus_load_profile",
        }
    )
    return documentation


def solve_dispatch(
    network: pypsa.Network,
    snapshots: pd.Index,
    solver: str,
) -> tuple[str, str]:
    print(f"Solving {len(snapshots)} snapshots with solver={solver}")
    result = network.optimize(
        snapshots=snapshots,
        solver_name=solver,
        include_objective_constant=False,
    )
    if isinstance(result, tuple) and len(result) >= 2:
        return str(result[0]), str(result[1])
    return "unknown", str(result)


def snapshot_weights_hours(network: pypsa.Network, snapshots: pd.Index) -> pd.Series:
    if hasattr(network, "snapshot_weightings") and "generators" in network.snapshot_weightings:
        weights = network.snapshot_weightings["generators"].reindex(snapshots)
    else:
        weights = pd.Series(1.0, index=snapshots)
    return weights.fillna(1.0).astype(float)


def load_shedding_dispatch(network: pypsa.Network, load_shedding_generators: pd.Index) -> pd.DataFrame:
    dispatch = network.generators_t.p.reindex(columns=load_shedding_generators, fill_value=0.0)
    dispatch = dispatch.clip(lower=0.0)
    dispatch.columns = network.generators.loc[dispatch.columns, "bus"].to_numpy()
    return dispatch.T.groupby(level=0).sum().T


def generator_dispatch_by_bus(network: pypsa.Network, generators: pd.Index) -> pd.DataFrame:
    dispatch = network.generators_t.p.reindex(columns=generators, fill_value=0.0)
    dispatch = dispatch.clip(lower=0.0)
    dispatch.columns = network.generators.loc[dispatch.columns, "bus"].to_numpy()
    return dispatch.T.groupby(level=0).sum().T


def summarize_dispatch_results(
    network: pypsa.Network,
    import_slack_generators: pd.Index,
    load_shedding_generators: pd.Index,
    snapshots: pd.Index,
    output_dir: Path,
    load_shedding_marginal_cost: float = VALUE_OF_LOST_LOAD_USD_PER_MWH,
    import_slack_marginal_cost: float = IMPORT_SLACK_MARGINAL_COST_USD_PER_MWH,
) -> pd.DataFrame:
    output_dir.mkdir(parents=True, exist_ok=True)

    import_dispatch_mw = generator_dispatch_by_bus(network, import_slack_generators).reindex(snapshots)
    bus_dispatch_mw = load_shedding_dispatch(network, load_shedding_generators).reindex(snapshots)
    weights = snapshot_weights_hours(network, snapshots)
    import_energy_mwh = import_dispatch_mw.multiply(weights, axis=0)
    bus_energy_mwh = bus_dispatch_mw.multiply(weights, axis=0)
    hourly_import_mw = import_dispatch_mw.sum(axis=1)
    hourly_import_mwh = import_energy_mwh.sum(axis=1)
    hourly_load_shed_mw = bus_dispatch_mw.sum(axis=1)
    hourly_load_shed_mwh = bus_energy_mwh.sum(axis=1)

    import_bus_summary = pd.DataFrame(
        {
            "bus": import_energy_mwh.columns,
            "total_import_slack_mwh": import_energy_mwh.sum(axis=0).to_numpy(),
            "max_hourly_import_slack_mw": import_dispatch_mw.max(axis=0).to_numpy(),
        }
    )
    import_bus_summary = import_bus_summary[
        import_bus_summary["total_import_slack_mwh"] > DISPATCH_TOLERANCE_MW
    ]
    import_bus_summary["import_slack_cost_usd"] = (
        import_bus_summary["total_import_slack_mwh"] * import_slack_marginal_cost
    )
    import_bus_summary = import_bus_summary.sort_values("total_import_slack_mwh", ascending=False)

    bus_summary = pd.DataFrame(
        {
            "bus": bus_energy_mwh.columns,
            "total_load_shed_mwh": bus_energy_mwh.sum(axis=0).to_numpy(),
            "max_hourly_load_shed_mw": bus_dispatch_mw.max(axis=0).to_numpy(),
        }
    )
    bus_summary = bus_summary[bus_summary["total_load_shed_mwh"] > DISPATCH_TOLERANCE_MW]
    bus_summary["load_shedding_cost_usd"] = (
        bus_summary["total_load_shed_mwh"] * load_shedding_marginal_cost
    )
    bus_summary = bus_summary.sort_values("total_load_shed_mwh", ascending=False)

    hourly_summary = pd.DataFrame(
        {
            "snapshot": snapshots,
            "import_slack_mw": hourly_import_mw.to_numpy(),
            "import_slack_mwh": hourly_import_mwh.to_numpy(),
            "import_slack_cost_usd": hourly_import_mwh.to_numpy() * import_slack_marginal_cost,
            "load_shed_mw": hourly_load_shed_mw.to_numpy(),
            "load_shed_mwh": hourly_load_shed_mwh.to_numpy(),
            "load_shedding_cost_usd": hourly_load_shed_mwh.to_numpy() * load_shedding_marginal_cost,
        }
    )

    total_import_slack_mwh = float(hourly_import_mwh.sum())
    hours_with_import_slack = int((hourly_import_mw > DISPATCH_TOLERANCE_MW).sum())
    import_slack_cost_usd = total_import_slack_mwh * import_slack_marginal_cost
    total_load_shed_mwh = float(hourly_load_shed_mwh.sum())
    max_hourly_load_shed_mw = float(hourly_load_shed_mw.max())
    total_cost_usd = total_load_shed_mwh * load_shedding_marginal_cost
    buses_with_load_shedding = int(len(bus_summary))
    load_shedding_occurs = total_load_shed_mwh > DISPATCH_TOLERANCE_MW

    summary = pd.DataFrame(
        [
            {
                "snapshots": len(snapshots),
                "first_snapshot": snapshots[0],
                "last_snapshot": snapshots[-1],
                "import_slack_marginal_cost_usd_per_mwh": import_slack_marginal_cost,
                "import_slack_generators": len(import_slack_generators),
                "import_slack_buses": ";".join(network.generators.loc[import_slack_generators, "bus"]),
                "total_import_slack_mwh": total_import_slack_mwh,
                "hours_with_import_slack": hours_with_import_slack,
                "total_import_slack_cost_usd": import_slack_cost_usd,
                "value_of_lost_load_usd_per_mwh": load_shedding_marginal_cost,
                "total_load_shed_mwh": total_load_shed_mwh,
                "maximum_hourly_load_shed_mw": max_hourly_load_shed_mw,
                "buses_experiencing_load_shedding": buses_with_load_shedding,
                "load_shedding_still_occurs_after_imports": load_shedding_occurs,
                "total_load_shedding_cost_usd": total_cost_usd,
            }
        ]
    )

    summary.to_csv(output_dir / "dispatch_summary.csv", index=False)
    import_bus_summary.to_csv(output_dir / "import_slack_by_bus.csv", index=False)
    bus_summary.to_csv(output_dir / "load_shedding_by_bus.csv", index=False)
    hourly_summary.to_csv(output_dir / "dispatch_by_snapshot.csv", index=False)

    print("\nEmergency import and load shedding results")
    print("Total import slack generation MWh:", round(total_import_slack_mwh, 6))
    print("Hours when import slack was used:", hours_with_import_slack)
    print("Total load shed MWh:", round(total_load_shed_mwh, 6))
    print("Maximum hourly load shed MW:", round(max_hourly_load_shed_mw, 6))
    print("Buses experiencing load shedding:", buses_with_load_shedding)
    print("Load shedding still occurs after imports:", load_shedding_occurs)
    print("Total import slack cost USD:", round(import_slack_cost_usd, 2))
    print("Total load shedding cost USD:", round(total_cost_usd, 2))
    print("Saved summary:", output_dir / "dispatch_summary.csv")
    print("Saved import report:", output_dir / "import_slack_by_bus.csv")
    print("Saved bus report:", output_dir / "load_shedding_by_bus.csv")
    print("Saved hourly report:", output_dir / "dispatch_by_snapshot.csv")
    return summary


def main() -> None:
    args = parse_args()

    if not args.network.exists():
        raise FileNotFoundError(args.network)

    network = pypsa.Network(args.network)
    if args.largest_component_only:
        keep_buses = largest_connected_component_buses(network)
        removed = restrict_to_buses(network, keep_buses)
        print("Restricted to largest connected component.")
        print("Kept buses:", len(keep_buses))
        print("Removed buses:", removed["removed_buses"])
        print("Removed lines:", removed["removed_lines"])
        print("Removed generators:", removed["removed_generators"])
        print("Removed loads:", removed["removed_loads"])

    snapshots = select_snapshots(
        network,
        start=args.start,
        periods=args.periods,
        all_snapshots=args.all_snapshots,
    )
    import_buses = infer_import_buses(network, args.import_bus, args.num_import_buses)
    generator_count_before_imports = len(network.generators)
    import_slack_generators = add_import_slack_generators(network, import_buses)
    load_shedding_generators = add_standard_load_shedding(network)

    print("Network:", args.network)
    print("Buses:", len(network.buses))
    print("Generators before imports/load shedding:", generator_count_before_imports)
    print("Generators before solve:", len(network.generators))
    print("Import slack buses:", ", ".join(import_buses))
    print("Import slack generators:", len(import_slack_generators))
    print("Import slack p_nom MW per bus:", IMPORT_SLACK_P_NOM_MW)
    print("Import slack marginal cost USD/MWh:", IMPORT_SLACK_MARGINAL_COST_USD_PER_MWH)
    print("Load-shedding generators:", len(load_shedding_generators))
    print("Load-shedding p_nom MW per bus:", LOAD_SHEDDING_P_NOM_MW)
    print("Load-shedding marginal cost USD/MWh:", VALUE_OF_LOST_LOAD_USD_PER_MWH)

    status, condition = solve_dispatch(network, snapshots, args.solver)
    print("Optimization status:", status)
    print("Optimization condition:", condition)

    if status.lower() not in {"ok", "warning"}:
        raise RuntimeError(f"Optimization did not finish successfully: {status}, {condition}")

    summarize_dispatch_results(
        network,
        import_slack_generators,
        load_shedding_generators,
        snapshots,
        args.output_dir,
    )

    if args.export_solved_network:
        solved_path = args.output_dir / "solved_network_with_load_shedding.nc"
        network.export_to_netcdf(solved_path)
        print("Saved solved network:", solved_path)


if __name__ == "__main__":
    main()
