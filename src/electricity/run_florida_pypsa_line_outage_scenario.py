"""
Run a first-pass natural-hazard line-outage scenario in PyPSA.

The scenario is intentionally simple and defensible:
  - keep buses, loads, and generators active
  - disable transmission lines exposed above a chosen hazard threshold
  - solve dispatch with emergency import slack and load shedding
  - compare unserved energy against a calibrated baseline run

Examples:

    python data/Electricity/run_florida_pypsa_line_outage_scenario.py ^
      --hazard tc_wind --threshold 35

    python data/Electricity/run_florida_pypsa_line_outage_scenario.py ^
      --hazard strong_wind --threshold 5
"""

from __future__ import annotations

import argparse
from pathlib import Path

import geopandas as gpd
import pandas as pd
import pypsa

from run_florida_pypsa_load_shedding_dispatch import (
    add_import_slack_generators,
    add_standard_load_shedding,
    infer_import_buses,
    largest_connected_component_buses,
    restrict_to_buses,
    select_snapshots,
    solve_dispatch,
    summarize_dispatch_results,
)


PROJECT_DIR = Path(r"C:\oxford_tc_project")
ELECTRICITY_DIR = PROJECT_DIR / "data" / "Electricity"
EXPOSURE_DIR = PROJECT_DIR / "data" / "Exposure"
PYPSA_DIR = ELECTRICITY_DIR / "pypsa_florida_network"

DEFAULT_NETWORK_FILE = PYPSA_DIR / "florida_network.nc"
DEFAULT_BASELINE_SUMMARY = PYPSA_DIR / "baseline_main_component_import10" / "dispatch_summary.csv"
LINE_CROSSWALK_FILE = ELECTRICITY_DIR / "florida_lines_with_s_nom.csv"

HAZARD_CONFIG = {
    "tc_wind": {
        "path": EXPOSURE_DIR / "lines_tropical_cyclone_exposure.gpkg",
        "value_column": "tc_wind_speed_max_ms",
        "id_column": "ID",
        "units": "m/s",
        "description": "tropical cyclone maximum wind speed",
    },
    "strong_wind": {
        "path": EXPOSURE_DIR / "lines_strong_wind_exposure_split.gpkg",
        "value_column": "strong_wind_frequency",
        "id_column": "ID",
        "units": "events/year",
        "description": "strong wind frequency",
    },
}

DISPATCH_TOLERANCE = 1e-6


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a PyPSA line-outage hazard scenario."
    )
    parser.add_argument("--hazard", choices=sorted(HAZARD_CONFIG), required=True)
    parser.add_argument("--threshold", type=float, required=True)
    parser.add_argument("--network", type=Path, default=DEFAULT_NETWORK_FILE)
    parser.add_argument("--baseline-summary", type=Path, default=DEFAULT_BASELINE_SUMMARY)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--solver", default="highs")
    parser.add_argument("--start", default=None)
    parser.add_argument("--periods", type=int, default=24)
    parser.add_argument("--all-snapshots", action="store_true")
    parser.add_argument("--num-import-buses", type=int, default=10)
    parser.add_argument("--import-bus", action="append", default=None)
    parser.add_argument(
        "--all-components",
        action="store_true",
        help="Use all connected components. Default uses largest connected component only.",
    )
    return parser.parse_args()


def load_line_crosswalk() -> pd.DataFrame:
    if not LINE_CROSSWALK_FILE.exists():
        raise FileNotFoundError(LINE_CROSSWALK_FILE)
    crosswalk = pd.read_csv(LINE_CROSSWALK_FILE)
    required = {"florida_line_id", "ID"}
    missing = required.difference(crosswalk.columns)
    if missing:
        raise ValueError(f"{LINE_CROSSWALK_FILE} missing columns: {sorted(missing)}")
    return crosswalk[["florida_line_id", "ID"]].drop_duplicates()


def exposed_hifld_line_ids(hazard: str, threshold: float) -> pd.DataFrame:
    config = HAZARD_CONFIG[hazard]
    if not config["path"].exists():
        raise FileNotFoundError(config["path"])

    exposure = gpd.read_file(config["path"], ignore_geometry=True)
    print(f"\n{hazard} exposure columns:")
    print(", ".join(exposure.columns.astype(str)))

    required = {config["id_column"], config["value_column"]}
    missing = required.difference(exposure.columns)
    if missing:
        raise ValueError(f"{config['path']} missing columns: {sorted(missing)}")

    values = pd.to_numeric(exposure[config["value_column"]], errors="coerce")
    exposure = exposure.assign(hazard_value=values)
    exposed = exposure[exposure["hazard_value"] >= threshold].copy()

    summary = (
        exposed.groupby(config["id_column"], as_index=False)
        .agg(
            hazard_value_max=("hazard_value", "max"),
            exposed_records=("hazard_value", "size"),
        )
        .rename(columns={config["id_column"]: "hifld_id"})
    )
    summary["hifld_id"] = pd.to_numeric(summary["hifld_id"], errors="coerce")
    summary = summary.dropna(subset=["hifld_id"])
    summary["hifld_id"] = summary["hifld_id"].astype(int)

    print(f"Hazard: {config['description']}")
    print(f"Threshold: {threshold} {config['units']}")
    print("Exposure records:", len(exposure))
    print("Exposed records:", len(exposed))
    print("Exposed unique HIFLD line IDs:", len(summary))
    return summary


def outage_pypsa_lines(
    network: pypsa.Network,
    hazard: str,
    threshold: float,
    output_dir: Path,
) -> pd.DataFrame:
    exposed_ids = exposed_hifld_line_ids(hazard, threshold)
    crosswalk = load_line_crosswalk()
    exposed_internal = exposed_ids.merge(
        crosswalk,
        left_on="hifld_id",
        right_on="ID",
        how="inner",
        validate="one_to_one",
    )

    line_source_ids = pd.to_numeric(network.lines["source_edge_id"], errors="coerce")
    outaged_lines = network.lines.index[line_source_ids.isin(exposed_internal["florida_line_id"])]
    outage_report = network.lines.loc[outaged_lines, ["bus0", "bus1", "v_nom", "s_nom", "length", "source_edge_id"]].copy()
    outage_report = outage_report.reset_index(names="line")
    outage_report = outage_report.merge(
        exposed_internal[["florida_line_id", "hifld_id", "hazard_value_max", "exposed_records"]],
        left_on="source_edge_id",
        right_on="florida_line_id",
        how="left",
    )
    outage_report.to_csv(output_dir / "outaged_lines.csv", index=False)

    network.lines.loc[outaged_lines, "active"] = False

    print("\nLine outage application")
    print("Matched exposed HIFLD IDs to internal line IDs:", len(exposed_internal))
    print("Outaged PyPSA lines active in scenario:", len(outaged_lines))
    print("Outaged line capacity MVA:", round(float(outage_report["s_nom"].sum()), 3))
    print("Outaged line length km:", round(float(outage_report["length"].sum()), 3))
    return outage_report


def read_baseline_summary(path: Path) -> pd.Series:
    if not path.exists():
        raise FileNotFoundError(path)
    baseline = pd.read_csv(path)
    if baseline.empty:
        raise ValueError(f"Baseline summary is empty: {path}")
    return baseline.iloc[0]


def write_incremental_summary(
    scenario_summary: pd.Series,
    baseline_summary: pd.Series,
    outage_report: pd.DataFrame,
    args: argparse.Namespace,
    output_dir: Path,
) -> pd.DataFrame:
    baseline_load_shed = float(baseline_summary.get("total_load_shed_mwh", 0.0))
    scenario_load_shed = float(scenario_summary.get("total_load_shed_mwh", 0.0))
    baseline_import = float(baseline_summary.get("total_import_slack_mwh", 0.0))
    scenario_import = float(scenario_summary.get("total_import_slack_mwh", 0.0))
    voll = float(scenario_summary.get("value_of_lost_load_usd_per_mwh", 10_000.0))
    import_cost = float(scenario_summary.get("import_slack_marginal_cost_usd_per_mwh", 5_000.0))

    incremental = pd.DataFrame(
        [
            {
                "hazard": args.hazard,
                "threshold": args.threshold,
                "outaged_lines": len(outage_report),
                "outaged_line_length_km": float(outage_report["length"].sum()) if not outage_report.empty else 0.0,
                "outaged_line_capacity_mva": float(outage_report["s_nom"].sum()) if not outage_report.empty else 0.0,
                "baseline_load_shed_mwh": baseline_load_shed,
                "scenario_load_shed_mwh": scenario_load_shed,
                "incremental_load_shed_mwh": scenario_load_shed - baseline_load_shed,
                "baseline_import_slack_mwh": baseline_import,
                "scenario_import_slack_mwh": scenario_import,
                "incremental_import_slack_mwh": scenario_import - baseline_import,
                "incremental_load_shedding_cost_usd": (scenario_load_shed - baseline_load_shed) * voll,
                "incremental_import_slack_cost_usd": (scenario_import - baseline_import) * import_cost,
                "load_shedding_still_occurs": bool(
                    scenario_summary.get("load_shedding_still_occurs_after_imports", False)
                ),
            }
        ]
    )
    incremental.to_csv(output_dir / "incremental_impact_summary.csv", index=False)
    return incremental


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir
    if output_dir is None:
        safe_threshold = str(args.threshold).replace(".", "p")
        output_dir = PYPSA_DIR / "hazard_scenarios" / f"{args.hazard}_threshold_{safe_threshold}"
    output_dir.mkdir(parents=True, exist_ok=True)

    network = pypsa.Network(args.network)
    if not args.all_components:
        keep_buses = largest_connected_component_buses(network)
        removed = restrict_to_buses(network, keep_buses)
        print("Restricted to largest connected component.")
        print("Kept buses:", len(keep_buses))
        print("Removed buses:", removed["removed_buses"])
        print("Removed lines:", removed["removed_lines"])
        print("Removed generators:", removed["removed_generators"])
        print("Removed loads:", removed["removed_loads"])

    outage_report = outage_pypsa_lines(network, args.hazard, args.threshold, output_dir)

    snapshots = select_snapshots(
        network,
        start=args.start,
        periods=args.periods,
        all_snapshots=args.all_snapshots,
    )
    import_buses = infer_import_buses(network, args.import_bus, args.num_import_buses)
    import_generators = add_import_slack_generators(network, import_buses)
    load_shedding_generators = add_standard_load_shedding(network)

    print("\nScenario dispatch setup")
    print("Network:", args.network)
    print("Output directory:", output_dir)
    print("Buses:", len(network.buses))
    print("Active lines after outage:", int(network.lines["active"].sum()))
    print("Inactive/outaged lines:", int((~network.lines["active"].astype(bool)).sum()))
    print("Import slack buses:", ", ".join(import_buses))
    print("Load-shedding generators:", len(load_shedding_generators))

    status, condition = solve_dispatch(network, snapshots, args.solver)
    print("Optimization status:", status)
    print("Optimization condition:", condition)
    if status.lower() not in {"ok", "warning"}:
        raise RuntimeError(f"Optimization did not finish successfully: {status}, {condition}")

    scenario_summary = summarize_dispatch_results(
        network,
        import_generators,
        load_shedding_generators,
        snapshots,
        output_dir,
    ).iloc[0]

    baseline_summary = read_baseline_summary(args.baseline_summary)
    incremental = write_incremental_summary(
        scenario_summary,
        baseline_summary,
        outage_report,
        args,
        output_dir,
    )

    print("\nIncremental hazard impact vs calibrated baseline")
    print(incremental.to_string(index=False))
    print("\nSaved:")
    print(output_dir / "outaged_lines.csv")
    print(output_dir / "dispatch_summary.csv")
    print(output_dir / "dispatch_by_snapshot.csv")
    print(output_dir / "load_shedding_by_bus.csv")
    print(output_dir / "incremental_impact_summary.csv")


if __name__ == "__main__":
    main()
