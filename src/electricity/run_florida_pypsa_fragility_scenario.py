"""
Run a voltage-specific wind fragility scenario in PyPSA.

This uses an expected-capacity approximation:

    s_nom_effective = s_nom * (1 - p_failure)

where p_failure is interpolated from a voltage-specific fragility curve.
The scenario keeps buses, loads, and generators active, then solves dispatch
with emergency imports and load shedding and compares against the calibrated
baseline.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import geopandas as gpd
import numpy as np
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
TC_WIND_EXPOSURE_FILE = EXPOSURE_DIR / "lines_tropical_cyclone_exposure.gpkg"


W6_3 = {
    # Overhead lines constructed by FPL and third-party lines with FPL equipment.
    "curve_id": "W6.3",
    "asset": "Overhead lines",
    "wind_speed_ms": list(range(0, 63)),
    "p_failure": [
        0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000,
        0.000, 0.000, 0.000, 0.000, 0.000, 0.001, 0.001, 0.001, 0.002, 0.002,
        0.003, 0.004, 0.005, 0.006, 0.007, 0.009, 0.011, 0.013, 0.016, 0.019,
        0.023, 0.027, 0.031, 0.037, 0.043, 0.050, 0.058, 0.067, 0.077, 0.088,
        0.100, 0.113, 0.129, 0.145, 0.164, 0.184, 0.206, 0.230, 0.257, 0.285,
        0.317, 0.351, 0.388, 0.428, 0.472, 0.519, 0.569, 0.624, 0.683, 0.746,
        0.814, 0.886, 0.964,
    ],
}

W3_51 = {
    # Transmission structure, wind loading standard of 105 mph.
    "curve_id": "W3.51",
    "asset": "Transmission structure, 105 mph wind loading standard",
    "wind_speed_ms": list(range(0, 77)),
    "p_failure": (
        [0.000] * 42
        + [
            0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.002, 0.002, 0.002,
            0.003, 0.003, 0.004, 0.005, 0.006, 0.007, 0.008, 0.010, 0.012,
            0.015, 0.018, 0.021, 0.025, 0.031, 0.037, 0.045, 0.054, 0.065,
            0.078, 0.094, 0.113, 0.136, 0.164, 0.198, 0.239, 0.288,
        ]
    ),
}

W3_52 = {
    # Hardened transmission structure, wind loading standard of 130 mph.
    "curve_id": "W3.52",
    "asset": "Hardened transmission structure, 130 mph wind loading standard",
    "wind_speed_ms": list(range(0, 77)),
    "p_failure": (
        [0.000] * 55
        + [
            0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.002, 0.002, 0.003,
            0.003, 0.004, 0.004, 0.005, 0.006, 0.008, 0.009, 0.011, 0.014,
            0.016, 0.020, 0.024, 0.029,
        ]
    ),
}

CURVES = {curve["curve_id"]: curve for curve in [W6_3, W3_51, W3_52]}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a voltage-specific TC wind fragility scenario."
    )
    parser.add_argument("--network", type=Path, default=DEFAULT_NETWORK_FILE)
    parser.add_argument("--baseline-summary", type=Path, default=DEFAULT_BASELINE_SUMMARY)
    parser.add_argument("--output-dir", type=Path, default=PYPSA_DIR / "hazard_scenarios" / "tc_wind_voltage_fragility")
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
    parser.add_argument(
        "--wind-scale",
        type=float,
        default=1.0,
        help="Optional multiplier on TC wind speed before applying fragility curves.",
    )
    return parser.parse_args()


def curve_for_voltage(v_nom: float) -> str:
    if pd.isna(v_nom) or v_nom <= 138:
        return "W6.3"
    if v_nom <= 345:
        return "W3.51"
    return "W3.52"


def interpolate_failure_probability(curve_id: str, wind_speed_ms: pd.Series) -> pd.Series:
    curve = CURVES[curve_id]
    speeds = np.array(curve["wind_speed_ms"], dtype=float)
    probs = np.array(curve["p_failure"], dtype=float)
    values = pd.to_numeric(wind_speed_ms, errors="coerce").fillna(0.0).to_numpy(dtype=float)
    return pd.Series(np.interp(values, speeds, probs, left=probs[0], right=probs[-1]), index=wind_speed_ms.index)


def load_wind_by_internal_line(wind_scale: float) -> pd.DataFrame:
    exposure = gpd.read_file(TC_WIND_EXPOSURE_FILE, ignore_geometry=True)
    print("\nTC wind exposure columns:")
    print(", ".join(exposure.columns.astype(str)))

    required = {"ID", "tc_wind_speed_max_ms"}
    missing = required.difference(exposure.columns)
    if missing:
        raise ValueError(f"{TC_WIND_EXPOSURE_FILE} missing columns: {sorted(missing)}")

    exposure["ID"] = pd.to_numeric(exposure["ID"], errors="coerce")
    exposure["tc_wind_speed_max_ms"] = pd.to_numeric(exposure["tc_wind_speed_max_ms"], errors="coerce")
    exposure = exposure.dropna(subset=["ID"])
    exposure["ID"] = exposure["ID"].astype(int)

    wind_by_hifld = (
        exposure.groupby("ID", as_index=False)["tc_wind_speed_max_ms"]
        .max()
        .rename(columns={"ID": "hifld_id"})
    )

    crosswalk = pd.read_csv(LINE_CROSSWALK_FILE)
    crosswalk = crosswalk[["florida_line_id", "ID"]].rename(columns={"ID": "hifld_id"})
    matched = crosswalk.merge(wind_by_hifld, on="hifld_id", how="left")
    matched["tc_wind_speed_scaled_ms"] = matched["tc_wind_speed_max_ms"].fillna(0.0) * wind_scale
    return matched


def apply_voltage_fragility(network: pypsa.Network, wind_scale: float, output_dir: Path) -> pd.DataFrame:
    wind = load_wind_by_internal_line(wind_scale)
    lines = network.lines.copy()
    lines["source_edge_id"] = pd.to_numeric(lines["source_edge_id"], errors="coerce")
    lines = lines.reset_index(names="line").merge(
        wind,
        left_on="source_edge_id",
        right_on="florida_line_id",
        how="left",
    )
    lines["tc_wind_speed_scaled_ms"] = lines["tc_wind_speed_scaled_ms"].fillna(0.0)
    lines["fragility_curve_id"] = lines["v_nom"].apply(curve_for_voltage)

    lines["failure_probability"] = 0.0
    for curve_id in CURVES:
        mask = lines["fragility_curve_id"].eq(curve_id)
        if mask.any():
            lines.loc[mask, "failure_probability"] = interpolate_failure_probability(
                curve_id,
                lines.loc[mask, "tc_wind_speed_scaled_ms"],
            ).to_numpy()

    lines["original_s_nom"] = lines["s_nom"]
    lines["effective_s_nom"] = lines["original_s_nom"] * (1.0 - lines["failure_probability"])
    lines["expected_capacity_loss_mva"] = lines["original_s_nom"] - lines["effective_s_nom"]

    network.lines.loc[lines["line"], "s_nom"] = lines.set_index("line")["effective_s_nom"]

    report_cols = [
        "line",
        "bus0",
        "bus1",
        "v_nom",
        "source_edge_id",
        "hifld_id",
        "tc_wind_speed_max_ms",
        "tc_wind_speed_scaled_ms",
        "fragility_curve_id",
        "failure_probability",
        "original_s_nom",
        "effective_s_nom",
        "expected_capacity_loss_mva",
        "length",
    ]
    report = lines[report_cols].copy()
    report.to_csv(output_dir / "line_fragility_capacity_adjustments.csv", index=False)

    curve_summary = (
        report.groupby("fragility_curve_id")
        .agg(
            lines=("line", "size"),
            mean_wind_ms=("tc_wind_speed_scaled_ms", "mean"),
            mean_failure_probability=("failure_probability", "mean"),
            max_failure_probability=("failure_probability", "max"),
            original_capacity_mva=("original_s_nom", "sum"),
            expected_capacity_loss_mva=("expected_capacity_loss_mva", "sum"),
        )
        .reset_index()
    )
    curve_summary.to_csv(output_dir / "fragility_curve_summary.csv", index=False)

    print("\nVoltage-specific fragility application")
    print("Wind scale:", wind_scale)
    print("Mean failure probability:", round(float(report["failure_probability"].mean()), 6))
    print("Max failure probability:", round(float(report["failure_probability"].max()), 6))
    print("Expected capacity loss MVA:", round(float(report["expected_capacity_loss_mva"].sum()), 3))
    print(curve_summary.to_string(index=False))
    return report


def read_baseline_summary(path: Path) -> pd.Series:
    baseline = pd.read_csv(path)
    if baseline.empty:
        raise ValueError(f"Baseline summary is empty: {path}")
    return baseline.iloc[0]


def write_incremental_summary(
    scenario_summary: pd.Series,
    baseline_summary: pd.Series,
    fragility_report: pd.DataFrame,
    args: argparse.Namespace,
    output_dir: Path,
) -> pd.DataFrame:
    baseline_load_shed = float(baseline_summary.get("total_load_shed_mwh", 0.0))
    scenario_load_shed = float(scenario_summary.get("total_load_shed_mwh", 0.0))
    baseline_import = float(baseline_summary.get("total_import_slack_mwh", 0.0))
    scenario_import = float(scenario_summary.get("total_import_slack_mwh", 0.0))
    voll = float(scenario_summary.get("value_of_lost_load_usd_per_mwh", 10_000.0))
    import_cost = float(scenario_summary.get("import_slack_marginal_cost_usd_per_mwh", 5_000.0))

    summary = pd.DataFrame(
        [
            {
                "hazard": "tc_wind",
                "method": "voltage_specific_fragility_expected_capacity",
                "wind_scale": args.wind_scale,
                "lines_adjusted": len(fragility_report),
                "mean_failure_probability": float(fragility_report["failure_probability"].mean()),
                "max_failure_probability": float(fragility_report["failure_probability"].max()),
                "expected_capacity_loss_mva": float(fragility_report["expected_capacity_loss_mva"].sum()),
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
    summary.to_csv(output_dir / "incremental_impact_summary.csv", index=False)
    return summary


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

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

    fragility_report = apply_voltage_fragility(network, args.wind_scale, args.output_dir)

    snapshots = select_snapshots(network, args.start, args.periods, args.all_snapshots)
    import_buses = infer_import_buses(network, args.import_bus, args.num_import_buses)
    import_generators = add_import_slack_generators(network, import_buses)
    load_shedding_generators = add_standard_load_shedding(network)

    print("\nScenario dispatch setup")
    print("Output directory:", args.output_dir)
    print("Buses:", len(network.buses))
    print("Lines:", len(network.lines))
    print("Import slack buses:", ", ".join(import_buses))

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
        args.output_dir,
    ).iloc[0]
    baseline_summary = read_baseline_summary(args.baseline_summary)
    incremental = write_incremental_summary(
        scenario_summary,
        baseline_summary,
        fragility_report,
        args,
        args.output_dir,
    )

    print("\nIncremental fragility impact vs calibrated baseline")
    print(incremental.to_string(index=False))
    print("\nSaved:")
    print(args.output_dir / "line_fragility_capacity_adjustments.csv")
    print(args.output_dir / "fragility_curve_summary.csv")
    print(args.output_dir / "dispatch_summary.csv")
    print(args.output_dir / "incremental_impact_summary.csv")


if __name__ == "__main__":
    main()
