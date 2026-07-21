"""
Run gradual-damage Florida PyPSA hazard scenarios across available return periods.

The suite uses:
  - SNAIL-derived JRC flood exposure tables with NHESS F6.2 vulnerability curve.
  - STORM fixed return-period TC wind exposure with NHESS W6.3 vulnerability curve.

Outputs are written to:
    data/Electricity/pypsa_florida_network/calibrated_hazard_scenarios/
        gradual_return_period_suite/

The W6.3 curve is an overhead-line fragility curve. TC generators use separate
W1.* generator curves assigned by carrier.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from run_florida_pypsa_calibrated_hazard_scenarios import (
    DEFAULT_OUTPUT_DIR,
    CALIBRATED_BASELINE_DIR,
    LINE_CAPACITY_MULTIPLIER,
    PROJECT_DIR,
    PYPSA_DIR,
    run_scenario,
)


EXPOSURE_DIR = PROJECT_DIR / "data" / "Exposure"
FLOOD_LINE_EXPOSURE = (
    EXPOSURE_DIR
    / "line_flood_exposure_with_ids"
    / "flood_line_exposure_by_line_return_period.csv"
)
FLOOD_GENERATOR_EXPOSURE = (
    EXPOSURE_DIR
    / "powerplant_polygon_flood_exposure"
    / "powerplant_polygon_flood_exposure.csv"
)
TC_LINE_EXPOSURE_DIR = EXPOSURE_DIR / "florida_clean_assets_tc_snail_intersection"
TC_GENERATOR_EXPOSURE = (
    EXPOSURE_DIR
    / "powerplant_polygon_tc_exposure"
    / "powerplant_polygon_tc_exposure.csv"
)
BUS_EXPOSURE_DIR = EXPOSURE_DIR / "pypsa_bus_hazard_exposure"
FLOOD_BUS_EXPOSURE = BUS_EXPOSURE_DIR / "pypsa_bus_flood_exposure_by_return_period.csv"
TC_BUS_EXPOSURE = BUS_EXPOSURE_DIR / "pypsa_bus_tc_exposure_by_return_period.csv"

OUTPUT_DIR = DEFAULT_OUTPUT_DIR / "gradual_return_period_suite"
MANIFEST_PATH = PYPSA_DIR / "gradual_return_period_suite_manifest.csv"
IMPROVED_NETWORK_DIR = (
    PROJECT_DIR
    / "data"
    / "Electricity"
    / "pypsa_florida_network_county_load_generator_overrides"
)

FLOOD_LINE_CURVE = "data/Cost/nhess_f62_distribution_elevated_crossings_flood_vulnerability_curve.csv"
FLOOD_GENERATOR_CURVE = "data/Cost/nhess_f11_f14_power_plant_flood_vulnerability_curves_long.csv"
FLOOD_SUBSTATION_CURVE = "data/Cost/nhess_f21_f23_substation_flood_vulnerability_curves_long.csv"
FLOOD_DISTRIBUTION_LINE_CURVE = "data/Cost/nhess_f51_f61_f62_distribution_line_flood_vulnerability_curves_long.csv"
TC_CURVE = "data/Cost/nhess_w63_fpl_overhead_lines_tc_fragility_curve.csv"
TC_GENERATOR_CURVE = "data/Cost/nhess_tc_generator_wind_fragility_curves_long.csv"
TC_SUBSTATION_CURVE = "data/Cost/nhess_w23_substation_open_area_tc_fragility_curve.csv"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run gradual return-period hazard suite.")
    parser.add_argument("--network-dir", type=Path, default=PYPSA_DIR)
    parser.add_argument("--baseline-dir", type=Path, default=CALIBRATED_BASELINE_DIR)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--manifest-path", type=Path, default=MANIFEST_PATH)
    parser.add_argument("--line-capacity-multiplier", type=float, default=LINE_CAPACITY_MULTIPLIER)
    parser.add_argument("--solver", default="highs")
    parser.add_argument("--highs-method", default="ipm")
    parser.add_argument("--start", default="2025-01-01 00:00:00")
    parser.add_argument("--periods", type=int, default=24)
    parser.add_argument("--chunk-size", type=int, default=24)
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip scenarios with an existing scenario_summary.csv.",
    )
    parser.add_argument(
        "--only-build-manifest",
        action="store_true",
        help="Write manifest and availability report without running PyPSA.",
    )
    return parser.parse_args()


def available_flood_return_periods() -> list[int]:
    line = pd.read_csv(FLOOD_LINE_EXPOSURE)
    gen = pd.read_csv(FLOOD_GENERATOR_EXPOSURE)
    bus = pd.read_csv(FLOOD_BUS_EXPOSURE) if FLOOD_BUS_EXPOSURE.exists() else pd.DataFrame()
    line_rps = set(pd.to_numeric(line["return_period"], errors="coerce").dropna().astype(int))
    gen_rps = set(pd.to_numeric(gen["return_period"], errors="coerce").dropna().astype(int))
    if not bus.empty and "return_period" in bus.columns:
        bus_rps = set(pd.to_numeric(bus["return_period"], errors="coerce").dropna().astype(int))
        return sorted(line_rps.intersection(gen_rps).intersection(bus_rps))
    return sorted(line_rps.intersection(gen_rps))


def available_tc_return_periods() -> list[int]:
    gen = pd.read_csv(TC_GENERATOR_EXPOSURE)
    bus = pd.read_csv(TC_BUS_EXPOSURE) if TC_BUS_EXPOSURE.exists() else pd.DataFrame()
    gen_rps = set(
        pd.to_numeric(
            gen.loc[gen["dataset"].astype(str).str.startswith("storm_rp"), "return_period"],
            errors="coerce",
        )
        .dropna()
        .astype(int)
    )
    line_rps = set()
    for path in TC_LINE_EXPOSURE_DIR.glob("snail_storm_rp*_florida_overhead_lines.gpkg"):
        rp_text = path.name.split("snail_storm_rp", 1)[1].split("_", 1)[0]
        if rp_text.isdigit():
            line_rps.add(int(rp_text))
    if not bus.empty and "return_period" in bus.columns:
        bus_rps = set(pd.to_numeric(bus["return_period"], errors="coerce").dropna().astype(int))
        return sorted(line_rps.intersection(gen_rps).intersection(bus_rps))
    return sorted(line_rps.intersection(gen_rps))


def build_manifest() -> tuple[pd.DataFrame, pd.DataFrame]:
    flood_rps = available_flood_return_periods()
    tc_rps = available_tc_return_periods()

    desired = [10, 20, 25, 50, 75, 100, 200, 250, 500]
    availability_rows = []
    for hazard, available in [("flood", flood_rps), ("tropical_cyclone", tc_rps)]:
        for rp in desired:
            availability_rows.append(
                {
                    "hazard": hazard,
                    "return_period": rp,
                    "available": rp in available,
                    "status": "will_run" if rp in available else "not_available",
                }
            )
    availability = pd.DataFrame(availability_rows)

    rows = []
    for rp in flood_rps:
        rows.append(
            {
                "scenario_id": f"flood_jrc_rp{rp}_f62_gradual",
                "hazard": "flood",
                "return_period": rp,
                "description": f"JRC RP{rp} flood: SNAIL line/raster exposure; transmission lines use F6.2 and generators use NHESS F1.1-F1.4 power-plant flood vulnerability curves",
                "line_damage_path": "data/Exposure/line_flood_exposure_with_ids/flood_line_exposure_by_line_return_period.csv",
                "line_filter_column": "return_period",
                "line_filter_value": rp,
                "line_id_column": "florida_line_id",
                "line_id_type": "florida_line_id",
                "line_value_column": "max_flood_depth_m",
                "line_threshold": "",
                "line_mode": "vulnerability_curve",
                "line_damage_fraction_column": "",
                "generator_damage_path": "data/Exposure/powerplant_polygon_flood_exposure/powerplant_polygon_flood_exposure.csv",
                "generator_filter_column": "return_period",
                "generator_filter_value": rp,
                "generator_value_column": "max_flood_depth_m",
                "generator_threshold": "",
                "generator_mode": "vulnerability_curve",
                "generator_damage_fraction_column": "",
                "generator_match_column": "gppd_ids",
                "line_curve_path": FLOOD_LINE_CURVE,
                "generator_curve_path": FLOOD_GENERATOR_CURVE,
                "bus_damage_path": "data/Exposure/pypsa_bus_hazard_exposure/pypsa_bus_flood_exposure_by_return_period.csv",
                "bus_filter_column": "return_period",
                "bus_filter_value": rp,
                "bus_id_column": "name",
                "bus_id_type": "pypsa_bus",
                "bus_value_column": "max_flood_depth_m",
                "bus_mode": "vulnerability_curve",
                "bus_curve_path": FLOOD_SUBSTATION_CURVE,
                "modelling_assumption": (
                    "Flood transmission lines use F6.2 distribution-circuit elevated-crossing vulnerability curve as the selected non-diked line-damage assumption; "
                    "generators use NHESS F1.1-F1.4 power-plant curves assigned by carrier and capacity. "
                    f"PyPSA buses/substations use {FLOOD_SUBSTATION_CURVE} with F2.1-F2.3 assigned by voltage and SNAIL bus/raster exposure. "
                    f"Distribution/minor-line curves {FLOOD_DISTRIBUTION_LINE_CURVE} are documented but not applied to the current high-voltage transmission-line model."
                ),
            }
        )

    for rp in tc_rps:
        rows.append(
            {
                "scenario_id": f"tc_storm_rp{rp}_w63_gradual",
                "hazard": "tropical_cyclone",
                "return_period": rp,
                "description": f"STORM RP{rp} TC wind: lines use NHESS W6.3 FPL overhead-line fragility curve and generators use NHESS TC generator wind fragility curves",
                "line_damage_path": f"data/Exposure/florida_clean_assets_tc_snail_intersection/snail_storm_rp{rp}_florida_overhead_lines.gpkg",
                "line_filter_column": "return_period",
                "line_filter_value": rp,
                "line_id_column": "florida_line_id",
                "line_id_type": "florida_line_id",
                "line_value_column": "tc_wind_speed_ms",
                "line_threshold": "",
                "line_mode": "vulnerability_curve",
                "line_damage_fraction_column": "",
                "generator_damage_path": "data/Exposure/powerplant_polygon_tc_exposure/powerplant_polygon_tc_exposure.csv",
                "generator_filter_column": "dataset",
                "generator_filter_value": f"storm_rp{rp}",
                "generator_value_column": "max_wind_speed_ms",
                "generator_threshold": "",
                "generator_mode": "vulnerability_curve",
                "generator_damage_fraction_column": "",
                "generator_match_column": "gppd_ids",
                "line_curve_path": TC_CURVE,
                "generator_curve_path": TC_GENERATOR_CURVE,
                "bus_damage_path": "data/Exposure/pypsa_bus_hazard_exposure/pypsa_bus_tc_exposure_by_return_period.csv",
                "bus_filter_column": "return_period",
                "bus_filter_value": rp,
                "bus_id_column": "name",
                "bus_id_type": "pypsa_bus",
                "bus_value_column": "tc_wind_speed_ms",
                "bus_mode": "vulnerability_curve",
                "bus_curve_path": TC_SUBSTATION_CURVE,
                "modelling_assumption": (
                    "TC transmission lines use W6.3 FPL overhead-line fragility curve. "
                    "TC generators use NHESS W1.10-W1.13 plus W1.6 wind-turbine fragility curves assigned by carrier; "
                    f"PyPSA buses/substations use {TC_SUBSTATION_CURVE} with W2.3 and SNAIL bus/raster exposure. "
                    "Fragility probability is used as the derating damage ratio for this first-pass operational scenario."
                ),
            }
        )

    manifest = pd.DataFrame(rows)
    MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)
    manifest.to_csv(MANIFEST_PATH, index=False)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    availability.to_csv(OUTPUT_DIR / "return_period_availability.csv", index=False)
    return manifest, availability


def fully_failed_count(path: Path, asset_type: str) -> int:
    file_name = f"{asset_type}_capacity_deratings.csv"
    derating_path = path / file_name
    if not derating_path.exists():
        return 0
    df = pd.read_csv(derating_path)
    return int(pd.to_numeric(df["damage_ratio"], errors="coerce").ge(1.0).sum())


def collect_suite_summary(manifest: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, scenario in manifest.iterrows():
        scenario_dir = OUTPUT_DIR / scenario["scenario_id"]
        summary_path = scenario_dir / "scenario_summary.csv"
        incremental_path = scenario_dir / "incremental_vs_calibrated_baseline.csv"
        if not summary_path.exists() or not incremental_path.exists():
            rows.append(
                {
                    "scenario_id": scenario["scenario_id"],
                    "hazard": scenario["hazard"],
                    "return_period": scenario["return_period"],
                    "status": "missing_output",
                    "solver_status": pd.NA,
                }
            )
            continue

        summary = pd.read_csv(summary_path).iloc[0]
        incremental = pd.read_csv(incremental_path).iloc[0]
        total_demand = float(summary["total_demand_mwh"])
        rows.append(
            {
                "scenario_id": scenario["scenario_id"],
                "hazard": scenario["hazard"],
                "return_period": int(scenario["return_period"]),
                "status": "complete",
                "solver_status": summary["solver_status"],
                "solver_condition": summary["solver_condition"],
                "derated_lines": int(summary["damaged_lines"]),
                "derated_generators": int(summary["damaged_generators"]),
                "fully_failed_lines": fully_failed_count(scenario_dir, "line"),
                "fully_failed_generators": fully_failed_count(scenario_dir, "generator"),
                "line_capacity_loss_mva": float(incremental.get("line_capacity_loss_mva", 0.0)),
                "generator_capacity_loss_mw": float(incremental.get("generator_capacity_loss_mw", 0.0)),
                "load_shed_mwh": float(summary["total_load_shed_mwh"]),
                "demand_served_mwh": float(summary["total_demand_served_mwh"]),
                "demand_served_percent": float(summary["total_demand_served_mwh"]) / total_demand * 100.0
                if total_demand
                else 0.0,
                "import_slack_mwh": float(summary["total_import_slack_mwh"]),
                "system_cost_usd": float(summary["total_system_cost_usd"]),
                "incremental_system_cost_usd": float(incremental.get("incremental_system_cost_usd", 0.0)),
                "modelling_assumption": scenario.get("modelling_assumption", ""),
            }
        )
    suite = pd.DataFrame(rows).sort_values(["hazard", "return_period"])
    suite.to_csv(OUTPUT_DIR / "gradual_return_period_suite_summary.csv", index=False)
    return suite


def plot_metric(
    suite: pd.DataFrame,
    metric: str,
    ylabel: str,
    title: str,
    output_name: str,
    combined: bool = True,
) -> None:
    complete = suite[suite["status"].eq("complete")].copy()
    fig, ax = plt.subplots(figsize=(8.5, 5.2))
    if combined:
        for hazard, group in complete.groupby("hazard"):
            group = group.sort_values("return_period")
            label = "Flood F6.2" if hazard == "flood" else "TC W6.3"
            ax.plot(group["return_period"], group[metric], marker="o", linewidth=2.2, label=label)
        ax.legend()
    else:
        group = complete.sort_values("return_period")
        ax.plot(group["return_period"], group[metric], marker="o", linewidth=2.2)
    ax.set_xlabel("Return period (years)")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / output_name, dpi=220)
    plt.close(fig)


def create_plots(suite: pd.DataFrame) -> None:
    metrics = [
        ("load_shed_mwh", "Load shed (MWh)", "Load shed by return period", "load_shed_by_return_period.png"),
        (
            "demand_served_percent",
            "Demand served (%)",
            "Percentage of demand served by return period",
            "demand_served_percent_by_return_period.png",
        ),
        ("system_cost_usd", "System cost (USD)", "System cost by return period", "system_cost_by_return_period.png"),
        (
            "line_capacity_loss_mva",
            "Transmission capacity lost (MVA)",
            "Transmission capacity lost by return period",
            "line_capacity_lost_by_return_period.png",
        ),
        (
            "generator_capacity_loss_mw",
            "Generation capacity lost (MW)",
            "Generation capacity lost by return period",
            "generator_capacity_lost_by_return_period.png",
        ),
        ("import_slack_mwh", "Import slack (MWh)", "Import slack by return period", "import_slack_by_return_period.png"),
    ]
    for metric, ylabel, title, output_name in metrics:
        plot_metric(suite, metric, ylabel, title, output_name, combined=True)

    plot_metric(
        suite,
        "load_shed_mwh",
        "Load shed (MWh)",
        "Flood vs tropical cyclone load shedding across return periods",
        "combined_flood_tc_load_shedding_by_return_period.png",
        combined=True,
    )


def main() -> None:
    global OUTPUT_DIR, MANIFEST_PATH
    args = parse_args()
    OUTPUT_DIR = args.output_dir
    MANIFEST_PATH = args.manifest_path
    manifest, availability = build_manifest()
    print("Saved suite manifest:", MANIFEST_PATH)
    print("Saved availability report:", OUTPUT_DIR / "return_period_availability.csv")
    print("\nAvailable return periods:")
    print(availability[availability["available"]].to_string(index=False))
    print("\nUnavailable requested return periods:")
    print(availability[~availability["available"]].to_string(index=False))

    if args.only_build_manifest:
        return

    run_args = argparse.Namespace(
        output_dir=OUTPUT_DIR,
        network_dir=args.network_dir,
        baseline_dir=args.baseline_dir,
        line_capacity_multiplier=args.line_capacity_multiplier,
        solver=args.solver,
        highs_method=args.highs_method,
        start=args.start,
        periods=args.periods,
        chunk_size=args.chunk_size,
    )

    scenario_summaries = []
    for _, scenario in manifest.iterrows():
        scenario_dir = OUTPUT_DIR / scenario["scenario_id"]
        if args.skip_existing and (scenario_dir / "scenario_summary.csv").exists():
            print("Skipping existing scenario:", scenario["scenario_id"])
            continue
        scenario_summaries.append(run_scenario(scenario, run_args))

    suite = collect_suite_summary(manifest)
    create_plots(suite)
    suite.to_csv(OUTPUT_DIR / "gradual_return_period_suite_summary.csv", index=False)
    pd.DataFrame(scenario_summaries).to_csv(OUTPUT_DIR / "raw_run_scenario_summaries.csv", index=False)
    print("\nSaved suite summary:", OUTPUT_DIR / "gradual_return_period_suite_summary.csv")
    print(suite.to_string(index=False, float_format=lambda value: f"{value:,.3f}"))


if __name__ == "__main__":
    main()
