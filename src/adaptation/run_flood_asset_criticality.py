"""Screen flood-exposed Florida PyPSA assets for adaptation criticality.

This script reads existing solved flood return-period scenarios, builds baseline
asset-impact tables, selects exposed candidate assets, optionally runs one-at-a-
time protection counterfactuals, and ranks assets by avoided expected annual
load shedding (EENS).

The protection assumption is deliberately simple: one asset is restored to its
pre-hazard capacity while all other flood damage remains unchanged. This is a
screening test, not a final engineering design or cost-benefit analysis.
"""

from __future__ import annotations

import argparse
import logging
import shutil
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

matplotlib.use("Agg")

PROJECT_ROOT = Path(r"C:\oxford_tc_project")
ELECTRICITY_DIR = PROJECT_ROOT / "data" / "Electricity"
SCRIPT_DIR = ELECTRICITY_DIR
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from run_florida_pypsa_baseline_validation import (  # noqa: E402
    dispatch_by_carrier,
    save_import_slack,
    save_line_loading,
    save_load_shedding,
    select_snapshots,
    solve_dispatch_in_chunks,
    write_baseline_summary,
)
from run_florida_pypsa_calibrated_hazard_scenarios import (  # noqa: E402
    apply_calibrated_import_caps,
    apply_generator_damage,
    apply_line_damage,
    calibrated_import_buses,
    load_calibrated_network,
    safe_name,
    system_cost_from_dispatch,
)
from run_florida_pypsa_load_shedding_dispatch import (  # noqa: E402
    add_import_slack_generators,
    add_standard_load_shedding,
)
import run_florida_pypsa_calibrated_hazard_scenarios as hazard_runner  # noqa: E402


DEFAULT_NETWORK_DIR = ELECTRICITY_DIR / "pypsa_florida_network"
DEFAULT_SUITE_DIR = DEFAULT_NETWORK_DIR / "calibrated_hazard_scenarios" / "gradual_return_period_suite"
DEFAULT_MANIFEST = DEFAULT_NETWORK_DIR / "gradual_return_period_suite_manifest.csv"
DEFAULT_OUTPUT_DIR = ELECTRICITY_DIR / "flood_adaptation_analysis" / "outputs"
DEFAULT_BASELINE_DIR = DEFAULT_NETWORK_DIR / "baseline_calibrated_no_hazard"
DEFAULT_ANALYSIS_ROOT = ELECTRICITY_DIR / "flood_adaptation_analysis"
LINE_CAPACITY_MULTIPLIER = 2.0


def load_vulnerability_curve_with_f63(path: Path) -> tuple[pd.DataFrame, str, str]:
    """Load vulnerability curves, including older F6.3 column names."""
    curve = pd.read_csv(path)
    intensity_candidates = ["flood_depth_m", "wind_speed_ms", "hazard_intensity", "intensity"]
    damage_candidates = [
        "nhess_f63_damage_ratio",
        "nhess_f62_damage_ratio",
        "damage_ratio",
        "vulnerability",
    ]
    intensity_col = next((col for col in intensity_candidates if col in curve.columns), None)
    damage_col = next((col for col in damage_candidates if col in curve.columns), None)
    if intensity_col is None or damage_col is None:
        raise ValueError(f"Could not identify intensity/damage columns in {path}. Columns are: {list(curve.columns)}")
    curve = curve[[intensity_col, damage_col]].copy()
    curve[intensity_col] = pd.to_numeric(curve[intensity_col], errors="coerce")
    curve[damage_col] = pd.to_numeric(curve[damage_col], errors="coerce")
    curve = curve.dropna().sort_values(intensity_col).drop_duplicates(intensity_col)
    curve[damage_col] = curve[damage_col].clip(0.0, 1.0)
    return curve, intensity_col, damage_col


hazard_runner.load_vulnerability_curve = load_vulnerability_curve_with_f63


@dataclass(frozen=True)
class ScenarioInfo:
    """A solved flood scenario discovered in the existing suite."""

    scenario_id: str
    return_period_years: int
    annual_exceedance_probability: float
    scenario_dir: Path


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--network-dir", type=Path, default=DEFAULT_NETWORK_DIR)
    parser.add_argument("--baseline-dir", type=Path, default=DEFAULT_BASELINE_DIR)
    parser.add_argument("--flood-scenario-manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--suite-dir", type=Path, default=DEFAULT_SUITE_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--asset-types", nargs="+", default=["line", "generator"])
    parser.add_argument("--maximum-number-of-candidates", type=int, default=20)
    parser.add_argument("--solver", default="highs")
    parser.add_argument("--highs-method", default="ipm")
    parser.add_argument("--start", default="2025-01-01 00:00:00")
    parser.add_argument("--periods", type=int, default=24)
    parser.add_argument("--chunk-size", type=int, default=24)
    parser.add_argument("--line-capacity-multiplier", type=float, default=LINE_CAPACITY_MULTIPLIER)
    parser.add_argument("--test-mode", action="store_true", help="Run only one candidate and one scenario.")
    parser.add_argument("--run-counterfactuals", action="store_true", help="Run PyPSA one-asset protection counterfactuals.")
    parser.add_argument("--resume", action="store_true", default=True)
    parser.add_argument("--hazard-model-token", default="f63", help="Filter flood scenarios by token, e.g. f63 or f62.")
    return parser.parse_args()


def setup_logging(output_dir: Path) -> None:
    """Configure file and console logging."""
    log_dir = DEFAULT_ANALYSIS_ROOT / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(log_dir / "run_flood_asset_criticality.log", mode="w", encoding="utf-8"),
        ],
    )


def ensure_structure(output_dir: Path) -> dict[str, Path]:
    """Create required output folders and return their paths."""
    paths = {
        "baseline": output_dir / "baseline",
        "counterfactual": output_dir / "individual_asset_protection",
        "rankings": output_dir / "rankings",
        "figures": output_dir / "figures",
    }
    for path in paths.values():
        path.mkdir(parents=True, exist_ok=True)
    (DEFAULT_ANALYSIS_ROOT / "scripts").mkdir(parents=True, exist_ok=True)
    return paths


def read_csv(path: Path) -> pd.DataFrame:
    """Read a CSV, returning an empty frame for empty files."""
    if not path.exists() or path.stat().st_size == 0:
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except pd.errors.EmptyDataError:
        return pd.DataFrame()


def find_flood_scenarios(suite_dir: Path, token: str) -> list[ScenarioInfo]:
    """Discover solved flood return-period scenarios in a suite folder."""
    scenarios: list[ScenarioInfo] = []
    for scenario_dir in sorted(suite_dir.glob(f"flood_jrc_rp*_{token}_gradual")):
        summary = read_csv(scenario_dir / "scenario_summary.csv")
        definition = read_csv(scenario_dir / "scenario_definition.csv")
        if summary.empty or definition.empty:
            continue
        return_period = int(pd.to_numeric(definition.loc[0, "return_period"], errors="coerce"))
        scenarios.append(
            ScenarioInfo(
                scenario_id=scenario_dir.name,
                return_period_years=return_period,
                annual_exceedance_probability=1.0 / float(return_period),
                scenario_dir=scenario_dir,
            )
        )
    scenarios.sort(key=lambda item: item.return_period_years)
    return scenarios


def inventory_existing_files(suite_dir: Path, scenarios: list[ScenarioInfo], output_path: Path) -> pd.DataFrame:
    """Create a concise inventory of relevant existing flood files."""
    rows = []
    relevant = {
        "scenario_summary.csv": "result/system-level",
        "scenario_definition.csv": "input/scenario",
        "line_capacity_deratings.csv": "result/asset-level",
        "generator_capacity_deratings.csv": "result/asset-level",
        "bus_substation_deratings.csv": "result/asset-level",
        "damaged_lines.csv": "result/asset-level",
        "damaged_generators.csv": "result/asset-level",
        "load_shedding.csv": "result/system-level",
        "load_shedding_by_bus.csv": "result/asset-level",
        "incremental_vs_calibrated_baseline.csv": "result/system-level",
        "line_loading.csv": "result/asset-level",
        "congested_corridors.csv": "result/asset-level",
    }
    for scenario in scenarios:
        for name, category in relevant.items():
            path = scenario.scenario_dir / name
            if not path.exists():
                continue
            df = read_csv(path)
            rows.append(
                {
                    "full_file_path": str(path),
                    "file_type": path.suffix.lower().lstrip("."),
                    "relevant_columns_or_variables": ", ".join(df.columns[:30]) if not df.empty else "empty file",
                    "scenario": scenario.scenario_id,
                    "return_period_years": scenario.return_period_years,
                    "role": category,
                    "contains_asset_level_information": "asset-level" in category,
                    "contains_system_level_information": "system-level" in category,
                }
            )
    inventory = pd.DataFrame(rows)
    inventory.to_csv(output_path, index=False)
    return inventory


def build_scenario_summary(scenarios: list[ScenarioInfo]) -> pd.DataFrame:
    """Standardize scenario-level flood summary outputs."""
    rows = []
    for scenario in scenarios:
        summary = read_csv(scenario.scenario_dir / "scenario_summary.csv")
        if summary.empty:
            continue
        row = summary.iloc[0].to_dict()
        rows.append(
            {
                "scenario": scenario.scenario_id,
                "return_period_years": scenario.return_period_years,
                "annual_exceedance_probability": scenario.annual_exceedance_probability,
                "total_load_shed_mwh": float(row.get("total_load_shed_mwh", np.nan)),
                "peak_load_shed_mw": float(row.get("maximum_hourly_load_shed_mw", np.nan)),
                "demand_served_mwh": float(row.get("total_demand_served_mwh", np.nan)),
                "system_cost": float(row.get("total_system_cost_usd", np.nan)),
                "total_capacity_lost": np.nan,
                "damaged_lines": int(row.get("damaged_lines", 0) or 0),
                "damaged_generators": int(row.get("damaged_generators", 0) or 0),
                "damaged_buses_or_substations": row.get("damaged_buses", np.nan),
                "physical_damage_usd": np.nan,
                "solver_status": row.get("solver_status", ""),
                "solver_condition": row.get("solver_condition", ""),
            }
        )
    out = pd.DataFrame(rows).sort_values("return_period_years")
    line_loss = aggregate_capacity_loss(scenarios, "line", "capacity_loss_mva")
    gen_loss = aggregate_capacity_loss(scenarios, "generator", "capacity_loss_mw")
    if not out.empty:
        out = out.merge(line_loss.rename(columns={"capacity_loss": "line_capacity_loss_mva"}), on="scenario", how="left")
        out = out.merge(gen_loss.rename(columns={"capacity_loss": "generator_capacity_loss_mw"}), on="scenario", how="left")
        out["total_capacity_lost"] = out[["line_capacity_loss_mva", "generator_capacity_loss_mw"]].sum(axis=1, min_count=1)
    return out


def aggregate_capacity_loss(scenarios: list[ScenarioInfo], asset_type: str, loss_col: str) -> pd.DataFrame:
    """Aggregate capacity loss for each scenario."""
    rows = []
    for scenario in scenarios:
        path = scenario.scenario_dir / f"{asset_type}_capacity_deratings.csv"
        df = read_csv(path)
        rows.append(
            {
                "scenario": scenario.scenario_id,
                "capacity_loss": float(pd.to_numeric(df.get(loss_col, pd.Series(dtype=float)), errors="coerce").fillna(0).sum())
                if not df.empty
                else 0.0,
            }
        )
    return pd.DataFrame(rows)


def load_network_tables(network_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load base network CSV tables for location and naming fields."""
    buses = read_csv(network_dir / "buses.csv")
    lines = read_csv(network_dir / "lines.csv")
    generators = read_csv(network_dir / "generators.csv")
    if "name" not in buses.columns and buses.index.name is None:
        pass
    return buses, lines, generators


def bus_lookup(buses: pd.DataFrame) -> pd.DataFrame:
    """Return bus lookup indexed by bus name."""
    buses = buses.copy()
    if "name" not in buses.columns:
        buses = buses.reset_index().rename(columns={"index": "name"})
    cols = [c for c in ["name", "x", "y", "substation_name", "county", "county_fips"] if c in buses.columns]
    return buses[cols].drop_duplicates("name")


def standardize_asset_impacts(
    scenarios: list[ScenarioInfo],
    network_dir: Path,
    asset_types: Iterable[str],
) -> pd.DataFrame:
    """Build a standardized asset-level impact table from existing derating outputs."""
    buses, _lines, _generators = load_network_tables(network_dir)
    bus_info = bus_lookup(buses)
    bus_xy = bus_info.set_index("name")[["x", "y"]] if {"name", "x", "y"}.issubset(bus_info.columns) else pd.DataFrame()
    frames = []
    for scenario in scenarios:
        if "line" in asset_types:
            df = read_csv(scenario.scenario_dir / "line_capacity_deratings.csv")
            if not df.empty:
                work = pd.DataFrame(
                    {
                        "scenario": scenario.scenario_id,
                        "return_period_years": scenario.return_period_years,
                        "annual_exceedance_probability": scenario.annual_exceedance_probability,
                        "asset_type": "line",
                        "asset_id": df["line"].astype(str),
                        "asset_name": df.get("source_edge_id", df["line"]).astype(str),
                        "bus0": df.get("bus0"),
                        "bus1": df.get("bus1"),
                        "bus": pd.NA,
                        "flood_depth_m": pd.to_numeric(df.get("hazard_intensity"), errors="coerce"),
                        "original_capacity": pd.to_numeric(df.get("original_s_nom_mva"), errors="coerce"),
                        "damage_ratio": pd.to_numeric(df.get("damage_ratio"), errors="coerce"),
                        "remaining_capacity": pd.to_numeric(df.get("reduced_s_nom_mva"), errors="coerce"),
                        "capacity_loss": pd.to_numeric(df.get("capacity_loss_mva"), errors="coerce"),
                        "failed_or_derated": pd.to_numeric(df.get("damage_ratio"), errors="coerce").fillna(0).gt(0),
                        "replacement_value_usd": np.nan,
                        "physical_damage_usd": np.nan,
                    }
                )
                if not bus_xy.empty:
                    b0 = work["bus0"].map(bus_xy["x"])
                    b1 = work["bus1"].map(bus_xy["x"])
                    work["longitude"] = pd.concat([b0, b1], axis=1).mean(axis=1)
                    b0y = work["bus0"].map(bus_xy["y"])
                    b1y = work["bus1"].map(bus_xy["y"])
                    work["latitude"] = pd.concat([b0y, b1y], axis=1).mean(axis=1)
                frames.append(work)
        if "generator" in asset_types:
            df = read_csv(scenario.scenario_dir / "generator_capacity_deratings.csv")
            if not df.empty:
                work = pd.DataFrame(
                    {
                        "scenario": scenario.scenario_id,
                        "return_period_years": scenario.return_period_years,
                        "annual_exceedance_probability": scenario.annual_exceedance_probability,
                        "asset_type": "generator",
                        "asset_id": df["generator"].astype(str),
                        "asset_name": df.get("source_name", df["generator"]).astype(str),
                        "bus0": pd.NA,
                        "bus1": pd.NA,
                        "bus": df.get("bus"),
                        "flood_depth_m": pd.to_numeric(df.get("hazard_intensity"), errors="coerce"),
                        "original_capacity": pd.to_numeric(df.get("original_p_nom_mw"), errors="coerce"),
                        "damage_ratio": pd.to_numeric(df.get("damage_ratio"), errors="coerce"),
                        "remaining_capacity": pd.to_numeric(df.get("reduced_p_nom_mw"), errors="coerce"),
                        "capacity_loss": pd.to_numeric(df.get("capacity_loss_mw"), errors="coerce"),
                        "failed_or_derated": pd.to_numeric(df.get("damage_ratio"), errors="coerce").fillna(0).gt(0),
                        "replacement_value_usd": np.nan,
                        "physical_damage_usd": np.nan,
                    }
                )
                if not bus_xy.empty:
                    work["longitude"] = work["bus"].map(bus_xy["x"])
                    work["latitude"] = work["bus"].map(bus_xy["y"])
                frames.append(work)
    impacts = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    for col in ["longitude", "latitude", "county", "county_fips"]:
        if col not in impacts.columns:
            impacts[col] = pd.NA
    return impacts


def load_bus_shedding_for_scenarios(scenarios: list[ScenarioInfo]) -> pd.DataFrame:
    """Load bus-level load shedding for scenario association metrics."""
    frames = []
    for scenario in scenarios:
        df = read_csv(scenario.scenario_dir / "load_shedding_by_bus.csv")
        if df.empty:
            continue
        df["scenario"] = scenario.scenario_id
        df["return_period_years"] = scenario.return_period_years
        frames.append(df)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def build_candidates(impacts: pd.DataFrame, scenario_summary: pd.DataFrame, bus_shed: pd.DataFrame) -> pd.DataFrame:
    """Build flood-exposed candidate asset summary."""
    if impacts.empty:
        return pd.DataFrame()
    work = impacts.copy()
    affected = (
        work["flood_depth_m"].fillna(0).gt(0)
        | work["damage_ratio"].fillna(0).gt(0)
        | work["capacity_loss"].fillna(0).gt(0)
        | work["failed_or_derated"].fillna(False)
    )
    work = work.loc[affected].copy()
    if bus_shed.empty:
        work["associated_baseline_load_shed_mwh"] = 0.0
    else:
        shed = bus_shed[["scenario", "bus", "total_load_shed_mwh"]].copy()
        line_b0 = shed.rename(columns={"bus": "bus0", "total_load_shed_mwh": "bus0_shed"})
        line_b1 = shed.rename(columns={"bus": "bus1", "total_load_shed_mwh": "bus1_shed"})
        gen_bus = shed.rename(columns={"total_load_shed_mwh": "bus_shed"})
        work = work.merge(line_b0, on=["scenario", "bus0"], how="left")
        work = work.merge(line_b1, on=["scenario", "bus1"], how="left")
        work = work.merge(gen_bus, on=["scenario", "bus"], how="left")
        work["associated_baseline_load_shed_mwh"] = work[["bus0_shed", "bus1_shed", "bus_shed"]].fillna(0).sum(axis=1)
    base_shed = scenario_summary.set_index("scenario")["total_load_shed_mwh"]
    work["scenario_baseline_load_shed_mwh"] = work["scenario"].map(base_shed).fillna(0)
    grouped = (
        work.groupby(["asset_type", "asset_id"], as_index=False)
        .agg(
            asset_name=("asset_name", "first"),
            bus0=("bus0", "first"),
            bus1=("bus1", "first"),
            bus=("bus", "first"),
            longitude=("longitude", "first"),
            latitude=("latitude", "first"),
            county=("county", "first"),
            county_fips=("county_fips", "first"),
            scenarios_affected=("scenario", "nunique"),
            first_damaging_return_period=("return_period_years", "min"),
            maximum_flood_depth_m=("flood_depth_m", "max"),
            maximum_damage_ratio=("damage_ratio", "max"),
            maximum_capacity_loss=("capacity_loss", "max"),
            average_damage_ratio=("damage_ratio", "mean"),
            baseline_load_shed_in_affected_scenarios=("scenario_baseline_load_shed_mwh", "sum"),
            associated_endpoint_or_bus_load_shed_mwh=("associated_baseline_load_shed_mwh", "sum"),
            total_capacity_loss_across_scenarios=("capacity_loss", "sum"),
        )
        .sort_values(
            ["associated_endpoint_or_bus_load_shed_mwh", "baseline_load_shed_in_affected_scenarios", "total_capacity_loss_across_scenarios"],
            ascending=False,
        )
    )
    return grouped


def expected_annual_loss(return_periods: pd.Series, losses: pd.Series) -> float:
    """Integrate a loss-exceedance curve using trapezoidal integration in AEP space.

    The curve uses p = 1 / return_period. A zero-loss point is added at p=1.
    The high-return-period tail is truncated at the smallest available AEP.
    """
    rp = pd.to_numeric(return_periods, errors="coerce")
    loss = pd.to_numeric(losses, errors="coerce")
    data = pd.DataFrame({"aep": 1.0 / rp, "loss": loss}).dropna().sort_values("aep")
    if data.empty:
        return 0.0
    if not np.isclose(data["aep"].max(), 1.0):
        data = pd.concat([data, pd.DataFrame([{"aep": 1.0, "loss": 0.0}])], ignore_index=True)
    data = data.sort_values("aep")
    return float(np.trapezoid(data["loss"].to_numpy(), data["aep"].to_numpy()))


def select_candidates_for_counterfactual(candidates: pd.DataFrame, maximum: int, test_mode: bool) -> pd.DataFrame:
    """Select candidates for counterfactual solving."""
    if candidates.empty:
        return candidates
    n = 1 if test_mode else maximum
    return candidates.head(n).copy()


def load_completed_results(path: Path) -> pd.DataFrame:
    """Load completed counterfactual rows for resume/checkpointing."""
    return read_csv(path)


def append_result(path: Path, row: dict) -> None:
    """Append a single result row to a CSV immediately."""
    df = pd.DataFrame([row])
    header = not path.exists() or path.stat().st_size == 0
    df.to_csv(path, mode="a", header=header, index=False)


def solve_damaged_case(
    scenario: ScenarioInfo,
    scenario_definition: pd.Series,
    args: argparse.Namespace,
    output_dir: Path,
    candidate: pd.Series | None = None,
) -> tuple[pd.Series, str]:
    """Solve one flood-damaged case, optionally restoring one protected asset."""
    output_dir.mkdir(parents=True, exist_ok=True)
    network = load_calibrated_network(args.network_dir, args.line_capacity_multiplier)
    snapshots = select_snapshots(network, args.start, args.periods, all_snapshots=False)
    damaged_lines = apply_line_damage(network, scenario_definition, output_dir)
    damaged_generators = apply_generator_damage(network, scenario_definition, output_dir)

    warning = ""
    if candidate is not None:
        asset_type = str(candidate["asset_type"])
        asset_id = str(candidate["asset_id"])
        if asset_type == "line":
            if asset_id not in network.lines.index:
                warning = "protected line missing from network"
            else:
                match = damaged_lines[damaged_lines["line"].astype(str).eq(asset_id)]
                if match.empty:
                    warning = "protected line was not damaged in this scenario"
                else:
                    original = float(match.iloc[0]["original_s_nom_mva"])
                    network.lines.loc[asset_id, "s_nom"] = original
                    network.lines.loc[asset_id, "active"] = True
        elif asset_type == "generator":
            if asset_id not in network.generators.index:
                warning = "protected generator missing from network"
            else:
                match = damaged_generators[damaged_generators["generator"].astype(str).eq(asset_id)]
                if match.empty:
                    warning = "protected generator was not damaged in this scenario"
                else:
                    original = float(match.iloc[0]["original_p_nom_mw"])
                    network.generators.loc[asset_id, "p_nom"] = original
        else:
            warning = f"counterfactual restore not implemented for asset_type={asset_type}"

    import_buses = calibrated_import_buses(args.baseline_dir)
    import_generators = add_import_slack_generators(network, import_buses)
    apply_calibrated_import_caps(network, import_generators, args.baseline_dir, output_dir)
    load_shedding_generators = add_standard_load_shedding(network)
    status, condition = solve_dispatch_in_chunks(
        network,
        snapshots,
        args.solver,
        args.highs_method,
        args.chunk_size,
    )
    _dispatch_hourly, generation_by_carrier = dispatch_by_carrier(network, snapshots, output_dir)
    load_shedding_hourly, load_shedding_by_bus = save_load_shedding(
        network,
        load_shedding_generators,
        snapshots,
        output_dir,
    )
    import_hourly, _import_by_bus = save_import_slack(network, import_generators, snapshots, output_dir)
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
    ).iloc[0]
    summary = summary.copy()
    summary["scenario_id"] = scenario.scenario_id
    summary["total_system_cost_usd"] = system_cost_from_dispatch(network, snapshots)
    summary["damaged_lines"] = len(damaged_lines)
    summary["damaged_generators"] = len(damaged_generators)
    pd.DataFrame([summary]).to_csv(output_dir / "scenario_summary.csv", index=False)
    scenario_definition.to_frame().T.to_csv(output_dir / "scenario_definition.csv", index=False)
    return summary, warning


def run_matching_baselines(
    scenarios: list[ScenarioInfo],
    args: argparse.Namespace,
    output_dir: Path,
) -> list[ScenarioInfo]:
    """Rerun flood baselines with the current network/code in the analysis folder."""
    baseline_root = output_dir / "current_flood_scenario_runs"
    baseline_root.mkdir(parents=True, exist_ok=True)
    scenarios_to_run = scenarios[:1] if args.test_mode else scenarios
    matched: list[ScenarioInfo] = []
    for scenario in scenarios_to_run:
        run_dir = baseline_root / scenario.scenario_id
        if args.resume and (run_dir / "scenario_summary.csv").exists():
            logging.info("Using existing matched baseline: %s", run_dir)
        else:
            definition = read_csv(scenario.scenario_dir / "scenario_definition.csv").iloc[0]
            logging.info("Running matched flood baseline: %s", scenario.scenario_id)
            solve_damaged_case(scenario, definition, args, run_dir)
        matched.append(
            ScenarioInfo(
                scenario_id=scenario.scenario_id,
                return_period_years=scenario.return_period_years,
                annual_exceedance_probability=scenario.annual_exceedance_probability,
                scenario_dir=run_dir,
            )
        )
    return matched


def run_counterfactual(
    scenario: ScenarioInfo,
    scenario_definition: pd.Series,
    candidate: pd.Series,
    args: argparse.Namespace,
    output_root: Path,
    baseline_summary: pd.Series,
) -> dict:
    """Run one protected-asset counterfactual."""
    asset_type = str(candidate["asset_type"])
    asset_id = str(candidate["asset_id"])
    run_id = f"{safe_name(scenario.scenario_id)}__{asset_type}__{safe_name(asset_id)}"
    run_dir = output_root / "runs" / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(prefix="flood_counterfactual_") as tmp:
        tmp_output = Path(tmp)
        protected_summary, warning = solve_damaged_case(scenario, scenario_definition, args, tmp_output, candidate)
        protected_cost = float(protected_summary.get("total_system_cost_usd", np.nan))
        for file in tmp_output.glob("*.csv"):
            shutil.copy2(file, run_dir / file.name)

    baseline_load = float(baseline_summary.get("total_load_shed_mwh", 0.0))
    protected_load = float(protected_summary.get("total_load_shed_mwh", np.nan))
    baseline_peak = float(baseline_summary.get("peak_load_shed_mw", baseline_summary.get("maximum_hourly_load_shed_mw", 0.0)))
    protected_peak = float(protected_summary.get("maximum_hourly_load_shed_mw", np.nan))
    baseline_cost = float(baseline_summary.get("system_cost", baseline_summary.get("total_system_cost_usd", np.nan)))
    avoided = baseline_load - protected_load
    row = {
        "asset_type": asset_type,
        "asset_id": asset_id,
        "asset_name": candidate.get("asset_name", ""),
        "scenario": scenario.scenario_id,
        "return_period_years": scenario.return_period_years,
        "annual_exceedance_probability": scenario.annual_exceedance_probability,
        "baseline_load_shed_mwh": baseline_load,
        "protected_load_shed_mwh": protected_load,
        "avoided_load_shed_mwh": avoided,
        "percent_load_shed_reduction": avoided / baseline_load * 100 if baseline_load > 0 else 0.0,
        "baseline_peak_load_shed_mw": baseline_peak,
        "protected_peak_load_shed_mw": protected_peak,
        "avoided_peak_load_shed_mw": baseline_peak - protected_peak,
        "baseline_system_cost": baseline_cost,
        "protected_system_cost": protected_cost,
        "avoided_system_cost": baseline_cost - protected_cost,
        "solver_status": protected_summary.get("solver_status", ""),
        "solver_condition": protected_summary.get("solver_condition", ""),
        "changed_result": abs(avoided) > 1e-6 or abs((baseline_cost - protected_cost)) > 1e-3,
        "warning": warning,
        "run_directory": str(run_dir),
    }
    if row["avoided_load_shed_mwh"] < -1e-6:
        row["warning"] = (row["warning"] + "; " if row["warning"] else "") + "protected run has more load shed than baseline"
    return row


def run_counterfactuals(
    scenarios: list[ScenarioInfo],
    candidates: pd.DataFrame,
    scenario_summary: pd.DataFrame,
    args: argparse.Namespace,
    output_path: Path,
) -> pd.DataFrame:
    """Run or recover individual protection counterfactuals."""
    selected = select_candidates_for_counterfactual(candidates, args.maximum_number_of_candidates, args.test_mode)
    scenarios_to_run = scenarios[:1] if args.test_mode else scenarios
    total_solves = len(selected) * len(scenarios_to_run)
    logging.info("Candidate-by-scenario solves requested: %s", total_solves)
    if selected.empty:
        return pd.DataFrame()
    completed = load_completed_results(output_path)
    completed_keys = set()
    if not completed.empty:
        usable_completed = completed[~completed.get("solver_status", "").astype(str).isin(["failed", "not_run"])]
        completed_keys = set(
            zip(
                usable_completed["asset_type"].astype(str),
                usable_completed["asset_id"].astype(str),
                usable_completed["scenario"].astype(str),
            )
        )

    if not args.run_counterfactuals:
        rows = []
        for _, candidate in selected.iterrows():
            for scenario in scenarios_to_run:
                baseline = scenario_summary[scenario_summary["scenario"].eq(scenario.scenario_id)].iloc[0]
                rows.append(
                    {
                        "asset_type": candidate["asset_type"],
                        "asset_id": candidate["asset_id"],
                        "asset_name": candidate.get("asset_name", ""),
                        "scenario": scenario.scenario_id,
                        "return_period_years": scenario.return_period_years,
                        "annual_exceedance_probability": scenario.annual_exceedance_probability,
                        "baseline_load_shed_mwh": baseline["total_load_shed_mwh"],
                        "protected_load_shed_mwh": np.nan,
                        "avoided_load_shed_mwh": np.nan,
                        "percent_load_shed_reduction": np.nan,
                        "baseline_peak_load_shed_mw": baseline["peak_load_shed_mw"],
                        "protected_peak_load_shed_mw": np.nan,
                        "avoided_peak_load_shed_mw": np.nan,
                        "baseline_system_cost": baseline["system_cost"],
                        "protected_system_cost": np.nan,
                        "avoided_system_cost": np.nan,
                        "solver_status": "not_run",
                        "changed_result": False,
                        "warning": "counterfactual not run; rerun with --run-counterfactuals",
                    }
                )
        out = pd.DataFrame(rows)
        out.to_csv(output_path, index=False)
        return out

    for _, candidate in selected.iterrows():
        for scenario in scenarios_to_run:
            key = (str(candidate["asset_type"]), str(candidate["asset_id"]), scenario.scenario_id)
            if key in completed_keys and args.resume:
                logging.info("Skipping completed run: %s", key)
                continue
            definition = read_csv(scenario.scenario_dir / "scenario_definition.csv").iloc[0]
            baseline = scenario_summary[scenario_summary["scenario"].eq(scenario.scenario_id)].iloc[0]
            logging.info("Running counterfactual: %s %s in %s", candidate["asset_type"], candidate["asset_id"], scenario.scenario_id)
            try:
                row = run_counterfactual(scenario, definition, candidate, args, output_path.parent, baseline)
            except Exception as exc:  # noqa: BLE001
                logging.exception("Counterfactual failed")
                row = {
                    "asset_type": candidate["asset_type"],
                    "asset_id": candidate["asset_id"],
                    "asset_name": candidate.get("asset_name", ""),
                    "scenario": scenario.scenario_id,
                    "return_period_years": scenario.return_period_years,
                    "annual_exceedance_probability": scenario.annual_exceedance_probability,
                    "baseline_load_shed_mwh": baseline["total_load_shed_mwh"],
                    "protected_load_shed_mwh": np.nan,
                    "avoided_load_shed_mwh": np.nan,
                    "percent_load_shed_reduction": np.nan,
                    "baseline_peak_load_shed_mw": baseline["peak_load_shed_mw"],
                    "protected_peak_load_shed_mw": np.nan,
                    "avoided_peak_load_shed_mw": np.nan,
                    "baseline_system_cost": baseline["system_cost"],
                    "protected_system_cost": np.nan,
                    "avoided_system_cost": np.nan,
                    "solver_status": "failed",
                    "changed_result": False,
                    "warning": str(exc),
                }
            append_result(output_path, row)
    return read_csv(output_path)


def annualize_benefits(results: pd.DataFrame, scenario_summary: pd.DataFrame, candidates: pd.DataFrame) -> pd.DataFrame:
    """Calculate annualized benefits from counterfactual rows."""
    baseline_eens = expected_annual_loss(scenario_summary["return_period_years"], scenario_summary["total_load_shed_mwh"])
    if results.empty:
        return pd.DataFrame()
    required_scenarios = int(scenario_summary["scenario"].nunique())
    rows = []
    for (asset_type, asset_id), group in results.groupby(["asset_type", "asset_id"]):
        valid = group.dropna(subset=["protected_load_shed_mwh"])
        solved_scenarios = int(valid["scenario"].nunique()) if "scenario" in valid else 0
        complete = solved_scenarios == required_scenarios
        if valid.empty or not complete:
            avoided_eens = np.nan
            protected_eens = np.nan
            avoided_cost = np.nan
        else:
            protected_eens = expected_annual_loss(valid["return_period_years"], valid["protected_load_shed_mwh"])
            avoided_eens = baseline_eens - protected_eens
            avoided_cost = expected_annual_loss(valid["return_period_years"], valid["avoided_system_cost"])
        max_benefit = pd.to_numeric(group["avoided_load_shed_mwh"], errors="coerce").max()
        if pd.isna(max_benefit):
            max_rp = np.nan
        else:
            max_rows = group[pd.to_numeric(group["avoided_load_shed_mwh"], errors="coerce").eq(max_benefit)]
            max_rp = max_rows["return_period_years"].iloc[0] if not max_rows.empty else np.nan
        rows.append(
            {
                "asset_type": asset_type,
                "asset_id": asset_id,
                "baseline_eens_mwh_per_year": baseline_eens,
                "protected_eens_mwh_per_year": protected_eens,
                "avoided_eens_mwh_per_year": avoided_eens,
                "risk_reduction_percent": avoided_eens / baseline_eens * 100 if pd.notna(avoided_eens) and baseline_eens > 0 else np.nan,
                "expected_annual_avoided_system_cost": avoided_cost,
                "solved_counterfactual_scenarios": solved_scenarios,
                "required_counterfactual_scenarios": required_scenarios,
                "counterfactual_complete": complete,
                "scenarios_with_nonzero_benefit": int((pd.to_numeric(group["avoided_load_shed_mwh"], errors="coerce").fillna(0) > 1e-6).sum()),
                "maximum_single_scenario_avoided_load_shed_mwh": max_benefit,
                "return_period_producing_largest_benefit": max_rp,
                "failed_or_unrun_scenarios": required_scenarios - solved_scenarios,
            }
        )
    annual = pd.DataFrame(rows)
    annual = annual.merge(candidates, on=["asset_type", "asset_id"], how="left")
    return annual


def create_ranking(annual: pd.DataFrame) -> pd.DataFrame:
    """Create the main adaptation ranking."""
    if annual.empty:
        return pd.DataFrame()
    ranking = annual.copy()
    ranking = ranking.sort_values(
        [
            "counterfactual_complete",
            "avoided_eens_mwh_per_year",
            "maximum_single_scenario_avoided_load_shed_mwh",
            "scenarios_with_nonzero_benefit",
            "associated_endpoint_or_bus_load_shed_mwh",
            "total_capacity_loss_across_scenarios",
        ],
        ascending=[False, False, False, False, False, False],
        na_position="last",
    ).reset_index(drop=True)
    ranking.insert(0, "rank", np.arange(1, len(ranking) + 1))
    ranking["data_quality_warning"] = np.where(
        ranking["failed_or_unrun_scenarios"].fillna(0).gt(0),
        "some or all counterfactual scenarios were not solved",
        "",
    )
    ranking["interpretation"] = np.where(
        ranking["avoided_eens_mwh_per_year"].fillna(0).gt(0),
        "candidate reduces modeled expected annual load shedding in one-at-a-time screening",
        "no solved avoided load-shedding benefit shown; treat as exposure/diagnostic candidate only",
    )
    return ranking


def top_five(ranking: pd.DataFrame) -> pd.DataFrame:
    """Select and describe the top five candidates."""
    if "counterfactual_complete" in ranking.columns:
        eligible = ranking[ranking["counterfactual_complete"].fillna(False)].copy()
    else:
        eligible = ranking.copy()
    top = eligible.head(5).copy()
    if top.empty:
        return top
    top["selection_explanation"] = top.apply(
        lambda row: (
            f"Selected by ranking order. Asset is a {row['asset_type']} affected in "
            f"{row.get('scenarios_affected', 'NA')} flood return-period scenarios; "
            f"avoided EENS is {row.get('avoided_eens_mwh_per_year', np.nan):,.3f} MWh/year. "
            "Physical interpretation is full flood protection/restoration screening only, not a final design."
        ),
        axis=1,
    )
    top["reasonable_physical_interpretation"] = np.where(
        top["asset_type"].eq("line"),
        "protect or elevate the vulnerable line/corridor segment or its exposed support/equipment",
        "protect flood-exposed plant equipment or associated interconnection/substation, subject to engineering review",
    )
    top["modeling_limitation"] = "individual benefits are not additive; interaction effects require a later portfolio run"
    return top


def make_figures(
    scenario_summary: pd.DataFrame,
    ranking: pd.DataFrame,
    results: pd.DataFrame,
    candidates: pd.DataFrame,
    paths: dict[str, Path],
) -> None:
    """Create diagnostic figures."""
    figures = paths["figures"]
    plt.rcParams.update({"font.size": 10})

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(scenario_summary["return_period_years"], scenario_summary["total_load_shed_mwh"], marker="o", color="#2f6c9f")
    ax.set_xscale("log")
    ax.set_xlabel("Flood return period (years)")
    ax.set_ylabel("Baseline load shed (MWh)")
    ax.set_title("Baseline Flood Load Shedding by Return Period", fontweight="bold")
    ax.grid(True, alpha=0.25)
    save(fig, figures / "01_baseline_load_shed_by_return_period")

    if not ranking.empty:
        top20 = ranking.head(20).sort_values("avoided_eens_mwh_per_year")
        fig, ax = plt.subplots(figsize=(9, max(5, len(top20) * 0.35)))
        colors = top20["asset_type"].map({"line": "#4c78a8", "generator": "#f58518"}).fillna("#777777")
        ax.barh(top20["asset_type"] + " " + top20["asset_id"].astype(str), top20["avoided_eens_mwh_per_year"].fillna(0), color=colors)
        ax.set_xlabel("Avoided expected annual load shed (MWh/year)")
        ax.set_title("Top Assets by Avoided Expected Annual Load Shedding", fontweight="bold")
        ax.grid(axis="x", alpha=0.25)
        save(fig, figures / "02_top_assets_avoided_eens")

        top5 = ranking.head(5)
        if not results.empty:
            fig, ax = plt.subplots(figsize=(8.5, 5.5))
            for _, row in top5.iterrows():
                sub = results[(results["asset_type"].eq(row["asset_type"])) & (results["asset_id"].astype(str).eq(str(row["asset_id"])))]
                if sub.empty:
                    continue
                ax.plot(sub["return_period_years"], sub["avoided_load_shed_mwh"], marker="o", label=f"{row['asset_type']} {row['asset_id']}")
            ax.set_xscale("log")
            ax.set_xlabel("Flood return period (years)")
            ax.set_ylabel("Avoided load shed (MWh)")
            ax.set_title("Top-Five Asset Benefit by Return Period", fontweight="bold")
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.25)
            save(fig, figures / "03_top_five_benefit_by_return_period")

        fig, ax = plt.subplots(figsize=(9, 5.5))
        plot = top5.sort_values("baseline_eens_mwh_per_year")
        y = np.arange(len(plot))
        ax.barh(y - 0.18, plot["baseline_eens_mwh_per_year"], height=0.36, label="Baseline", color="#666666")
        ax.barh(y + 0.18, plot["protected_eens_mwh_per_year"].fillna(0), height=0.36, label="Protected", color="#2f6c9f")
        ax.set_yticks(y, plot["asset_type"] + " " + plot["asset_id"].astype(str))
        ax.set_xlabel("Expected annual load shed (MWh/year)")
        ax.set_title("Baseline vs Individually Protected Annualized Risk", fontweight="bold")
        ax.legend()
        ax.grid(axis="x", alpha=0.25)
        save(fig, figures / "04_top_five_baseline_vs_protected_eens")

        fig, ax = plt.subplots(figsize=(8, 5.5))
        ax.scatter(candidates["maximum_flood_depth_m"], ranking.set_index(["asset_type", "asset_id"]).reindex(candidates.set_index(["asset_type", "asset_id"]).index)["avoided_eens_mwh_per_year"], s=45, color="#4c78a8")
        for _, row in ranking.head(5).iterrows():
            ax.annotate(f"{row['asset_type']} {row['asset_id']}", (row["maximum_flood_depth_m"], row["avoided_eens_mwh_per_year"] if pd.notna(row["avoided_eens_mwh_per_year"]) else 0), fontsize=8, xytext=(4, 4), textcoords="offset points")
        ax.set_xlabel("Maximum flood depth (m)")
        ax.set_ylabel("Avoided expected annual load shed (MWh/year)")
        ax.set_title("Flood Exposure vs System Benefit", fontweight="bold")
        ax.grid(True, alpha=0.25)
        save(fig, figures / "05_exposure_vs_system_benefit")

        if {"longitude", "latitude"}.issubset(top5.columns) and top5[["longitude", "latitude"]].notna().all(axis=None):
            fig, ax = plt.subplots(figsize=(7, 7))
            ax.scatter(candidates["longitude"], candidates["latitude"], s=6, color="#bbbbbb", alpha=0.35, label="exposed candidates")
            for asset_type, group in top5.groupby("asset_type"):
                ax.scatter(group["longitude"], group["latitude"], s=90, label=asset_type)
                for _, row in group.iterrows():
                    ax.annotate(str(row["asset_id"]), (row["longitude"], row["latitude"]), fontsize=8, xytext=(4, 4), textcoords="offset points")
            ax.set_xlabel("Longitude")
            ax.set_ylabel("Latitude")
            ax.set_title("Top Five Flood-Adaptation Candidates", fontweight="bold")
            ax.legend()
            ax.grid(True, alpha=0.2)
            save(fig, figures / "06_top_five_candidate_map")


def save(fig: plt.Figure, stem: Path) -> None:
    """Save a figure as PNG and PDF."""
    fig.tight_layout()
    fig.savefig(stem.with_suffix(".png"), dpi=300)
    fig.savefig(stem.with_suffix(".pdf"))
    plt.close(fig)


def write_method_validation(
    paths: dict[str, Path],
    results: pd.DataFrame,
    scenarios: list[ScenarioInfo],
    selected: pd.DataFrame,
    ran_counterfactuals: bool,
) -> pd.DataFrame:
    """Write a method validation check table."""
    if ran_counterfactuals and not results.empty:
        usable = results[results.get("solver_status", "").astype(str).eq("ok")].copy()
        row = (usable.iloc[0] if not usable.empty else results.iloc[0]).to_dict()
        avoided = pd.to_numeric(pd.Series([row.get("avoided_load_shed_mwh")]), errors="coerce").iloc[0]
        status = "completed" if pd.notna(avoided) and avoided >= -1e-6 else "failed_negative_benefit"
        row.update(
            {
                "validation_status": status,
                "only_intended_asset_changed_check": "manual: damage restored for selected asset only after baseline damage application",
                "baseline_output_matches_saved_baseline": "not_applicable: matched baseline rerun in adaptation folder",
                "network_copy_reset_afterward": "fresh network loaded per run",
            }
        )
        out = pd.DataFrame([row])
    else:
        out = pd.DataFrame(
            [
                {
                    "validation_status": "not_run",
                    "reason": "counterfactual test not launched; rerun with --test-mode --run-counterfactuals",
                    "selected_scenario": scenarios[0].scenario_id if scenarios else "",
                    "selected_asset": f"{selected.iloc[0]['asset_type']} {selected.iloc[0]['asset_id']}" if not selected.empty else "",
                }
            ]
        )
    out.to_csv(paths["counterfactual"] / "method_validation_check.csv", index=False)
    return out


def write_readme(
    paths: dict[str, Path],
    scenarios: list[ScenarioInfo],
    baseline_eens: float,
    candidates: pd.DataFrame,
    ranking: pd.DataFrame,
    results: pd.DataFrame,
    args: argparse.Namespace,
) -> None:
    """Write analysis README."""
    readme = DEFAULT_ANALYSIS_ROOT / "README.md"
    top = ranking.head(5) if not ranking.empty else pd.DataFrame()
    lines = [
        "# Florida Flood Adaptation Asset Criticality Screening",
        "",
        "## Purpose",
        "This analysis identifies flood-exposed PyPSA assets that may provide the largest resilience benefit if individually protected.",
        "",
        "## Existing baseline flood model used",
        f"- Scenario suite: `{args.suite_dir}`",
        f"- Hazard model token: `{args.hazard_model_token}`",
        f"- Return periods: {', '.join(str(s.return_period_years) for s in scenarios)}",
        "- F6.3 is used here because it is the solved suite with nonzero flood load shedding and matches the requested flood vulnerability relationship.",
        "",
        "## Method",
        "- Existing scenario outputs are copied/standardized; original results are not modified.",
        "- Assets are first screened by flood exposure, damage ratio, capacity loss, and association with baseline load-shed buses.",
        "- One-at-a-time protection restores a selected asset to pre-hazard capacity while leaving all other flood damage unchanged.",
        "- Expected annual load shedding is integrated with trapezoidal integration over annual exceedance probability p = 1 / return period.",
        "- A zero-loss point at p = 1 is added; the high-return-period tail is truncated at the largest modeled return period.",
        "",
        "## Why not rank by flood depth alone",
        "Flood depth and damage ratio measure exposure/vulnerability. The ranking criterion is avoided system disruption, measured as avoided expected annual load shedding where counterfactual solves are available.",
        "",
        "## Screening limitations",
        "- This is not an engineering design and does not estimate upgrade cost.",
        "- Individual benefits cannot be added together; portfolio interaction effects require a later combined protection run.",
        "- If counterfactuals are not run for all candidates, rankings are limited to the solved/reduced candidate set.",
        "- Generator selections may indicate that the associated plant interconnection or substation should be reviewed physically.",
        "",
        "## Run summary",
        f"- Exposed candidate assets: {len(candidates)}",
        f"- Counterfactual result rows: {len(results)}",
        f"- Baseline EENS: {baseline_eens:,.3f} MWh/year",
        "",
        "## Top five candidates",
    ]
    if top.empty:
        lines.append("- none")
    else:
        for _, row in top.iterrows():
            lines.append(
                f"- Rank {int(row['rank'])}: {row['asset_type']} `{row['asset_id']}`; "
                f"avoided EENS {row.get('avoided_eens_mwh_per_year', np.nan):,.3f} MWh/year; "
                f"risk reduction {row.get('risk_reduction_percent', np.nan):,.3f}%."
            )
    readme.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    """Run the flood adaptation screening workflow."""
    args = parse_args()
    setup_logging(args.output_dir)
    paths = ensure_structure(args.output_dir)
    scenarios = find_flood_scenarios(args.suite_dir, args.hazard_model_token)
    if not scenarios:
        raise FileNotFoundError(f"No flood scenarios matching token {args.hazard_model_token!r} found under {args.suite_dir}")
    logging.info("Flood return periods found: %s", [s.return_period_years for s in scenarios])

    inventory_existing_files(args.suite_dir, scenarios, paths["baseline"] / "flood_workflow_inventory.csv")
    if args.run_counterfactuals:
        scenarios = run_matching_baselines(scenarios, args, paths["baseline"])
        logging.info("Using matched baseline runs for adaptation calculations.")
    scenario_summary = build_scenario_summary(scenarios)
    scenario_summary.to_csv(paths["baseline"] / "baseline_flood_scenario_summary.csv", index=False)
    impacts = standardize_asset_impacts(scenarios, args.network_dir, args.asset_types)
    impacts.to_csv(paths["baseline"] / "baseline_flood_asset_impacts.csv", index=False)
    bus_shed = load_bus_shedding_for_scenarios(scenarios)
    candidates = build_candidates(impacts, scenario_summary, bus_shed)
    candidates.to_csv(paths["rankings"] / "flood_exposed_asset_candidates.csv", index=False)

    selected = select_candidates_for_counterfactual(candidates, args.maximum_number_of_candidates, args.test_mode)
    total_required_solves = len(selected) * (1 if args.test_mode else len(scenarios))
    logging.info("Estimated counterfactual solves for this run: %s", total_required_solves)

    counterfactual_path = paths["counterfactual"] / "individual_asset_counterfactual_results.csv"
    results = run_counterfactuals(scenarios, candidates, scenario_summary, args, counterfactual_path)
    validation = write_method_validation(paths, results, scenarios, selected, args.run_counterfactuals)
    baseline_eens = expected_annual_loss(scenario_summary["return_period_years"], scenario_summary["total_load_shed_mwh"])
    annual = annualize_benefits(results, scenario_summary, candidates)
    annual.to_csv(paths["rankings"] / "individual_asset_annualized_benefits.csv", index=False)
    ranking = create_ranking(annual)
    ranking.to_csv(paths["rankings"] / "flood_asset_criticality_ranking.csv", index=False)
    top = top_five(ranking)
    top.to_csv(paths["rankings"] / "top_five_flood_adaptation_candidates.csv", index=False)
    make_figures(scenario_summary, ranking, results, candidates, paths)
    write_readme(paths, scenarios, baseline_eens, candidates, ranking, results, args)

    summary = pd.DataFrame(
        [
            {
                "return_periods_analyzed": ", ".join(str(s.return_period_years) for s in scenarios),
                "asset_types_included": ", ".join(args.asset_types),
                "exposed_candidate_assets": len(candidates),
                "total_counterfactual_solves_estimated_for_this_run": total_required_solves,
                "counterfactual_rows": len(results),
                "successful_solves": int(results["solver_status"].eq("ok").sum()) if not results.empty and "solver_status" in results else 0,
                "failed_solves": int(results["solver_status"].eq("failed").sum()) if not results.empty and "solver_status" in results else 0,
                "baseline_eens_mwh_per_year": baseline_eens,
                "method_validation_status": validation.loc[0, "validation_status"] if not validation.empty else "",
            }
        ]
    )
    summary.to_csv(args.output_dir / "flood_adaptation_run_summary.csv", index=False)
    print(summary.to_string(index=False))
    if not top.empty:
        print("\nTop five candidates:")
        print(top[["rank", "asset_type", "asset_id", "avoided_eens_mwh_per_year", "risk_reduction_percent", "return_period_producing_largest_benefit"]].to_string(index=False))


if __name__ == "__main__":
    main()
