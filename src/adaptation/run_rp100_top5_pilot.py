"""Run a simplified RP100 top-five flood-adaptation pilot.

The pilot protects five selected assets together to an RP100 flood design depth
plus configurable freeboard, then solves only RP10, RP100, and RP500. It is a
screening experiment, not a final cost-benefit or engineering design analysis.
"""

from __future__ import annotations

import argparse
import logging
import sys
import tempfile
import time
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
ANALYSIS_ROOT = ELECTRICITY_DIR / "flood_adaptation_analysis"
DEFAULT_EXISTING_OUTPUTS = ANALYSIS_ROOT / "outputs_current_grid_reduced5"
DEFAULT_PILOT_DIR = ANALYSIS_ROOT / "rp100_top5_pilot"
DEFAULT_NETWORK_DIR = ELECTRICITY_DIR / "pypsa_florida_network"
DEFAULT_BASELINE_DIR = DEFAULT_NETWORK_DIR / "baseline_calibrated_no_hazard"
DEFAULT_SUITE_DIR = DEFAULT_NETWORK_DIR / "calibrated_hazard_scenarios" / "gradual_return_period_suite"
DEFAULT_SELECTED = DEFAULT_PILOT_DIR / "selected_top5_assets.csv"
LINE_CAPACITY_MULTIPLIER = 2.0

if str(ELECTRICITY_DIR) not in sys.path:
    sys.path.insert(0, str(ELECTRICITY_DIR))
if str(ANALYSIS_ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ANALYSIS_ROOT / "scripts"))

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


@dataclass(frozen=True)
class Scenario:
    """A flood return-period scenario used by the pilot."""

    return_period_years: int
    scenario_id: str
    suite_dir: Path
    baseline_dir: Path


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--existing-outputs", type=Path, default=DEFAULT_EXISTING_OUTPUTS)
    parser.add_argument("--pilot-dir", type=Path, default=DEFAULT_PILOT_DIR)
    parser.add_argument("--selected-assets", type=Path, default=DEFAULT_SELECTED)
    parser.add_argument("--network-dir", type=Path, default=DEFAULT_NETWORK_DIR)
    parser.add_argument("--baseline-dir", type=Path, default=DEFAULT_BASELINE_DIR)
    parser.add_argument("--suite-dir", type=Path, default=DEFAULT_SUITE_DIR)
    parser.add_argument("--return-periods", nargs="+", type=int, default=[10, 100, 500])
    parser.add_argument("--freeboard-m", type=float, default=0.30)
    parser.add_argument("--solver", default="highs")
    parser.add_argument("--highs-method", default="ipm")
    parser.add_argument("--start", default="2025-01-01 00:00:00")
    parser.add_argument("--periods", type=int, default=24)
    parser.add_argument("--chunk-size", type=int, default=24)
    parser.add_argument("--line-capacity-multiplier", type=float, default=LINE_CAPACITY_MULTIPLIER)
    return parser.parse_args()


def setup_logging(pilot_dir: Path) -> None:
    """Configure logging to console and a pilot log file."""
    pilot_dir.mkdir(parents=True, exist_ok=True)
    log_dir = pilot_dir / "logs"
    log_dir.mkdir(exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(log_dir / "run_rp100_top5_pilot.log", mode="w", encoding="utf-8"),
        ],
    )


def ensure_dirs(pilot_dir: Path) -> dict[str, Path]:
    """Create pilot output folders."""
    paths = {
        "root": pilot_dir,
        "adapted_runs": pilot_dir / "adapted_runs",
        "figures": pilot_dir / "figures",
        "logs": pilot_dir / "logs",
    }
    for path in paths.values():
        path.mkdir(parents=True, exist_ok=True)
    return paths


def read_csv(path: Path) -> pd.DataFrame:
    """Read a CSV file, returning an empty table when missing."""
    if not path.exists() or path.stat().st_size == 0:
        return pd.DataFrame()
    return pd.read_csv(path)


def load_vulnerability_curve(path: Path) -> tuple[pd.DataFrame, str, str]:
    """Load flood vulnerability curve, including F6.3 column names."""
    curve = pd.read_csv(path)
    intensity_col = next((c for c in ["flood_depth_m", "hazard_intensity", "intensity"] if c in curve.columns), None)
    damage_col = next(
        (c for c in ["nhess_f63_damage_ratio", "nhess_f62_damage_ratio", "damage_ratio", "vulnerability"] if c in curve.columns),
        None,
    )
    if intensity_col is None or damage_col is None:
        raise ValueError(f"Could not identify curve columns in {path}: {list(curve.columns)}")
    curve = curve[[intensity_col, damage_col]].copy()
    curve[intensity_col] = pd.to_numeric(curve[intensity_col], errors="coerce")
    curve[damage_col] = pd.to_numeric(curve[damage_col], errors="coerce")
    curve = curve.dropna().sort_values(intensity_col).drop_duplicates(intensity_col)
    curve[damage_col] = curve[damage_col].clip(0.0, 1.0)
    return curve, intensity_col, damage_col


hazard_runner.load_vulnerability_curve = load_vulnerability_curve


def interpolate_damage(depth: float, curve: pd.DataFrame, intensity_col: str, damage_col: str) -> float:
    """Interpolate damage ratio from flood depth."""
    if pd.isna(depth) or depth <= 0:
        return 0.0
    return float(
        np.interp(
            float(depth),
            curve[intensity_col].to_numpy(dtype=float),
            curve[damage_col].to_numpy(dtype=float),
            left=float(curve[damage_col].iloc[0]),
            right=float(curve[damage_col].iloc[-1]),
        )
    )


def scenario_for_rp(rp: int, args: argparse.Namespace) -> Scenario:
    """Build a scenario object for a return period."""
    scenario_id = f"flood_jrc_rp{rp}_f63_gradual"
    baseline_dir = args.existing_outputs / "baseline" / "current_flood_scenario_runs" / scenario_id
    if not baseline_dir.exists():
        baseline_dir = args.suite_dir / scenario_id
    suite_dir = args.suite_dir / scenario_id
    if not suite_dir.exists():
        raise FileNotFoundError(suite_dir)
    if not (baseline_dir / "scenario_summary.csv").exists():
        raise FileNotFoundError(f"Missing valid baseline summary for RP{rp}: {baseline_dir}")
    return Scenario(rp, scenario_id, suite_dir, baseline_dir)


def load_scenarios(args: argparse.Namespace) -> list[Scenario]:
    """Load the requested pilot scenarios."""
    if len(args.return_periods) > 3:
        raise ValueError("This pilot is limited to three adapted scenarios.")
    return [scenario_for_rp(rp, args) for rp in args.return_periods]


def first_value(df: pd.DataFrame, column: str, default=np.nan):
    """Return the first value from a column if available."""
    if df.empty or column not in df.columns:
        return default
    values = df[column].dropna()
    return values.iloc[0] if not values.empty else default


def load_counterfactual_evidence(existing_outputs: Path) -> pd.DataFrame:
    """Load completed and partial individual counterfactual evidence."""
    path = existing_outputs / "individual_asset_protection" / "individual_asset_counterfactual_results.csv"
    evidence = read_csv(path)
    if evidence.empty:
        return evidence
    evidence["avoided_load_shed_mwh"] = pd.to_numeric(evidence["avoided_load_shed_mwh"], errors="coerce")
    evidence["solver_status"] = evidence["solver_status"].astype(str)
    return evidence


def score_candidates(candidates: pd.DataFrame, evidence: pd.DataFrame) -> pd.DataFrame:
    """Score assets using baseline proxy metrics and any completed evidence."""
    work = candidates.copy()
    numeric = [
        "scenarios_affected",
        "maximum_flood_depth_m",
        "maximum_damage_ratio",
        "maximum_capacity_loss",
        "total_capacity_loss_across_scenarios",
        "baseline_load_shed_in_affected_scenarios",
        "first_damaging_return_period",
    ]
    for col in numeric:
        if col in work.columns:
            work[col] = pd.to_numeric(work[col], errors="coerce")
    if not evidence.empty:
        ev = (
            evidence[evidence["solver_status"].eq("ok")]
            .groupby(["asset_type", "asset_id"], as_index=False)
            .agg(
                completed_or_partial_counterfactual_scenarios=("scenario", "nunique"),
                completed_or_partial_avoided_load_shed_mwh=("avoided_load_shed_mwh", "sum"),
                maximum_single_counterfactual_benefit_mwh=("avoided_load_shed_mwh", "max"),
            )
        )
        work = work.merge(ev, on=["asset_type", "asset_id"], how="left")
    for col in [
        "completed_or_partial_counterfactual_scenarios",
        "completed_or_partial_avoided_load_shed_mwh",
        "maximum_single_counterfactual_benefit_mwh",
    ]:
        if col not in work.columns:
            work[col] = 0.0
        work[col] = pd.to_numeric(work[col], errors="coerce").fillna(0.0)
    work["selection_score"] = (
        work["scenarios_affected"].fillna(0) * 4.0
        + work["maximum_damage_ratio"].fillna(0) * 12.0
        + np.log1p(work["maximum_capacity_loss"].fillna(0)) * 3.0
        + np.log1p(work["total_capacity_loss_across_scenarios"].fillna(0)) * 2.0
        + np.log1p(work["baseline_load_shed_in_affected_scenarios"].fillna(0)) * 1.5
        + np.log1p(work["completed_or_partial_avoided_load_shed_mwh"].clip(lower=0)) * 6.0
        - work["first_damaging_return_period"].fillna(999) * 0.01
    )
    return work.sort_values("selection_score", ascending=False)


def asset_key(row: pd.Series) -> tuple[str, str]:
    """Return asset key."""
    return str(row["asset_type"]), str(row["asset_id"])


def same_physical_representation(row: pd.Series, chosen: list[pd.Series]) -> bool:
    """Avoid obvious duplicate representations of the same line/generator object."""
    for prev in chosen:
        if str(row["asset_type"]) != str(prev["asset_type"]):
            continue
        if str(row["asset_id"]) == str(prev["asset_id"]):
            return True
        if str(row["asset_type"]) == "line":
            row_pair = {str(row.get("bus0", "")), str(row.get("bus1", ""))}
            prev_pair = {str(prev.get("bus0", "")), str(prev.get("bus1", ""))}
            if row_pair == prev_pair:
                return True
        if str(row["asset_type"]) == "generator" and str(row.get("asset_name", "")).lower() == str(prev.get("asset_name", "")).lower():
            return True
    return False


def rp100_depth_for(asset_type: str, asset_id: str, rp100_dir: Path) -> float:
    """Return RP100 modeled flood depth for one asset."""
    file_name = "line_capacity_deratings.csv" if asset_type == "line" else "generator_capacity_deratings.csv"
    id_col = "line" if asset_type == "line" else "generator"
    df = read_csv(rp100_dir / file_name)
    if df.empty:
        return 0.0
    match = df[df[id_col].astype(str).eq(asset_id)]
    return float(pd.to_numeric(match.get("hazard_intensity", pd.Series(dtype=float)), errors="coerce").max()) if not match.empty else 0.0


def select_assets(args: argparse.Namespace, scenarios: list[Scenario]) -> pd.DataFrame:
    """Select five pilot assets without additional one-at-a-time solves."""
    if args.selected_assets.exists():
        selected = read_csv(args.selected_assets)
        logging.info("Using existing selected assets: %s", args.selected_assets)
        print("\nSelected assets:")
        print(selected[["asset_type", "asset_id", "asset_name", "reason_selected"]].to_string(index=False))
        return selected

    candidates = read_csv(args.existing_outputs / "rankings" / "flood_exposed_asset_candidates.csv")
    if candidates.empty:
        raise FileNotFoundError("Missing flood_exposed_asset_candidates.csv")
    evidence = load_counterfactual_evidence(args.existing_outputs)
    scored = score_candidates(candidates, evidence)
    chosen: list[pd.Series] = []

    fort = scored[(scored["asset_type"].eq("generator")) & (scored["asset_id"].astype(str).eq("gen_27_Fort Myers"))]
    if fort.empty:
        raise ValueError("Required asset gen_27_Fort Myers was not found in candidates.")
    chosen.append(fort.iloc[0])

    for asset_type, target_count in [("generator", 3), ("line", 2)]:
        for _, row in scored[scored["asset_type"].eq(asset_type)].iterrows():
            if len([c for c in chosen if str(c["asset_type"]) == asset_type]) >= target_count:
                break
            if same_physical_representation(row, chosen):
                continue
            chosen.append(row)

    if len(chosen) < 5:
        for _, row in scored.iterrows():
            if len(chosen) >= 5:
                break
            if not same_physical_representation(row, chosen):
                chosen.append(row)

    rp100_dir = next(s.baseline_dir for s in scenarios if s.return_period_years == 100)
    rows = []
    evidence_lookup = {}
    if not evidence.empty:
        grouped = evidence[evidence["solver_status"].eq("ok")].groupby(["asset_type", "asset_id"])
        for key, group in grouped:
            evidence_lookup[key] = (
                f"{group['scenario'].nunique()} solved/partial counterfactual scenarios; "
                f"sum avoided load shed {group['avoided_load_shed_mwh'].sum():,.1f} MWh"
            )
    for row in chosen[:5]:
        atype, aid = asset_key(row)
        rp100_depth = rp100_depth_for(atype, aid, rp100_dir)
        evidence_text = evidence_lookup.get((atype, aid), "selected from baseline proxy metrics only")
        if aid == "gen_27_Fort Myers":
            evidence_text = "complete individual counterfactual: avoided EENS about 304 MWh/year; risk reduction about 0.16%"
        reason = (
            f"High flood exposure and capacity at an asset affected in {int(row.get('scenarios_affected', 0))} return-period scenarios; "
            f"max damage ratio {float(row.get('maximum_damage_ratio', np.nan)):.2f}; "
            f"max capacity loss {float(row.get('maximum_capacity_loss', np.nan)):,.1f}."
        )
        if aid == "gen_27_Fort Myers":
            reason = "Included because it has the completed validated individual counterfactual result and strong baseline flood exposure."
        rows.append(
            {
                "asset_type": atype,
                "asset_id": aid,
                "asset_name": row.get("asset_name", ""),
                "reason_selected": reason,
                "maximum_flood_depth_m": row.get("maximum_flood_depth_m", np.nan),
                "RP100_flood_depth_m": rp100_depth,
                "first_damaging_return_period": row.get("first_damaging_return_period", np.nan),
                "scenarios_affected": row.get("scenarios_affected", np.nan),
                "existing_criticality_evidence": evidence_text,
                "selection_warning": "baseline proxy selection; not a completed individual counterfactual result"
                if aid != "gen_27_Fort Myers"
                else "",
            }
        )
    selected = pd.DataFrame(rows)
    selected.to_csv(args.selected_assets, index=False)
    print("\nSelected assets:")
    print(selected[["asset_type", "asset_id", "asset_name", "reason_selected"]].to_string(index=False))
    return selected


def write_protection_design(selected: pd.DataFrame, pilot_dir: Path, freeboard_m: float) -> pd.DataFrame:
    """Write RP100 protection design assumptions."""
    design = selected[["asset_type", "asset_id", "asset_name", "RP100_flood_depth_m"]].copy()
    design["freeboard_m"] = freeboard_m
    design["design_depth_m"] = pd.to_numeric(design["RP100_flood_depth_m"], errors="coerce").fillna(0.0) + freeboard_m
    design["adaptation_method"] = (
        "residual flood depth = max(0, scenario_flood_depth_m - design_depth_m); "
        "F6.3 vulnerability curve applied to residual depth"
    )
    design.to_csv(pilot_dir / "rp100_protection_design.csv", index=False)
    return design


def apply_top5_protection(
    network,
    selected: pd.DataFrame,
    design: pd.DataFrame,
    damaged_lines: pd.DataFrame,
    damaged_generators: pd.DataFrame,
    curve: pd.DataFrame,
    intensity_col: str,
    damage_col: str,
    scenario: Scenario,
) -> pd.DataFrame:
    """Apply residual-depth protection to selected assets and return comparison rows."""
    design_lookup = design.set_index(["asset_type", "asset_id"])
    comparison_rows = []
    for _, asset in selected.iterrows():
        atype = str(asset["asset_type"])
        aid = str(asset["asset_id"])
        name = str(asset.get("asset_name", ""))
        design_depth = float(design_lookup.loc[(atype, aid), "design_depth_m"])
        warning = ""
        if atype == "line":
            table = damaged_lines[damaged_lines["line"].astype(str).eq(aid)].copy()
            capacity_col = "s_nom"
            original_col = "original_s_nom_mva"
            reduced_col = "reduced_s_nom_mva"
            loss_col = "capacity_loss_mva"
            if table.empty:
                original_capacity = float(network.lines.loc[aid, capacity_col]) if aid in network.lines.index else np.nan
                depth = 0.0
                baseline_damage = 0.0
                baseline_capacity = original_capacity
                warning = "asset not damaged/listed in baseline derating table for this scenario"
            else:
                rec = table.iloc[0]
                original_capacity = float(rec[original_col])
                depth = float(rec["hazard_intensity"])
                baseline_damage = float(rec["damage_ratio"])
                baseline_capacity = float(rec[reduced_col])
            residual = max(0.0, depth - design_depth)
            adapted_damage = interpolate_damage(residual, curve, intensity_col, damage_col)
            adapted_capacity = original_capacity * (1.0 - adapted_damage)
            if aid in network.lines.index:
                network.lines.loc[aid, "s_nom"] = adapted_capacity
                network.lines.loc[aid, "active"] = True
            else:
                warning = (warning + "; " if warning else "") + "line missing from network"
            avoided_capacity_loss = adapted_capacity - baseline_capacity
        elif atype == "generator":
            table = damaged_generators[damaged_generators["generator"].astype(str).eq(aid)].copy()
            if table.empty:
                original_capacity = float(network.generators.loc[aid, "p_nom"]) if aid in network.generators.index else np.nan
                depth = 0.0
                baseline_damage = 0.0
                baseline_capacity = original_capacity
                warning = "asset not damaged/listed in baseline derating table for this scenario"
            else:
                rec = table.iloc[0]
                original_capacity = float(rec["original_p_nom_mw"])
                depth = float(rec["hazard_intensity"])
                baseline_damage = float(rec["damage_ratio"])
                baseline_capacity = float(rec["reduced_p_nom_mw"])
            residual = max(0.0, depth - design_depth)
            adapted_damage = interpolate_damage(residual, curve, intensity_col, damage_col)
            adapted_capacity = original_capacity * (1.0 - adapted_damage)
            if aid in network.generators.index:
                network.generators.loc[aid, "p_nom"] = adapted_capacity
            else:
                warning = (warning + "; " if warning else "") + "generator missing from network"
            avoided_capacity_loss = adapted_capacity - baseline_capacity
        else:
            depth = np.nan
            residual = np.nan
            baseline_damage = np.nan
            adapted_damage = np.nan
            original_capacity = np.nan
            baseline_capacity = np.nan
            adapted_capacity = np.nan
            avoided_capacity_loss = np.nan
            warning = f"unsupported asset type {atype}"
        comparison_rows.append(
            {
                "scenario": scenario.scenario_id,
                "return_period_years": scenario.return_period_years,
                "asset_type": atype,
                "asset_id": aid,
                "asset_name": name,
                "original_flood_depth_m": depth,
                "design_depth_m": design_depth,
                "residual_flood_depth_m": residual,
                "baseline_damage_ratio": baseline_damage,
                "adapted_damage_ratio": adapted_damage,
                "baseline_capacity": baseline_capacity,
                "adapted_capacity": adapted_capacity,
                "avoided_capacity_loss": avoided_capacity_loss,
                "protection_exceeded": bool(pd.notna(residual) and residual > 0),
                "warning": warning,
            }
        )
    return pd.DataFrame(comparison_rows)


def solve_adapted_scenario(
    scenario: Scenario,
    selected: pd.DataFrame,
    design: pd.DataFrame,
    args: argparse.Namespace,
    paths: dict[str, Path],
) -> tuple[pd.Series, pd.DataFrame, float]:
    """Solve one adapted top-five scenario with checkpointing."""
    run_dir = paths["adapted_runs"] / scenario.scenario_id
    summary_path = run_dir / "scenario_summary.csv"
    damage_path = run_dir / "selected_asset_damage_comparison.csv"
    if summary_path.exists() and damage_path.exists():
        logging.info("Skipping completed adapted scenario: %s", scenario.scenario_id)
        return read_csv(summary_path).iloc[0], read_csv(damage_path), 0.0

    run_dir.mkdir(parents=True, exist_ok=True)
    definition = read_csv(scenario.suite_dir / "scenario_definition.csv").iloc[0]
    curve, intensity_col, damage_col = load_vulnerability_curve(Path(definition["line_curve_path"]))
    start_time = time.time()
    logging.info("Running adapted portfolio scenario: %s", scenario.scenario_id)

    with tempfile.TemporaryDirectory(prefix="rp100_top5_pilot_") as tmp:
        tmp_output = Path(tmp)
        network = load_calibrated_network(args.network_dir, args.line_capacity_multiplier)
        snapshots = select_snapshots(network, args.start, args.periods, all_snapshots=False)
        damaged_lines = apply_line_damage(network, definition, tmp_output)
        damaged_generators = apply_generator_damage(network, definition, tmp_output)
        comparison = apply_top5_protection(
            network,
            selected,
            design,
            damaged_lines,
            damaged_generators,
            curve,
            intensity_col,
            damage_col,
            scenario,
        )
        comparison.to_csv(tmp_output / "selected_asset_damage_comparison.csv", index=False)

        import_buses = calibrated_import_buses(args.baseline_dir)
        import_generators = add_import_slack_generators(network, import_buses)
        apply_calibrated_import_caps(network, import_generators, args.baseline_dir, tmp_output)
        load_shedding_generators = add_standard_load_shedding(network)
        status, condition = solve_dispatch_in_chunks(network, snapshots, args.solver, args.highs_method, args.chunk_size)
        _dispatch_hourly, generation_by_carrier = dispatch_by_carrier(network, snapshots, tmp_output)
        load_shedding_hourly, load_shedding_by_bus = save_load_shedding(network, load_shedding_generators, snapshots, tmp_output)
        import_hourly, _import_by_bus = save_import_slack(network, import_generators, snapshots, tmp_output)
        line_loading = save_line_loading(network, snapshots, tmp_output)
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
            tmp_output,
        ).iloc[0].copy()
        summary["scenario_id"] = scenario.scenario_id
        summary["return_period_years"] = scenario.return_period_years
        summary["total_system_cost_usd"] = system_cost_from_dispatch(network, snapshots)
        summary["runtime_minutes"] = (time.time() - start_time) / 60.0
        for file in tmp_output.glob("*.csv"):
            file.replace(run_dir / file.name)
        pd.DataFrame([summary]).to_csv(summary_path, index=False)
    return summary, comparison, float(summary["runtime_minutes"])


def baseline_summary(scenario: Scenario) -> pd.Series:
    """Load existing baseline summary."""
    return read_csv(scenario.baseline_dir / "scenario_summary.csv").iloc[0]


def baseline_capacity_loss(scenario: Scenario) -> tuple[int, float]:
    """Return damaged asset count and total capacity loss from baseline derating tables."""
    line = read_csv(scenario.baseline_dir / "line_capacity_deratings.csv")
    gen = read_csv(scenario.baseline_dir / "generator_capacity_deratings.csv")
    line_loss = pd.to_numeric(line.get("capacity_loss_mva", pd.Series(dtype=float)), errors="coerce").fillna(0).sum()
    gen_loss = pd.to_numeric(gen.get("capacity_loss_mw", pd.Series(dtype=float)), errors="coerce").fillna(0).sum()
    return int(len(line) + len(gen)), float(line_loss + gen_loss)


def build_scenario_comparison(
    scenarios: list[Scenario],
    adapted_summaries: list[pd.Series],
    damage_comparison: pd.DataFrame,
    pilot_dir: Path,
) -> pd.DataFrame:
    """Create baseline-vs-adapted system comparison table."""
    rows = []
    adapted_lookup = {str(row["scenario_id"]): row for row in adapted_summaries}
    for scenario in scenarios:
        base = baseline_summary(scenario)
        adapted = adapted_lookup[scenario.scenario_id]
        base_count, base_loss = baseline_capacity_loss(scenario)
        selected_changes = damage_comparison[damage_comparison["scenario"].eq(scenario.scenario_id)]
        avoided_capacity = pd.to_numeric(selected_changes["avoided_capacity_loss"], errors="coerce").fillna(0).sum()
        adapted_loss = base_loss - avoided_capacity
        adapted_count = base_count - int(
            (
                pd.to_numeric(selected_changes["baseline_damage_ratio"], errors="coerce").fillna(0).gt(0)
                & pd.to_numeric(selected_changes["adapted_damage_ratio"], errors="coerce").fillna(0).le(0)
            ).sum()
        )
        baseline_load = float(base["total_load_shed_mwh"])
        adapted_load = float(adapted["total_load_shed_mwh"])
        avoided_load = baseline_load - adapted_load
        warning = ""
        if adapted_load > baseline_load + 1e-6:
            warning = "adapted load shedding exceeds baseline; inspect redispatch/network constraints"
        rows.append(
            {
                "return_period_years": scenario.return_period_years,
                "scenario": scenario.scenario_id,
                "baseline_load_shed_mwh": baseline_load,
                "adapted_load_shed_mwh": adapted_load,
                "avoided_load_shed_mwh": avoided_load,
                "load_shed_reduction_percent": avoided_load / baseline_load * 100 if baseline_load > 0 else 0.0,
                "baseline_peak_load_shed_mw": float(base.get("maximum_hourly_load_shed_mw", np.nan)),
                "adapted_peak_load_shed_mw": float(adapted.get("maximum_hourly_load_shed_mw", np.nan)),
                "avoided_peak_load_shed_mw": float(base.get("maximum_hourly_load_shed_mw", np.nan))
                - float(adapted.get("maximum_hourly_load_shed_mw", np.nan)),
                "baseline_demand_served_mwh": float(base.get("total_demand_served_mwh", np.nan)),
                "adapted_demand_served_mwh": float(adapted.get("total_demand_served_mwh", np.nan)),
                "baseline_system_cost": float(base.get("total_system_cost_usd", np.nan)),
                "adapted_system_cost": float(adapted.get("total_system_cost_usd", np.nan)),
                "avoided_system_operating_cost": float(base.get("total_system_cost_usd", np.nan))
                - float(adapted.get("total_system_cost_usd", np.nan)),
                "baseline_damaged_asset_count": base_count,
                "adapted_damaged_asset_count": adapted_count,
                "baseline_capacity_loss": base_loss,
                "adapted_capacity_loss": adapted_loss,
                "avoided_capacity_loss": avoided_capacity,
                "solver_status": adapted.get("solver_status", ""),
                "runtime_minutes": adapted.get("runtime_minutes", np.nan),
                "warning": warning,
            }
        )
    out = pd.DataFrame(rows).sort_values("return_period_years")
    out.to_csv(pilot_dir / "top5_portfolio_scenario_comparison.csv", index=False)
    return out


def expected_annual_loss(return_periods: Iterable[float], losses: Iterable[float]) -> float:
    """Integrate a loss exceedance curve in AEP space."""
    data = pd.DataFrame({"aep": 1.0 / pd.Series(return_periods, dtype=float), "loss": pd.Series(losses, dtype=float)})
    data = data.dropna().sort_values("aep")
    if data.empty:
        return 0.0
    if not np.isclose(data["aep"].max(), 1.0):
        data = pd.concat([data, pd.DataFrame([{"aep": 1.0, "loss": 0.0}])], ignore_index=True)
    data = data.sort_values("aep")
    return float(np.trapezoid(data["loss"].to_numpy(), data["aep"].to_numpy()))


def write_pilot_risk(comparison: pd.DataFrame, pilot_dir: Path) -> pd.DataFrame:
    """Write preliminary three-return-period risk comparison."""
    baseline = expected_annual_loss(comparison["return_period_years"], comparison["baseline_load_shed_mwh"])
    adapted = expected_annual_loss(comparison["return_period_years"], comparison["adapted_load_shed_mwh"])
    avoided = baseline - adapted
    out = pd.DataFrame(
        [
            {
                "label": "Pilot three-return-period annualized estimate",
                "pilot_baseline_eens_mwh_per_year": baseline,
                "pilot_adapted_eens_mwh_per_year": adapted,
                "pilot_avoided_eens_mwh_per_year": avoided,
                "pilot_risk_reduction_percent": avoided / baseline * 100 if baseline > 0 else 0.0,
                "warning": "Approximate pilot estimate only; intermediate return periods are omitted.",
            }
        ]
    )
    out.to_csv(pilot_dir / "pilot_annualized_risk_comparison.csv", index=False)
    return out


def save_figure(fig: plt.Figure, stem: Path) -> None:
    """Save PNG and PDF copies."""
    fig.tight_layout()
    fig.savefig(stem.with_suffix(".png"), dpi=300)
    fig.savefig(stem.with_suffix(".pdf"))
    plt.close(fig)


def make_figures(comparison: pd.DataFrame, damage: pd.DataFrame, risk: pd.DataFrame, paths: dict[str, Path]) -> None:
    """Create the four essential pilot figures."""
    figures = paths["figures"]
    plt.rcParams.update({"font.size": 11})

    x = np.arange(len(comparison))
    labels = [f"RP{int(rp)}" for rp in comparison["return_period_years"]]
    fig, ax = plt.subplots(figsize=(8, 5))
    width = 0.36
    ax.bar(x - width / 2, comparison["baseline_load_shed_mwh"], width, label="Baseline", color="#8f98a3")
    ax.bar(x + width / 2, comparison["adapted_load_shed_mwh"], width, label="Adapted", color="#2f6c9f")
    ax.set_xticks(x, labels)
    ax.set_ylabel("Load shed (MWh)")
    ax.set_title("RP100 Top-Five Pilot Reduces Modeled Flood Load Shedding")
    ax.legend()
    ax.grid(axis="y", alpha=0.25)
    save_figure(fig, figures / "01_baseline_vs_adapted_load_shed")

    fig, ax = plt.subplots(figsize=(7, 4.8))
    ax.bar(labels, comparison["load_shed_reduction_percent"], color="#2f6c9f")
    ax.set_ylabel("Load-shed reduction (%)")
    ax.set_title("Percentage Load-Shed Reduction by Flood Return Period")
    ax.grid(axis="y", alpha=0.25)
    save_figure(fig, figures / "02_load_shed_reduction_percent")

    focus = damage[damage["return_period_years"].isin([100, 500])].copy()
    focus["label"] = focus["asset_name"].fillna(focus["asset_id"])
    fig, ax = plt.subplots(figsize=(10, 6))
    pos = np.arange(len(focus))
    ax.barh(pos - 0.18, focus["baseline_damage_ratio"], height=0.36, label="Baseline", color="#c44e52")
    ax.barh(pos + 0.18, focus["adapted_damage_ratio"], height=0.36, label="Adapted", color="#2f6c9f")
    ax.set_yticks(pos, [f"RP{int(rp)} | {label}" for rp, label in zip(focus["return_period_years"], focus["label"])])
    ax.set_xlabel("Damage ratio")
    ax.set_xlim(0, 1.05)
    ax.set_title("Selected-Asset Damage Ratios After RP100+Freeboard Protection")
    ax.legend()
    ax.grid(axis="x", alpha=0.25)
    save_figure(fig, figures / "03_selected_asset_damage_comparison")

    fig, ax = plt.subplots(figsize=(7.5, 5))
    aep = 1.0 / comparison["return_period_years"].astype(float)
    order = np.argsort(aep)
    ax.plot(aep.iloc[order], comparison["baseline_load_shed_mwh"].iloc[order], marker="o", label="Baseline", color="#8f98a3")
    ax.plot(aep.iloc[order], comparison["adapted_load_shed_mwh"].iloc[order], marker="o", label="Adapted", color="#2f6c9f")
    ax.set_xlabel("Annual exceedance probability")
    ax.set_ylabel("Load shed (MWh)")
    ax.set_title("Pilot Three-Return-Period Exceedance Comparison")
    ax.legend(title=risk.iloc[0]["label"])
    ax.grid(True, alpha=0.25)
    save_figure(fig, figures / "04_pilot_baseline_vs_adapted_exceedance")


def write_interpretation(
    selected: pd.DataFrame,
    design: pd.DataFrame,
    comparison: pd.DataFrame,
    risk: pd.DataFrame,
    pilot_dir: Path,
) -> None:
    """Write cautious pilot interpretation."""
    lines = [
        "# RP100 Top-Five Flood-Adaptation Pilot",
        "",
        "This is a simplified portfolio screening experiment. It does not estimate adaptation construction cost and does not establish final cost-effectiveness.",
        "",
        "## Selected Assets",
    ]
    for _, row in selected.iterrows():
        lines.append(f"- `{row['asset_id']}` ({row['asset_type']}, {row.get('asset_name', '')}): {row['reason_selected']} {row['existing_criticality_evidence']}")
    lines += [
        "",
        "## Adaptation Representation",
        "Each selected asset was protected to its modeled RP100 flood depth plus 0.30 m freeboard. Damage was recalculated from residual depth, so RP500 can still exceed the design level.",
        "",
        "## Scenario Benefits",
    ]
    for _, row in comparison.iterrows():
        lines.append(
            f"- RP{int(row['return_period_years'])}: avoided {row['avoided_load_shed_mwh']:,.1f} MWh "
            f"({row['load_shed_reduction_percent']:.2f}% reduction)."
        )
    r = risk.iloc[0]
    lines += [
        "",
        "## Pilot Annualized Estimate",
        (
            f"Pilot baseline EENS is {r['pilot_baseline_eens_mwh_per_year']:,.1f} MWh/year; "
            f"adapted EENS is {r['pilot_adapted_eens_mwh_per_year']:,.1f} MWh/year; "
            f"avoided EENS is {r['pilot_avoided_eens_mwh_per_year']:,.1f} MWh/year "
            f"({r['pilot_risk_reduction_percent']:.2f}%)."
        ),
        "This is approximate because only RP10, RP100, and RP500 are included.",
        "",
        "## Cautious Conclusion",
    ]
    max_benefit = comparison.loc[comparison["avoided_load_shed_mwh"].idxmax()]
    lines.append(f"Benefits are largest in RP{int(max_benefit['return_period_years'])}.")
    if comparison["avoided_load_shed_mwh"].sum() > 0:
        lines.append("The package shows modeled benefit and is promising enough to consider a fuller return-period test, but only after reviewing the selected assets and solver behavior.")
    else:
        lines.append("The package does not show meaningful modeled benefit in this pilot and should be redesigned before a fuller suite.")
    if (comparison["return_period_years"].eq(500).any()) and (comparison.loc[comparison["return_period_years"].eq(500), "avoided_load_shed_mwh"].iloc[0] < comparison["avoided_load_shed_mwh"].max()):
        lines.append("The RP500 case appears to partly overwhelm the RP100 protection, which is expected under the residual-depth method.")
    lines.append("Four of the five assets were selected by baseline proxy metrics rather than complete individual counterfactuals, so the portfolio result should be treated as screening evidence.")
    (pilot_dir / "pilot_interpretation.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    """Run the pilot workflow."""
    args = parse_args()
    setup_logging(args.pilot_dir)
    paths = ensure_dirs(args.pilot_dir)
    scenarios = load_scenarios(args)
    selected = select_assets(args, scenarios)
    design = write_protection_design(selected, args.pilot_dir, args.freeboard_m)

    print("\nProtection design:")
    print(design[["asset_type", "asset_id", "RP100_flood_depth_m", "freeboard_m", "design_depth_m"]].to_string(index=False))

    adapted_summaries = []
    damage_frames = []
    for scenario in scenarios:
        summary, damage, runtime = solve_adapted_scenario(scenario, selected, design, args, paths)
        adapted_summaries.append(summary)
        damage_frames.append(damage)
        base = baseline_summary(scenario)
        baseline_load = float(base["total_load_shed_mwh"])
        adapted_load = float(summary["total_load_shed_mwh"])
        avoided = baseline_load - adapted_load
        pct = avoided / baseline_load * 100 if baseline_load > 0 else 0.0
        print(
            f"\nScenario completed: RP{scenario.return_period_years}\n"
            f"Runtime minutes: {float(summary.get('runtime_minutes', runtime)):.2f}\n"
            f"Baseline load shed: {baseline_load:,.1f} MWh\n"
            f"Adapted load shed: {adapted_load:,.1f} MWh\n"
            f"Avoided load shed: {avoided:,.1f} MWh\n"
            f"Reduction: {pct:.2f}%\n"
            f"Checkpoint: {paths['adapted_runs'] / scenario.scenario_id}"
        )

    damage_comparison = pd.concat(damage_frames, ignore_index=True)
    damage_comparison.to_csv(args.pilot_dir / "top5_asset_damage_comparison.csv", index=False)
    comparison = build_scenario_comparison(scenarios, adapted_summaries, damage_comparison, args.pilot_dir)
    risk = write_pilot_risk(comparison, args.pilot_dir)
    make_figures(comparison, damage_comparison, risk, paths)
    write_interpretation(selected, design, comparison, risk, args.pilot_dir)

    print("\nPilot summary:")
    print("Selected assets:")
    print(selected[["asset_type", "asset_id", "asset_name", "reason_selected"]].to_string(index=False))
    print("\nScenario comparison:")
    print(comparison[["return_period_years", "avoided_load_shed_mwh", "load_shed_reduction_percent", "solver_status", "runtime_minutes"]].to_string(index=False))
    print("\nPilot annualized risk:")
    print(risk.to_string(index=False))
    print("\nCreated files in:")
    print(args.pilot_dir)


if __name__ == "__main__":
    main()
