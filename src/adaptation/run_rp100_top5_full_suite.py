"""Run the RP100 top-five adaptation package across the full flood suite.

This workflow reuses completed RP10/RP100/RP500 pilot runs when valid, solves
only missing adapted return periods, recomputes expected annual load shedding
from the full return-period suite, and updates the planning-level economics.
It never runs individual-asset counterfactuals.
"""

from __future__ import annotations

import argparse
import logging
import shutil
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Iterable

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

matplotlib.use("Agg")

PROJECT_ROOT = Path(r"C:\oxford_tc_project")
ELECTRICITY_DIR = PROJECT_ROOT / "data" / "Electricity"
ANALYSIS_ROOT = ELECTRICITY_DIR / "flood_adaptation_analysis"
SCRIPTS_DIR = ANALYSIS_ROOT / "scripts"
PILOT_DIR = ANALYSIS_ROOT / "rp100_top5_pilot"
COST_BENEFIT_DIR = ANALYSIS_ROOT / "rp100_top5_cost_benefit"
FULL_SUITE_DIR = ANALYSIS_ROOT / "rp100_top5_full_suite"
DEFAULT_EXISTING_OUTPUTS = ANALYSIS_ROOT / "outputs_current_grid_reduced5"
DEFAULT_NETWORK_DIR = ELECTRICITY_DIR / "pypsa_florida_network"
DEFAULT_BASELINE_DIR = DEFAULT_NETWORK_DIR / "baseline_calibrated_no_hazard"
DEFAULT_SUITE_DIR = DEFAULT_NETWORK_DIR / "calibrated_hazard_scenarios" / "gradual_return_period_suite"
DEFAULT_SELECTED = PILOT_DIR / "selected_top5_assets.csv"
DEFAULT_COST_ASSUMPTIONS = COST_BENEFIT_DIR / "adaptation_cost_assumptions.csv"
DEFAULT_ECON_ASSUMPTIONS = COST_BENEFIT_DIR / "economic_assumptions.csv"
LINE_CAPACITY_MULTIPLIER = 2.0

if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))
if str(ELECTRICITY_DIR) not in sys.path:
    sys.path.insert(0, str(ELECTRICITY_DIR))

import run_rp100_top5_pilot as pilot  # noqa: E402


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--full-suite-dir", type=Path, default=FULL_SUITE_DIR)
    parser.add_argument("--pilot-dir", type=Path, default=PILOT_DIR)
    parser.add_argument("--existing-outputs", type=Path, default=DEFAULT_EXISTING_OUTPUTS)
    parser.add_argument("--selected-assets", type=Path, default=DEFAULT_SELECTED)
    parser.add_argument("--network-dir", type=Path, default=DEFAULT_NETWORK_DIR)
    parser.add_argument("--baseline-dir", type=Path, default=DEFAULT_BASELINE_DIR)
    parser.add_argument("--suite-dir", type=Path, default=DEFAULT_SUITE_DIR)
    parser.add_argument("--cost-assumptions", type=Path, default=DEFAULT_COST_ASSUMPTIONS)
    parser.add_argument("--economic-assumptions", type=Path, default=DEFAULT_ECON_ASSUMPTIONS)
    parser.add_argument("--return-periods", nargs="*", type=int, default=None)
    parser.add_argument("--freeboard-m", type=float, default=0.30)
    parser.add_argument("--solver", default="highs")
    parser.add_argument("--highs-method", default="ipm")
    parser.add_argument("--start", default="2025-01-01 00:00:00")
    parser.add_argument("--periods", type=int, default=24)
    parser.add_argument("--chunk-size", type=int, default=24)
    parser.add_argument("--line-capacity-multiplier", type=float, default=LINE_CAPACITY_MULTIPLIER)
    parser.add_argument("--skip-economics", action="store_true")
    return parser.parse_args()


def setup_logging(output_dir: Path) -> None:
    """Create folders and configure logging."""
    (output_dir / "adapted_runs").mkdir(parents=True, exist_ok=True)
    (output_dir / "figures").mkdir(exist_ok=True)
    (output_dir / "economics").mkdir(exist_ok=True)
    (output_dir / "logs").mkdir(exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(output_dir / "logs" / "run_rp100_top5_full_suite.log", mode="w", encoding="utf-8"),
        ],
    )


def read_csv(path: Path) -> pd.DataFrame:
    """Read a required CSV."""
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_csv(path)


def discover_return_periods(suite_dir: Path) -> list[int]:
    """Discover available F6.3 flood return periods from scenario folders."""
    periods: list[int] = []
    for path in suite_dir.glob("flood_jrc_rp*_f63_gradual"):
        stem = path.name.replace("flood_jrc_rp", "").replace("_f63_gradual", "")
        if stem.isdigit():
            periods.append(int(stem))
    return sorted(set(periods))


def scenario_for_rp(rp: int, args: argparse.Namespace) -> pilot.Scenario:
    """Build the pilot Scenario object for a return period."""
    ns = SimpleNamespace(existing_outputs=args.existing_outputs, suite_dir=args.suite_dir)
    return pilot.scenario_for_rp(rp, ns)


def validate_selected_assets(selected_path: Path, pilot_dir: Path) -> pd.DataFrame:
    """Validate that the selected five assets match the completed pilot."""
    selected = read_csv(selected_path)
    pilot_selected = read_csv(pilot_dir / "selected_top5_assets.csv")
    expected = set(zip(pilot_selected["asset_type"].astype(str), pilot_selected["asset_id"].astype(str)))
    actual = set(zip(selected["asset_type"].astype(str), selected["asset_id"].astype(str)))
    if actual != expected:
        raise ValueError(f"Selected assets differ from pilot. Expected {expected}, got {actual}")
    if len(actual) != 5:
        raise ValueError(f"Expected exactly five selected assets, got {len(actual)}")
    return selected


def validate_or_write_design(selected: pd.DataFrame, output_dir: Path, pilot_dir: Path, freeboard_m: float) -> pd.DataFrame:
    """Write full-suite design table and validate against the pilot design."""
    pilot_design = read_csv(pilot_dir / "rp100_protection_design.csv")
    design = pilot.write_protection_design(selected, output_dir, freeboard_m)
    merged = design.merge(
        pilot_design[["asset_type", "asset_id", "design_depth_m"]],
        on=["asset_type", "asset_id"],
        suffixes=("_full", "_pilot"),
    )
    if not np.allclose(merged["design_depth_m_full"], merged["design_depth_m_pilot"], rtol=0, atol=1e-9):
        raise ValueError("Full-suite design depths do not match completed pilot design depths.")
    return design


def run_complete(run_dir: Path) -> bool:
    """Return whether an adapted scenario run has the required checkpoint files."""
    summary = run_dir / "scenario_summary.csv"
    damage = run_dir / "selected_asset_damage_comparison.csv"
    if not summary.exists() or not damage.exists():
        return False
    try:
        row = pd.read_csv(summary).iloc[0]
    except Exception:
        return False
    return str(row.get("solver_status", "")).lower() == "ok"


def copy_reusable_pilot_runs(scenarios: list[pilot.Scenario], args: argparse.Namespace) -> dict[str, str]:
    """Copy completed pilot scenarios into full-suite adapted run folder."""
    sources: dict[str, str] = {}
    target_root = args.full_suite_dir / "adapted_runs"
    for scenario in scenarios:
        target = target_root / scenario.scenario_id
        if run_complete(target):
            sources[scenario.scenario_id] = "existing_full_suite_checkpoint"
            continue
        source = args.pilot_dir / "adapted_runs" / scenario.scenario_id
        if run_complete(source):
            shutil.copytree(source, target, dirs_exist_ok=True)
            sources[scenario.scenario_id] = "reused_completed_pilot"
        else:
            sources[scenario.scenario_id] = "new_solve"
    return sources


def build_inventory(scenarios: list[pilot.Scenario], sources: dict[str, str], output_dir: Path) -> pd.DataFrame:
    """Build scenario inventory before solving missing runs."""
    rows: list[dict[str, object]] = []
    for scenario in scenarios:
        baseline_summary = scenario.baseline_dir / "scenario_summary.csv"
        adapted_dir = output_dir / "adapted_runs" / scenario.scenario_id
        definition_path = scenario.suite_dir / "scenario_definition.csv"
        definition = read_csv(definition_path).iloc[0] if definition_path.exists() else pd.Series(dtype=object)
        adapted_available = run_complete(adapted_dir)
        warning = ""
        if not baseline_summary.exists():
            warning = "missing baseline scenario summary"
        rows.append(
            {
                "scenario": scenario.scenario_id,
                "return_period_years": scenario.return_period_years,
                "annual_exceedance_probability": 1.0 / float(scenario.return_period_years),
                "baseline_result_available": baseline_summary.exists(),
                "adapted_result_available": adapted_available,
                "adapted_result_source": sources.get(scenario.scenario_id, "new_solve"),
                "new_solve_required": not adapted_available,
                "baseline_summary_path": str(baseline_summary),
                "adapted_run_path": str(adapted_dir),
                "flood_depth_input": definition.get("line_damage_path", ""),
                "solver_settings": "highs/ipm, 24 snapshots, chunk size 24 unless overridden",
                "snapshots_and_objective": "same PyPSA dispatch objective and selected snapshots as RP100 top-five pilot",
                "warning": warning,
            }
        )
    inventory = pd.DataFrame(rows)
    inventory.to_csv(output_dir / "full_suite_scenario_inventory.csv", index=False)
    return inventory


def solve_missing(
    scenarios: list[pilot.Scenario],
    selected: pd.DataFrame,
    design: pd.DataFrame,
    args: argparse.Namespace,
    sources: dict[str, str],
) -> tuple[list[pd.Series], list[pd.DataFrame], dict[str, str]]:
    """Run only missing adapted scenarios and return all summaries/damage tables."""
    paths = {"adapted_runs": args.full_suite_dir / "adapted_runs"}
    solve_args = SimpleNamespace(
        network_dir=args.network_dir,
        baseline_dir=args.baseline_dir,
        line_capacity_multiplier=args.line_capacity_multiplier,
        start=args.start,
        periods=args.periods,
        solver=args.solver,
        highs_method=args.highs_method,
        chunk_size=args.chunk_size,
    )
    summaries: list[pd.Series] = []
    damage_tables: list[pd.DataFrame] = []
    for scenario in scenarios:
        run_dir = paths["adapted_runs"] / scenario.scenario_id
        before_complete = run_complete(run_dir)
        summary, damage, runtime = pilot.solve_adapted_scenario(scenario, selected, design, solve_args, paths)
        if not before_complete:
            sources[scenario.scenario_id] = "new_full_suite_solve"
        summaries.append(summary)
        damage_tables.append(damage)
        base = pilot.baseline_summary(scenario)
        baseline_load = float(base["total_load_shed_mwh"])
        adapted_load = float(summary["total_load_shed_mwh"])
        avoided = baseline_load - adapted_load
        print(
            f"\nScenario completed: RP{scenario.return_period_years}\n"
            f"Runtime: {runtime:.2f} minutes\n"
            f"Baseline load shed: {baseline_load:,.1f} MWh\n"
            f"Adapted load shed: {adapted_load:,.1f} MWh\n"
            f"Avoided load shed: {avoided:,.1f} MWh\n"
            f"Reduction: {(avoided / baseline_load * 100 if baseline_load > 0 else 0.0):.2f}%\n"
            f"Solver status: {summary.get('solver_status', '')}\n"
            f"Checkpoint: {run_dir}"
        )
    return summaries, damage_tables, sources


def build_full_comparison(
    scenarios: list[pilot.Scenario],
    summaries: list[pd.Series],
    damage: pd.DataFrame,
    sources: dict[str, str],
    output_dir: Path,
) -> pd.DataFrame:
    """Build full-suite scenario comparison with source and AEP columns."""
    comparison = pilot.build_scenario_comparison(scenarios, summaries, damage, output_dir)
    comparison["annual_exceedance_probability"] = 1.0 / comparison["return_period_years"].astype(float)
    comparison["result_source"] = comparison["scenario"].map(sources)
    comparison["avoided_system_cost"] = comparison["avoided_system_operating_cost"]
    ordered_cols = [
        "scenario",
        "return_period_years",
        "annual_exceedance_probability",
        "baseline_load_shed_mwh",
        "adapted_load_shed_mwh",
        "avoided_load_shed_mwh",
        "load_shed_reduction_percent",
        "baseline_peak_load_shed_mw",
        "adapted_peak_load_shed_mw",
        "baseline_demand_served_mwh",
        "adapted_demand_served_mwh",
        "baseline_system_cost",
        "adapted_system_cost",
        "avoided_system_cost",
        "baseline_capacity_loss",
        "adapted_capacity_loss",
        "avoided_capacity_loss",
        "solver_status",
        "runtime_minutes",
        "result_source",
        "warning",
    ]
    comparison = comparison[[c for c in ordered_cols if c in comparison.columns]]
    comparison.to_csv(output_dir / "full_suite_scenario_comparison.csv", index=False)
    return comparison


def annualized_risk(comparison: pd.DataFrame, output_dir: Path, pilot_dir: Path) -> pd.DataFrame:
    """Calculate full-suite annualized risk using the pilot trapezoidal method."""
    baseline = pilot.expected_annual_loss(comparison["return_period_years"], comparison["baseline_load_shed_mwh"])
    adapted = pilot.expected_annual_loss(comparison["return_period_years"], comparison["adapted_load_shed_mwh"])
    avoided = baseline - adapted
    baseline_cost = pilot.expected_annual_loss(comparison["return_period_years"], comparison["baseline_system_cost"])
    adapted_cost = pilot.expected_annual_loss(comparison["return_period_years"], comparison["adapted_system_cost"])
    pilot_risk = read_csv(pilot_dir / "pilot_annualized_risk_comparison.csv").iloc[0]
    pilot_avoided = float(pilot_risk["pilot_avoided_eens_mwh_per_year"])
    out = pd.DataFrame(
        [
            {
                "baseline_eens_mwh_per_year": baseline,
                "adapted_eens_mwh_per_year": adapted,
                "avoided_eens_mwh_per_year": avoided,
                "risk_reduction_percent": avoided / baseline * 100 if baseline > 0 else 0.0,
                "baseline_expected_annual_system_cost": baseline_cost,
                "adapted_expected_annual_system_cost": adapted_cost,
                "expected_annual_avoided_system_cost": baseline_cost - adapted_cost,
                "pilot_avoided_eens_mwh_per_year": pilot_avoided,
                "full_suite_avoided_eens_mwh_per_year": avoided,
                "absolute_difference": avoided - pilot_avoided,
                "percentage_difference": (avoided - pilot_avoided) / pilot_avoided * 100 if pilot_avoided else np.nan,
                "probability_values": "; ".join(
                    f"RP{int(rp)} AEP={1 / float(rp):.6f}" for rp in comparison["return_period_years"]
                ),
                "integration_method": "trapezoidal integration in annual-exceedance-probability space using pilot.expected_annual_loss",
                "ordering": "points sorted by AEP ascending before integration",
                "endpoint_treatment": "adds AEP=1.0 with zero loss when max modeled AEP is below 1.0",
                "tail_treatment": "no extrapolation beyond smallest AEP/highest return period",
            }
        ]
    )
    out.to_csv(output_dir / "full_suite_annualized_risk.csv", index=False)
    return out


def write_sanity_checks(comparison: pd.DataFrame, risk: pd.DataFrame, output_dir: Path) -> None:
    """Write required sanity checks."""
    load = comparison.sort_values("return_period_years")["baseline_load_shed_mwh"].to_numpy()
    avoided = comparison.sort_values("return_period_years")["avoided_load_shed_mwh"].to_numpy()
    monotonic = bool(np.all(np.diff(load) >= -1e-6))
    max_avoided_row = comparison.loc[comparison["avoided_load_shed_mwh"].idxmax()]
    rp10 = comparison[comparison["return_period_years"].eq(10)]
    rp10_share = float(rp10["avoided_load_shed_mwh"].iloc[0] / comparison["avoided_load_shed_mwh"].sum()) if not rp10.empty and comparison["avoided_load_shed_mwh"].sum() else np.nan
    text = f"""# Full-Suite Sanity Checks

1. Load shed monotonic with return period: {monotonic}. Baseline load shedding is broadly increasing if this is true.
2. Irregular dispatch/solver changes: review scenario comparison warnings. Current warnings: {comparison['warning'].replace('', np.nan).dropna().tolist()}.
3. Avoided load shed is largest at RP{int(max_avoided_row['return_period_years'])}: {float(max_avoided_row['avoided_load_shed_mwh']):,.1f} MWh.
4. Integration uses all available return periods and trapezoidal AEP integration; no simple summation is used.
5. RP10 share of summed avoided scenario MWh is {rp10_share:.2%}; this is checked because high-AEP points can strongly affect annualized EENS.
6. RP10 is treated as a modeled annual-exceedance loss point because it is part of the existing gradual flood suite.
7. Every scenario uses 24 snapshots, so load shedding is measured over the same event duration.
8. Load shedding is MWh over the scenario window, not MW.
9. Scenario probabilities are integrated in AEP space rather than summed.
10. Outage benefits are calculated once from avoided EENS, not once per snapshot or asset.

Full-suite avoided EENS: {float(risk['avoided_eens_mwh_per_year'].iloc[0]):,.1f} MWh/year.
"""
    (output_dir / "full_suite_sanity_checks.md").write_text(text, encoding="utf-8")


def pv_factor(rate: float, years: int, growth: float = 0.0) -> float:
    """Present value factor for annual values."""
    if abs(rate - growth) < 1e-12:
        return years / (1.0 + rate)
    return (1.0 - ((1.0 + growth) / (1.0 + rate)) ** years) / (rate - growth)


def discounted_payback(capex: float, net_annual: float, rate: float, years: int) -> float:
    """Calculate discounted payback year."""
    cumulative = 0.0
    for year in range(1, years + 1):
        cumulative += net_annual / ((1.0 + rate) ** year)
        if cumulative >= capex:
            return float(year)
    return np.nan


def update_economics(avoided_eens: float, args: argparse.Namespace, output_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Update cost-benefit tables using full-suite avoided EENS and existing assumptions."""
    econ_dir = output_dir / "economics"
    econ_dir.mkdir(exist_ok=True)
    costs = read_csv(args.cost_assumptions)
    econ = read_csv(args.economic_assumptions).iloc[0]
    lifetime = int(econ["analysis_lifetime_years"])
    rate = float(econ["real_discount_rate"])
    maintenance_pct = float(econ["annual_maintenance_percent_of_capex"])
    growth = float(econ["annual_benefit_growth_rate"])
    benefit_factor = pv_factor(rate, lifetime, growth)
    maintenance_factor = pv_factor(rate, lifetime, 0.0)
    package_costs = pd.DataFrame(
        [
            {"package_cost_scenario": label, "capital_cost_usd": float(costs[f"{label}_cost_usd"].sum())}
            for label in ["low", "central", "high"]
        ]
    )
    rows: list[dict[str, object]] = []
    for _, cost_row in package_costs.iterrows():
        capex = float(cost_row["capital_cost_usd"])
        annual_maintenance = capex * maintenance_pct
        pv_maintenance = annual_maintenance * maintenance_factor
        for label, voll in {
            "low": float(econ["low_voll_usd_per_mwh"]),
            "central": float(econ["central_voll_usd_per_mwh"]),
            "high": float(econ["high_voll_usd_per_mwh"]),
        }.items():
            annual_benefit = avoided_eens * voll
            pv_benefits = annual_benefit * benefit_factor
            total_cost = capex + pv_maintenance
            net_annual = annual_benefit - annual_maintenance
            rows.append(
                {
                    "package_cost_scenario": cost_row["package_cost_scenario"],
                    "voll_scenario": label,
                    "avoided_eens_mwh_per_year": avoided_eens,
                    "voll_usd_per_mwh": voll,
                    "capital_cost_usd": capex,
                    "annual_avoided_outage_value_usd": annual_benefit,
                    "annual_maintenance_cost_usd": annual_maintenance,
                    "present_value_benefits_usd": pv_benefits,
                    "present_value_maintenance_usd": pv_maintenance,
                    "benefit_cost_ratio": pv_benefits / total_cost,
                    "net_present_value_usd": pv_benefits - total_cost,
                    "simple_payback_years": capex / net_annual if net_annual > 0 else np.nan,
                    "discounted_payback_years": discounted_payback(capex, net_annual, rate, lifetime),
                    "break_even_capital_cost_usd": pv_benefits / (1 + maintenance_pct * maintenance_factor),
                    "lifecycle_cost_per_avoided_mwh_usd_per_mwh": total_cost / (avoided_eens * maintenance_factor),
                }
            )
    results = pd.DataFrame(rows)
    results.to_csv(econ_dir / "full_suite_cost_benefit_results.csv", index=False)

    pilot_results = read_csv(COST_BENEFIT_DIR / "cost_benefit_results.csv")
    pilot_central = pilot_results[
        (pilot_results["package_cost_scenario"].eq("central")) & (pilot_results["voll_scenario"].eq("central"))
    ].iloc[0]
    full_central = results[
        (results["package_cost_scenario"].eq("central")) & (results["voll_scenario"].eq("central"))
    ].iloc[0]
    comparison = pd.DataFrame(
        [
            {
                "metric": "avoided_eens_mwh_per_year",
                "pilot_value": float(pilot_central.get("annual_avoided_outage_benefit_usd"))
                / float(pilot_central.get("annual_avoided_outage_benefit_usd"))
                * float(read_csv(PILOT_DIR / "pilot_annualized_risk_comparison.csv")["pilot_avoided_eens_mwh_per_year"].iloc[0]),
                "full_suite_value": avoided_eens,
            },
            {
                "metric": "central_annual_avoided_outage_value_usd",
                "pilot_value": float(pilot_central["annual_avoided_outage_benefit_usd"]),
                "full_suite_value": float(full_central["annual_avoided_outage_value_usd"]),
            },
            {
                "metric": "central_bcr",
                "pilot_value": float(pilot_central["benefit_cost_ratio"]),
                "full_suite_value": float(full_central["benefit_cost_ratio"]),
            },
            {
                "metric": "central_npv_usd",
                "pilot_value": float(pilot_central["net_present_value_usd"]),
                "full_suite_value": float(full_central["net_present_value_usd"]),
            },
            {
                "metric": "central_simple_payback_years",
                "pilot_value": float(pilot_central["simple_payback_years"]),
                "full_suite_value": float(full_central["simple_payback_years"]),
            },
            {
                "metric": "central_break_even_capital_cost_usd",
                "pilot_value": float(pilot_central["break_even_capital_cost_usd"]),
                "full_suite_value": float(full_central["break_even_capital_cost_usd"]),
            },
        ]
    )
    comparison["absolute_change"] = comparison["full_suite_value"] - comparison["pilot_value"]
    comparison["percentage_change_from_pilot"] = comparison["absolute_change"] / comparison["pilot_value"] * 100
    comparison.to_csv(econ_dir / "pilot_vs_full_suite_economics.csv", index=False)
    write_plausibility_check(avoided_eens, results, output_dir)
    return results, comparison


def write_plausibility_check(avoided_eens: float, economics: pd.DataFrame, output_dir: Path) -> None:
    """Write economic plausibility diagnostic."""
    central = economics[
        (economics["package_cost_scenario"].eq("central")) & (economics["voll_scenario"].eq("central"))
    ].iloc[0]
    pilot_risk = read_csv(PILOT_DIR / "pilot_annualized_risk_comparison.csv").iloc[0]
    baseline_implied = float(pilot_risk["pilot_baseline_eens_mwh_per_year"])
    annual_value = float(central["annual_avoided_outage_value_usd"])
    capex = float(central["capital_cost_usd"])
    text = f"""# Economic Plausibility Check

- Central annual avoided outage value as a share of central package capital cost: {annual_value / capex:.1%}.
- Implied simple payback period: {float(central['simple_payback_years']):.2f} years.
- Avoided MWh per dollar of central capital expenditure: {avoided_eens / capex:.8f} MWh/year per USD.
- Pilot baseline annual EENS implied by the reported 2.10% pilot reduction: {baseline_implied:,.1f} MWh/year.
- The benefit is a societal avoided outage value based on VOLL; it should not be interpreted as direct utility cash savings.
- The 30-year constant annual benefit assumption is a simplifying planning assumption and may not be reasonable without more hazard, demand, and asset-aging analysis.
- Climate change, asset degradation, and demand growth are excluded unless included in the underlying PyPSA/hazard scenarios.
- VOLL is a customer welfare measure for interruption losses, not utility revenue.
"""
    (output_dir / "economics" / "economic_plausibility_check.md").write_text(text, encoding="utf-8")


def save_figure(fig: plt.Figure, path: Path) -> None:
    """Save PNG/PDF versions of a figure."""
    fig.tight_layout()
    fig.savefig(path.with_suffix(".png"), dpi=300)
    fig.savefig(path.with_suffix(".pdf"))
    plt.close(fig)


def make_figures(comparison: pd.DataFrame, risk: pd.DataFrame, economics_comparison: pd.DataFrame, output_dir: Path) -> None:
    """Create four final full-suite figures."""
    fig_dir = output_dir / "figures"
    plt.rcParams.update({"font.size": 11})
    comparison = comparison.sort_values("return_period_years")
    labels = [f"RP{int(rp)}" for rp in comparison["return_period_years"]]
    x = np.arange(len(labels))

    fig, ax = plt.subplots(figsize=(9, 5.2))
    width = 0.38
    ax.bar(x - width / 2, comparison["baseline_load_shed_mwh"], width, label="Baseline", color="#7f8c8d")
    ax.bar(x + width / 2, comparison["adapted_load_shed_mwh"], width, label="Adapted", color="#2f6c9f")
    ax.set_xticks(x, labels)
    ax.set_ylabel("Load shed (MWh)")
    ax.set_title("Full Suite: RP100 Top-Five Package Reduces Flood Load Shedding")
    ax.legend()
    ax.grid(axis="y", alpha=0.25)
    save_figure(fig, fig_dir / "01_full_suite_baseline_vs_adapted_load_shedding")

    fig, ax1 = plt.subplots(figsize=(9, 5.2))
    ax1.bar(x, comparison["avoided_load_shed_mwh"], color="#2f6c9f", label="Avoided load shed")
    ax1.set_ylabel("Avoided load shed (MWh)")
    ax1.set_xticks(x, labels)
    ax2 = ax1.twinx()
    ax2.plot(x, comparison["load_shed_reduction_percent"], color="#d77a22", marker="o", linewidth=2.5, label="Reduction")
    ax2.set_ylabel("Reduction (%)")
    ax1.set_title("Avoided Load Shedding Peaks in the Middle Return Periods")
    ax1.grid(axis="y", alpha=0.25)
    lines1, labs1 = ax1.get_legend_handles_labels()
    lines2, labs2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labs1 + labs2, loc="upper left")
    save_figure(fig, fig_dir / "02_avoided_load_shedding_by_return_period")

    fig, ax = plt.subplots(figsize=(8, 5.2))
    aep = comparison["annual_exceedance_probability"]
    order = np.argsort(aep)
    ax.plot(aep.iloc[order], comparison["baseline_load_shed_mwh"].iloc[order], marker="o", linewidth=2.5, label="Baseline")
    ax.plot(aep.iloc[order], comparison["adapted_load_shed_mwh"].iloc[order], marker="o", linewidth=2.5, label="Adapted")
    ax.fill_between(
        aep.iloc[order],
        comparison["adapted_load_shed_mwh"].iloc[order],
        comparison["baseline_load_shed_mwh"].iloc[order],
        alpha=0.18,
        color="#2f6c9f",
        label="Avoided EENS area",
    )
    ax.set_xlabel("Annual exceedance probability")
    ax.set_ylabel("Load shed (MWh)")
    ax.set_title("Full-Suite Loss-Exceedance Curves")
    ax.legend()
    ax.grid(alpha=0.25)
    save_figure(fig, fig_dir / "03_full_suite_loss_exceedance_curves")

    metrics = economics_comparison.set_index("metric")
    fig, axes = plt.subplots(1, 3, figsize=(12, 4.4))
    plot_specs = [
        ("avoided_eens_mwh_per_year", "Avoided EENS\n(MWh/year)", 1.0),
        ("central_bcr", "Central BCR", 1.0),
        ("central_npv_usd", "Central NPV\n(million USD)", 1e6),
    ]
    for ax, (metric, title, scale) in zip(axes, plot_specs):
        values = [metrics.loc[metric, "pilot_value"] / scale, metrics.loc[metric, "full_suite_value"] / scale]
        ax.bar(["Pilot", "Full suite"], values, color=["#9aa3ad", "#2f6c9f"])
        ax.set_title(title)
        ax.grid(axis="y", alpha=0.25)
        for i, value in enumerate(values):
            ax.text(i, value, f"{value:,.1f}", ha="center", va="bottom", fontweight="bold")
    fig.suptitle("Full-Suite Economics Replace the Three-Point Pilot Estimate", fontweight="bold")
    save_figure(fig, fig_dir / "04_pilot_vs_full_suite_economics")


def write_final_interpretation(
    comparison: pd.DataFrame,
    risk: pd.DataFrame,
    economics: pd.DataFrame,
    econ_comp: pd.DataFrame,
    output_dir: Path,
) -> None:
    """Write final full-suite interpretation."""
    risk_row = risk.iloc[0]
    central = economics[
        (economics["package_cost_scenario"].eq("central")) & (economics["voll_scenario"].eq("central"))
    ].iloc[0]
    low_high = economics[
        (economics["package_cost_scenario"].eq("high")) & (economics["voll_scenario"].eq("low"))
    ].iloc[0]
    max_effect = comparison.loc[comparison["avoided_load_shed_mwh"].idxmax()]
    rp500 = comparison[comparison["return_period_years"].eq(500)].iloc[0]
    text = f"""# Final Full-Suite Interpretation

1. The full-return-period suite estimates {float(risk_row['avoided_eens_mwh_per_year']):,.1f} MWh/year of avoided expected annual load shedding.
2. This differs from the three-point pilot estimate by {float(risk_row['absolute_difference']):,.1f} MWh/year ({float(risk_row['percentage_difference']):.1f}%).
3. The package removes {float(risk_row['risk_reduction_percent']):.2f}% of modeled baseline annual flood load-shedding risk.
4. The package is most effective at RP{int(max_effect['return_period_years'])}, where it avoids {float(max_effect['avoided_load_shed_mwh']):,.1f} MWh.
5. RP100 plus freeboard still provides some RP500 benefit, avoiding {float(rp500['avoided_load_shed_mwh']):,.1f} MWh, but the protection can be exceeded in extreme events.
6. The updated central BCR is {float(central['benefit_cost_ratio']):.2f}, with central NPV of ${float(central['net_present_value_usd']) / 1e6:,.1f} million.
7. Under low VOLL and high cost, BCR is {float(low_high['benefit_cost_ratio']):.2f}; this is {'above' if float(low_high['benefit_cost_ratio']) > 1 else 'below'} 1.
8. The short payback is mainly driven by the illustrative VOLL assumption and the model-estimated avoided EENS, so it should be treated cautiously.
9. Strong limitations remain: conceptual adaptation costs, illustrative VOLL, transmission-scale outage modeling, no avoided repair-cost savings, and no engineering design validation.
10. Further modeling is needed before presenting this as a final thesis result, especially review of scenario integration, site-specific costs, and whether the full return-period suite should include additional hazard probabilities.

These are model-estimated, planning-level results for a conceptual adaptation package. The monetized benefit is societal avoided outage value and should not be interpreted as direct utility revenue.
"""
    (output_dir / "final_full_suite_interpretation.md").write_text(text, encoding="utf-8")


def main() -> None:
    """Run the full-suite adaptation workflow."""
    args = parse_args()
    setup_logging(args.full_suite_dir)
    return_periods = args.return_periods or discover_return_periods(args.suite_dir)
    scenarios = [scenario_for_rp(rp, args) for rp in return_periods]
    selected = validate_selected_assets(args.selected_assets, args.pilot_dir)
    selected.to_csv(args.full_suite_dir / "selected_top5_assets.csv", index=False)
    design = validate_or_write_design(selected, args.full_suite_dir, args.pilot_dir, args.freeboard_m)

    sources = copy_reusable_pilot_runs(scenarios, args)
    inventory = build_inventory(scenarios, sources, args.full_suite_dir)
    new_solves = int(inventory["new_solve_required"].sum())
    reused = int((inventory["adapted_result_source"].eq("reused_completed_pilot")).sum())
    print(f"Complete return-period set: {', '.join('RP' + str(rp) for rp in return_periods)}")
    print(f"Reused completed pilot solves: {reused}")
    print(f"New solves required before beginning: {new_solves}")

    start = time.time()
    summaries, damage_tables, sources = solve_missing(scenarios, selected, design, args, sources)
    damage = pd.concat(damage_tables, ignore_index=True)
    damage.to_csv(args.full_suite_dir / "full_suite_selected_asset_damage_comparison.csv", index=False)
    comparison = build_full_comparison(scenarios, summaries, damage, sources, args.full_suite_dir)
    build_inventory(scenarios, sources, args.full_suite_dir)
    risk = annualized_risk(comparison, args.full_suite_dir, args.pilot_dir)
    write_sanity_checks(comparison, risk, args.full_suite_dir)

    economics = pd.DataFrame()
    econ_comp = pd.DataFrame()
    if not args.skip_economics:
        economics, econ_comp = update_economics(float(risk["avoided_eens_mwh_per_year"].iloc[0]), args, args.full_suite_dir)
        make_figures(comparison, risk, econ_comp, args.full_suite_dir)
        write_final_interpretation(comparison, risk, economics, econ_comp, args.full_suite_dir)

    successful = comparison["solver_status"].astype(str).str.lower().eq("ok").sum()
    failed = len(comparison) - successful
    risk_row = risk.iloc[0]
    central = economics[
        (economics["package_cost_scenario"].eq("central")) & (economics["voll_scenario"].eq("central"))
    ].iloc[0]
    annual_values = economics[economics["package_cost_scenario"].eq("central")].set_index("voll_scenario")[
        "annual_avoided_outage_value_usd"
    ]
    low_high = economics[
        (economics["package_cost_scenario"].eq("high")) & (economics["voll_scenario"].eq("low"))
    ].iloc[0]
    robust = bool(float(low_high["benefit_cost_ratio"]) > 1.0)
    print("\nFull-suite RP100 top-five workflow complete")
    print(f"Complete return-period set: {', '.join('RP' + str(rp) for rp in return_periods)}")
    print(f"Reused pilot solves: {reused}")
    print(f"New solves: {new_solves}")
    print(f"Successful solves: {successful}; failed solves: {failed}")
    print(f"Baseline full-suite EENS: {float(risk_row['baseline_eens_mwh_per_year']):,.1f} MWh/year")
    print(f"Adapted full-suite EENS: {float(risk_row['adapted_eens_mwh_per_year']):,.1f} MWh/year")
    print(f"Avoided full-suite EENS: {float(risk_row['avoided_eens_mwh_per_year']):,.1f} MWh/year")
    print(f"Risk reduction: {float(risk_row['risk_reduction_percent']):.2f}%")
    print(
        f"Pilot vs full-suite difference: {float(risk_row['absolute_difference']):,.1f} MWh/year "
        f"({float(risk_row['percentage_difference']):.1f}%)"
    )
    print(f"Updated low annual benefit: ${float(annual_values['low']) / 1e6:,.1f}M/year")
    print(f"Updated central annual benefit: ${float(annual_values['central']) / 1e6:,.1f}M/year")
    print(f"Updated high annual benefit: ${float(annual_values['high']) / 1e6:,.1f}M/year")
    print(f"Updated central BCR: {float(central['benefit_cost_ratio']):.2f}")
    print(f"Updated central NPV: ${float(central['net_present_value_usd']) / 1e6:,.1f}M")
    print(f"Updated payback period: {float(central['simple_payback_years']):.1f} years")
    print(f"Break-even package cost: ${float(central['break_even_capital_cost_usd']) / 1e6:,.1f}M")
    print(f"High preliminary economics remained robust under low VOLL/high cost: {robust}")
    print(f"Outputs written to: {args.full_suite_dir}")
    print(f"Elapsed workflow time: {(time.time() - start) / 60:.2f} minutes")


if __name__ == "__main__":
    main()
