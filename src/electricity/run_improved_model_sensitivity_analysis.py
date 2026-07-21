"""
Run focused sensitivity tests for the improved Florida PyPSA hazard model.

The sensitivity suite varies the two most important calibration assumptions:
  - transmission line capacity multiplier
  - northern boundary import cap fraction

For each assumption set, it:
  1. builds a calibrated no-hazard baseline,
  2. runs Flood RP500 and TC RP500 gradual hazard scenarios,
  3. collects load shedding, import slack, cost, and capacity-loss metrics,
  4. writes comparison plots and a short report.

This is intentionally focused on the two extreme return-period cases so the
run is computationally useful without exploding into dozens of scenarios.
"""

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import pandas as pd


PROJECT_DIR = Path(r"C:\oxford_tc_project")
ELECTRICITY_DIR = PROJECT_DIR / "data" / "Electricity"
NETWORK_DIR = ELECTRICITY_DIR / "pypsa_florida_network_county_load_generator_overrides"
BOUNDARY_BUSES = (
    ELECTRICITY_DIR / "pypsa_florida_network" / "boundary_import_buses" / "selected_boundary_import_buses.csv"
)
MANIFEST = NETWORK_DIR / "gradual_return_period_suite_manifest.csv"
OUTPUT_DIR = NETWORK_DIR / "sensitivity_analysis"


SENSITIVITY_CASES = [
    {
        "case_id": "line_x150_import_cap100",
        "line_capacity_multiplier": 1.50,
        "boundary_import_cap_fraction": 1.00,
        "description": "Lower transmission capacity multiplier, baseline import cap.",
    },
    {
        "case_id": "line_x175_import_cap075",
        "line_capacity_multiplier": 1.75,
        "boundary_import_cap_fraction": 0.75,
        "description": "Final line multiplier, lower boundary import cap.",
    },
    {
        "case_id": "line_x175_import_cap100",
        "line_capacity_multiplier": 1.75,
        "boundary_import_cap_fraction": 1.00,
        "description": "Final calibrated assumption set.",
    },
    {
        "case_id": "line_x175_import_cap125",
        "line_capacity_multiplier": 1.75,
        "boundary_import_cap_fraction": 1.25,
        "description": "Final line multiplier, higher boundary import cap.",
    },
    {
        "case_id": "line_x200_import_cap100",
        "line_capacity_multiplier": 2.00,
        "boundary_import_cap_fraction": 1.00,
        "description": "Higher transmission capacity multiplier, baseline import cap.",
    },
]

SCENARIO_IDS = [
    "flood_jrc_rp500_f62_gradual",
    "tc_storm_rp500_w63_gradual",
]


def run_command(command: list[str]) -> None:
    print("\nRunning:")
    print(" ".join(command))
    subprocess.run(command, cwd=PROJECT_DIR, check=True)


def read_first(path: Path) -> pd.Series:
    if not path.exists():
        raise FileNotFoundError(path)
    df = pd.read_csv(path)
    if df.empty:
        raise ValueError(f"Empty file: {path}")
    return df.iloc[0]


def build_baseline(case: dict) -> Path:
    case_dir = OUTPUT_DIR / case["case_id"]
    baseline_build = case_dir / "baseline_build"
    official_baseline = case_dir / "baseline_calibrated_no_hazard"
    if baseline_build.exists():
        shutil.rmtree(baseline_build)
    if official_baseline.exists():
        shutil.rmtree(official_baseline)
    baseline_build.mkdir(parents=True, exist_ok=True)

    run_command(
        [
            sys.executable,
            "data/Electricity/run_boundary_import_baseline.py",
            "--network-dir",
            str(NETWORK_DIR),
            "--boundary-buses",
            str(BOUNDARY_BUSES),
            "--output-dir",
            str(baseline_build),
            "--line-limit-multiplier",
            str(case["line_capacity_multiplier"]),
            "--add-artifact-island-imports",
            "--boundary-import-cap-fraction",
            str(case["boundary_import_cap_fraction"]),
            "--artifact-import-peak-load-margin",
            "1.10",
        ]
    )

    source = baseline_build / "boundary_imports_plus_artifact_islands"
    shutil.copytree(source, official_baseline)
    return official_baseline


def run_hazard_cases(case: dict, baseline_dir: Path) -> Path:
    scenario_output = OUTPUT_DIR / case["case_id"] / "hazard_scenarios"
    if scenario_output.exists():
        shutil.rmtree(scenario_output)
    scenario_output.mkdir(parents=True, exist_ok=True)

    command = [
        sys.executable,
        "data/Electricity/run_florida_pypsa_calibrated_hazard_scenarios.py",
        "--manifest",
        str(MANIFEST),
        "--output-dir",
        str(scenario_output),
        "--network-dir",
        str(NETWORK_DIR),
        "--baseline-dir",
        str(baseline_dir),
        "--line-capacity-multiplier",
        str(case["line_capacity_multiplier"]),
        "--solver",
        "highs",
        "--highs-method",
        "ipm",
        "--start",
        "2025-01-01 00:00:00",
        "--periods",
        "24",
        "--chunk-size",
        "24",
    ]
    for scenario_id in SCENARIO_IDS:
        command.extend(["--scenario-id", scenario_id])
    run_command(command)
    return scenario_output


def collect_case_results(case: dict, baseline_dir: Path, scenario_output: Path) -> list[dict]:
    baseline = read_first(baseline_dir / "baseline_summary.csv")
    rows = [
        {
            "case_id": case["case_id"],
            "description": case["description"],
            "line_capacity_multiplier": case["line_capacity_multiplier"],
            "boundary_import_cap_fraction": case["boundary_import_cap_fraction"],
            "scenario_id": "baseline",
            "hazard": "baseline",
            "return_period": 0,
            "solver_status": baseline["solver_status"],
            "solver_condition": baseline["solver_condition"],
            "load_shed_mwh": float(baseline["total_load_shed_mwh"]),
            "demand_served_mwh": float(baseline["total_demand_served_mwh"]),
            "import_slack_mwh": float(baseline["total_import_slack_mwh"]),
            "system_cost_usd": float(baseline["total_system_cost_usd"]),
            "incremental_system_cost_usd": 0.0,
            "line_capacity_loss_mva": 0.0,
            "generator_capacity_loss_mw": 0.0,
        }
    ]

    for scenario_id in SCENARIO_IDS:
        scenario_dir = scenario_output / scenario_id
        summary = read_first(scenario_dir / "scenario_summary.csv")
        incremental = read_first(scenario_dir / "incremental_vs_calibrated_baseline.csv")
        rows.append(
            {
                "case_id": case["case_id"],
                "description": case["description"],
                "line_capacity_multiplier": case["line_capacity_multiplier"],
                "boundary_import_cap_fraction": case["boundary_import_cap_fraction"],
                "scenario_id": scenario_id,
                "hazard": "flood" if scenario_id.startswith("flood") else "tropical_cyclone",
                "return_period": 500,
                "solver_status": summary["solver_status"],
                "solver_condition": summary["solver_condition"],
                "load_shed_mwh": float(summary["total_load_shed_mwh"]),
                "demand_served_mwh": float(summary["total_demand_served_mwh"]),
                "import_slack_mwh": float(summary["total_import_slack_mwh"]),
                "system_cost_usd": float(summary["total_system_cost_usd"]),
                "incremental_system_cost_usd": float(incremental["incremental_system_cost_usd"]),
                "line_capacity_loss_mva": float(incremental["line_capacity_loss_mva"]),
                "generator_capacity_loss_mw": float(incremental["generator_capacity_loss_mw"]),
            }
        )
    return rows


def money_axis(value: float, _pos=None) -> str:
    if abs(value) >= 1e9:
        return f"${value / 1e9:.1f}B"
    if abs(value) >= 1e6:
        return f"${value / 1e6:.1f}M"
    if abs(value) >= 1e3:
        return f"${value / 1e3:.0f}k"
    return f"${value:.0f}"


def plot_sensitivity(results: pd.DataFrame) -> None:
    hazard_results = results[results["hazard"].ne("baseline")].copy()
    order = [case["case_id"] for case in SENSITIVITY_CASES]
    hazard_results["case_id"] = pd.Categorical(hazard_results["case_id"], categories=order, ordered=True)
    hazard_results = hazard_results.sort_values(["case_id", "hazard"])

    for metric, ylabel, output_name, money in [
        ("load_shed_mwh", "Load shed (MWh)", "sensitivity_load_shed_mwh.png", False),
        ("incremental_system_cost_usd", "Incremental system cost (USD)", "sensitivity_incremental_cost.png", True),
        ("import_slack_mwh", "Import slack (MWh)", "sensitivity_import_slack.png", False),
    ]:
        fig, ax = plt.subplots(figsize=(11, 5.8))
        for hazard, group in hazard_results.groupby("hazard", observed=False):
            label = "Flood RP500" if hazard == "flood" else "TC RP500"
            ax.plot(group["case_id"].astype(str), group[metric], marker="o", linewidth=2.2, label=label)
        ax.set_title(f"Sensitivity of {ylabel.lower()}")
        ax.set_xlabel("Sensitivity case")
        ax.set_ylabel(ylabel)
        ax.tick_params(axis="x", rotation=25)
        ax.grid(alpha=0.25)
        ax.legend()
        if money:
            ax.yaxis.set_major_formatter(FuncFormatter(money_axis))
        fig.tight_layout()
        fig.savefig(OUTPUT_DIR / output_name, dpi=240)
        plt.close(fig)


def write_report(results: pd.DataFrame) -> None:
    hazards = results[results["hazard"].ne("baseline")].copy()
    baseline = results[results["hazard"].eq("baseline")].copy()
    flood = hazards[hazards["hazard"].eq("flood")]
    tc = hazards[hazards["hazard"].eq("tropical_cyclone")]

    lines = [
        "# Improved Model Sensitivity Analysis",
        "",
        "Sensitivity dimensions:",
        "- Transmission capacity multiplier: 1.50, 1.75, 2.00",
        "- Boundary import cap fraction: 0.75, 1.00, 1.25",
        "- Artifact-island import margin fixed at 1.10 x peak island load",
        "- Scenarios tested: Flood RP500 and TC RP500",
        "",
        "## Main Findings",
        "",
        f"- All sensitivity baselines solved successfully; maximum baseline load shed = {baseline['load_shed_mwh'].max():,.2f} MWh.",
        f"- Flood RP500 load shed range = {flood['load_shed_mwh'].min():,.2f} to {flood['load_shed_mwh'].max():,.2f} MWh.",
        f"- TC RP500 load shed range = {tc['load_shed_mwh'].min():,.2f} to {tc['load_shed_mwh'].max():,.2f} MWh.",
        f"- Flood RP500 incremental cost range = ${flood['incremental_system_cost_usd'].min():,.0f} to ${flood['incremental_system_cost_usd'].max():,.0f}.",
        f"- TC RP500 incremental cost range = ${tc['incremental_system_cost_usd'].min():,.0f} to ${tc['incremental_system_cost_usd'].max():,.0f}.",
        "",
        "## Interpretation",
        "",
        "This focused sensitivity test checks whether the headline result depends strongly on the calibrated line multiplier or import cap. Flood RP500 remains a capacity-derating/cost impact rather than a load-shedding impact across the tested cases if its load shed stays near zero. TC RP500 is expected to be more sensitive because it is the only updated scenario that already produces load shedding under final assumptions.",
        "",
        "## Files",
        "",
        "- `sensitivity_summary.csv`",
        "- `sensitivity_load_shed_mwh.png`",
        "- `sensitivity_incremental_cost.png`",
        "- `sensitivity_import_slack.png`",
    ]
    (OUTPUT_DIR / "sensitivity_analysis_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    rows = []
    pd.DataFrame(SENSITIVITY_CASES).to_csv(OUTPUT_DIR / "sensitivity_case_definitions.csv", index=False)

    for case in SENSITIVITY_CASES:
        print(f"\n=== Sensitivity case: {case['case_id']} ===")
        baseline_dir = build_baseline(case)
        scenario_output = run_hazard_cases(case, baseline_dir)
        rows.extend(collect_case_results(case, baseline_dir, scenario_output))

    results = pd.DataFrame(rows)
    results.to_csv(OUTPUT_DIR / "sensitivity_summary.csv", index=False)
    plot_sensitivity(results)
    write_report(results)

    print("\nSaved sensitivity analysis:", OUTPUT_DIR)
    print(results.to_string(index=False, float_format=lambda value: f"{value:,.3f}"))


if __name__ == "__main__":
    main()
