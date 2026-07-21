"""Run seasonal no-hazard validation for the improved Florida PyPSA model.

The official calibrated baseline currently validates a 24-hour January window.
This script checks whether the same calibrated assumptions also avoid
no-hazard load shedding in representative seasonal and load-stress windows.

It reuses ``run_boundary_import_baseline.py`` so the import placement, artifact
island support, load shedding, marginal costs, and ``s_nom`` multiplier remain
identical to the official baseline.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


PROJECT_DIR = Path(__file__).resolve().parents[2]
NETWORK_DIR = PROJECT_DIR / "data" / "Electricity" / "pypsa_florida_network_county_load_generator_overrides"
BOUNDARY_BUSES = PROJECT_DIR / "data" / "Electricity" / "pypsa_florida_network" / "boundary_import_buses" / "selected_boundary_import_buses.csv"
OUTPUT_DIR = NETWORK_DIR / "seasonal_baseline_validation"
LOADS_P_SET = NETWORK_DIR / "loads-p_set.csv"

LINE_MULTIPLIER = 1.75
BOUNDARY_IMPORT_CAP_FRACTION = 1.0
ARTIFACT_IMPORT_PEAK_LOAD_MARGIN = 1.10
PERIODS = 24
CHUNK_SIZE = 24


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run seasonal no-hazard validation.")
    parser.add_argument("--network-dir", type=Path, default=NETWORK_DIR)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--boundary-buses", type=Path, default=BOUNDARY_BUSES)
    parser.add_argument("--line-multiplier", type=float, default=LINE_MULTIPLIER)
    parser.add_argument("--boundary-import-cap-fraction", type=float, default=BOUNDARY_IMPORT_CAP_FRACTION)
    parser.add_argument("--artifact-import-peak-load-margin", type=float, default=ARTIFACT_IMPORT_PEAK_LOAD_MARGIN)
    return parser.parse_args()


def load_statewide_hourly_load(loads_p_set: Path) -> pd.DataFrame:
    loads = pd.read_csv(loads_p_set)
    loads["snapshot"] = pd.to_datetime(loads["snapshot"], errors="raise")
    load_cols = [c for c in loads.columns if c != "snapshot"]
    loads["statewide_load_mw"] = loads[load_cols].sum(axis=1)
    loads["date"] = loads["snapshot"].dt.date
    return loads[["snapshot", "date", "statewide_load_mw"]]


def day_start_from_peak(loads: pd.DataFrame, mask: pd.Series, label: str) -> dict:
    subset = loads[mask].copy()
    if subset.empty:
        raise ValueError(f"No snapshots found for {label}.")
    peak_row = subset.loc[subset["statewide_load_mw"].idxmax()]
    day = pd.Timestamp(peak_row["date"])
    day_load = loads[loads["date"] == peak_row["date"]]
    return {
        "case_id": label,
        "start": day.strftime("%Y-%m-%d 00:00:00"),
        "selection_method": "data_driven_peak_day",
        "peak_snapshot": peak_row["snapshot"],
        "peak_load_mw": float(peak_row["statewide_load_mw"]),
        "daily_energy_mwh": float(day_load["statewide_load_mw"].sum()),
    }


def day_start_from_low(loads: pd.DataFrame, label: str) -> dict:
    daily = (
        loads.groupby("date", as_index=False)
        .agg(daily_energy_mwh=("statewide_load_mw", "sum"), peak_load_mw=("statewide_load_mw", "max"))
        .sort_values("daily_energy_mwh")
    )
    row = daily.iloc[0]
    return {
        "case_id": label,
        "start": pd.Timestamp(row["date"]).strftime("%Y-%m-%d 00:00:00"),
        "selection_method": "data_driven_low_load_day",
        "peak_snapshot": pd.NaT,
        "peak_load_mw": float(row["peak_load_mw"]),
        "daily_energy_mwh": float(row["daily_energy_mwh"]),
    }


def fixed_day(loads: pd.DataFrame, case_id: str, start: str) -> dict:
    start_ts = pd.Timestamp(start)
    day = start_ts.date()
    day_load = loads[loads["date"] == day]
    if day_load.empty:
        raise ValueError(f"No snapshots found for fixed day {start}.")
    peak_row = day_load.loc[day_load["statewide_load_mw"].idxmax()]
    return {
        "case_id": case_id,
        "start": start_ts.strftime("%Y-%m-%d 00:00:00"),
        "selection_method": "fixed_representative_day",
        "peak_snapshot": peak_row["snapshot"],
        "peak_load_mw": float(peak_row["statewide_load_mw"]),
        "daily_energy_mwh": float(day_load["statewide_load_mw"].sum()),
    }


def choose_validation_windows(loads: pd.DataFrame) -> pd.DataFrame:
    windows = [
        fixed_day(loads, "winter_representative_jan15", "2025-01-15 00:00:00"),
        fixed_day(loads, "spring_representative_apr15", "2025-04-15 00:00:00"),
        fixed_day(loads, "summer_representative_jul15", "2025-07-15 00:00:00"),
        fixed_day(loads, "fall_representative_oct15", "2025-10-15 00:00:00"),
        day_start_from_peak(loads, loads["snapshot"].dt.month.isin([12, 1, 2]), "winter_peak_load_day"),
        day_start_from_peak(loads, loads["snapshot"].dt.month.isin([6, 7, 8, 9]), "summer_peak_load_day"),
        day_start_from_peak(loads, pd.Series(True, index=loads.index), "annual_peak_load_day"),
        day_start_from_low(loads, "annual_low_load_day"),
    ]
    windows_df = pd.DataFrame(windows)
    windows_df = windows_df.drop_duplicates(subset=["start"], keep="first").reset_index(drop=True)
    return windows_df


def run_case(
    row: pd.Series,
    network_dir: Path,
    output_dir: Path,
    boundary_buses: Path,
    line_multiplier: float,
    boundary_import_cap_fraction: float,
    artifact_import_peak_load_margin: float,
) -> pd.Series:
    case_dir = output_dir / str(row["case_id"])
    case_dir.mkdir(parents=True, exist_ok=True)
    log_path = case_dir / "run.log"

    cmd = [
        sys.executable,
        "data/Electricity/run_boundary_import_baseline.py",
        "--network-dir",
        str(network_dir),
        "--boundary-buses",
        str(boundary_buses),
        "--output-dir",
        str(case_dir),
        "--line-limit-multiplier",
        str(line_multiplier),
        "--boundary-import-cap-fraction",
        str(boundary_import_cap_fraction),
        "--artifact-import-peak-load-margin",
        str(artifact_import_peak_load_margin),
        "--add-artifact-island-imports",
        "--start",
        str(row["start"]),
        "--periods",
        str(PERIODS),
        "--chunk-size",
        str(CHUNK_SIZE),
        "--solver",
        "highs",
        "--highs-method",
        "ipm",
    ]

    with log_path.open("w", encoding="utf-8") as log_file:
        completed = subprocess.run(
            cmd,
            cwd=PROJECT_DIR,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
        )

    summary_path = (
        case_dir
        / "boundary_imports_plus_artifact_islands"
        / "baseline_summary.csv"
    )
    if not summary_path.exists():
        return pd.Series(
            {
                "case_id": row["case_id"],
                "start": row["start"],
                "return_code": completed.returncode,
                "solver_status": "missing_summary",
                "solver_condition": "missing_summary",
            }
        )

    summary = pd.read_csv(summary_path).iloc[0]
    return pd.Series(
        {
            "case_id": row["case_id"],
            "start": row["start"],
            "selection_method": row["selection_method"],
            "input_peak_load_mw": row["peak_load_mw"],
            "input_daily_energy_mwh": row["daily_energy_mwh"],
            "return_code": completed.returncode,
            "solver_status": summary.get("solver_status"),
            "solver_condition": summary.get("solver_condition"),
            "total_demand_mwh": summary.get("total_demand_mwh"),
            "demand_served_mwh": summary.get("total_demand_served_mwh"),
            "load_shed_mwh": summary.get("total_load_shed_mwh"),
            "import_slack_mwh": summary.get("total_import_slack_mwh"),
            "number_overloaded_lines": summary.get("number_overloaded_lines"),
            "max_line_loading_pu": summary.get("max_line_loading_pu"),
            "system_cost_usd": summary.get("total_system_cost_usd"),
            "case_output_dir": str(case_dir),
        }
    )


def plot_validation(summary: pd.DataFrame, output_dir: Path) -> None:
    plot_df = summary.sort_values("input_daily_energy_mwh").copy()

    fig, ax = plt.subplots(figsize=(10, 5), dpi=180)
    ax.bar(plot_df["case_id"], plot_df["load_shed_mwh"], color="#1f77b4")
    ax.set_title("Seasonal No-Hazard Baseline Validation: Load Shedding")
    ax.set_ylabel("Load shed (MWh)")
    ax.set_xlabel("Validation window")
    ax.tick_params(axis="x", rotation=35)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_dir / "seasonal_baseline_load_shedding.png")
    plt.close(fig)

    fig, ax1 = plt.subplots(figsize=(10, 5), dpi=180)
    ax1.plot(
        plot_df["case_id"],
        plot_df["total_demand_mwh"],
        marker="o",
        linewidth=2,
        label="Demand",
        color="#1f77b4",
    )
    ax1.plot(
        plot_df["case_id"],
        plot_df["import_slack_mwh"],
        marker="o",
        linewidth=2,
        label="Import slack",
        color="#ff7f0e",
    )
    ax1.set_title("Seasonal No-Hazard Baseline Validation: Demand and Import Slack")
    ax1.set_ylabel("MWh over 24-hour window")
    ax1.set_xlabel("Validation window")
    ax1.tick_params(axis="x", rotation=35)
    ax1.grid(alpha=0.25)
    ax1.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "seasonal_baseline_demand_import_slack.png")
    plt.close(fig)


def write_report(
    windows: pd.DataFrame,
    summary: pd.DataFrame,
    network_dir: Path,
    output_dir: Path,
    line_multiplier: float,
    boundary_import_cap_fraction: float,
    artifact_import_peak_load_margin: float,
) -> None:
    max_shed = float(pd.to_numeric(summary["load_shed_mwh"], errors="coerce").max())
    max_loading = float(pd.to_numeric(summary["max_line_loading_pu"], errors="coerce").max())
    failed = summary[summary["solver_status"].ne("ok")]

    report = f"""# Seasonal No-Hazard Baseline Validation

This validation checks whether the official improved no-hazard baseline remains feasible outside the original January 1 24-hour window.

## Assumptions

- Network: `{network_dir.name}`
- Line capacity multiplier: `s_nom x {line_multiplier}`
- Boundary import cap fraction: `{boundary_import_cap_fraction}`
- Artifact-island import margin: `{artifact_import_peak_load_margin} x peak island load`
- Window length: `{PERIODS}` hours

## Result

- Validation windows run: `{len(summary)}`
- Maximum no-hazard load shedding: `{max_shed:,.6f} MWh`
- Maximum line loading reported by solved cases: `{max_loading:,.3f} pu`
- Solver failures or missing summaries: `{len(failed)}`

## Interpretation

"""
    if max_shed <= 1e-6 and failed.empty:
        report += (
            "The calibrated baseline passes the seasonal validation: no tested "
            "seasonal or peak-load no-hazard window sheds load. This supports "
            f"using `s_nom x {line_multiplier}` as the central calibrated baseline for the "
            "hazard scenarios.\n"
        )
    else:
        report += (
            "At least one tested no-hazard window still has load shedding or did "
            "not solve cleanly. The calibrated baseline should be revisited before "
            "interpreting hazard impacts for those seasons.\n"
        )

    report += "\n## Validation Summary\n\n"
    report_table = summary[
        [
            "case_id",
            "start",
            "total_demand_mwh",
            "load_shed_mwh",
            "import_slack_mwh",
            "number_overloaded_lines",
            "max_line_loading_pu",
            "solver_status",
            "solver_condition",
        ]
    ].copy()
    report += "| " + " | ".join(report_table.columns) + " |\n"
    report += "| " + " | ".join(["---"] * len(report_table.columns)) + " |\n"
    for _, row in report_table.iterrows():
        values = []
        for col in report_table.columns:
            value = row[col]
            if isinstance(value, float):
                values.append(f"{value:,.3f}")
            else:
                values.append(str(value))
        report += "| " + " | ".join(values) + " |\n"
    report += "\n\n## Files\n\n"
    report += "- `seasonal_validation_windows.csv`\n"
    report += "- `seasonal_baseline_validation_summary.csv`\n"
    report += "- `seasonal_baseline_load_shedding.png`\n"
    report += "- `seasonal_baseline_demand_import_slack.png`\n"

    (output_dir / "seasonal_baseline_validation_report.md").write_text(report, encoding="utf-8")


def main() -> None:
    args = parse_args()
    network_dir = args.network_dir.resolve()
    output_dir = (args.output_dir or network_dir / "seasonal_baseline_validation").resolve()
    loads_p_set = network_dir / "loads-p_set.csv"
    output_dir.mkdir(parents=True, exist_ok=True)
    loads = load_statewide_hourly_load(loads_p_set)
    windows = choose_validation_windows(loads)
    windows.to_csv(output_dir / "seasonal_validation_windows.csv", index=False)

    rows = []
    for _, row in windows.iterrows():
        print(f"Running {row['case_id']} starting {row['start']}")
        rows.append(
            run_case(
                row,
                network_dir,
                output_dir,
                args.boundary_buses.resolve(),
                args.line_multiplier,
                args.boundary_import_cap_fraction,
                args.artifact_import_peak_load_margin,
            )
        )

    summary = pd.DataFrame(rows)
    summary.to_csv(output_dir / "seasonal_baseline_validation_summary.csv", index=False)
    plot_validation(summary, output_dir)
    write_report(
        windows,
        summary,
        network_dir,
        output_dir,
        args.line_multiplier,
        args.boundary_import_cap_fraction,
        args.artifact_import_peak_load_margin,
    )
    print("Saved seasonal validation:", output_dir)
    print(summary[["case_id", "start", "load_shed_mwh", "import_slack_mwh", "solver_status"]].to_string(index=False))


if __name__ == "__main__":
    main()
