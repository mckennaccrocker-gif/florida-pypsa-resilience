"""
Summarize tropical-cyclone wind PyPSA line-outage scenarios.

Reads per-threshold `incremental_impact_summary.csv` files produced by
run_florida_pypsa_line_outage_scenario.py and writes a comparison table/plot.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


PROJECT_DIR = Path(r"C:\oxford_tc_project")
SCENARIO_DIR = PROJECT_DIR / "data" / "Electricity" / "pypsa_florida_network" / "hazard_scenarios"
OUTPUT_DIR = SCENARIO_DIR / "tc_wind_threshold_comparison"


def load_summaries() -> pd.DataFrame:
    records = []
    for path in sorted(SCENARIO_DIR.glob("tc_wind_threshold_*/incremental_impact_summary.csv")):
        frame = pd.read_csv(path)
        if frame.empty:
            continue
        row = frame.iloc[0].to_dict()
        row["scenario_dir"] = str(path.parent)
        records.append(row)
    if not records:
        raise FileNotFoundError("No TC wind incremental impact summaries found.")
    return pd.DataFrame(records).sort_values("threshold")


def plot_comparison(summary: pd.DataFrame, output_path: Path) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(9, 8), sharex=True)

    axes[0].plot(
        summary["threshold"],
        summary["incremental_load_shed_mwh"],
        marker="o",
        color="#b3261e",
        linewidth=2,
    )
    axes[0].set_ylabel("Incremental load shed (MWh)")
    axes[0].grid(True, alpha=0.25)

    axes[1].bar(
        summary["threshold"].astype(str),
        summary["outaged_line_length_km"],
        color="#355c7d",
    )
    axes[1].set_xlabel("TC wind outage threshold (m/s)")
    axes[1].set_ylabel("Outaged line length (km)")
    axes[1].grid(True, axis="y", alpha=0.25)

    fig.suptitle("Florida PyPSA TC Wind Line-Outage Sensitivity")
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    summary = load_summaries()

    keep_cols = [
        "hazard",
        "threshold",
        "outaged_lines",
        "outaged_line_length_km",
        "outaged_line_capacity_mva",
        "baseline_load_shed_mwh",
        "scenario_load_shed_mwh",
        "incremental_load_shed_mwh",
        "baseline_import_slack_mwh",
        "scenario_import_slack_mwh",
        "incremental_import_slack_mwh",
        "incremental_load_shedding_cost_usd",
        "incremental_import_slack_cost_usd",
        "load_shedding_still_occurs",
        "scenario_dir",
    ]
    keep_cols = [col for col in keep_cols if col in summary.columns]
    summary = summary[keep_cols]

    output_csv = OUTPUT_DIR / "tc_wind_threshold_comparison.csv"
    output_png = OUTPUT_DIR / "tc_wind_threshold_sensitivity.png"
    summary.to_csv(output_csv, index=False)
    plot_comparison(summary, output_png)

    print("TC wind threshold comparison")
    print(summary.to_string(index=False))
    print("\nSaved:")
    print(output_csv)
    print(output_png)


if __name__ == "__main__":
    main()
