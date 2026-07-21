"""
Create polished return-period comparison plots from the latest gradual suite.

These figures replace the older F6.3/W3.10 plots with the current model setup:
  - Flood: F6.2 lines, F1.* generators, F2.* buses/substations
  - Tropical cyclone: W6.3 lines, W1.* generators, W2.3 buses/substations
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import pandas as pd


PROJECT_DIR = Path(r"C:\oxford_tc_project")
OUT = (
    PROJECT_DIR
    / "data"
    / "Electricity"
    / "pypsa_florida_network"
    / "calibrated_hazard_scenarios"
    / "gradual_return_period_suite"
)
SUMMARY = OUT / "gradual_return_period_suite_summary.csv"

HAZARD_LABELS = {
    "flood": "Flood F6.2",
    "tropical_cyclone": "TC W6.3",
}
HAZARD_COLORS = {
    "flood": "#1f77b4",
    "tropical_cyclone": "#ff7f0e",
}


def money_axis(value: float, _pos=None) -> str:
    if abs(value) >= 1e9:
        return f"${value / 1e9:.1f}B"
    if abs(value) >= 1e6:
        return f"${value / 1e6:.0f}M"
    if abs(value) >= 1e3:
        return f"${value / 1e3:.0f}k"
    return f"${value:.0f}"


def plain_number(value: float, _pos=None) -> str:
    if abs(value) >= 1e6:
        return f"{value / 1e6:.1f}M"
    if abs(value) >= 1e3:
        return f"{value / 1e3:.0f}k"
    return f"{value:.0f}"


def plot_metric(
    summary: pd.DataFrame,
    metric: str,
    ylabel: str,
    title: str,
    output_name: str,
    money: bool = False,
    percent: bool = False,
) -> None:
    fig, ax = plt.subplots(figsize=(9.6, 5.8))
    for hazard, group in summary.groupby("hazard", sort=False):
        group = group.sort_values("return_period")
        ax.plot(
            group["return_period"],
            group[metric],
            marker="o",
            linewidth=2.4,
            markersize=5.8,
            color=HAZARD_COLORS.get(hazard),
            label=HAZARD_LABELS.get(hazard, hazard),
        )

    ax.set_title(title, fontsize=14, fontweight="bold", pad=11)
    ax.set_xlabel("Return period (years)", fontsize=11)
    ax.set_ylabel(ylabel, fontsize=11)
    ax.grid(True, alpha=0.22)
    ax.legend(loc="best", frameon=True)

    if money:
        ax.yaxis.set_major_formatter(FuncFormatter(money_axis))
    elif not percent:
        ax.yaxis.set_major_formatter(FuncFormatter(plain_number))
    if percent:
        ax.set_ylim(bottom=max(0, summary[metric].min() - 0.1), top=100.05)

    fig.text(
        0.5,
        0.018,
        "Updated gradual-damage PyPSA scenarios using SNAIL exposure and current NHESS curve assignments.",
        ha="center",
        fontsize=8.5,
        color="#555555",
    )
    fig.tight_layout(rect=(0, 0.045, 1, 1))
    fig.savefig(OUT / output_name, dpi=260)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create polished gradual return-period plots.")
    parser.add_argument("--suite-dir", type=Path, default=OUT)
    return parser.parse_args()


def main() -> None:
    global OUT
    args = parse_args()
    OUT = args.suite_dir
    summary = pd.read_csv(OUT / "gradual_return_period_suite_summary.csv")
    summary = summary[summary["status"].eq("complete")].copy()
    summary["hazard"] = pd.Categorical(
        summary["hazard"],
        categories=["flood", "tropical_cyclone"],
        ordered=True,
    )
    summary = summary.sort_values(["hazard", "return_period"])

    specs = [
        (
            "import_slack_mwh",
            "Import slack (MWh)",
            "Import slack by return period",
            "import_slack_by_return_period.png",
            False,
            False,
        ),
        (
            "system_cost_usd",
            "System cost (USD)",
            "System cost by return period",
            "system_cost_by_return_period.png",
            True,
            False,
        ),
        (
            "incremental_system_cost_usd",
            "Incremental system cost (USD)",
            "Incremental system cost by return period",
            "incremental_system_cost_by_return_period.png",
            True,
            False,
        ),
        (
            "load_shed_mwh",
            "Load shed (MWh)",
            "Load shed by return period",
            "load_shed_by_return_period.png",
            False,
            False,
        ),
        (
            "demand_served_percent",
            "Demand served (%)",
            "Percentage of demand served by return period",
            "demand_served_percent_by_return_period.png",
            False,
            True,
        ),
        (
            "line_capacity_loss_mva",
            "Transmission capacity lost (MVA)",
            "Transmission capacity lost by return period",
            "line_capacity_lost_by_return_period.png",
            False,
            False,
        ),
        (
            "generator_capacity_loss_mw",
            "Generation capacity lost (MW)",
            "Generation capacity lost by return period",
            "generator_capacity_lost_by_return_period.png",
            False,
            False,
        ),
    ]
    for metric, ylabel, title, output_name, money, percent in specs:
        plot_metric(summary, metric, ylabel, title, output_name, money=money, percent=percent)

    plot_metric(
        summary,
        "load_shed_mwh",
        "Load shed (MWh)",
        "Flood vs tropical cyclone load shedding across return periods",
        "combined_flood_tc_load_shedding_by_return_period.png",
    )

    print("Saved updated return-period plots to:", OUT)
    for _, _, _, output_name, _, _ in specs:
        print(OUT / output_name)
    print(OUT / "combined_flood_tc_load_shedding_by_return_period.png")


if __name__ == "__main__":
    main()
