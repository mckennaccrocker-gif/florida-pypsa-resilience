"""
Plot updated expected annual electricity system cost graphs.

Reads the latest gradual return-period suite annualized-risk outputs and
overwrites the polished PNGs in interpretation_and_annualized_risk.
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
    / "interpretation_and_annualized_risk"
)

CURVE_POINTS = OUT / "annualized_risk_exceedance_curve_points.csv"
RISK_SUMMARY = OUT / "annualized_risk_summary.csv"


def money_axis(value: float, _pos=None) -> str:
    if abs(value) >= 1e9:
        return f"${value / 1e9:.1f}B"
    if abs(value) >= 1e6:
        return f"${value / 1e6:.0f}M"
    if abs(value) >= 1e3:
        return f"${value / 1e3:.0f}k"
    return f"${value:.0f}"


def ead_label(value: float) -> str:
    if abs(value) >= 1e6:
        return f"EAD = ${value / 1e6:.2f}M/year"
    if abs(value) >= 1e3:
        return f"EAD = ${value / 1e3:.0f}k/year"
    return f"EAD = ${value:.0f}/year"


def title_for_hazard(hazard: str) -> str:
    if hazard == "flood":
        return "Flood Expected Annual Electricity System Cost"
    return "Tropical Cyclone Expected Annual Electricity System Cost"


def output_name_for_hazard(hazard: str) -> str:
    if hazard == "flood":
        return "flood_expected_annual_electricity_cost_ead.png"
    return "tc_expected_annual_electricity_cost_ead.png"


def plot_hazard_cost(hazard: str, curves: pd.DataFrame, risk: pd.DataFrame) -> None:
    metric = "expected_annual_incremental_system_cost_usd_per_year"
    curve = curves[
        curves["hazard"].eq(hazard)
        & curves["metric"].eq(metric)
        & curves["point_type"].isin(["available_return_period", "rare_tail_constant_at_largest_available_rp"])
    ].copy()
    curve = curve.sort_values("annual_exceedance_probability")
    hazard_risk = risk[risk["hazard"].eq(hazard)].iloc[0]
    ead = float(hazard_risk[metric])

    color = "#1f77b4" if hazard == "flood" else "#ff7f0e"
    fill = "#cfe8f3" if hazard == "flood" else "#fde0c5"

    fig, ax = plt.subplots(figsize=(10.5, 6.3))
    ax.fill_between(
        curve["annual_exceedance_probability"],
        curve["metric_value"],
        color=fill,
        alpha=0.9,
        label=ead_label(ead),
    )
    ax.plot(
        curve["annual_exceedance_probability"],
        curve["metric_value"],
        color=color,
        marker="o",
        linewidth=2.3,
        markersize=5.5,
    )

    available = curve[curve["point_type"].eq("available_return_period")]
    for _, row in available.iterrows():
        ax.annotate(
            f"RP{int(row['return_period'])}",
            (row["annual_exceedance_probability"], row["metric_value"]),
            textcoords="offset points",
            xytext=(0, 8),
            ha="center",
            fontsize=8,
            fontweight="bold",
            color="#333333",
        )

    ax.set_title(title_for_hazard(hazard), fontsize=15, fontweight="bold", pad=12)
    ax.set_xlabel("Annual exceedance probability (AEP)", fontsize=11)
    ax.set_ylabel("Incremental electricity system cost (USD)", fontsize=11)
    ax.yaxis.set_major_formatter(FuncFormatter(money_axis))
    ax.grid(True, alpha=0.22)
    ax.legend(loc="upper right", frameon=True)
    ax.set_xlim(-0.003, 0.105)
    ax.set_ylim(bottom=0)
    fig.text(
        0.5,
        0.025,
        "Area under curve calculated with trapezoidal integration. AEP = 1 / return period; available return periods shown.",
        ha="center",
        fontsize=9,
        color="#555555",
    )
    fig.tight_layout(rect=(0, 0.055, 1, 1))
    fig.savefig(OUT / output_name_for_hazard(hazard), dpi=260)
    plt.close(fig)


def plot_dashboard(curves: pd.DataFrame, risk: pd.DataFrame) -> None:
    metric = "expected_annual_incremental_system_cost_usd_per_year"
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.6), sharey=False)
    for ax, hazard in zip(axes, ["flood", "tropical_cyclone"]):
        curve = curves[
            curves["hazard"].eq(hazard)
            & curves["metric"].eq(metric)
            & curves["point_type"].isin(["available_return_period", "rare_tail_constant_at_largest_available_rp"])
        ].copy()
        curve = curve.sort_values("annual_exceedance_probability")
        ead = float(risk.loc[risk["hazard"].eq(hazard), metric].iloc[0])
        color = "#1f77b4" if hazard == "flood" else "#ff7f0e"
        fill = "#cfe8f3" if hazard == "flood" else "#fde0c5"
        ax.fill_between(curve["annual_exceedance_probability"], curve["metric_value"], color=fill, alpha=0.9)
        ax.plot(curve["annual_exceedance_probability"], curve["metric_value"], color=color, marker="o", linewidth=2)
        ax.set_title(title_for_hazard(hazard), fontsize=12, fontweight="bold")
        ax.set_xlabel("AEP")
        ax.set_ylabel("Incremental cost (USD)")
        ax.yaxis.set_major_formatter(FuncFormatter(money_axis))
        ax.grid(True, alpha=0.22)
        ax.text(
            0.98,
            0.92,
            ead_label(ead),
            transform=ax.transAxes,
            ha="right",
            bbox={"facecolor": "white", "alpha": 0.85, "edgecolor": "#dddddd"},
            fontsize=10,
        )
        ax.set_xlim(-0.003, 0.105)
        ax.set_ylim(bottom=0)
    fig.suptitle("Expected Annual Incremental Electricity System Cost", fontsize=15, fontweight="bold")
    fig.tight_layout()
    fig.savefig(OUT / "annualized_risk_summary_dashboard.png", dpi=260)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot expected annual electricity system cost graphs.")
    parser.add_argument("--risk-dir", type=Path, default=OUT)
    return parser.parse_args()


def main() -> None:
    global OUT
    args = parse_args()
    OUT = args.risk_dir
    curves = pd.read_csv(OUT / "annualized_risk_exceedance_curve_points.csv")
    risk = pd.read_csv(OUT / "annualized_risk_summary.csv")
    for hazard in ["flood", "tropical_cyclone"]:
        plot_hazard_cost(hazard, curves, risk)
    plot_dashboard(curves, risk)
    print("Saved updated annualized-risk PNGs:")
    print(OUT / "flood_expected_annual_electricity_cost_ead.png")
    print(OUT / "tc_expected_annual_electricity_cost_ead.png")
    print(OUT / "annualized_risk_summary_dashboard.png")


if __name__ == "__main__":
    main()
