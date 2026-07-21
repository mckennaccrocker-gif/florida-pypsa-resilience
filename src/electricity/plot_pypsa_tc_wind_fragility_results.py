"""
Create publication-style figures for Florida PyPSA TC wind results.

Figures combine deterministic threshold scenarios and the voltage-specific
fragility expected-capacity scenario.
"""

from __future__ import annotations

from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PROJECT_DIR = Path(r"C:\oxford_tc_project")
PYPSA_DIR = PROJECT_DIR / "data" / "Electricity" / "pypsa_florida_network"
SCENARIO_DIR = PYPSA_DIR / "hazard_scenarios"
FRAGILITY_DIR = SCENARIO_DIR / "tc_wind_voltage_fragility"
COMPARISON_DIR = SCENARIO_DIR / "tc_wind_threshold_comparison"
OUTPUT_DIR = SCENARIO_DIR / "tc_wind_visual_summary"
LINES_GPKG = PROJECT_DIR / "data" / "Electricity" / "florida_lines_with_s_nom.gpkg"


COLOR_RED = "#b3261e"
COLOR_BLUE = "#355c7d"
COLOR_GREEN = "#2f7d59"
COLOR_GOLD = "#d59f0f"
COLOR_GRAY = "#5f6368"


def load_threshold_summary() -> pd.DataFrame:
    path = COMPARISON_DIR / "tc_wind_threshold_comparison.csv"
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_csv(path).sort_values("threshold")


def load_fragility_summary() -> tuple[pd.Series, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    incremental = pd.read_csv(FRAGILITY_DIR / "incremental_impact_summary.csv").iloc[0]
    curve_summary = pd.read_csv(FRAGILITY_DIR / "fragility_curve_summary.csv")
    line_adjustments = pd.read_csv(FRAGILITY_DIR / "line_fragility_capacity_adjustments.csv")
    dispatch = pd.read_csv(FRAGILITY_DIR / "dispatch_by_snapshot.csv")
    dispatch["snapshot"] = pd.to_datetime(dispatch["snapshot"])
    return incremental, curve_summary, line_adjustments, dispatch


def savefig(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path, dpi=220, bbox_inches="tight")
    plt.close()
    print("Saved:", path)


def plot_threshold_vs_fragility(threshold: pd.DataFrame, fragility: pd.Series) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))

    ax = axes[0, 0]
    ax.plot(threshold["threshold"], threshold["incremental_load_shed_mwh"], marker="o", color=COLOR_RED, linewidth=2.5)
    ax.axhline(fragility["incremental_load_shed_mwh"], color=COLOR_GREEN, linestyle="--", linewidth=2, label="Voltage fragility")
    ax.set_title("Incremental unserved energy")
    ax.set_ylabel("MWh above baseline")
    ax.legend(frameon=False)
    ax.grid(alpha=0.25)

    ax = axes[0, 1]
    ax.plot(threshold["threshold"], threshold["outaged_lines"], marker="o", color=COLOR_BLUE, linewidth=2.5)
    ax.set_title("Deterministic line outages")
    ax.set_ylabel("Outaged PyPSA lines")
    ax.grid(alpha=0.25)

    ax = axes[1, 0]
    ax.bar(threshold["threshold"].astype(str), threshold["outaged_line_capacity_mva"], color=COLOR_GOLD)
    ax.axhline(fragility["expected_capacity_loss_mva"], color=COLOR_GREEN, linestyle="--", linewidth=2, label="Expected capacity loss")
    ax.set_title("Transmission capacity affected")
    ax.set_xlabel("TC wind threshold (m/s)")
    ax.set_ylabel("MVA")
    ax.legend(frameon=False)
    ax.grid(axis="y", alpha=0.25)

    ax = axes[1, 1]
    ax.plot(threshold["threshold"], threshold["incremental_import_slack_mwh"], marker="o", color=COLOR_GRAY, linewidth=2.5)
    ax.axhline(fragility["incremental_import_slack_mwh"], color=COLOR_GREEN, linestyle="--", linewidth=2, label="Voltage fragility")
    ax.set_title("Incremental emergency imports")
    ax.set_xlabel("TC wind threshold (m/s)")
    ax.set_ylabel("MWh above baseline")
    ax.legend(frameon=False)
    ax.grid(alpha=0.25)

    fig.suptitle("Florida PyPSA TC Wind: Threshold Outages vs Voltage-Specific Fragility", fontsize=15, fontweight="bold")
    fig.tight_layout()
    savefig(OUTPUT_DIR / "01_threshold_vs_fragility_dashboard.png")


def plot_fragility_curve_breakdown(curve_summary: pd.DataFrame, line_adjustments: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    curve_summary = curve_summary.sort_values("expected_capacity_loss_mva", ascending=False)
    axes[0].bar(
        curve_summary["fragility_curve_id"],
        curve_summary["expected_capacity_loss_mva"],
        color=[COLOR_GREEN, COLOR_BLUE, COLOR_GOLD],
    )
    axes[0].set_title("Expected capacity loss by fragility curve")
    axes[0].set_ylabel("MVA")
    axes[0].grid(axis="y", alpha=0.25)

    bins = np.linspace(0, max(0.18, line_adjustments["failure_probability"].max()), 18)
    for curve_id, group in line_adjustments.groupby("fragility_curve_id"):
        axes[1].hist(
            group["failure_probability"],
            bins=bins,
            alpha=0.65,
            label=curve_id,
        )
    axes[1].set_title("Line failure-probability distribution")
    axes[1].set_xlabel("Failure probability")
    axes[1].set_ylabel("Number of lines")
    axes[1].legend(frameon=False)
    axes[1].grid(axis="y", alpha=0.25)

    fig.suptitle("Voltage-Specific Fragility Drivers", fontsize=15, fontweight="bold")
    fig.tight_layout()
    savefig(OUTPUT_DIR / "02_fragility_curve_breakdown.png")


def plot_hourly_dispatch(dispatch: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(12, 5))
    x = dispatch["snapshot"]
    ax.plot(x, dispatch["import_slack_mwh"], color=COLOR_BLUE, linewidth=2.3, label="Emergency imports")
    ax.plot(x, dispatch["load_shed_mwh"], color=COLOR_RED, linewidth=2.3, label="Load shed")
    ax.fill_between(x, dispatch["load_shed_mwh"], color=COLOR_RED, alpha=0.15)
    ax.set_title("Hourly emergency response under voltage-specific fragility")
    ax.set_ylabel("MWh per hour")
    ax.set_xlabel("Snapshot")
    ax.legend(frameon=False)
    ax.grid(alpha=0.25)
    fig.autofmt_xdate()
    savefig(OUTPUT_DIR / "03_fragility_hourly_dispatch.png")


def plot_fragility_map(line_adjustments: pd.DataFrame) -> None:
    lines = gpd.read_file(LINES_GPKG)
    lines = lines.merge(
        line_adjustments,
        left_on="florida_line_id",
        right_on="source_edge_id",
        how="inner",
    )
    if lines.empty:
        print("No matched lines for map; skipping.")
        return

    fig, axes = plt.subplots(1, 2, figsize=(14, 8))

    lines.plot(
        ax=axes[0],
        column="failure_probability",
        cmap="inferno_r",
        linewidth=0.8,
        legend=True,
        legend_kwds={"label": "Failure probability"},
    )
    axes[0].set_title("Voltage-specific failure probability")
    axes[0].set_axis_off()

    lines.plot(
        ax=axes[1],
        column="expected_capacity_loss_mva",
        cmap="magma_r",
        linewidth=0.8,
        legend=True,
        legend_kwds={"label": "Expected capacity loss (MVA)"},
    )
    axes[1].set_title("Expected capacity loss by line")
    axes[1].set_axis_off()

    fig.suptitle("Spatial Pattern of TC Wind Fragility Effects", fontsize=15, fontweight="bold")
    fig.tight_layout()
    savefig(OUTPUT_DIR / "04_fragility_spatial_map.png")


def write_key_metrics(threshold: pd.DataFrame, fragility: pd.Series, curve_summary: pd.DataFrame) -> None:
    metrics = {
        "fragility_incremental_load_shed_mwh": float(fragility["incremental_load_shed_mwh"]),
        "fragility_incremental_import_slack_mwh": float(fragility["incremental_import_slack_mwh"]),
        "fragility_expected_capacity_loss_mva": float(fragility["expected_capacity_loss_mva"]),
        "fragility_mean_failure_probability": float(fragility["mean_failure_probability"]),
        "fragility_max_failure_probability": float(fragility["max_failure_probability"]),
        "largest_threshold_incremental_load_shed_mwh": float(threshold["incremental_load_shed_mwh"].max()),
        "largest_threshold_outaged_lines": int(threshold["outaged_lines"].max()),
    }
    rows = [{"metric": key, "value": value} for key, value in metrics.items()]
    pd.DataFrame(rows).to_csv(OUTPUT_DIR / "key_visual_metrics.csv", index=False)
    curve_summary.to_csv(OUTPUT_DIR / "curve_capacity_loss_summary.csv", index=False)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    threshold = load_threshold_summary()
    fragility, curve_summary, line_adjustments, dispatch = load_fragility_summary()

    plot_threshold_vs_fragility(threshold, fragility)
    plot_fragility_curve_breakdown(curve_summary, line_adjustments)
    plot_hourly_dispatch(dispatch)
    plot_fragility_map(line_adjustments)
    write_key_metrics(threshold, fragility, curve_summary)

    print("\nVisual summary written to:", OUTPUT_DIR)


if __name__ == "__main__":
    main()
