"""
Return-period exposure plots for Florida electricity nodes.

Creates bar charts in the same style as the earlier node exposure figure:
counts on the y-axis, return period on the x-axis, and labels showing both
exposed node counts and percentages of all nodes.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


PROJECT_DIR = Path(r"C:\oxford_tc_project")
EXPOSURE_DIR = PROJECT_DIR / "data" / "Exposure"
OUTPUT_DIR = EXPOSURE_DIR / "node_return_period_exposure_graphs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

FLOOD_SUMMARY = EXPOSURE_DIR / "nodes_flood_exposure_by_return_period.csv"
TC_SUMMARY = EXPOSURE_DIR / "nodes_tc_exposure_by_return_period.csv"


def style_axis(ax: plt.Axes) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", color="#dddddd", linewidth=0.7, alpha=0.8)
    ax.set_axisbelow(True)


def annotate_bars(
    ax: plt.Axes,
    x_values: list[str],
    counts: pd.Series,
    percents: pd.Series,
) -> None:
    ymax = max(float(counts.max()), 1.0)
    ax.set_ylim(0, ymax * 1.17)
    for idx, (count, percent) in enumerate(zip(counts, percents)):
        ax.text(
            idx,
            count + ymax * 0.025,
            f"{int(count):,}\n({percent:.1f}%)",
            ha="center",
            va="bottom",
            fontsize=9,
        )


def plot_single_return_period_chart(
    df: pd.DataFrame,
    count_col: str,
    percent_col: str,
    title: str,
    xlabel: str,
    ylabel: str,
    output_name: str,
    color: str = "#2b7bbb",
) -> Path:
    df = df.sort_values("return_period").copy()
    x_labels = df["return_period"].astype(int).astype(str).tolist()

    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.bar(x_labels, df[count_col], color=color, edgecolor="white", linewidth=1.0)
    annotate_bars(ax, x_labels, df[count_col], df[percent_col])
    ax.set_title(title, fontsize=13, pad=12)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    style_axis(ax)
    fig.tight_layout()

    path = OUTPUT_DIR / output_name
    fig.savefig(path, dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Wrote {path}")
    return path


def plot_combined_return_period_chart(flood: pd.DataFrame, tc: pd.DataFrame) -> Path:
    flood = flood.sort_values("return_period").copy()
    tc = tc.sort_values("return_period").copy()

    fig, axes = plt.subplots(1, 2, figsize=(15, 5.5))
    fig.suptitle(
        "Florida Electricity Node Exposure by Return Period",
        fontsize=16,
        fontweight="bold",
    )

    ax = axes[0]
    flood_x = flood["return_period"].astype(int).astype(str).tolist()
    ax.bar(
        flood_x,
        flood["exposed_node_count_gt_0_5m"],
        color="#2b7bbb",
        edgecolor="white",
        linewidth=1.0,
    )
    annotate_bars(
        ax,
        flood_x,
        flood["exposed_node_count_gt_0_5m"],
        flood["exposed_node_percent_gt_0_5m"],
    )
    ax.set_title("JRC flood exposure")
    ax.set_xlabel("Flood return period (years)")
    ax.set_ylabel("Flood-exposed electricity nodes (>0.5 m)")
    style_axis(ax)

    ax = axes[1]
    tc_x = tc["return_period"].astype(int).astype(str).tolist()
    ax.bar(
        tc_x,
        tc["exposed_node_count_ge_25ms"],
        color="#d95f02",
        edgecolor="white",
        linewidth=1.0,
    )
    annotate_bars(
        ax,
        tc_x,
        tc["exposed_node_count_ge_25ms"],
        tc["exposed_node_percent_ge_25ms"],
    )
    ax.set_title("STORM tropical cyclone wind exposure")
    ax.set_xlabel("TC wind return period (years)")
    ax.set_ylabel("TC-exposed electricity nodes (>=25 m/s)")
    style_axis(ax)

    fig.tight_layout(rect=[0, 0, 1, 0.92])
    path = OUTPUT_DIR / "nodes_flood_and_storm_tc_exposure_by_return_period.png"
    fig.savefig(path, dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Wrote {path}")
    return path


def main() -> None:
    print(f"Reading {FLOOD_SUMMARY}")
    flood = pd.read_csv(FLOOD_SUMMARY)
    print(f"Reading {TC_SUMMARY}")
    tc = pd.read_csv(TC_SUMMARY)

    outputs = [
        plot_single_return_period_chart(
            flood,
            count_col="exposed_node_count_gt_0_5m",
            percent_col="exposed_node_percent_gt_0_5m",
            title="Electricity Node Flood Exposure by Return Period",
            xlabel="Flood return period (years)",
            ylabel="Flood-exposed electricity nodes (>0.5 m)",
            output_name="nodes_jrc_flood_exposure_by_return_period.png",
            color="#2b7bbb",
        ),
        plot_single_return_period_chart(
            tc,
            count_col="exposed_node_count_ge_25ms",
            percent_col="exposed_node_percent_ge_25ms",
            title="Electricity Node STORM TC Wind Exposure by Return Period",
            xlabel="TC wind return period (years)",
            ylabel="TC-exposed electricity nodes (>=25 m/s)",
            output_name="nodes_storm_tc_exposure_by_return_period.png",
            color="#d95f02",
        ),
        plot_combined_return_period_chart(flood, tc),
    ]

    print("\nCreated node return-period exposure figures:")
    for path in outputs:
        print(f"  - {path}")


if __name__ == "__main__":
    main()
