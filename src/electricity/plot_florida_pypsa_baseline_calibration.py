"""Plot Florida PyPSA baseline calibration diagnostics from saved CSV outputs."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


PROJECT_DIR = Path(r"C:\oxford_tc_project")
CALIBRATION_DIR = PROJECT_DIR / "data" / "Electricity" / "pypsa_florida_network" / "baseline_calibration"
PLOT_DIR = CALIBRATION_DIR / "plots"


CASE_LABELS = {
    "case_1_import_slack_each_load_island": "Import in each\nload island",
    "case_2_load_only_largest_component": "Largest-component\nload only",
    "case_3_island_imports_line_limits_x2": "Island imports +\n2x line limits",
}


def read_case_summary() -> pd.DataFrame:
    summary = pd.read_csv(CALIBRATION_DIR / "calibration_case_summary.csv")
    summary["case_label"] = summary["case"].map(CASE_LABELS).fillna(summary["case"])
    return summary


def plot_case_summary(summary: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), constrained_layout=True)
    x = range(len(summary))

    axes[0].bar(x, summary["total_load_shed_mwh"], color="#c44e52")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(summary["case_label"], rotation=0)
    axes[0].set_ylabel("Load shedding (MWh)")
    axes[0].set_title("No-hazard baseline load shedding")
    for i, value in enumerate(summary["total_load_shed_mwh"]):
        axes[0].text(i, value, f"{value:,.0f}", ha="center", va="bottom", fontsize=9)

    axes[1].bar(x, summary["total_import_slack_mwh"], color="#4c78a8")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(summary["case_label"], rotation=0)
    axes[1].set_ylabel("Emergency imports (MWh)")
    axes[1].set_title("Import slack used")
    for i, value in enumerate(summary["total_import_slack_mwh"]):
        axes[1].text(i, value, f"{value:,.0f}", ha="center", va="bottom", fontsize=9)

    fig.savefig(PLOT_DIR / "calibration_load_shed_and_imports.png", dpi=220)
    plt.close(fig)


def plot_case_dispatch(case_dir: Path, output_name: str) -> None:
    dispatch = pd.read_csv(case_dir / "baseline_generation_by_carrier_summary.csv")
    dispatch = dispatch[dispatch["carrier"] != "load_shedding"].sort_values("generation_mwh")

    fig, ax = plt.subplots(figsize=(10, 6))
    colors = ["#4c78a8" if c != "import_slack" else "#f58518" for c in dispatch["carrier"]]
    ax.barh(dispatch["carrier"], dispatch["generation_mwh"], color=colors)
    ax.set_xlabel("Generation / import energy (MWh)")
    ax.set_title(case_dir.name.replace("_", " ").title())
    fig.tight_layout()
    fig.savefig(PLOT_DIR / output_name, dpi=220)
    plt.close(fig)


def plot_congested_corridors(case_dir: Path, output_name: str) -> None:
    corridors = pd.read_csv(case_dir / "top_congested_corridors.csv").head(20)
    corridors = corridors.sort_values("max_loading_pu")
    labels = corridors["line"].astype(str)

    fig, ax = plt.subplots(figsize=(10, 7))
    ax.barh(labels, corridors["max_loading_pu"], color="#59a14f")
    ax.axvline(1.0, color="#333333", linestyle="--", linewidth=1)
    ax.set_xlabel("Maximum line loading (p.u.)")
    ax.set_title("Top congested corridors")
    fig.tight_layout()
    fig.savefig(PLOT_DIR / output_name, dpi=220)
    plt.close(fig)


def plot_main_island_load_shedding(case_dir: Path, output_name: str) -> None:
    path = case_dir / "top_main_island_load_shedding_buses.csv"
    buses = pd.read_csv(path)
    buses = buses[buses["total_load_shed_mwh"] > 1e-6].head(20)
    if buses.empty:
        return
    buses = buses.sort_values("total_load_shed_mwh")

    fig, ax = plt.subplots(figsize=(10, 7))
    ax.barh(buses["bus"], buses["total_load_shed_mwh"], color="#c44e52")
    ax.set_xlabel("Load shedding (MWh)")
    ax.set_title("Main-island load-shedding buses")
    fig.tight_layout()
    fig.savefig(PLOT_DIR / output_name, dpi=220)
    plt.close(fig)


def main() -> None:
    PLOT_DIR.mkdir(parents=True, exist_ok=True)
    summary = read_case_summary()
    plot_case_summary(summary)

    case1 = CALIBRATION_DIR / "case_1_import_slack_each_load_island"
    case3 = CALIBRATION_DIR / "case_3_island_imports_line_limits_x2"
    plot_case_dispatch(case1, "case1_dispatch_by_carrier.png")
    plot_case_dispatch(case3, "case3_dispatch_by_carrier.png")
    plot_congested_corridors(case1, "case1_top_congested_corridors.png")
    plot_congested_corridors(case3, "case3_top_congested_corridors.png")
    plot_main_island_load_shedding(case1, "case1_main_island_load_shedding_buses.png")

    print("Saved baseline calibration plots to:", PLOT_DIR)


if __name__ == "__main__":
    main()
