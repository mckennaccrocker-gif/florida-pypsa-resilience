"""Compare official-current network against voltage-layer transformer variant."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


PROJECT_DIR = Path(__file__).resolve().parents[2]
OFFICIAL = PROJECT_DIR / "data" / "Electricity" / "pypsa_florida_network_official_current"
TRANSFORMER = PROJECT_DIR / "data" / "Electricity" / "pypsa_florida_network_voltage_transformers"
OUTPUT_DIR = PROJECT_DIR / "data" / "Electricity" / "voltage_transformer_network_comparison"


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    official = pd.read_csv(OFFICIAL / "summer_peak_multiplier_sweep" / "summer_peak_multiplier_sweep_summary.csv")
    transformer = pd.read_csv(TRANSFORMER / "summer_peak_multiplier_sweep_v2" / "summer_peak_multiplier_sweep_summary.csv")
    official["network_variant"] = "official_current_no_transformers"
    transformer["network_variant"] = "voltage_layer_transformers"
    combined = pd.concat([official, transformer], ignore_index=True)
    combined.to_csv(OUTPUT_DIR / "summer_peak_official_vs_transformer_network.csv", index=False)

    comparison = official[
        ["line_capacity_multiplier", "load_shed_mwh", "import_slack_mwh", "boundary_import_slack_mwh", "artifact_import_slack_mwh"]
    ].rename(
        columns={
            "load_shed_mwh": "official_load_shed_mwh",
            "import_slack_mwh": "official_import_slack_mwh",
            "boundary_import_slack_mwh": "official_boundary_import_mwh",
            "artifact_import_slack_mwh": "official_artifact_import_mwh",
        }
    ).merge(
        transformer[
            [
                "line_capacity_multiplier",
                "load_shed_mwh",
                "import_slack_mwh",
                "boundary_import_slack_mwh",
                "artifact_import_slack_mwh",
            ]
        ].rename(
            columns={
                "load_shed_mwh": "transformer_load_shed_mwh",
                "import_slack_mwh": "transformer_import_slack_mwh",
                "boundary_import_slack_mwh": "transformer_boundary_import_mwh",
                "artifact_import_slack_mwh": "transformer_artifact_import_mwh",
            }
        ),
        on="line_capacity_multiplier",
    )
    comparison["load_shed_reduction_mwh"] = (
        comparison["official_load_shed_mwh"] - comparison["transformer_load_shed_mwh"]
    )
    comparison["load_shed_reduction_percent"] = (
        comparison["load_shed_reduction_mwh"] / comparison["official_load_shed_mwh"].replace(0, pd.NA) * 100
    )
    comparison.to_csv(OUTPUT_DIR / "summer_peak_transformer_improvement_table.csv", index=False)

    fig, ax = plt.subplots(figsize=(8.8, 5.2), dpi=180)
    ax.plot(
        official["line_capacity_multiplier"],
        official["load_shed_mwh"],
        marker="o",
        linewidth=2.4,
        label="Official current",
        color="#4c78a8",
    )
    ax.plot(
        transformer["line_capacity_multiplier"],
        transformer["load_shed_mwh"],
        marker="s",
        linewidth=2.4,
        label="Voltage-layer transformers",
        color="#54a24b",
    )
    ax.set_title("Summer Peak Load Shedding: Official vs Transformer Network")
    ax.set_xlabel("Line capacity multiplier applied to s_nom")
    ax.set_ylabel("Load shed (MWh)")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "summer_peak_load_shed_official_vs_transformers.png")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8.8, 5.2), dpi=180)
    ax.plot(
        official["line_capacity_multiplier"],
        official["boundary_import_slack_mwh"],
        marker="o",
        linewidth=2.2,
        label="Official boundary import",
        color="#4c78a8",
    )
    ax.plot(
        transformer["line_capacity_multiplier"],
        transformer["boundary_import_slack_mwh"],
        marker="s",
        linewidth=2.2,
        label="Transformer boundary import",
        color="#54a24b",
    )
    ax.set_title("Summer Peak Boundary Import: Official vs Transformer Network")
    ax.set_xlabel("Line capacity multiplier applied to s_nom")
    ax.set_ylabel("Boundary import slack (MWh)")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "summer_peak_boundary_import_official_vs_transformers.png")
    plt.close(fig)

    transformer_validation = pd.read_csv(
        TRANSFORMER
        / "baseline_x1p75_transformer_validation_v2"
        / "boundary_imports_plus_artifact_islands"
        / "baseline_summary.csv"
    ).iloc[0]

    report = "# Voltage-Layer Transformer Network Comparison\n\n"
    report += "## What Changed\n\n"
    report += (
        "The transformer variant adds explicit voltage-layer auxiliary buses and PyPSA transformers "
        "where line voltage differs from bus voltage. It created 608 transformers and reassigned "
        "1,341 line endpoints.\n\n"
    )
    report += "## January Calibration Check\n\n"
    report += (
        f"At `s_nom x1.75`, the transformer network has "
        f"{transformer_validation['total_load_shed_mwh']:,.3f} MWh load shedding for the January "
        "no-hazard calibration window with boundary + artifact-island imports.\n\n"
    )
    report += "## Summer Peak Improvement\n\n"
    report += "| multiplier | official shed MWh | transformer shed MWh | reduction MWh | reduction % |\n"
    report += "| ---: | ---: | ---: | ---: | ---: |\n"
    for row in comparison.itertuples(index=False):
        pct = "" if pd.isna(row.load_shed_reduction_percent) else f"{row.load_shed_reduction_percent:,.2f}%"
        report += (
            f"| {row.line_capacity_multiplier:.2f} | {row.official_load_shed_mwh:,.3f} | "
            f"{row.transformer_load_shed_mwh:,.3f} | {row.load_shed_reduction_mwh:,.3f} | {pct} |\n"
        )
    report += "\n## Recommendation\n\n"
    report += (
        "The transformer variant is a strong candidate for the next official network because it "
        "preserves the January no-hazard calibration and substantially reduces summer peak shedding "
        "without increasing the global line multiplier. Before rerunning hazards, run at least one "
        "seasonal validation on the transformer network and decide whether the higher boundary import "
        "use under summer peak is acceptable or requires further import-cap calibration.\n"
    )
    (OUTPUT_DIR / "voltage_transformer_network_comparison_report.md").write_text(report, encoding="utf-8")

    print("Saved comparison:", OUTPUT_DIR)
    print(comparison.to_string(index=False))


if __name__ == "__main__":
    main()
