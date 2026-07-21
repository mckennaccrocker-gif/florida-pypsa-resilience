"""Compare summer peak multiplier sweeps for cleanup variants."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


PROJECT_DIR = Path(__file__).resolve().parents[2]
OUTPUT_DIR = PROJECT_DIR / "data" / "Electricity" / "summer_peak_cleanup_variant_comparison"

VARIANTS = {
    "original_county_generator_overrides": PROJECT_DIR
    / "data"
    / "Electricity"
    / "pypsa_florida_network_county_load_generator_overrides"
    / "summer_peak_multiplier_sweep"
    / "summer_peak_multiplier_sweep_summary.csv",
    "local_load_adjusted": PROJECT_DIR
    / "data"
    / "Electricity"
    / "pypsa_florida_network_county_load_local_adjusted"
    / "summer_peak_multiplier_sweep"
    / "summer_peak_multiplier_sweep_summary.csv",
    "bus311_load_only_cleanup": PROJECT_DIR
    / "data"
    / "Electricity"
    / "pypsa_florida_network_bus311_load_only_cleanup"
    / "summer_peak_multiplier_sweep"
    / "summer_peak_multiplier_sweep_summary.csv",
    "targeted_line_deactivation_cleanup": PROJECT_DIR
    / "data"
    / "Electricity"
    / "pypsa_florida_network_targeted_topology_cleanup"
    / "summer_peak_multiplier_sweep"
    / "summer_peak_multiplier_sweep_summary.csv",
}


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    frames = []
    for variant, path in VARIANTS.items():
        df = pd.read_csv(path)
        df["variant"] = variant
        frames.append(df)
    combined = pd.concat(frames, ignore_index=True)
    combined.to_csv(OUTPUT_DIR / "summer_peak_cleanup_variant_comparison.csv", index=False)

    pivot = combined.pivot_table(
        index="line_capacity_multiplier",
        columns="variant",
        values="load_shed_mwh",
    ).reset_index()
    pivot.to_csv(OUTPUT_DIR / "summer_peak_cleanup_load_shed_pivot.csv", index=False)

    fig, ax = plt.subplots(figsize=(9, 5.4), dpi=180)
    for variant, group in combined.groupby("variant"):
        style = {
            "original_county_generator_overrides": ("#4c78a8", "o"),
            "local_load_adjusted": ("#f58518", "o"),
            "bus311_load_only_cleanup": ("#54a24b", "s"),
            "targeted_line_deactivation_cleanup": ("#e45756", "^"),
        }.get(variant, ("#777777", "o"))
        ax.plot(
            group["line_capacity_multiplier"],
            group["load_shed_mwh"],
            marker=style[1],
            color=style[0],
            linewidth=2,
            label=variant.replace("_", " "),
        )
    ax.set_title("Summer Peak Load Shedding by Cleanup Variant")
    ax.set_xlabel("Line capacity multiplier applied to s_nom")
    ax.set_ylabel("Load shed (MWh)")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "summer_peak_cleanup_variant_load_shed.png")
    plt.close(fig)

    local = combined[combined["variant"].eq("local_load_adjusted")][
        ["line_capacity_multiplier", "load_shed_mwh"]
    ].rename(columns={"load_shed_mwh": "local_load_adjusted_mwh"})
    bus311 = combined[combined["variant"].eq("bus311_load_only_cleanup")][
        ["line_capacity_multiplier", "load_shed_mwh"]
    ].rename(columns={"load_shed_mwh": "bus311_load_only_mwh"})
    compare = local.merge(bus311, on="line_capacity_multiplier")
    compare["absolute_reduction_mwh"] = (
        compare["local_load_adjusted_mwh"] - compare["bus311_load_only_mwh"]
    )
    compare["percent_reduction"] = (
        compare["absolute_reduction_mwh"] / compare["local_load_adjusted_mwh"] * 100.0
    )
    compare.to_csv(OUTPUT_DIR / "local_adjusted_vs_bus311_load_only.csv", index=False)

    report = "# Summer Peak Cleanup Variant Comparison\n\n"
    report += "## Result\n\n"
    report += (
        "The conservative `bus311_load_only_cleanup` improves the current local-adjusted "
        "network slightly at every multiplier. The targeted low-voltage line deactivation "
        "cleanup makes the summer peak shedding worse and should not be adopted as the "
        "official network treatment.\n\n"
    )
    report += "## Local-Adjusted vs Bus-311-Load-Only\n\n"
    report += "| multiplier | local adjusted MWh | bus311 load-only MWh | reduction MWh | reduction % |\n"
    report += "| ---: | ---: | ---: | ---: | ---: |\n"
    for row in compare.itertuples(index=False):
        report += (
            f"| {row.line_capacity_multiplier:.2f} | {row.local_load_adjusted_mwh:,.3f} | "
            f"{row.bus311_load_only_mwh:,.3f} | {row.absolute_reduction_mwh:,.3f} | "
            f"{row.percent_reduction:,.2f}% |\n"
        )
    report += "\n## Recommendation\n\n"
    report += (
        "Adopt the bus_311 load-only cleanup as the next conservative model update. "
        "Do not adopt the line-deactivation cleanup. The line-deactivation experiment "
        "showed that those low-voltage lines are numerically stressed, but simply "
        "removing them reduces deliverability rather than fixing the model. A future "
        "topology improvement should represent voltage layers/transformers more "
        "carefully instead of deleting those paths.\n"
    )
    (OUTPUT_DIR / "summer_peak_cleanup_variant_comparison_report.md").write_text(report, encoding="utf-8")

    print("Saved comparison:", OUTPUT_DIR)
    print(compare.to_string(index=False))


if __name__ == "__main__":
    main()
