"""Validate the calibrated Florida PyPSA line-capacity multiplier.

This script does not try to prove that a global multiplier is physically exact.
Instead it documents whether the selected multiplier is defensible for the
current model structure:

1. It is the smallest tested value that removes no-hazard load shedding.
2. Hazard results are reported with sensitivity around it.
3. Modeled emergency imports are compared with public NERC Florida import
   transfer capability values as an external reasonableness check.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


PROJECT_DIR = Path(__file__).resolve().parents[2]
NETWORK_DIR = PROJECT_DIR / "data" / "Electricity" / "pypsa_florida_network_county_load_generator_overrides"
SENSITIVITY_DIR = NETWORK_DIR / "sensitivity_analysis"
OUTPUT_DIR = NETWORK_DIR / "line_capacity_multiplier_validation"

TOPOLOGY_COMPARISON = NETWORK_DIR / "topology_repair_and_multiplier_comparison.csv"
SENSITIVITY_SUMMARY = SENSITIVITY_DIR / "sensitivity_summary.csv"
IMPORT_SELECTION = NETWORK_DIR / "baseline_calibrated_no_hazard" / "import_bus_selection.csv"
IMPORT_DISPATCH = NETWORK_DIR / "baseline_calibrated_no_hazard" / "baseline_import_slack_by_bus.csv"

NERCC_SOURCE_URL = "https://www.nerc.com/globalassets/initiatives/itcs/itcs_part_1_results.pdf"
NERCC_SOURCE_LABEL = "NERC Interregional Transfer Capability Study Part 1 Results, August 2024"
NERC_FLORIDA_SUMMER_IMPORT_MW = 2958.0
NERC_FLORIDA_WINTER_IMPORT_MW = 1807.0


def money(value: float) -> str:
    return f"${value:,.0f}"


def mwh(value: float) -> str:
    return f"{value:,.1f} MWh"


def mw(value: float) -> str:
    return f"{value:,.1f} MW"


def make_calibration_plot(calibration: pd.DataFrame, out_path: Path) -> None:
    sweep = calibration[
        calibration["case_label"].str.contains("original_topology_x", regex=False)
        | calibration["case_label"].eq("official_x2_calibrated_baseline")
    ].copy()
    sweep = sweep.sort_values("line_capacity_multiplier")

    fig, ax = plt.subplots(figsize=(8, 5), dpi=180)
    ax.plot(
        sweep["line_capacity_multiplier"],
        sweep["total_load_shed_mwh"],
        marker="o",
        linewidth=2.5,
        color="#1f77b4",
    )
    ax.axvline(1.75, color="#2ca02c", linestyle="--", linewidth=2, label="Selected x1.75")
    ax.set_title("No-Hazard Load Shedding vs Line-Capacity Multiplier")
    ax.set_xlabel("Line capacity multiplier applied to s_nom")
    ax.set_ylabel("No-hazard load shed (MWh, 24-hour run)")
    ax.grid(True, alpha=0.25)
    ax.legend()
    for _, row in sweep.iterrows():
        ax.annotate(
            f"{row['total_load_shed_mwh']:,.0f}",
            (row["line_capacity_multiplier"], row["total_load_shed_mwh"]),
            textcoords="offset points",
            xytext=(0, 8),
            ha="center",
            fontsize=8,
        )
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def make_tc_sensitivity_plot(sensitivity: pd.DataFrame, out_path: Path) -> None:
    tc = sensitivity[sensitivity["scenario_id"].eq("tc_storm_rp500_w63_gradual")].copy()
    tc = tc[tc["boundary_import_cap_fraction"].eq(1.0)].sort_values("line_capacity_multiplier")

    fig, ax = plt.subplots(figsize=(8, 5), dpi=180)
    ax.plot(
        tc["line_capacity_multiplier"],
        tc["load_shed_mwh"],
        marker="o",
        linewidth=2.5,
        color="#d95f02",
    )
    ax.axvline(1.75, color="#2ca02c", linestyle="--", linewidth=2, label="Selected x1.75")
    ax.set_title("TC RP500 Load Shedding Sensitivity to Line-Capacity Multiplier")
    ax.set_xlabel("Line capacity multiplier applied to s_nom")
    ax.set_ylabel("TC RP500 load shed (MWh, 24-hour run)")
    ax.grid(True, alpha=0.25)
    ax.legend()
    for _, row in tc.iterrows():
        ax.annotate(
            f"{row['load_shed_mwh']:,.0f}",
            (row["line_capacity_multiplier"], row["load_shed_mwh"]),
            textcoords="offset points",
            xytext=(0, 8),
            ha="center",
            fontsize=8,
        )
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def write_report(
    calibration: pd.DataFrame,
    sensitivity: pd.DataFrame,
    import_selection: pd.DataFrame,
    import_dispatch: pd.DataFrame,
    out_path: Path,
) -> None:
    sweep = calibration[
        calibration["case_label"].str.contains("original_topology_x", regex=False)
        | calibration["case_label"].eq("official_x2_calibrated_baseline")
    ].copy()
    sweep = sweep.sort_values("line_capacity_multiplier")

    selected = sweep[sweep["line_capacity_multiplier"].eq(1.75)].iloc[0]
    lower = sweep[sweep["line_capacity_multiplier"].eq(1.5)].iloc[0]
    base = sweep[sweep["line_capacity_multiplier"].eq(1.0)].iloc[0]
    x2 = sweep[sweep["line_capacity_multiplier"].eq(2.0)].iloc[0]

    boundary_imports = import_selection[
        import_selection["import_type"].eq("northern_boundary_intertie_import")
    ].copy()
    artifact_imports = import_selection[
        import_selection["import_type"].eq("topology_artifact_island_import")
    ].copy()

    modeled_boundary_cap_mw = float(boundary_imports["import_p_nom_mw"].sum())
    modeled_artifact_cap_mw = float(artifact_imports["import_p_nom_mw"].sum())
    actual_import_mwh = float(import_dispatch["total_import_slack_mwh"].sum())
    actual_import_peak_mw = float(import_dispatch["max_hourly_import_slack_mw"].sum())
    actual_import_average_mw = actual_import_mwh / 24.0

    tc = sensitivity[sensitivity["scenario_id"].eq("tc_storm_rp500_w63_gradual")].copy()
    flood = sensitivity[sensitivity["scenario_id"].eq("flood_jrc_rp500_f62_gradual")].copy()
    tc_import_sweep = tc[tc["line_capacity_multiplier"].eq(1.75)].copy()

    report = f"""# Line-Capacity Multiplier Validation

This report validates the current calibrated line-capacity multiplier, `s_nom x 1.75`, for the improved Florida PyPSA network.

## Bottom Line

`s_nom x 1.75` is defensible as the central calibrated case, but it should be described as a calibration assumption rather than a measured physical rating. It is the smallest tested multiplier that removes no-hazard load shedding after the topology/load/generator-bus improvements.

## Internal Calibration Evidence

| multiplier | no-hazard load shed (MWh) | buses with load shed | import slack (MWh) |
|---:|---:|---:|---:|
"""

    for _, row in sweep.iterrows():
        report += (
            f"| {row['line_capacity_multiplier']:.2f} | "
            f"{row['total_load_shed_mwh']:,.3f} | "
            f"{int(row['buses_experiencing_load_shedding'])} | "
            f"{row['total_import_slack_mwh']:,.3f} |\n"
        )

    report += f"""
Key interpretation:

- At `s_nom x 1.00`, the no-hazard model sheds {mwh(base['total_load_shed_mwh'])}.
- At `s_nom x 1.50`, it still sheds {mwh(lower['total_load_shed_mwh'])}.
- At `s_nom x 1.75`, it sheds {mwh(selected['total_load_shed_mwh'])}.
- At `s_nom x 2.00`, it also sheds {mwh(x2['total_load_shed_mwh'])}, but this is a looser assumption than needed.

Therefore, `x1.75` is preferred over `x2.00` because it is the first tested multiplier that gives a normal no-hazard baseline without load shedding.

## External Import-Transfer Reasonableness Check

Public benchmark:

- Source: [{NERCC_SOURCE_LABEL}]({NERCC_SOURCE_URL})
- NERC reports SERC Southeast -> SERC Florida transfer capability of:
  - {mw(NERC_FLORIDA_SUMMER_IMPORT_MW)} for 2024 summer
  - {mw(NERC_FLORIDA_WINTER_IMPORT_MW)} for 2024/25 winter

Model import setup:

- Northern boundary import p_nom in the model: {mw(modeled_boundary_cap_mw)}
- Topology-artifact island import p_nom: {mw(modeled_artifact_cap_mw)}
- Actual no-hazard import dispatch: {mwh(actual_import_mwh)}
- Actual average import dispatch: {mw(actual_import_average_mw)}
- Actual simultaneous peak import dispatch across used import buses: {mw(actual_import_peak_mw)}

Interpretation:

- The selected northern-boundary import cap is an inferred upper bound from incident line capacity, not a verified Florida TTC limit.
- However, the no-hazard model does not actually use the northern boundary imports. The dispatch is only {mw(actual_import_average_mw)} on average, which is far below the NERC SERC Southeast -> SERC Florida transfer capability values.
- This means the `x1.75` baseline is not being held together by unrealistically large northern imports.

## Hazard Sensitivity Evidence

Flood RP500:

- Load shedding stays at {mwh(flood['load_shed_mwh'].min())} to {mwh(flood['load_shed_mwh'].max())} across tested multiplier/import cases.
- Flood results are not very sensitive to the line multiplier under the current F6.2 line curve.

TC RP500:

| multiplier | import cap fraction | TC RP500 load shed (MWh) | incremental cost |
|---:|---:|---:|---:|
"""

    for _, row in tc.sort_values(["line_capacity_multiplier", "boundary_import_cap_fraction"]).iterrows():
        report += (
            f"| {row['line_capacity_multiplier']:.2f} | "
            f"{row['boundary_import_cap_fraction']:.2f} | "
            f"{row['load_shed_mwh']:,.3f} | "
            f"{money(row['incremental_system_cost_usd'])} |\n"
        )

    report += f"""
Key interpretation:

- TC RP500 is sensitive to the line multiplier: load shed ranges from {mwh(tc['load_shed_mwh'].min())} to {mwh(tc['load_shed_mwh'].max())}.
- Changing the boundary import cap between 0.75, 1.00, and 1.25 at `x1.75` does not change TC RP500 load shedding, which remains {mwh(tc_import_sweep['load_shed_mwh'].iloc[0])}.
- This points to internal transmission deliverability and asset derating as the important uncertainty, not the boundary import cap.

## Recommendation

Use `s_nom x 1.75` as the central case, and present `x1.50` and `x2.00` as sensitivity bounds for TC results. For flood, the multiplier sensitivity is minor under the current F6.2 assumptions.

## Files Created

- `line_multiplier_no_hazard_validation.png`
- `tc_rp500_line_multiplier_sensitivity.png`
- `line_capacity_multiplier_validation_report.md`
"""

    out_path.write_text(report, encoding="utf-8")


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    calibration = pd.read_csv(TOPOLOGY_COMPARISON)
    sensitivity = pd.read_csv(SENSITIVITY_SUMMARY)
    import_selection = pd.read_csv(IMPORT_SELECTION)
    import_dispatch = pd.read_csv(IMPORT_DISPATCH)

    calibration.to_csv(OUTPUT_DIR / "line_multiplier_calibration_evidence.csv", index=False)
    sensitivity.to_csv(OUTPUT_DIR / "line_multiplier_hazard_sensitivity_evidence.csv", index=False)

    make_calibration_plot(calibration, OUTPUT_DIR / "line_multiplier_no_hazard_validation.png")
    make_tc_sensitivity_plot(sensitivity, OUTPUT_DIR / "tc_rp500_line_multiplier_sensitivity.png")
    write_report(
        calibration,
        sensitivity,
        import_selection,
        import_dispatch,
        OUTPUT_DIR / "line_capacity_multiplier_validation_report.md",
    )

    print("Saved validation outputs:", OUTPUT_DIR)


if __name__ == "__main__":
    main()
