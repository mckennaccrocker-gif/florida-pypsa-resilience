"""Create thesis-ready summary tables for the flood adaptation workflow.

This script only reads completed outputs. It does not run PyPSA, alter model
results, or recalculate adaptation scenarios.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(r"C:\oxford_tc_project")
ANALYSIS_ROOT = PROJECT_ROOT / "data" / "Electricity" / "flood_adaptation_analysis"
FULL_SUITE = ANALYSIS_ROOT / "rp100_top5_full_suite"
PILOT = ANALYSIS_ROOT / "rp100_top5_pilot"
COST_BENEFIT = ANALYSIS_ROOT / "rp100_top5_cost_benefit"
OUTPUT = ANALYSIS_ROOT / "final_summary_tables"


def read_csv(path: Path) -> pd.DataFrame:
    """Read a required CSV file."""
    if not path.exists():
        raise FileNotFoundError(f"Required file not found: {path}")
    return pd.read_csv(path)


def money(value: float) -> str:
    """Format a monetary value for thesis summary tables."""
    if abs(value) >= 1e9:
        return f"${value / 1e9:,.2f}B"
    if abs(value) >= 1e6:
        return f"${value / 1e6:,.1f}M"
    return f"${value:,.0f}"


def number(value: float, decimals: int = 1) -> str:
    """Format a number with thousands separators."""
    return f"{value:,.{decimals}f}"


def export_table(df: pd.DataFrame, stem: str) -> list[Path]:
    """Export a table as CSV and XLSX."""
    csv_path = OUTPUT / f"{stem}.csv"
    xlsx_path = OUTPUT / f"{stem}.xlsx"
    df.to_csv(csv_path, index=False)
    df.to_excel(xlsx_path, index=False)
    return [csv_path, xlsx_path]


def build_table1(
    selected: pd.DataFrame,
    design: pd.DataFrame,
    comparison: pd.DataFrame,
    risk: pd.Series,
) -> pd.DataFrame:
    """Build Table 1: flood adaptation package summary."""
    return_periods = comparison["return_period_years"].astype(int).tolist()
    selected_ids = "; ".join(selected["asset_id"].astype(str).tolist())
    strategy = (
        "Protect the same five modeled assets using residual-depth flood protection; "
        "damage is recalculated after subtracting the design depth."
    )
    design_standard = "RP100 flood depth plus 0.30 m freeboard"
    freeboard = design["freeboard_m"].dropna().iloc[0]
    rows = [
        ("Number of selected assets", len(selected)),
        ("Selected asset IDs", selected_ids),
        ("Adaptation strategy", strategy),
        ("Design standard", design_standard),
        ("Freeboard", f"{freeboard:.2f} m"),
        ("Number of return periods analysed", len(return_periods)),
        ("Return periods included", ", ".join(f"RP{rp}" for rp in return_periods)),
        ("Baseline EENS (MWh/year)", number(float(risk["baseline_eens_mwh_per_year"]))),
        ("Adapted EENS (MWh/year)", number(float(risk["adapted_eens_mwh_per_year"]))),
        ("Avoided EENS (MWh/year)", number(float(risk["avoided_eens_mwh_per_year"]))),
        ("Percentage risk reduction", f"{float(risk['risk_reduction_percent']):.2f}%"),
    ]
    return pd.DataFrame(rows, columns=["Metric", "Value"])


def build_table2(comparison: pd.DataFrame) -> pd.DataFrame:
    """Build Table 2: scenario-by-scenario comparison."""
    out = comparison.sort_values("return_period_years")[
        [
            "return_period_years",
            "baseline_load_shed_mwh",
            "adapted_load_shed_mwh",
            "avoided_load_shed_mwh",
            "load_shed_reduction_percent",
            "baseline_system_cost",
            "adapted_system_cost",
            "avoided_system_cost",
        ]
    ].copy()
    out.columns = [
        "Return period",
        "Baseline load shed (MWh)",
        "Adapted load shed (MWh)",
        "Avoided load shed (MWh)",
        "Percent reduction",
        "Baseline system cost",
        "Adapted system cost",
        "Avoided system cost",
    ]
    out["Return period"] = out["Return period"].astype(int).map(lambda x: f"RP{x}")
    for col in ["Baseline load shed (MWh)", "Adapted load shed (MWh)", "Avoided load shed (MWh)"]:
        out[col] = out[col].map(lambda x: round(float(x), 1))
    out["Percent reduction"] = out["Percent reduction"].map(lambda x: round(float(x), 2))
    for col in ["Baseline system cost", "Adapted system cost", "Avoided system cost"]:
        out[col] = out[col].map(lambda x: round(float(x), 0))
    return out


def build_table3(economics: pd.DataFrame) -> pd.DataFrame:
    """Build Table 3: economic summary."""
    def get(cost_case: str, voll_case: str, column: str) -> float:
        row = economics[
            economics["package_cost_scenario"].eq(cost_case) & economics["voll_scenario"].eq(voll_case)
        ].iloc[0]
        return float(row[column])

    central = economics[
        economics["package_cost_scenario"].eq("central") & economics["voll_scenario"].eq("central")
    ].iloc[0]
    low_high_bcr = get("high", "low", "benefit_cost_ratio")
    summary = {
        "Row": "Full-suite result",
        "Low package cost": money(get("low", "central", "capital_cost_usd")),
        "Central package cost": money(get("central", "central", "capital_cost_usd")),
        "High package cost": money(get("high", "central", "capital_cost_usd")),
        "Low annual benefit": money(get("central", "low", "annual_avoided_outage_value_usd")),
        "Central annual benefit": money(get("central", "central", "annual_avoided_outage_value_usd")),
        "High annual benefit": money(get("central", "high", "annual_avoided_outage_value_usd")),
        "Central BCR": f"{float(central['benefit_cost_ratio']):.2f}",
        "Central NPV": money(float(central["net_present_value_usd"])),
        "Simple payback": f"{float(central['simple_payback_years']):.1f} years",
        "Break-even package cost": money(float(central["break_even_capital_cost_usd"])),
        "Low VOLL / High Cost BCR": f"{low_high_bcr:.2f}",
        "Interpretation": "",
    }
    interpretation = {key: "" for key in summary}
    interpretation["Row"] = "Key Interpretation"
    interpretation["Interpretation"] = (
        "Under the model assumptions, the adaptation package remained cost-effective "
        "across all tested sensitivity cases."
    )
    return pd.DataFrame([summary, interpretation])


def build_key_findings(
    selected: pd.DataFrame,
    design: pd.DataFrame,
    comparison: pd.DataFrame,
    risk: pd.Series,
    economics: pd.DataFrame,
) -> str:
    """Write a one-page Markdown summary for the adaptation chapter."""
    central = economics[
        economics["package_cost_scenario"].eq("central") & economics["voll_scenario"].eq("central")
    ].iloc[0]
    low_high = economics[
        economics["package_cost_scenario"].eq("high") & economics["voll_scenario"].eq("low")
    ].iloc[0]
    max_effect = comparison.loc[comparison["avoided_load_shed_mwh"].idxmax()]
    asset_lines = "\n".join(
        f"- `{row.asset_id}` ({row.asset_type}; {row.asset_name})"
        for row in selected[["asset_id", "asset_type", "asset_name"]].itertuples(index=False)
    )
    return f"""# Flood Adaptation Key Findings

## Objective

The objective of this workflow was to evaluate a conceptual five-asset flood adaptation package for the Florida PyPSA transmission model. The analysis estimates how much expected annual load shedding could be avoided if the selected assets were protected against RP100 flood depth plus freeboard, and then translates the modeled avoided load shedding into a preliminary planning-level economic assessment.

## Adaptation Approach

The adaptation package protected the same five modeled assets used in the RP100 top-five pilot. For each selected asset, the protection target was defined as the modeled RP100 flood depth plus 0.30 m of freeboard. Damage was recalculated using residual flood depth, meaning that the selected assets were not assumed to become permanently invulnerable and larger flood events could still exceed the design level.

## Five Selected Assets

{asset_lines}

## Main Engineering Assumptions

The analysis used conceptual adaptation categories rather than detailed engineering designs. Generator assets were interpreted as protection of modeled generating facilities and associated electrical equipment. Line assets were interpreted as protection of the modeled exposed line representation and associated endpoint or structure-level equipment where relevant. The cost assumptions remain planning-level ranges and should not be interpreted as site-specific construction estimates.

## Main Flood-Risk Findings

Across the full return-period suite ({', '.join('RP' + str(int(rp)) for rp in comparison['return_period_years'])}), baseline expected annual load shedding was {number(float(risk['baseline_eens_mwh_per_year']))} MWh/year and adapted expected annual load shedding was {number(float(risk['adapted_eens_mwh_per_year']))} MWh/year. The package avoided {number(float(risk['avoided_eens_mwh_per_year']))} MWh/year, corresponding to a {float(risk['risk_reduction_percent']):.2f}% reduction in modeled annual flood load-shedding risk. The largest scenario-level reduction occurred at RP{int(max_effect['return_period_years'])}, where the package avoided {number(float(max_effect['avoided_load_shed_mwh']))} MWh.

## Main Economic Findings

Using the full-suite avoided EENS and the existing illustrative Value of Lost Load assumptions, the central annual avoided outage value was {money(float(central['annual_avoided_outage_value_usd']))} per year. Under the central cost and central VOLL case, the benefit-cost ratio was {float(central['benefit_cost_ratio']):.2f}, the net present value was {money(float(central['net_present_value_usd']))}, and the simple payback period was {float(central['simple_payback_years']):.1f} years. Even under the low VOLL and high cost case, the BCR was {float(low_high['benefit_cost_ratio']):.2f}, suggesting that the package remains potentially cost-effective within the tested sensitivity range.

## Limitations

These findings should be interpreted cautiously. The PyPSA model represents transmission-scale operational consequences and does not capture all distribution-level outages. The VOLL assumptions are illustrative societal outage-value assumptions, not verified Florida-specific utility revenue. The adaptation costs are conceptual planning ranges, and avoided physical repair costs are not included. The result also depends on the modeled flood exposure data, the F6.3 damage relationship, and the annualized-risk integration method.

## Suggested Future Work

Future work should replace the conceptual cost ranges with site-specific engineering estimates, test additional adaptation packages, and compare model-estimated load shedding with observed outage datasets where possible. The adaptation results would also be strengthened by evaluating additional hazard probabilities, considering distribution-network impacts, and testing whether the selected assets remain important under alternative demand, import, and restoration assumptions.
"""


def main() -> None:
    """Create all final summary tables."""
    OUTPUT.mkdir(parents=True, exist_ok=True)
    selected = read_csv(FULL_SUITE / "selected_top5_assets.csv")
    design = read_csv(FULL_SUITE / "rp100_protection_design.csv")
    comparison = read_csv(FULL_SUITE / "full_suite_scenario_comparison.csv")
    risk = read_csv(FULL_SUITE / "full_suite_annualized_risk.csv").iloc[0]
    economics = read_csv(FULL_SUITE / "economics" / "full_suite_cost_benefit_results.csv")

    created: list[Path] = []
    created.extend(export_table(build_table1(selected, design, comparison, risk), "Table1_Flood_Adaptation_Summary"))
    created.extend(export_table(build_table2(comparison), "Table2_Return_Period_Comparison"))
    created.extend(export_table(build_table3(economics), "Table3_Cost_Benefit_Summary"))

    findings_path = OUTPUT / "Flood_Adaptation_Key_Findings.md"
    findings_path.write_text(build_key_findings(selected, design, comparison, risk, economics), encoding="utf-8")
    created.append(findings_path)

    print("Created final summary table files:")
    for path in created:
        print(path)


if __name__ == "__main__":
    main()
