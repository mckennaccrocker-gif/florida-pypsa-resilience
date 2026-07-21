"""
Finalize Florida PyPSA generator marginal costs.

This script starts from the component-based cost table and fills only remaining
missing values using documented technology assumptions. It does not overwrite
plant-specific EIA-923/PyPSA-USA component values or fuel-type average fallback
values.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_DIR = Path(r"C:\oxford_tc_project")
PYPSA_DIR = PROJECT_DIR / "data" / "Electricity" / "pypsa_florida_network"
INPUT_GENERATORS = PYPSA_DIR / "generators_with_component_marginal_costs.csv"
OUTPUT_GENERATORS = PYPSA_DIR / "generators_with_final_marginal_costs.csv"
SUMMARY_OUTPUT = (
    PROJECT_DIR
    / "data"
    / "Electricity"
    / "generator_marginal_costs"
    / "final_generator_marginal_cost_summary_by_fuel.csv"
)

PLANT_SPECIFIC_SOURCE_OLD = "plant_specific_eia923_heat_rate_and_fuel_cost_plus_pypsa_usa_ads_vom"
PLANT_SPECIFIC_SOURCE_NEW = "eia923_plant_specific"
FUEL_TYPE_FALLBACK_SOURCE = "fuel_type_average_fallback"
ZERO_SOURCE = "zero_variable_cost_renewable"
DOCUMENTED_SOURCE = "documented_assumption_fallback"
MISSING_SOURCE = "missing_component_inputs"

ZERO_COST_CARRIERS = {"Solar", "Wind", "Hydro", "Storage"}

# Documented non-EIA fallback assumptions used only when component data is absent.
# Solar/wind/hydro/storage are treated as zero dispatch-cost resources here.
# Nuclear uses the PyPSA-USA ADS median VOM observed for nuclear units because
# EIA-923 fuel cost was not available in the current early-release workbook.
# Biomass/waste use the PyPSA-USA ADS median VOM as a conservative non-fuel
# operating-cost fallback when EIA-923 fuel cost was unavailable.
DOCUMENTED_FALLBACK_COSTS_USD_PER_MWH = {
    "Nuclear": 5.633296,
    "Biomass": 2.027986,
    "Waste": 2.027986,
}


def load_generators() -> pd.DataFrame:
    if not INPUT_GENERATORS.exists():
        raise FileNotFoundError(
            f"Missing {INPUT_GENERATORS}. Run populate_florida_generator_marginal_costs.py first."
        )
    generators = pd.read_csv(INPUT_GENERATORS)
    generators["marginal_cost_source"] = generators["marginal_cost_source"].replace(
        {PLANT_SPECIFIC_SOURCE_OLD: PLANT_SPECIFIC_SOURCE_NEW}
    )
    return generators


def finalize_costs(generators: pd.DataFrame) -> pd.DataFrame:
    final = generators.copy()

    protected_sources = {PLANT_SPECIFIC_SOURCE_NEW, FUEL_TYPE_FALLBACK_SOURCE}
    protected = final["marginal_cost_source"].isin(protected_sources) & final["marginal_cost"].notna()

    missing = final["marginal_cost"].isna() & ~protected
    zero_mask = missing & final["carrier"].isin(ZERO_COST_CARRIERS)
    final.loc[zero_mask, "marginal_cost"] = 0.0
    final.loc[zero_mask, "marginal_cost_source"] = ZERO_SOURCE

    missing = final["marginal_cost"].isna() & ~protected
    for carrier, cost in DOCUMENTED_FALLBACK_COSTS_USD_PER_MWH.items():
        mask = missing & final["carrier"].eq(carrier)
        final.loc[mask, "marginal_cost"] = cost
        final.loc[mask, "marginal_cost_source"] = DOCUMENTED_SOURCE

    final.loc[final["marginal_cost"].isna(), "marginal_cost_source"] = MISSING_SOURCE

    # Guardrail: verify component-based rows were not changed.
    changed_protected = (
        protected
        & generators["marginal_cost"].notna()
        & ~np.isclose(final["marginal_cost"], generators["marginal_cost"], equal_nan=True)
    )
    if changed_protected.any():
        changed = final.loc[changed_protected, ["name", "carrier", "marginal_cost_source"]]
        raise RuntimeError(
            "Protected component-based marginal costs were changed unexpectedly:\n"
            + changed.head(10).to_string(index=False)
        )

    return final


def summarize(final: pd.DataFrame) -> pd.DataFrame:
    summary = (
        final.groupby("carrier", dropna=False)
        .agg(
            generators=("name", "size"),
            average_marginal_cost=("marginal_cost", "mean"),
            min_marginal_cost=("marginal_cost", "min"),
            max_marginal_cost=("marginal_cost", "max"),
            eia923_plant_specific=(
                "marginal_cost_source",
                lambda s: int((s == PLANT_SPECIFIC_SOURCE_NEW).sum()),
            ),
            fuel_type_average_fallback=(
                "marginal_cost_source",
                lambda s: int((s == FUEL_TYPE_FALLBACK_SOURCE).sum()),
            ),
            zero_variable_cost_renewable=(
                "marginal_cost_source",
                lambda s: int((s == ZERO_SOURCE).sum()),
            ),
            documented_assumption_fallback=(
                "marginal_cost_source",
                lambda s: int((s == DOCUMENTED_SOURCE).sum()),
            ),
            still_missing=("marginal_cost", lambda s: int(s.isna().sum())),
        )
        .reset_index()
        .sort_values("generators", ascending=False)
    )
    SUMMARY_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(SUMMARY_OUTPUT, index=False)
    return summary


def print_diagnostics(final: pd.DataFrame, summary: pd.DataFrame) -> None:
    print("\nFinal marginal cost assumptions:")
    print("- Preserved eia923_plant_specific component values.")
    print("- Preserved fuel_type_average_fallback component values.")
    print("- Solar, wind, hydro, storage: 0 USD/MWh dispatch-cost assumption.")
    print("- Nuclear, biomass, waste: PyPSA-USA ADS median VOM fallback where EIA components are missing.")

    print("\nFinal diagnostics:")
    print(f"Total generators: {len(final)}")
    print("Count by marginal_cost_source:")
    print(final["marginal_cost_source"].value_counts(dropna=False).to_string())
    print(f"Count still missing marginal_cost: {int(final['marginal_cost'].isna().sum())}")

    print("\nSummary by fuel type/carrier:")
    print(summary.to_string(index=False, float_format=lambda x: f"{x:,.3f}"))
    print(f"\nSaved final generators: {OUTPUT_GENERATORS}")
    print(f"Saved summary: {SUMMARY_OUTPUT}")


def main() -> None:
    generators = load_generators()
    final = finalize_costs(generators)
    final.to_csv(OUTPUT_GENERATORS, index=False)
    summary = summarize(final)
    print_diagnostics(final, summary)


if __name__ == "__main__":
    main()
