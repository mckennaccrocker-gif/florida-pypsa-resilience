"""
Populate Florida PyPSA generator marginal costs from documented components.

Method:
    marginal_cost_usd_per_mwh =
        vom_usd_per_mwh + fuel_cost_usd_per_mmbtu * heat_rate_mmbtu_per_mwh

Heat rates are calculated from EIA-923 Schedule 2/3/4/5 Page 1:
    Elec Fuel Consumption MMBtu / Net Generation MWh

Fuel costs are calculated from EIA-923 Page 5 fuel receipts:
    FUEL_COST is reported in cents/MMBtu, converted to USD/MMBtu.
    Monthly rows are weighted by received fuel energy where possible.

VOM:
    EIA-923 does not provide variable O&M. The lookup table is built from the
    PyPSA-USA input file `.codex_tmp/pypsa-usa/workflow/repo_data/plants/
    eia860_ads_merged.csv`, using median ADS `ads_vom_cost` by carrier/fuel.
    This follows the same PyPSA-USA convention that existing plant marginal
    costs are VOM plus fuel cost times heat rate.

No arbitrary placeholder marginal costs are assigned. Missing plant-specific
values are filled only with fuel-type averages derived from EIA-923 values and
marked as `fuel_type_average_fallback`.
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_DIR = Path(r"C:\oxford_tc_project")
ELECTRICITY_DIR = PROJECT_DIR / "data" / "Electricity"
PYPSA_DIR = ELECTRICITY_DIR / "pypsa_florida_network"
EIA923_DIR = ELECTRICITY_DIR / "f923_2025er"
OUTPUT_DIR = ELECTRICITY_DIR / "generator_marginal_costs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

EIA923_MAIN = EIA923_DIR / "EIA923_Schedules_2_3_4_5_M_12_2025_Early_Release_30JUN2026.xlsx"
GENERATORS_CSV = PYPSA_DIR / "generators.csv"
PLANTS_CSV = ELECTRICITY_DIR / "florida_powerplants_osm_matched_plus_unmatched.csv"
PYPSA_USA_ADS_PLANTS = (
    PROJECT_DIR
    / ".codex_tmp"
    / "pypsa-usa"
    / "workflow"
    / "repo_data"
    / "plants"
    / "eia860_ads_merged.csv"
)

OUTPUT_GENERATORS = PYPSA_DIR / "generators_with_component_marginal_costs.csv"
OUTPUT_SUMMARY = OUTPUT_DIR / "generator_marginal_cost_summary_by_fuel.csv"
OUTPUT_COMPONENTS = OUTPUT_DIR / "generator_marginal_cost_components.csv"
OUTPUT_VOM = OUTPUT_DIR / "vom_lookup_from_pypsa_usa_ads.csv"
OUTPUT_PLANT_COSTS = OUTPUT_DIR / "eia923_florida_plant_fuel_cost_components.csv"

EIA_WARNING_COLUMN_PREFIX = "Early release data"
MONTHS = [
    "January",
    "February",
    "March",
    "April",
    "May",
    "June",
    "July",
    "August",
    "September",
    "October",
    "November",
    "December",
]

CARRIER_TO_FUEL_TYPE = {
    "Gas": "gas",
    "Coal": "coal",
    "Oil": "oil",
    "Biomass": "biomass",
    "Waste": "waste",
    "Nuclear": "nuclear",
    "Hydro": "hydro",
    "Solar": "solar",
    "Wind": "wind",
    "Storage": "battery",
    "Cogeneration": "gas",
    "Unknown": "other",
}

EIA_FUEL_TO_FUEL_TYPE = {
    "NG": "gas",
    "BFG": "gas",
    "OG": "gas",
    "PG": "gas",
    "BIT": "coal",
    "SUB": "coal",
    "LIG": "coal",
    "WC": "coal",
    "RC": "coal",
    "DFO": "oil",
    "RFO": "oil",
    "JF": "oil",
    "KER": "oil",
    "PC": "oil",
    "WDS": "biomass",
    "WDL": "biomass",
    "OBS": "biomass",
    "OBL": "biomass",
    "AB": "biomass",
    "MSW": "waste",
    "LFG": "waste",
    "NUC": "nuclear",
    "WAT": "hydro",
    "SUN": "solar",
    "WND": "wind",
}


def normalize_name(value: object) -> str:
    if pd.isna(value):
        return ""
    text = str(value).lower()
    text = re.sub(r"[^a-z0-9]+", " ", text)
    text = re.sub(r"\b(power|plant|facility|generating|generation|station|energy|solar|center)\b", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def clean_eia_columns(df: pd.DataFrame) -> pd.DataFrame:
    warning_cols = [col for col in df.columns if str(col).startswith(EIA_WARNING_COLUMN_PREFIX)]
    df = df.drop(columns=warning_cols, errors="ignore")
    df = df.loc[df["Plant Id"].astype(str).ne("Plant Id")].copy()
    df.columns = [str(col).replace("\n", " ").strip() for col in df.columns]
    return df


def numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series.replace(".", np.nan), errors="coerce")


def extract_eia_plant_id(gppd_id: object) -> float:
    if pd.isna(gppd_id):
        return np.nan
    match = re.search(r"USA0*(\d+)", str(gppd_id))
    return float(match.group(1)) if match else np.nan


def load_generators_with_plant_ids() -> pd.DataFrame:
    generators = pd.read_csv(GENERATORS_CSV)
    plants = pd.read_csv(PLANTS_CSV)
    plants["source_name_norm"] = plants["name"].map(normalize_name)
    plants["eia_plant_id"] = plants["gppd_idnr"].map(extract_eia_plant_id)
    plants_lookup = (
        plants.sort_values(["source_name_norm", "match_score"], ascending=[True, False])
        .drop_duplicates("source_name_norm")
        [[
            "source_name_norm",
            "gppd_idnr",
            "eia_plant_id",
            "name",
            "gppd_name",
            "primary_fuel",
        ]]
    )

    generators["source_name_norm"] = generators["source_name"].map(normalize_name)
    generators = generators.merge(plants_lookup, on="source_name_norm", how="left", suffixes=("", "_plant"))
    generators["fuel_type"] = generators["carrier"].map(CARRIER_TO_FUEL_TYPE).fillna("other")
    generators["generator_original_marginal_cost"] = generators["marginal_cost"]
    generators["marginal_cost"] = np.nan
    return generators


def load_eia923_heat_rates() -> pd.DataFrame:
    page1 = pd.read_excel(EIA923_MAIN, sheet_name="Page 1 Generation and Fuel Data", header=6)
    page1 = clean_eia_columns(page1)
    page1 = page1[page1["Plant State"].eq("FL")].copy()

    page1["plant_id_eia"] = numeric(page1["Plant Id"])
    page1["fuel_type"] = page1["Reported Fuel Type Code"].map(EIA_FUEL_TO_FUEL_TYPE).fillna(
        page1["MER Fuel Type Code"].map(EIA_FUEL_TO_FUEL_TYPE)
    )
    page1["elec_mmbtu"] = numeric(page1["Elec Fuel Consumption MMBtu"])
    page1["net_generation_mwh"] = numeric(page1["Net Generation (Megawatthours)"])

    valid = page1[
        page1["plant_id_eia"].notna()
        & page1["fuel_type"].notna()
        & page1["elec_mmbtu"].gt(0)
        & page1["net_generation_mwh"].gt(0)
    ].copy()

    grouped = (
        valid.groupby(["plant_id_eia", "Plant Name", "fuel_type"], as_index=False)
        .agg(
            elec_mmbtu=("elec_mmbtu", "sum"),
            net_generation_mwh=("net_generation_mwh", "sum"),
        )
    )
    grouped["heat_rate_mmbtu_per_mwh"] = grouped["elec_mmbtu"] / grouped["net_generation_mwh"]
    return grouped


def load_eia923_fuel_costs() -> pd.DataFrame:
    receipts = pd.read_excel(EIA923_MAIN, sheet_name="Page 5 Fuel Receipts and Costs", header=5)
    receipts = clean_eia_columns(receipts)
    receipts = receipts[receipts["Plant State"].eq("FL")].copy()
    receipts["plant_id_eia"] = numeric(receipts["Plant Id"])
    receipts["fuel_type"] = receipts["ENERGY_SOURCE"].map(EIA_FUEL_TO_FUEL_TYPE)
    receipts["fuel_cost_usd_per_mmbtu"] = numeric(receipts["FUEL_COST"]) / 100.0
    receipts["quantity"] = numeric(receipts["QUANTITY"])
    receipts["average_heat_content_mmbtu_per_unit"] = numeric(receipts["Average Heat Content"])
    receipts["received_mmbtu"] = receipts["quantity"] * receipts["average_heat_content_mmbtu_per_unit"]

    valid = receipts[
        receipts["plant_id_eia"].notna()
        & receipts["fuel_type"].notna()
        & receipts["fuel_cost_usd_per_mmbtu"].gt(0)
    ].copy()
    valid["weight"] = valid["received_mmbtu"].where(valid["received_mmbtu"].gt(0), valid["quantity"])
    valid["weight"] = valid["weight"].where(valid["weight"].gt(0), 1.0)

    grouped = (
        valid.groupby(["plant_id_eia", "Plant Name", "fuel_type"], as_index=False)
        .apply(
            lambda g: pd.Series(
                {
                    "fuel_cost_usd_per_mmbtu": np.average(
                        g["fuel_cost_usd_per_mmbtu"],
                        weights=g["weight"],
                    ),
                    "fuel_receipt_rows": len(g),
                    "received_mmbtu": g["received_mmbtu"].sum(),
                }
            ),
            include_groups=False,
        )
        .reset_index(drop=True)
    )
    return grouped


def build_vom_lookup() -> pd.DataFrame:
    if not PYPSA_USA_ADS_PLANTS.exists():
        raise FileNotFoundError(
            "PyPSA-USA ADS plant table not found. Clone PyPSA-USA before running this script: "
            f"{PYPSA_USA_ADS_PLANTS}"
        )
    cols = ["carrier", "fuel_type", "ads_vom_cost"]
    ads = pd.read_csv(PYPSA_USA_ADS_PLANTS, usecols=cols)
    ads["ads_vom_cost"] = numeric(ads["ads_vom_cost"])
    ads = ads[ads["ads_vom_cost"].notna()].copy()

    by_fuel = (
        ads.groupby("fuel_type", as_index=False)
        .agg(
            vom_usd_per_mwh=("ads_vom_cost", "median"),
            vom_rows=("ads_vom_cost", "size"),
            vom_source_carriers=("carrier", lambda s: "; ".join(sorted(set(map(str, s))))),
        )
    )
    by_fuel["vom_source"] = (
        "PyPSA-USA eia860_ads_merged.csv median ads_vom_cost by fuel_type; "
        "ADS-derived VOM because EIA-923 does not report VOM"
    )
    by_fuel.to_csv(OUTPUT_VOM, index=False)
    return by_fuel[["fuel_type", "vom_usd_per_mwh", "vom_source"]]


def build_component_table() -> tuple[pd.DataFrame, pd.DataFrame]:
    heat_rates = load_eia923_heat_rates()
    fuel_costs = load_eia923_fuel_costs()
    plant_components = heat_rates.merge(
        fuel_costs,
        on=["plant_id_eia", "fuel_type"],
        how="outer",
        suffixes=("_heat_rate", "_fuel_cost"),
    )
    plant_components["Plant Name"] = plant_components["Plant Name_heat_rate"].combine_first(
        plant_components["Plant Name_fuel_cost"]
    )
    plant_components = plant_components.drop(columns=["Plant Name_heat_rate", "Plant Name_fuel_cost"])
    plant_components.to_csv(OUTPUT_PLANT_COSTS, index=False)

    fuel_averages = (
        plant_components[
            plant_components["heat_rate_mmbtu_per_mwh"].notna()
            & plant_components["fuel_cost_usd_per_mmbtu"].notna()
        ]
        .groupby("fuel_type", as_index=False)
        .agg(
            heat_rate_mmbtu_per_mwh=("heat_rate_mmbtu_per_mwh", "mean"),
            fuel_cost_usd_per_mmbtu=("fuel_cost_usd_per_mmbtu", "mean"),
            fallback_plant_fuel_rows=("plant_id_eia", "size"),
        )
    )
    return plant_components, fuel_averages


def populate_costs() -> pd.DataFrame:
    generators = load_generators_with_plant_ids()
    plant_components, fuel_averages = build_component_table()
    vom_lookup = build_vom_lookup()

    components = generators.merge(
        plant_components[[
            "plant_id_eia",
            "fuel_type",
            "heat_rate_mmbtu_per_mwh",
            "fuel_cost_usd_per_mmbtu",
            "Plant Name",
        ]],
        left_on=["eia_plant_id", "fuel_type"],
        right_on=["plant_id_eia", "fuel_type"],
        how="left",
    )
    components = components.merge(vom_lookup, on="fuel_type", how="left")
    components["marginal_cost_source"] = np.where(
        components["heat_rate_mmbtu_per_mwh"].notna()
        & components["fuel_cost_usd_per_mmbtu"].notna()
        & components["vom_usd_per_mwh"].notna(),
        "plant_specific_eia923_heat_rate_and_fuel_cost_plus_pypsa_usa_ads_vom",
        pd.NA,
    )

    needs_fallback = components["marginal_cost_source"].isna()
    fallback = fuel_averages.rename(
        columns={
            "heat_rate_mmbtu_per_mwh": "fallback_heat_rate_mmbtu_per_mwh",
            "fuel_cost_usd_per_mmbtu": "fallback_fuel_cost_usd_per_mmbtu",
        }
    )
    components = components.merge(fallback, on="fuel_type", how="left")
    fallback_mask = (
        needs_fallback
        & components["fallback_heat_rate_mmbtu_per_mwh"].notna()
        & components["fallback_fuel_cost_usd_per_mmbtu"].notna()
        & components["vom_usd_per_mwh"].notna()
    )
    components.loc[fallback_mask, "heat_rate_mmbtu_per_mwh"] = components.loc[
        fallback_mask,
        "fallback_heat_rate_mmbtu_per_mwh",
    ]
    components.loc[fallback_mask, "fuel_cost_usd_per_mmbtu"] = components.loc[
        fallback_mask,
        "fallback_fuel_cost_usd_per_mmbtu",
    ]
    components.loc[fallback_mask, "marginal_cost_source"] = "fuel_type_average_fallback"

    calc_mask = (
        components["heat_rate_mmbtu_per_mwh"].notna()
        & components["fuel_cost_usd_per_mmbtu"].notna()
        & components["vom_usd_per_mwh"].notna()
    )
    components.loc[calc_mask, "marginal_cost"] = (
        components.loc[calc_mask, "vom_usd_per_mwh"]
        + components.loc[calc_mask, "fuel_cost_usd_per_mmbtu"]
        * components.loc[calc_mask, "heat_rate_mmbtu_per_mwh"]
    )

    components.loc[components["marginal_cost_source"].isna(), "marginal_cost_source"] = "missing_component_inputs"

    output_columns = [
        "name",
        "bus",
        "carrier",
        "p_nom",
        "marginal_cost",
        "heat_rate_mmbtu_per_mwh",
        "fuel_cost_usd_per_mmbtu",
        "vom_usd_per_mwh",
        "marginal_cost_source",
        "source_name",
        "matched_source",
        "location_confidence",
        "gppd_idnr",
        "eia_plant_id",
        "fuel_type",
        "vom_source",
        "generator_original_marginal_cost",
    ]
    output = components[output_columns].copy()
    output.to_csv(OUTPUT_GENERATORS, index=False)
    components.to_csv(OUTPUT_COMPONENTS, index=False)
    return output


def print_diagnostics(output: pd.DataFrame) -> None:
    source_counts = output["marginal_cost_source"].value_counts(dropna=False)
    plant_specific = int(source_counts.get("plant_specific_eia923_heat_rate_and_fuel_cost_plus_pypsa_usa_ads_vom", 0))
    fallback = int(source_counts.get("fuel_type_average_fallback", 0))
    missing = int(source_counts.get("missing_component_inputs", 0))

    print("\nMarginal cost component sources:")
    print("- Heat rate: EIA-923 Page 1 Elec Fuel Consumption MMBtu / Net Generation MWh")
    print("- Fuel cost: EIA-923 Page 5 FUEL_COST, converted from cents/MMBtu to USD/MMBtu")
    print("- VOM: PyPSA-USA repo_data/plants/eia860_ads_merged.csv median ADS ads_vom_cost by fuel_type")
    print("- Formula: marginal_cost = VOM + fuel_cost * heat_rate")

    print("\nDiagnostics:")
    print(f"Generators with plant-specific marginal cost: {plant_specific}")
    print(f"Generators using fuel-type average fallback: {fallback}")
    print(f"Generators still missing marginal cost: {missing}")
    print(f"Output generator table: {OUTPUT_GENERATORS}")
    print(f"Component audit table: {OUTPUT_COMPONENTS}")
    print(f"VOM lookup table: {OUTPUT_VOM}")

    summary = (
        output.groupby("carrier", dropna=False)
        .agg(
            generators=("name", "size"),
            plant_specific=(
                "marginal_cost_source",
                lambda s: int((s == "plant_specific_eia923_heat_rate_and_fuel_cost_plus_pypsa_usa_ads_vom").sum()),
            ),
            fuel_type_average_fallback=(
                "marginal_cost_source",
                lambda s: int((s == "fuel_type_average_fallback").sum()),
            ),
            still_missing=("marginal_cost_source", lambda s: int((s == "missing_component_inputs").sum())),
            average_marginal_cost=("marginal_cost", "mean"),
            min_marginal_cost=("marginal_cost", "min"),
            max_marginal_cost=("marginal_cost", "max"),
            average_heat_rate=("heat_rate_mmbtu_per_mwh", "mean"),
            average_fuel_cost=("fuel_cost_usd_per_mmbtu", "mean"),
            average_vom=("vom_usd_per_mwh", "mean"),
        )
        .reset_index()
        .sort_values("generators", ascending=False)
    )
    summary.to_csv(OUTPUT_SUMMARY, index=False)

    print("\nSummary by fuel type/carrier:")
    print(summary.to_string(index=False, float_format=lambda x: f"{x:,.3f}"))
    print(f"\nSummary CSV: {OUTPUT_SUMMARY}")


def main() -> None:
    print("Building Florida generator marginal costs from EIA-923 components.")
    print(f"EIA-923 workbook: {EIA923_MAIN}")
    output = populate_costs()
    print_diagnostics(output)


if __name__ == "__main__":
    main()
