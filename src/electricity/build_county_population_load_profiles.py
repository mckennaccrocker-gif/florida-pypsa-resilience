"""
Create county-population-weighted PyPSA load profiles for Florida.

Inputs:
  - Existing PyPSA buses/loads/loads-p_set from pypsa_florida_network
  - Cleaned 2025 Florida county population extracted from the EDR PDF
  - Census TIGER county boundaries, downloaded if needed

Method:
  1. Sum the existing hourly Florida load to keep the same statewide demand.
  2. Assign each PyPSA bus to a Florida county.
  3. Allocate statewide load to counties by 2025 county population.
  4. Allocate each county's load to buses using a voltage/connectivity weight.

This writes a separate network folder and does not overwrite the original model.
"""

from __future__ import annotations

import shutil
from pathlib import Path
from urllib.error import URLError
from urllib.request import urlretrieve

import geopandas as gpd
import numpy as np
import pandas as pd


PROJECT_DIR = Path(r"C:\oxford_tc_project")
ELECTRICITY_DIR = PROJECT_DIR / "data" / "Electricity"
BOUNDARY_DIR = PROJECT_DIR / "data" / "Boundaries"
BASE_NETWORK_DIR = ELECTRICITY_DIR / "pypsa_florida_network"
OUTPUT_NETWORK_DIR = ELECTRICITY_DIR / "pypsa_florida_network_county_population_load"
COUNTY_POPULATION = ELECTRICITY_DIR / "florida_county_population_2025_clean.csv"
PROJECTED_CRS = "EPSG:3086"

COUNTY_ZIP = BOUNDARY_DIR / "tl_us_county.zip"
COUNTY_URLS = [
    "https://www2.census.gov/geo/tiger/TIGER2025/COUNTY/tl_2025_us_county.zip",
    "https://www2.census.gov/geo/tiger/TIGER2024/COUNTY/tl_2024_us_county.zip",
    "https://www2.census.gov/geo/tiger/TIGER2023/COUNTY/tl_2023_us_county.zip",
]


def copy_network_folder() -> None:
    OUTPUT_NETWORK_DIR.mkdir(parents=True, exist_ok=True)
    for path in BASE_NETWORK_DIR.glob("*.csv"):
        shutil.copy2(path, OUTPUT_NETWORK_DIR / path.name)


def download_counties() -> Path:
    BOUNDARY_DIR.mkdir(parents=True, exist_ok=True)
    if COUNTY_ZIP.exists():
        return COUNTY_ZIP
    errors = []
    for url in COUNTY_URLS:
        try:
            print("Downloading county boundaries:", url)
            urlretrieve(url, COUNTY_ZIP)
            return COUNTY_ZIP
        except (URLError, OSError) as exc:
            errors.append(f"{url}: {exc}")
    raise RuntimeError("Could not download county boundaries:\n" + "\n".join(errors))


def load_florida_counties() -> gpd.GeoDataFrame:
    county_zip = download_counties()
    counties = gpd.read_file(county_zip).to_crs("EPSG:4326")
    florida = counties[counties["STATEFP"].astype(str).eq("12")].copy()
    if florida.empty:
        raise ValueError("No Florida counties found in Census county boundary file.")
    florida["county_name"] = florida["NAME"].astype(str) + " County"
    return florida[["GEOID", "NAME", "county_name", "geometry"]].copy()


def assign_buses_to_counties(buses: pd.DataFrame, counties: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    buses_geo = gpd.GeoDataFrame(
        buses.copy(),
        geometry=gpd.points_from_xy(buses["x"], buses["y"]),
        crs="EPSG:4326",
    )
    assigned = gpd.sjoin(
        buses_geo,
        counties[["GEOID", "NAME", "county_name", "geometry"]],
        how="left",
        predicate="within",
    ).drop(columns=["index_right"], errors="ignore")

    missing = assigned["county_name"].isna()
    if missing.any():
        nearest = gpd.sjoin_nearest(
            buses_geo.loc[missing].to_crs(PROJECTED_CRS),
            counties[["GEOID", "NAME", "county_name", "geometry"]].to_crs(PROJECTED_CRS),
            how="left",
            distance_col="nearest_county_distance_m",
        ).to_crs("EPSG:4326")
        for col in ["GEOID", "NAME", "county_name", "nearest_county_distance_m"]:
            assigned.loc[missing, col] = nearest[col].to_numpy()
    assigned["county_assignment_method"] = np.where(missing, "nearest_county", "within_county")
    return assigned


def build_bus_load_weights(assigned_buses: pd.DataFrame, population: pd.DataFrame) -> pd.DataFrame:
    buses = assigned_buses.copy()
    buses["v_nom"] = pd.to_numeric(buses["v_nom"], errors="coerce").fillna(230.0)
    buses["connected_edge_count"] = pd.to_numeric(
        buses["connected_edge_count"], errors="coerce"
    ).fillna(0.0)
    buses["bus_voltage_weight"] = (buses["v_nom"] / 230.0).clip(lower=0.25, upper=2.5)
    buses["bus_connectivity_weight"] = buses["connected_edge_count"].clip(lower=1.0)
    buses["within_county_weight_raw"] = buses["bus_voltage_weight"] * buses["bus_connectivity_weight"]

    population = population.copy()
    population["county_name"] = population["county_name"].astype(str)
    population["population_2025"] = pd.to_numeric(population["population_2025"], errors="raise")
    population["county_population_share"] = population["population_2025"] / population[
        "population_2025"
    ].sum()

    merged = buses.merge(
        population[["county_name", "population_2025", "county_population_share"]],
        on="county_name",
        how="left",
        validate="many_to_one",
    )
    missing_population = merged[merged["population_2025"].isna()]["county_name"].dropna().unique()
    if len(missing_population):
        raise ValueError(f"Bus counties missing population data: {sorted(missing_population)}")

    county_weight_sum = merged.groupby("county_name")["within_county_weight_raw"].transform("sum")
    merged["within_county_load_share"] = merged["within_county_weight_raw"] / county_weight_sum
    merged["statewide_bus_load_share"] = (
        merged["county_population_share"] * merged["within_county_load_share"]
    )
    merged["statewide_bus_load_share"] = merged["statewide_bus_load_share"] / merged[
        "statewide_bus_load_share"
    ].sum()
    return merged


def rebuild_load_timeseries(bus_weights: pd.DataFrame) -> pd.DataFrame:
    loads = pd.read_csv(BASE_NETWORK_DIR / "loads.csv")
    loads_p_set = pd.read_csv(BASE_NETWORK_DIR / "loads-p_set.csv")
    snapshots = loads_p_set["snapshot"].copy()
    statewide_hourly_mw = loads_p_set.drop(columns=["snapshot"]).sum(axis=1)

    load_to_bus = loads.set_index("name")["bus"]
    bus_share = bus_weights.set_index("name")["statewide_bus_load_share"]

    output = pd.DataFrame({"snapshot": snapshots})
    for load_name, bus in load_to_bus.items():
        output[load_name] = statewide_hourly_mw * float(bus_share.get(bus, 0.0))
    return output


def main() -> None:
    if not COUNTY_POPULATION.exists():
        raise FileNotFoundError(COUNTY_POPULATION)

    copy_network_folder()
    buses = pd.read_csv(BASE_NETWORK_DIR / "buses.csv")
    population = pd.read_csv(COUNTY_POPULATION)
    counties = load_florida_counties()
    assigned = assign_buses_to_counties(buses, counties)
    bus_weights = build_bus_load_weights(assigned.drop(columns="geometry"), population)
    new_loads = rebuild_load_timeseries(bus_weights)

    new_loads.to_csv(OUTPUT_NETWORK_DIR / "loads-p_set.csv", index=False)
    bus_weights.drop(columns=["geometry"], errors="ignore").to_csv(
        OUTPUT_NETWORK_DIR / "bus_county_population_load_weights.csv", index=False
    )

    county_summary = (
        bus_weights.groupby("county_name", as_index=False)
        .agg(
            buses=("name", "size"),
            population_2025=("population_2025", "first"),
            county_population_share=("county_population_share", "first"),
            statewide_bus_load_share=("statewide_bus_load_share", "sum"),
        )
        .sort_values("population_2025", ascending=False)
    )
    county_summary.to_csv(OUTPUT_NETWORK_DIR / "county_load_allocation_summary.csv", index=False)

    old_loads = pd.read_csv(BASE_NETWORK_DIR / "loads-p_set.csv")
    diagnostics = pd.DataFrame(
        [
            {
                "county_population_rows": len(population),
                "bus_rows": len(buses),
                "assigned_bus_rows": len(bus_weights),
                "county_load_share_sum": county_summary["statewide_bus_load_share"].sum(),
                "old_annual_load_mwh": old_loads.drop(columns=["snapshot"]).sum(axis=1).sum(),
                "new_annual_load_mwh": new_loads.drop(columns=["snapshot"]).sum(axis=1).sum(),
                "max_hourly_absolute_load_difference_mw": (
                    old_loads.drop(columns=["snapshot"]).sum(axis=1)
                    - new_loads.drop(columns=["snapshot"]).sum(axis=1)
                )
                .abs()
                .max(),
                "buses_assigned_by_nearest_county": int(
                    (bus_weights["county_assignment_method"] == "nearest_county").sum()
                ),
            }
        ]
    )
    diagnostics.to_csv(OUTPUT_NETWORK_DIR / "county_population_load_diagnostics.csv", index=False)

    print("Saved county-population load network:", OUTPUT_NETWORK_DIR)
    print(diagnostics.to_string(index=False))
    print("\nTop county load shares:")
    print(county_summary.head(15).to_string(index=False))


if __name__ == "__main__":
    main()
