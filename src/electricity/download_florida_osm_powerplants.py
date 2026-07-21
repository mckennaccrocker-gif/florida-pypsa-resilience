"""
Download OpenStreetMap power plants and keep only plants located in Florida.

Source repository:
https://github.com/open-energy-transition/osm-powerplants

The repository publishes a ready-to-use global CSV at the repo root:
osm_global.csv.gz

Outputs:
  - data/Electricity/florida_osm_powerplants.gpkg
  - data/Electricity/florida_osm_powerplants.csv
"""

from __future__ import annotations

from pathlib import Path
from urllib.request import urlretrieve

import geopandas as gpd
import pandas as pd


PROJECT_DIR = Path(r"C:\oxford_tc_project")
OUTPUT_DIR = PROJECT_DIR / "data" / "Electricity"
BOUNDARY_DIR = PROJECT_DIR / "data" / "Boundaries"

OUTPUT_GPKG = OUTPUT_DIR / "florida_osm_powerplants.gpkg"
OUTPUT_CSV = OUTPUT_DIR / "florida_osm_powerplants.csv"

OSM_POWERPLANTS_CSV_URLS = [
    "https://raw.githubusercontent.com/open-energy-transition/osm-powerplants/main/osm_global.csv.gz",
    "https://raw.githubusercontent.com/open-energy-transition/osm-powerplants/master/osm_global.csv.gz",
    "https://raw.githubusercontent.com/open-energy-transition/osm-powerplants/main/osm_global.csv",
    "https://raw.githubusercontent.com/open-energy-transition/osm-powerplants/master/osm_global.csv",
]

# Census TIGER/Line state boundaries. Florida is selected from the states file.
CENSUS_STATES_ZIP_URL = (
    "https://www2.census.gov/geo/tiger/TIGER2023/STATE/tl_2023_us_state.zip"
)
CENSUS_STATES_ZIP = BOUNDARY_DIR / "tl_2023_us_state.zip"


def find_column(columns: list[str], candidates: list[str]) -> str | None:
    """Find a column by trying exact and case-insensitive candidate names."""
    column_lookup = {column.lower(): column for column in columns}
    for candidate in candidates:
        if candidate in columns:
            return candidate
        if candidate.lower() in column_lookup:
            return column_lookup[candidate.lower()]
    return None


def load_osm_powerplants() -> pd.DataFrame:
    """Load the OSM global power plants CSV from GitHub raw URLs."""
    last_error = None
    for url in OSM_POWERPLANTS_CSV_URLS:
        try:
            print("Trying OSM power plants CSV:", url)
            return pd.read_csv(url)
        except Exception as exc:  # noqa: BLE001 - print all download/parser failures
            last_error = exc
            print("  Failed:", exc)

    raise RuntimeError(
        "Could not load OSM global power plants CSV from known raw GitHub URLs. "
        "Check the repository for the current global CSV name."
    ) from last_error


def powerplants_to_geodataframe(powerplants: pd.DataFrame) -> gpd.GeoDataFrame:
    """Convert a power plant table into point geometry using lon/lat columns."""
    lon_column = find_column(
        list(powerplants.columns),
        ["lon", "longitude", "lng", "x", "Lon", "Longitude"],
    )
    lat_column = find_column(
        list(powerplants.columns),
        ["lat", "latitude", "y", "Lat", "Latitude"],
    )

    if lon_column is None or lat_column is None:
        print("\nCould not identify longitude/latitude columns.")
        print("Available columns:")
        print(list(powerplants.columns))
        raise ValueError("Missing longitude/latitude columns in OSM power plants CSV.")

    print(f"Using coordinate columns: longitude={lon_column}, latitude={lat_column}")
    powerplants = powerplants.copy()
    powerplants[lon_column] = pd.to_numeric(powerplants[lon_column], errors="coerce")
    powerplants[lat_column] = pd.to_numeric(powerplants[lat_column], errors="coerce")
    powerplants = powerplants.dropna(subset=[lon_column, lat_column])

    return gpd.GeoDataFrame(
        powerplants,
        geometry=gpd.points_from_xy(powerplants[lon_column], powerplants[lat_column]),
        crs="EPSG:4326",
    )


def load_florida_boundary() -> gpd.GeoDataFrame:
    """Download/read Census state boundaries and return Florida in EPSG:4326."""
    BOUNDARY_DIR.mkdir(parents=True, exist_ok=True)
    if not CENSUS_STATES_ZIP.exists():
        print("Downloading Census state boundaries:", CENSUS_STATES_ZIP_URL)
        urlretrieve(CENSUS_STATES_ZIP_URL, CENSUS_STATES_ZIP)

    states = gpd.read_file(CENSUS_STATES_ZIP)
    states = states.to_crs("EPSG:4326")

    if "STUSPS" in states.columns:
        florida = states[states["STUSPS"] == "FL"].copy()
    elif "NAME" in states.columns:
        florida = states[states["NAME"].str.lower() == "florida"].copy()
    else:
        raise ValueError("Could not identify Florida in Census states boundary file.")

    if florida.empty:
        raise ValueError("Florida boundary was not found in the Census states file.")

    return florida


def print_fuel_counts(florida_plants: gpd.GeoDataFrame) -> None:
    """Print counts by a likely fuel/type/source column if available."""
    fuel_candidates = [
        "Fueltype",
        "fuel_type",
        "fuel",
        "primary_fuel",
        "source",
        "Source",
        "Technology",
    ]
    fuel_column = find_column(list(florida_plants.columns), fuel_candidates)
    if fuel_column is None:
        print("\nNo fuel/fuel_type/source-like column found for counts.")
        return

    print(f"\nCounts by {fuel_column}:")
    print(florida_plants[fuel_column].fillna("Unknown").value_counts().to_string())


def safe_console_print(text: str) -> None:
    """Print text even when the Windows console cannot encode some characters."""
    print(text.encode("cp1252", errors="replace").decode("cp1252"))


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # 1-3. Download/read the global OSM power plants CSV.
    powerplants = load_osm_powerplants()
    print("\nNumber of global OSM power plants loaded:", len(powerplants))
    print("\nColumn names in global OSM power plants dataset:")
    print(list(powerplants.columns))

    # 4. Convert the CSV into a GeoDataFrame using longitude/latitude columns.
    powerplants_gdf = powerplants_to_geodataframe(powerplants)

    # 5. Load a reliable Florida boundary from Census TIGER/Line states.
    florida_boundary = load_florida_boundary()

    # 6. Clip/filter OSM power plants to only those located in Florida.
    florida_plants = gpd.sjoin(
        powerplants_gdf,
        florida_boundary[["geometry"]],
        how="inner",
        predicate="within",
    ).drop(columns=["index_right"])
    florida_plants = florida_plants.to_crs("EPSG:4326")

    print("\nNumber of Florida power plants after filtering:", len(florida_plants))
    print("\nFirst few Florida OSM power plant rows:")
    safe_console_print(florida_plants.head().drop(columns="geometry").to_string())
    print_fuel_counts(florida_plants)

    # 7-8. Save Florida-only OSM power plants as GeoPackage and CSV.
    florida_plants.to_file(
        OUTPUT_GPKG,
        layer="florida_osm_powerplants",
        driver="GPKG",
    )
    florida_plants.drop(columns="geometry").to_csv(OUTPUT_CSV, index=False)

    print("\nSaved Florida OSM power plants GeoPackage:", OUTPUT_GPKG)
    print("Saved Florida OSM power plants CSV:", OUTPUT_CSV)


if __name__ == "__main__":
    main()
