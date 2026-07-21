"""
Merge Florida OSM power plants with the existing Global Power Plant Database nodes.

Inputs:
  - data/Electricity/florida_osm_powerplants.gpkg
  - data/Electricty/Nodes2.gpkg

Outputs:
  - data/Electricity/florida_powerplants_cleaned.gpkg
  - data/Electricity/florida_powerplants_cleaned.csv
  - data/Electricity/florida_powerplant_matches_review.csv

Matching uses a conservative score based on name similarity, distance, fuel type,
and capacity. Matched records use OSM geometry and GPPD attributes where available.
"""

from __future__ import annotations

import re
from difflib import SequenceMatcher
from pathlib import Path
from urllib.request import urlretrieve

import geopandas as gpd
import numpy as np
import pandas as pd


PROJECT_DIR = Path(r"C:\oxford_tc_project")
ELECTRICITY_DIR = PROJECT_DIR / "data" / "Electricity"
OLD_ELECTRICTY_DIR = PROJECT_DIR / "data" / "Electricty"
BOUNDARY_DIR = PROJECT_DIR / "data" / "Boundaries"

OSM_FLORIDA_FILE = ELECTRICITY_DIR / "florida_osm_powerplants.gpkg"
GPPD_FILE = OLD_ELECTRICTY_DIR / "Nodes2.gpkg"

OUTPUT_GPKG = ELECTRICITY_DIR / "florida_powerplants_cleaned.gpkg"
OUTPUT_CSV = ELECTRICITY_DIR / "florida_powerplants_cleaned.csv"
OSM_TRUSTED_GPKG = ELECTRICITY_DIR / "florida_powerplants_osm_matched_plus_unmatched.gpkg"
OSM_TRUSTED_CSV = ELECTRICITY_DIR / "florida_powerplants_osm_matched_plus_unmatched.csv"
MATCH_REVIEW_CSV = ELECTRICITY_DIR / "florida_powerplant_matches_review.csv"

CENSUS_STATES_ZIP_URL = (
    "https://www2.census.gov/geo/tiger/TIGER2023/STATE/tl_2023_us_state.zip"
)
CENSUS_STATES_ZIP = BOUNDARY_DIR / "tl_2023_us_state.zip"

PROJECTED_CRS = "EPSG:3086"  # Florida Albers, meters
MAX_MATCH_DISTANCE_M = 20_000
MIN_MATCH_SCORE = 0.52


FUEL_MAP = {
    "gas": "Gas",
    "natural gas": "Gas",
    "solar": "Solar",
    "pv": "Solar",
    "photovoltaic": "Solar",
    "hydro": "Hydro",
    "hydroelectric": "Hydro",
    "wind": "Wind",
    "oil": "Oil",
    "diesel": "Oil",
    "coal": "Coal",
    "hard coal": "Coal",
    "lignite": "Coal",
    "nuclear": "Nuclear",
    "biomass": "Biomass",
    "solid biomass": "Biomass",
    "biogas": "Biomass",
    "waste": "Waste",
    "battery": "Storage",
    "storage": "Storage",
    "geothermal": "Geothermal",
}

STOPWORDS = {
    "power",
    "plant",
    "station",
    "generating",
    "generation",
    "energy",
    "facility",
    "solar",
    "farm",
    "center",
    "electric",
    "project",
    "llc",
    "inc",
    "company",
    "co",
}


def clean_name(value) -> str:
    """Normalize names for approximate matching."""
    if pd.isna(value):
        return ""
    text = str(value).lower()
    text = re.sub(r"[^a-z0-9 ]+", " ", text)
    tokens = [token for token in text.split() if token not in STOPWORDS]
    return " ".join(tokens)


def name_similarity(left, right) -> float:
    left_clean = clean_name(left)
    right_clean = clean_name(right)
    if not left_clean or not right_clean:
        return 0.0
    return SequenceMatcher(None, left_clean, right_clean).ratio()


def normalize_fuel(value) -> str:
    if pd.isna(value):
        return "Unknown"
    text = str(value).strip().lower()
    if text in FUEL_MAP:
        return FUEL_MAP[text]
    for key, mapped in FUEL_MAP.items():
        if key in text:
            return mapped
    return str(value).strip() if str(value).strip() else "Unknown"


def capacity_similarity(osm_capacity, gppd_capacity) -> float:
    if pd.isna(osm_capacity) or pd.isna(gppd_capacity):
        return 0.5
    osm_capacity = float(osm_capacity)
    gppd_capacity = float(gppd_capacity)
    if osm_capacity <= 0 or gppd_capacity <= 0:
        return 0.5
    relative_diff = abs(osm_capacity - gppd_capacity) / max(osm_capacity, gppd_capacity)
    return max(0.0, 1.0 - relative_diff)


def distance_score(distance_m: float) -> float:
    if pd.isna(distance_m):
        return 0.0
    return max(0.0, 1.0 - distance_m / MAX_MATCH_DISTANCE_M)


def load_florida_boundary() -> gpd.GeoDataFrame:
    BOUNDARY_DIR.mkdir(parents=True, exist_ok=True)
    if not CENSUS_STATES_ZIP.exists():
        print("Downloading Census state boundaries:", CENSUS_STATES_ZIP_URL)
        urlretrieve(CENSUS_STATES_ZIP_URL, CENSUS_STATES_ZIP)

    states = gpd.read_file(CENSUS_STATES_ZIP).to_crs("EPSG:4326")
    if "STUSPS" in states.columns:
        florida = states[states["STUSPS"] == "FL"].copy()
    else:
        florida = states[states["NAME"].str.lower() == "florida"].copy()
    if florida.empty:
        raise ValueError("Florida boundary not found.")
    return florida


def load_gppd_florida() -> gpd.GeoDataFrame:
    gppd = gpd.read_file(GPPD_FILE).to_crs("EPSG:4326")
    if "country" in gppd.columns:
        gppd = gppd[gppd["country"] == "USA"].copy()

    florida = load_florida_boundary()
    gppd_fl = gpd.sjoin(
        gppd,
        florida[["geometry"]],
        how="inner",
        predicate="within",
    ).drop(columns=["index_right"])
    return gppd_fl.reset_index(drop=True)


def prepare_inputs() -> tuple[gpd.GeoDataFrame, gpd.GeoDataFrame]:
    if not OSM_FLORIDA_FILE.exists():
        raise FileNotFoundError(
            f"Missing {OSM_FLORIDA_FILE}. Run download_florida_osm_powerplants.py first."
        )
    if not GPPD_FILE.exists():
        raise FileNotFoundError(f"Missing {GPPD_FILE}")

    osm = gpd.read_file(OSM_FLORIDA_FILE).to_crs("EPSG:4326").reset_index(drop=True)
    gppd = load_gppd_florida()

    osm["osm_id"] = osm.index
    gppd["gppd_row_id"] = gppd.index

    osm["match_name"] = osm["Name"] if "Name" in osm.columns else ""
    gppd["match_name"] = gppd["name"] if "name" in gppd.columns else ""
    osm["fuel_norm"] = osm["Fueltype"].apply(normalize_fuel) if "Fueltype" in osm.columns else "Unknown"
    gppd["fuel_norm"] = gppd["primary_fuel"].apply(normalize_fuel) if "primary_fuel" in gppd.columns else "Unknown"
    osm["capacity_mw_osm"] = pd.to_numeric(osm.get("Capacity"), errors="coerce")
    gppd["capacity_mw_gppd"] = pd.to_numeric(gppd.get("capacity_mw"), errors="coerce")
    return osm, gppd


def build_candidate_matches(
    osm: gpd.GeoDataFrame,
    gppd: gpd.GeoDataFrame,
) -> pd.DataFrame:
    osm_m = osm.to_crs(PROJECTED_CRS)
    gppd_m = gppd.to_crs(PROJECTED_CRS)

    joined = gpd.sjoin_nearest(
        osm_m,
        gppd_m[
            [
                "gppd_row_id",
                "match_name",
                "fuel_norm",
                "capacity_mw_gppd",
                "geometry",
            ]
        ].rename(
            columns={
                "match_name": "gppd_name",
                "fuel_norm": "gppd_fuel_norm",
            }
        ),
        how="left",
        max_distance=MAX_MATCH_DISTANCE_M,
        distance_col="match_distance_m",
    )

    records = []
    for row in joined.itertuples(index=False):
        if pd.isna(row.gppd_row_id):
            continue
        osm_record = osm.loc[int(row.osm_id)]
        gppd_record = gppd.loc[int(row.gppd_row_id)]
        name_score = name_similarity(osm_record.get("match_name"), gppd_record.get("match_name"))
        fuel_match = (
            osm_record.get("fuel_norm") != "Unknown"
            and gppd_record.get("fuel_norm") != "Unknown"
            and osm_record.get("fuel_norm") == gppd_record.get("fuel_norm")
        )
        fuel_score = 1.0 if fuel_match else 0.35
        cap_score = capacity_similarity(
            osm_record.get("capacity_mw_osm"),
            gppd_record.get("capacity_mw_gppd"),
        )
        dist_score = distance_score(row.match_distance_m)
        total_score = (
            0.45 * name_score
            + 0.30 * dist_score
            + 0.15 * fuel_score
            + 0.10 * cap_score
        )
        records.append(
            {
                "osm_id": int(row.osm_id),
                "gppd_row_id": int(row.gppd_row_id),
                "osm_name": osm_record.get("Name"),
                "gppd_name": gppd_record.get("name"),
                "osm_fuel": osm_record.get("Fueltype"),
                "gppd_fuel": gppd_record.get("primary_fuel"),
                "osm_capacity_mw": osm_record.get("capacity_mw_osm"),
                "gppd_capacity_mw": gppd_record.get("capacity_mw_gppd"),
                "match_distance_m": float(row.match_distance_m),
                "name_score": name_score,
                "fuel_match": fuel_match,
                "capacity_score": cap_score,
                "distance_score": dist_score,
                "match_score": total_score,
            }
        )

    candidates = pd.DataFrame(records)
    if candidates.empty:
        return candidates
    return candidates.sort_values("match_score", ascending=False)


def greedy_select_matches(candidates: pd.DataFrame) -> pd.DataFrame:
    if candidates.empty:
        return candidates

    selected = []
    used_osm = set()
    used_gppd = set()
    for row in candidates.itertuples(index=False):
        if row.match_score < MIN_MATCH_SCORE:
            continue
        if row.osm_id in used_osm or row.gppd_row_id in used_gppd:
            continue
        selected.append(row._asdict())
        used_osm.add(row.osm_id)
        used_gppd.add(row.gppd_row_id)

    return pd.DataFrame(selected)


def merged_record_from_match(
    osm_row: pd.Series,
    gppd_row: pd.Series,
    match_row: pd.Series,
) -> dict:
    capacity = gppd_row.get("capacity_mw")
    if pd.isna(capacity):
        capacity = osm_row.get("Capacity")
    primary_fuel = gppd_row.get("primary_fuel")
    if pd.isna(primary_fuel) or str(primary_fuel).strip() == "":
        primary_fuel = osm_row.get("Fueltype")
    primary_fuel = normalize_fuel(primary_fuel)

    return {
        "name": gppd_row.get("name") if pd.notna(gppd_row.get("name")) else osm_row.get("Name"),
        "osm_name": osm_row.get("Name"),
        "gppd_name": gppd_row.get("name"),
        "capacity_mw": capacity,
        "primary_fuel": primary_fuel,
        "osm_fuel": osm_row.get("Fueltype"),
        "gppd_fuel": gppd_row.get("primary_fuel"),
        "osm_projectID": osm_row.get("projectID"),
        "gppd_idnr": gppd_row.get("gppd_idnr"),
        "matched_source": "OSM + GPPD",
        "location_source": "OSM",
        "attribute_source": "GPPD where available; OSM fallback",
        "match_distance_m": match_row.get("match_distance_m"),
        "match_score": match_row.get("match_score"),
        "location_confidence": "high",
        "geometry": osm_row.geometry,
    }


def merged_record_from_osm(osm_row: pd.Series) -> dict:
    return {
        "name": osm_row.get("Name"),
        "osm_name": osm_row.get("Name"),
        "gppd_name": np.nan,
        "capacity_mw": osm_row.get("Capacity"),
        "primary_fuel": normalize_fuel(osm_row.get("Fueltype")),
        "osm_fuel": osm_row.get("Fueltype"),
        "gppd_fuel": np.nan,
        "osm_projectID": osm_row.get("projectID"),
        "gppd_idnr": np.nan,
        "matched_source": "OSM only",
        "location_source": "OSM",
        "attribute_source": "OSM",
        "match_distance_m": np.nan,
        "match_score": np.nan,
        "location_confidence": "high",
        "geometry": osm_row.geometry,
    }


def merged_record_from_gppd(gppd_row: pd.Series) -> dict:
    return {
        "name": gppd_row.get("name"),
        "osm_name": np.nan,
        "gppd_name": gppd_row.get("name"),
        "capacity_mw": gppd_row.get("capacity_mw"),
        "primary_fuel": normalize_fuel(gppd_row.get("primary_fuel")),
        "osm_fuel": np.nan,
        "gppd_fuel": gppd_row.get("primary_fuel"),
        "osm_projectID": np.nan,
        "gppd_idnr": gppd_row.get("gppd_idnr"),
        "matched_source": "GPPD only",
        "location_source": "GPPD",
        "attribute_source": "GPPD",
        "match_distance_m": np.nan,
        "match_score": np.nan,
        "location_confidence": "lower",
        "geometry": gppd_row.geometry,
    }


def create_cleaned_dataset(
    osm: gpd.GeoDataFrame,
    gppd: gpd.GeoDataFrame,
    matches: pd.DataFrame,
) -> gpd.GeoDataFrame:
    records = []
    matched_osm = set()
    matched_gppd = set()

    for match in matches.itertuples(index=False):
        osm_row = osm.loc[int(match.osm_id)]
        gppd_row = gppd.loc[int(match.gppd_row_id)]
        records.append(merged_record_from_match(osm_row, gppd_row, pd.Series(match._asdict())))
        matched_osm.add(int(match.osm_id))
        matched_gppd.add(int(match.gppd_row_id))

    for _, osm_row in osm.loc[~osm["osm_id"].isin(matched_osm)].iterrows():
        records.append(merged_record_from_osm(osm_row))

    for _, gppd_row in gppd.loc[~gppd["gppd_row_id"].isin(matched_gppd)].iterrows():
        records.append(merged_record_from_gppd(gppd_row))

    cleaned = gpd.GeoDataFrame(records, geometry="geometry", crs="EPSG:4326")
    cleaned["capacity_mw"] = pd.to_numeric(cleaned["capacity_mw"], errors="coerce")
    cleaned["primary_fuel"] = cleaned["primary_fuel"].fillna("Unknown")
    cleaned["longitude"] = cleaned.geometry.x
    cleaned["latitude"] = cleaned.geometry.y
    return cleaned


def save_outputs(cleaned: gpd.GeoDataFrame, candidates: pd.DataFrame, matches: pd.DataFrame) -> None:
    ELECTRICITY_DIR.mkdir(parents=True, exist_ok=True)
    cleaned.to_file(OUTPUT_GPKG, layer="florida_powerplants_cleaned", driver="GPKG")
    cleaned.drop(columns="geometry").to_csv(OUTPUT_CSV, index=False)

    osm_trusted = cleaned[cleaned["matched_source"] != "GPPD only"].copy()
    osm_trusted.to_file(
        OSM_TRUSTED_GPKG,
        layer="florida_powerplants_osm_matched_plus_unmatched",
        driver="GPKG",
    )
    osm_trusted.drop(columns="geometry").to_csv(OSM_TRUSTED_CSV, index=False)

    review = candidates.copy()
    if not review.empty:
        review["selected_match"] = False
        selected_pairs = set(zip(matches["osm_id"], matches["gppd_row_id"])) if not matches.empty else set()
        review["selected_match"] = [
            (row.osm_id, row.gppd_row_id) in selected_pairs
            for row in review.itertuples(index=False)
        ]
    review.to_csv(MATCH_REVIEW_CSV, index=False)


def print_summary(
    osm: gpd.GeoDataFrame,
    gppd: gpd.GeoDataFrame,
    cleaned: gpd.GeoDataFrame,
    matches: pd.DataFrame,
) -> None:
    matched_count = len(matches)
    unmatched_osm = len(osm) - matched_count
    unmatched_gppd = len(gppd) - matched_count
    avg_distance = matches["match_distance_m"].mean() if matched_count else np.nan

    print("\nSummary statistics")
    print("Matched plants:", matched_count)
    print("Unmatched OSM plants:", unmatched_osm)
    print("Unmatched Global Power Plant Database plants:", unmatched_gppd)
    print("OSM-trusted output plants, excluding GPPD-only:", len(cleaned[cleaned["matched_source"] != "GPPD only"]))
    print("Average match distance (m):", round(avg_distance, 1) if pd.notna(avg_distance) else "NA")

    print("\nCounts by fuel type:")
    print(cleaned["primary_fuel"].fillna("Unknown").value_counts().to_string())

    print("\nTotal capacity by fuel type (MW):")
    print(
        cleaned.groupby("primary_fuel", dropna=False)["capacity_mw"]
        .sum()
        .sort_values(ascending=False)
        .round(2)
        .to_string()
    )


def main() -> None:
    print("Loading Florida OSM power plants and Global Power Plant Database nodes...")
    osm, gppd = prepare_inputs()
    print("Florida OSM plants:", len(osm))
    print("Florida GPPD plants:", len(gppd))

    print("\nBuilding candidate matches...")
    candidates = build_candidate_matches(osm, gppd)
    matches = greedy_select_matches(candidates)

    print("Creating cleaned merged dataset...")
    cleaned = create_cleaned_dataset(osm, gppd, matches)
    save_outputs(cleaned, candidates, matches)
    print_summary(osm, gppd, cleaned, matches)

    print("\nSaved cleaned GeoPackage:", OUTPUT_GPKG)
    print("Saved cleaned CSV:", OUTPUT_CSV)
    print("Saved OSM-trusted GeoPackage:", OSM_TRUSTED_GPKG)
    print("Saved OSM-trusted CSV:", OSM_TRUSTED_CSV)
    print("Saved match review CSV:", MATCH_REVIEW_CSV)


if __name__ == "__main__":
    main()
