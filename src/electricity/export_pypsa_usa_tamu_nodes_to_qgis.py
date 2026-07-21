from pathlib import Path

import geopandas as gpd
import pandas as pd
from shapely.geometry import LineString, Point


ROOT = Path(__file__).resolve().parents[2]
BASE_GRID = ROOT / "external" / "pypsa-usa" / "workflow" / "data" / "breakthrough_network" / "base_grid"
OUT_DIR = ROOT / "data" / "Electricity" / "pypsa_usa_tamu_qgis"
OUT_GPKG = OUT_DIR / "pypsa_usa_tamu_network_for_qgis.gpkg"

# Slightly expanded Florida box so northern tie-line nodes in GA/AL are visible too.
FLORIDA_REGION_BOUNDS = {
    "lon_min": -88.0,
    "lon_max": -79.0,
    "lat_min": 24.0,
    "lat_max": 32.5,
}


def in_florida_region(df: pd.DataFrame) -> pd.Series:
    return (
        df["lon"].between(FLORIDA_REGION_BOUNDS["lon_min"], FLORIDA_REGION_BOUNDS["lon_max"])
        & df["lat"].between(FLORIDA_REGION_BOUNDS["lat_min"], FLORIDA_REGION_BOUNDS["lat_max"])
    )


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    if OUT_GPKG.exists():
        OUT_GPKG.unlink()

    buses = pd.read_csv(BASE_GRID / "bus.csv")
    substations = pd.read_csv(BASE_GRID / "sub.csv")
    bus2sub = pd.read_csv(BASE_GRID / "bus2sub.csv")
    branches = pd.read_csv(BASE_GRID / "branch.csv")

    substations = substations.rename(columns={"name": "substation_name"})
    buses = (
        buses.merge(bus2sub[["bus_id", "sub_id"]], on="bus_id", how="left")
        .merge(
            substations[["sub_id", "substation_name", "lat", "lon", "interconnect"]],
            on="sub_id",
            how="left",
            suffixes=("", "_sub"),
        )
    )

    buses = buses.dropna(subset=["lat", "lon"]).copy()
    buses["geometry"] = [Point(xy) for xy in zip(buses["lon"], buses["lat"])]
    buses_gdf = gpd.GeoDataFrame(buses, geometry="geometry", crs="EPSG:4326")

    substations = substations.dropna(subset=["lat", "lon"]).copy()
    substations["geometry"] = [Point(xy) for xy in zip(substations["lon"], substations["lat"])]
    substations_gdf = gpd.GeoDataFrame(substations, geometry="geometry", crs="EPSG:4326")

    bus_points = buses_gdf.set_index("bus_id")["geometry"]
    branches = branches[
        branches["from_bus_id"].isin(bus_points.index) & branches["to_bus_id"].isin(bus_points.index)
    ].copy()
    branches["from_lon"] = branches["from_bus_id"].map(buses_gdf.set_index("bus_id")["lon"])
    branches["from_lat"] = branches["from_bus_id"].map(buses_gdf.set_index("bus_id")["lat"])
    branches["to_lon"] = branches["to_bus_id"].map(buses_gdf.set_index("bus_id")["lon"])
    branches["to_lat"] = branches["to_bus_id"].map(buses_gdf.set_index("bus_id")["lat"])
    branches["geometry"] = [
        LineString([bus_points.loc[row.from_bus_id], bus_points.loc[row.to_bus_id]])
        for row in branches.itertuples()
    ]
    branches_gdf = gpd.GeoDataFrame(branches, geometry="geometry", crs="EPSG:4326")

    florida_bus_ids = set(buses_gdf.loc[in_florida_region(buses_gdf), "bus_id"])
    florida_sub_ids = set(buses_gdf.loc[buses_gdf["bus_id"].isin(florida_bus_ids), "sub_id"].dropna())
    florida_branches = branches_gdf[
        branches_gdf["from_bus_id"].isin(florida_bus_ids) | branches_gdf["to_bus_id"].isin(florida_bus_ids)
    ].copy()

    buses_gdf.to_file(OUT_GPKG, layer="tamu_buses_all", driver="GPKG")
    substations_gdf.to_file(OUT_GPKG, layer="tamu_substations_all", driver="GPKG")
    branches_gdf.to_file(OUT_GPKG, layer="tamu_branches_all", driver="GPKG")
    buses_gdf[buses_gdf["bus_id"].isin(florida_bus_ids)].to_file(
        OUT_GPKG,
        layer="florida_region_tamu_buses",
        driver="GPKG",
    )
    substations_gdf[substations_gdf["sub_id"].isin(florida_sub_ids)].to_file(
        OUT_GPKG,
        layer="florida_region_tamu_substations",
        driver="GPKG",
    )
    florida_branches.to_file(OUT_GPKG, layer="florida_region_tamu_branches", driver="GPKG")

    print(f"Saved {OUT_GPKG}")
    print(f"All buses: {len(buses_gdf):,}")
    print(f"All substations: {len(substations_gdf):,}")
    print(f"All branches: {len(branches_gdf):,}")
    print(f"Florida-region buses: {len(florida_bus_ids):,}")
    print(f"Florida-region substations: {len(florida_sub_ids):,}")
    print(f"Florida-region branches: {len(florida_branches):,}")


if __name__ == "__main__":
    main()
