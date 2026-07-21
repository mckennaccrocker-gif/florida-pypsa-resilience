"""
Polygon-based flood exposure for Florida power plant footprints.

This uses OSM power plant polygons with transferred point/GPPD attributes and
samples JRC flood-depth rasters by polygon area. It is separate from the older
point-based node exposure so plant footprints can capture partial inundation.
"""

from __future__ import annotations

from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rasterio
from rasterio.mask import mask


PROJECT_DIR = Path(r"C:\oxford_tc_project")
ELECTRICITY_DIR = PROJECT_DIR / "data" / "Electricity"
HAZARD_DIR = PROJECT_DIR / "data" / "Hazards" / "Flood"
OUTPUT_DIR = PROJECT_DIR / "data" / "Exposure" / "powerplant_polygon_flood_exposure"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

POWERPLANT_POLYGONS = ELECTRICITY_DIR / "florida_osm_powerplant_polygons_with_point_attributes.gpkg"
RETURN_PERIOD_RASTERS = {
    10: HAZARD_DIR / "JRC_RP10_USA_assets.vrt",
    20: HAZARD_DIR / "JRC_RP20_USA_assets.vrt",
    50: HAZARD_DIR / "JRC_RP50_USA_assets.vrt",
    75: HAZARD_DIR / "JRC_RP75_USA_assets.vrt",
    100: HAZARD_DIR / "JRC_RP100_USA_assets.vrt",
    200: HAZARD_DIR / "JRC_RP200_USA_assets.vrt",
    500: HAZARD_DIR / "JRC_RP500_USA_assets.vrt",
}

FLOOD_CLASS_ORDER = ["0 m", "0-0.5 m", "0.5-1 m", "1-2 m", "2-5 m", ">5 m"]
FLOOD_CLASS_COLORS = {
    "0 m": "#d9d9d9",
    "0-0.5 m": "#abd9e9",
    "0.5-1 m": "#74add1",
    "1-2 m": "#4575b4",
    "2-5 m": "#fdae61",
    ">5 m": "#d73027",
}
FUEL_COLORS = {
    "Solar": "#f4c430",
    "Gas": "#7f7f7f",
    "Hydro": "#2b8cbe",
    "Wind": "#41ab5d",
    "Oil": "#252525",
    "Waste": "#8c6bb1",
    "Coal": "#4d4d4d",
    "Biomass": "#a1d99b",
    "Storage": "#fb6a4a",
    "Nuclear": "#31a354",
    "Unknown": "#bdbdbd",
}


def classify_flood_depth(depth: float) -> str:
    if pd.isna(depth) or depth <= 0:
        return "0 m"
    if depth <= 0.5:
        return "0-0.5 m"
    if depth <= 1:
        return "0.5-1 m"
    if depth <= 2:
        return "1-2 m"
    if depth <= 5:
        return "2-5 m"
    return ">5 m"


def clean_raster_values(values: np.ndarray, nodata: float | None) -> np.ndarray:
    values = values.astype("float64")
    if nodata is not None:
        values = np.where(values == nodata, np.nan, values)
    values = np.where(values < 0, np.nan, values)
    return values


def load_powerplant_polygons() -> gpd.GeoDataFrame:
    if not POWERPLANT_POLYGONS.exists():
        raise FileNotFoundError(POWERPLANT_POLYGONS)

    plants = gpd.read_file(POWERPLANT_POLYGONS).to_crs("EPSG:4326")
    print("\nPower plant polygon columns:")
    print(", ".join(plants.columns.astype(str)))
    print("Power plant polygon geometry types:")
    print(plants.geom_type.value_counts().to_string())

    plants = plants[plants.geometry.notna() & ~plants.geometry.is_empty].copy()
    plants["plant_polygon_id"] = np.arange(len(plants))
    plants["capacity_mw"] = pd.to_numeric(plants.get("capacity_mw"), errors="coerce")
    plants["primary_fuel"] = plants.get("primary_fuel", "Unknown").fillna("Unknown")
    return plants


def polygon_flood_stats(plants: gpd.GeoDataFrame, raster_path: Path, return_period: int) -> pd.DataFrame:
    if not raster_path.exists():
        raise FileNotFoundError(raster_path)

    records = []
    with rasterio.open(raster_path) as src:
        plants_raster = plants.to_crs(src.crs)
        print(f"\nSampling RP{return_period}: {raster_path}")
        print("Raster CRS:", src.crs, "| nodata:", src.nodata)

        for row in plants_raster.itertuples():
            base = {
                "plant_polygon_id": int(row.plant_polygon_id),
                "return_period": return_period,
            }
            try:
                out_image, _ = mask(
                    src,
                    [row.geometry],
                    crop=True,
                    filled=False,
                    all_touched=True,
                )
                data = out_image[0]
                if np.ma.isMaskedArray(data):
                    values = data.compressed().astype("float64")
                else:
                    values = data.ravel().astype("float64")
                values = clean_raster_values(values, src.nodata)
                values = values[np.isfinite(values)]
            except ValueError:
                values = np.array([], dtype="float64")

            if values.size == 0:
                base.update(
                    {
                        "sampled_pixel_count": 0,
                        "wet_pixel_count": 0,
                        "wet_pixel_fraction": 0.0,
                        "max_flood_depth_m": 0.0,
                        "mean_flood_depth_m": 0.0,
                        "mean_wet_flood_depth_m": 0.0,
                    }
                )
            else:
                wet = values > 0
                base.update(
                    {
                        "sampled_pixel_count": int(values.size),
                        "wet_pixel_count": int(wet.sum()),
                        "wet_pixel_fraction": float(wet.mean()),
                        "max_flood_depth_m": float(values.max()),
                        "mean_flood_depth_m": float(values.mean()),
                        "mean_wet_flood_depth_m": float(values[wet].mean()) if wet.any() else 0.0,
                    }
                )
            base["flood_class"] = classify_flood_depth(base["max_flood_depth_m"])
            records.append(base)

    return pd.DataFrame(records)


def build_exposure_tables(plants: gpd.GeoDataFrame) -> tuple[gpd.GeoDataFrame, pd.DataFrame]:
    stats = []
    for return_period, raster_path in RETURN_PERIOD_RASTERS.items():
        stats.append(polygon_flood_stats(plants, raster_path, return_period))
    stats = pd.concat(stats, ignore_index=True)

    attr_cols = [
        "plant_polygon_id",
        "polygon_osm_id",
        "name",
        "capacity_mw",
        "capacity_mw_max",
        "primary_fuel",
        "owner",
        "source",
        "polygon_area_m2",
        "matched_point_count",
        "gppd_ids",
        "osm_project_ids",
    ]
    attr_cols = [col for col in attr_cols if col in plants.columns]
    attrs = pd.DataFrame(plants.drop(columns="geometry"))[attr_cols]
    exposure = plants.merge(stats, on="plant_polygon_id", how="left")
    exposure["flood_class"] = pd.Categorical(
        exposure["flood_class"],
        categories=FLOOD_CLASS_ORDER,
        ordered=True,
    )
    table = attrs.merge(stats, on="plant_polygon_id", how="left")
    return exposure, table


def summarize_exposure(table: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    table["is_exposed"] = table["max_flood_depth_m"] > 0
    table["exposed_capacity_mw"] = np.where(
        table["is_exposed"],
        table["capacity_mw"].fillna(0.0),
        0.0,
    )

    rp_summary = (
        table.groupby("return_period", as_index=False)
        .agg(
            plant_polygons=("plant_polygon_id", "nunique"),
            exposed_polygons=("is_exposed", "sum"),
            exposed_capacity_mw=("exposed_capacity_mw", "sum"),
            max_flood_depth_m=("max_flood_depth_m", "max"),
            mean_wet_flood_depth_m=("mean_wet_flood_depth_m", "mean"),
        )
        .sort_values("return_period")
    )
    rp_summary["exposed_polygon_percent"] = (
        rp_summary["exposed_polygons"] / rp_summary["plant_polygons"] * 100
    )

    class_summary = (
        table.groupby(["return_period", "flood_class"], observed=False)
        .agg(
            plant_polygons=("plant_polygon_id", "nunique"),
            capacity_mw=("capacity_mw", "sum"),
        )
        .reset_index()
    )

    fuel_summary = (
        table[table["is_exposed"]]
        .groupby(["return_period", "primary_fuel"], observed=False)
        .agg(
            exposed_polygons=("plant_polygon_id", "nunique"),
            exposed_capacity_mw=("capacity_mw", "sum"),
            max_flood_depth_m=("max_flood_depth_m", "max"),
        )
        .reset_index()
        .sort_values(["return_period", "exposed_capacity_mw"], ascending=[True, False])
    )
    return rp_summary, class_summary, fuel_summary


def plot_return_period_summary(rp_summary: pd.DataFrame) -> None:
    fig, ax1 = plt.subplots(figsize=(9, 5))
    ax2 = ax1.twinx()

    ax1.bar(
        rp_summary["return_period"].astype(str),
        rp_summary["exposed_capacity_mw"],
        color="#4575b4",
        label="Exposed capacity",
    )
    ax2.plot(
        rp_summary["return_period"].astype(str),
        rp_summary["exposed_polygons"],
        color="#d73027",
        marker="o",
        linewidth=2,
        label="Exposed polygons",
    )

    ax1.set_xlabel("Flood return period (years)")
    ax1.set_ylabel("Exposed plant capacity (MW)")
    ax2.set_ylabel("Exposed plant polygons")
    ax1.set_title("Power Plant Polygon Flood Exposure by Return Period")
    ax1.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "powerplant_polygon_flood_exposure_by_return_period.png", dpi=220)
    plt.close(fig)


def plot_class_summary(class_summary: pd.DataFrame) -> None:
    pivot = class_summary.pivot_table(
        index="return_period",
        columns="flood_class",
        values="plant_polygons",
        aggfunc="sum",
        fill_value=0,
        observed=False,
    )
    pivot = pivot.reindex(columns=FLOOD_CLASS_ORDER, fill_value=0)

    fig, ax = plt.subplots(figsize=(10, 5.5))
    bottom = np.zeros(len(pivot))
    x = np.arange(len(pivot))
    for flood_class in FLOOD_CLASS_ORDER:
        vals = pivot[flood_class].to_numpy()
        ax.bar(
            x,
            vals,
            bottom=bottom,
            label=flood_class,
            color=FLOOD_CLASS_COLORS[flood_class],
        )
        bottom += vals
    ax.set_xticks(x)
    ax.set_xticklabels(pivot.index.astype(str))
    ax.set_xlabel("Flood return period (years)")
    ax.set_ylabel("Power plant polygons")
    ax.set_title("Power Plant Polygons by Maximum Flood Depth Class")
    ax.legend(title="Max flood depth", bbox_to_anchor=(1.02, 1), loc="upper left")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "powerplant_polygon_flood_depth_classes.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_fuel_summary(fuel_summary: pd.DataFrame) -> None:
    if fuel_summary.empty:
        return

    pivot = fuel_summary.pivot_table(
        index="return_period",
        columns="primary_fuel",
        values="exposed_capacity_mw",
        aggfunc="sum",
        fill_value=0,
        observed=False,
    )
    fig, ax = plt.subplots(figsize=(10, 5.5))
    bottom = np.zeros(len(pivot))
    x = np.arange(len(pivot))
    for fuel in pivot.columns:
        vals = pivot[fuel].to_numpy()
        ax.bar(x, vals, bottom=bottom, label=fuel, color=FUEL_COLORS.get(fuel, "#969696"))
        bottom += vals
    ax.set_xticks(x)
    ax.set_xticklabels(pivot.index.astype(str))
    ax.set_xlabel("Flood return period (years)")
    ax.set_ylabel("Exposed capacity (MW)")
    ax.set_title("Flood-Exposed Power Plant Capacity by Fuel")
    ax.legend(title="Fuel", bbox_to_anchor=(1.02, 1), loc="upper left")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "powerplant_polygon_flood_exposed_capacity_by_fuel.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    print("Checking files...")
    print(POWERPLANT_POLYGONS, "exists?", POWERPLANT_POLYGONS.exists())
    for rp, path in RETURN_PERIOD_RASTERS.items():
        print(f"RP{rp}", path, "exists?", path.exists())

    plants = load_powerplant_polygons()
    exposure, table = build_exposure_tables(plants)
    rp_summary, class_summary, fuel_summary = summarize_exposure(table)

    exposure.to_file(
        OUTPUT_DIR / "powerplant_polygon_flood_exposure.gpkg",
        layer="powerplant_polygon_flood_exposure",
        driver="GPKG",
    )
    table.to_csv(OUTPUT_DIR / "powerplant_polygon_flood_exposure.csv", index=False)
    rp_summary.to_csv(OUTPUT_DIR / "powerplant_polygon_flood_summary_by_return_period.csv", index=False)
    class_summary.to_csv(OUTPUT_DIR / "powerplant_polygon_flood_summary_by_class.csv", index=False)
    fuel_summary.to_csv(OUTPUT_DIR / "powerplant_polygon_flood_summary_by_fuel.csv", index=False)

    plot_return_period_summary(rp_summary)
    plot_class_summary(class_summary)
    plot_fuel_summary(fuel_summary)

    print("\nReturn-period summary:")
    print(rp_summary.to_string(index=False))
    print("\nTop exposed fuel rows:")
    print(fuel_summary.head(20).to_string(index=False))
    print("\nSaved outputs in:", OUTPUT_DIR)


if __name__ == "__main__":
    main()
