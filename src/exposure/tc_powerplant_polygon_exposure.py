"""
Polygon-based tropical cyclone wind exposure for Florida power plant footprints.

This samples TC wind rasters over OSM power plant polygons with transferred
point/GPPD attributes. It reports max/mean wind across each facility footprint
and area fractions above wind thresholds.

Inputs:
  - data/Electricity/florida_osm_powerplant_polygons_with_point_attributes.gpkg
  - data/Hazards/Tropical_cyclones/STORM_constant_RP*_US_crop.tif
  - data/Exposure/florida_clean_assets_tc_snail_intersection/
      fred_open_gira_ibtracs_historical_max_wind_florida.tif

Outputs:
  - data/Exposure/powerplant_polygon_tc_exposure/
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
TC_DIR = PROJECT_DIR / "data" / "Hazards" / "Tropical_cyclones"
OUTPUT_DIR = PROJECT_DIR / "data" / "Exposure" / "powerplant_polygon_tc_exposure"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

POWERPLANT_POLYGONS = ELECTRICITY_DIR / "florida_osm_powerplant_polygons_with_point_attributes.gpkg"
OPEN_GIRA_HISTORICAL_MAX_RASTER = (
    PROJECT_DIR
    / "data"
    / "Exposure"
    / "florida_clean_assets_tc_snail_intersection"
    / "fred_open_gira_ibtracs_historical_max_wind_florida.tif"
)
STORM_RP_RASTERS = {
    10: TC_DIR / "STORM_constant_RP10_US_crop.tif",
    20: TC_DIR / "STORM_constant_RP20_US_crop.tif",
    50: TC_DIR / "STORM_constant_RP50_US_crop.tif",
    100: TC_DIR / "STORM_constant_RP100_US_crop.tif",
    200: TC_DIR / "STORM_constant_RP200_US_crop.tif",
    500: TC_DIR / "STORM_constant_RP500_US_crop.tif",
}

TC_BINS = [0, 25, 30, 35, 40, np.inf]
TC_LABELS = ["<25", "25-30", "30-35", "35-40", ">40"]
TC_CLASS_COLORS = {
    "<25": "#2c7bb6",
    "25-30": "#abd9e9",
    "30-35": "#ffffbf",
    "35-40": "#fdae61",
    ">40": "#d7191c",
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


def classify_wind_speed(speed: float) -> str:
    if pd.isna(speed):
        return np.nan
    return pd.cut([speed], bins=TC_BINS, labels=TC_LABELS, include_lowest=True)[0]


def clean_raster_values(values: np.ndarray, nodata: float | None) -> np.ndarray:
    values = values.astype("float64")
    if nodata is not None:
        values = np.where(values == nodata, np.nan, values)
    values = np.where(values < 0, np.nan, values)
    return values[np.isfinite(values)]


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


def raster_sources() -> list[dict]:
    sources = [
        {
            "dataset": "open_gira_historical_max",
            "return_period": np.nan,
            "raster_path": OPEN_GIRA_HISTORICAL_MAX_RASTER,
        }
    ]
    for return_period, raster_path in STORM_RP_RASTERS.items():
        sources.append(
            {
                "dataset": f"storm_rp{return_period}",
                "return_period": return_period,
                "raster_path": raster_path,
            }
        )
    return sources


def polygon_wind_stats(
    plants: gpd.GeoDataFrame,
    raster_path: Path,
    dataset: str,
    return_period,
) -> pd.DataFrame:
    if not raster_path.exists():
        raise FileNotFoundError(raster_path)

    records = []
    with rasterio.open(raster_path) as src:
        plants_raster = plants.to_crs(src.crs)
        print(f"\nSampling {dataset}: {raster_path}")
        print("Raster CRS:", src.crs, "| nodata:", src.nodata)

        for row in plants_raster.itertuples():
            base = {
                "plant_polygon_id": int(row.plant_polygon_id),
                "dataset": dataset,
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
            except ValueError:
                values = np.array([], dtype="float64")

            if values.size == 0:
                base.update(
                    {
                        "sampled_pixel_count": 0,
                        "max_wind_speed_ms": np.nan,
                        "mean_wind_speed_ms": np.nan,
                        "fraction_ge_25ms": 0.0,
                        "fraction_ge_30ms": 0.0,
                        "fraction_ge_35ms": 0.0,
                        "fraction_ge_40ms": 0.0,
                    }
                )
            else:
                base.update(
                    {
                        "sampled_pixel_count": int(values.size),
                        "max_wind_speed_ms": float(values.max()),
                        "mean_wind_speed_ms": float(values.mean()),
                        "fraction_ge_25ms": float((values >= 25).mean()),
                        "fraction_ge_30ms": float((values >= 30).mean()),
                        "fraction_ge_35ms": float((values >= 35).mean()),
                        "fraction_ge_40ms": float((values >= 40).mean()),
                    }
                )
            base["tc_wind_class"] = classify_wind_speed(base["max_wind_speed_ms"])
            records.append(base)

    return pd.DataFrame(records)


def build_exposure_tables(plants: gpd.GeoDataFrame) -> tuple[gpd.GeoDataFrame, pd.DataFrame]:
    stats = []
    for source in raster_sources():
        stats.append(
            polygon_wind_stats(
                plants,
                source["raster_path"],
                source["dataset"],
                source["return_period"],
            )
        )
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
    exposure["tc_wind_class"] = pd.Categorical(
        exposure["tc_wind_class"],
        categories=TC_LABELS,
        ordered=True,
    )
    table = attrs.merge(stats, on="plant_polygon_id", how="left")
    return exposure, table


def summarize_exposure(table: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    table["is_exposed_ge_25ms"] = table["max_wind_speed_ms"] >= 25
    table["capacity_exposed_ge_25ms"] = np.where(
        table["is_exposed_ge_25ms"],
        table["capacity_mw"].fillna(0.0),
        0.0,
    )

    dataset_summary = (
        table.groupby(["dataset", "return_period"], dropna=False, as_index=False)
        .agg(
            plant_polygons=("plant_polygon_id", "nunique"),
            polygons_ge_25ms=("is_exposed_ge_25ms", "sum"),
            capacity_ge_25ms=("capacity_exposed_ge_25ms", "sum"),
            max_wind_speed_ms=("max_wind_speed_ms", "max"),
            mean_max_wind_speed_ms=("max_wind_speed_ms", "mean"),
            mean_fraction_ge_40ms=("fraction_ge_40ms", "mean"),
        )
    )
    dataset_summary["polygons_ge_25ms_percent"] = (
        dataset_summary["polygons_ge_25ms"] / dataset_summary["plant_polygons"] * 100
    )

    class_summary = (
        table.groupby(["dataset", "return_period", "tc_wind_class"], observed=False, dropna=False)
        .agg(
            plant_polygons=("plant_polygon_id", "nunique"),
            capacity_mw=("capacity_mw", "sum"),
        )
        .reset_index()
    )

    fuel_summary = (
        table[table["is_exposed_ge_25ms"]]
        .groupby(["dataset", "return_period", "primary_fuel"], observed=False, dropna=False)
        .agg(
            exposed_polygons=("plant_polygon_id", "nunique"),
            exposed_capacity_mw=("capacity_mw", "sum"),
            max_wind_speed_ms=("max_wind_speed_ms", "max"),
        )
        .reset_index()
        .sort_values(["dataset", "return_period", "exposed_capacity_mw"], ascending=[True, True, False])
    )
    return dataset_summary, class_summary, fuel_summary


def sorted_plot_labels(summary: pd.DataFrame) -> pd.DataFrame:
    out = summary.copy()
    out["plot_label"] = np.where(
        out["dataset"].eq("open_gira_historical_max"),
        "OpenGIRA\nhistorical max",
        "STORM RP" + out["return_period"].fillna(0).astype(int).astype(str),
    )
    out["sort_key"] = np.where(
        out["dataset"].eq("open_gira_historical_max"),
        -1,
        out["return_period"].fillna(0),
    )
    return out.sort_values("sort_key")


def plot_dataset_summary(dataset_summary: pd.DataFrame) -> None:
    plot_data = sorted_plot_labels(dataset_summary)
    fig, ax1 = plt.subplots(figsize=(10, 5.5))
    ax2 = ax1.twinx()

    ax1.bar(
        plot_data["plot_label"],
        plot_data["capacity_ge_25ms"],
        color="#4575b4",
        label="Capacity >=25 m/s",
    )
    ax2.plot(
        plot_data["plot_label"],
        plot_data["polygons_ge_25ms"],
        color="#d73027",
        marker="o",
        linewidth=2,
        label="Polygons >=25 m/s",
    )

    ax1.set_ylabel("Plant capacity exposed >=25 m/s (MW)")
    ax2.set_ylabel("Plant polygons exposed >=25 m/s")
    ax1.set_title("Power Plant Polygon TC Wind Exposure")
    ax1.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "powerplant_polygon_tc_exposure_by_dataset.png", dpi=220)
    plt.close(fig)


def plot_class_summary(class_summary: pd.DataFrame) -> None:
    data = class_summary.copy()
    data["plot_label"] = np.where(
        data["dataset"].eq("open_gira_historical_max"),
        "OpenGIRA\nhistorical max",
        "STORM RP" + data["return_period"].fillna(0).astype(int).astype(str),
    )
    data["sort_key"] = np.where(
        data["dataset"].eq("open_gira_historical_max"),
        -1,
        data["return_period"].fillna(0),
    )

    pivot = data.pivot_table(
        index=["sort_key", "plot_label"],
        columns="tc_wind_class",
        values="plant_polygons",
        aggfunc="sum",
        fill_value=0,
        observed=False,
    ).sort_index()
    pivot = pivot.reindex(columns=TC_LABELS, fill_value=0)

    fig, ax = plt.subplots(figsize=(11, 5.5))
    bottom = np.zeros(len(pivot))
    x = np.arange(len(pivot))
    for wind_class in TC_LABELS:
        vals = pivot[wind_class].to_numpy()
        ax.bar(
            x,
            vals,
            bottom=bottom,
            label=wind_class,
            color=TC_CLASS_COLORS[wind_class],
        )
        bottom += vals
    ax.set_xticks(x)
    ax.set_xticklabels([label for _, label in pivot.index])
    ax.set_ylabel("Power plant polygons")
    ax.set_title("Power Plant Polygons by Maximum TC Wind Class")
    ax.legend(title="Max wind speed", bbox_to_anchor=(1.02, 1), loc="upper left")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "powerplant_polygon_tc_wind_classes.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_fuel_summary(fuel_summary: pd.DataFrame) -> None:
    if fuel_summary.empty:
        return

    data = fuel_summary.copy()
    data["plot_label"] = np.where(
        data["dataset"].eq("open_gira_historical_max"),
        "OpenGIRA\nhistorical max",
        "STORM RP" + data["return_period"].fillna(0).astype(int).astype(str),
    )
    data["sort_key"] = np.where(
        data["dataset"].eq("open_gira_historical_max"),
        -1,
        data["return_period"].fillna(0),
    )
    pivot = data.pivot_table(
        index=["sort_key", "plot_label"],
        columns="primary_fuel",
        values="exposed_capacity_mw",
        aggfunc="sum",
        fill_value=0,
        observed=False,
    ).sort_index()

    fig, ax = plt.subplots(figsize=(11, 5.5))
    bottom = np.zeros(len(pivot))
    x = np.arange(len(pivot))
    for fuel in pivot.columns:
        vals = pivot[fuel].to_numpy()
        ax.bar(x, vals, bottom=bottom, label=fuel, color=FUEL_COLORS.get(fuel, "#969696"))
        bottom += vals
    ax.set_xticks(x)
    ax.set_xticklabels([label for _, label in pivot.index])
    ax.set_ylabel("Exposed capacity >=25 m/s (MW)")
    ax.set_title("TC-Wind-Exposed Power Plant Capacity by Fuel")
    ax.legend(title="Fuel", bbox_to_anchor=(1.02, 1), loc="upper left")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "powerplant_polygon_tc_exposed_capacity_by_fuel.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    print("Checking files...")
    print(POWERPLANT_POLYGONS, "exists?", POWERPLANT_POLYGONS.exists())
    print(OPEN_GIRA_HISTORICAL_MAX_RASTER, "exists?", OPEN_GIRA_HISTORICAL_MAX_RASTER.exists())
    for rp, path in STORM_RP_RASTERS.items():
        print(f"STORM RP{rp}", path, "exists?", path.exists())

    plants = load_powerplant_polygons()
    exposure, table = build_exposure_tables(plants)
    dataset_summary, class_summary, fuel_summary = summarize_exposure(table)

    exposure.to_file(
        OUTPUT_DIR / "powerplant_polygon_tc_exposure.gpkg",
        layer="powerplant_polygon_tc_exposure",
        driver="GPKG",
    )
    table.to_csv(OUTPUT_DIR / "powerplant_polygon_tc_exposure.csv", index=False)
    dataset_summary.to_csv(OUTPUT_DIR / "powerplant_polygon_tc_summary_by_dataset.csv", index=False)
    class_summary.to_csv(OUTPUT_DIR / "powerplant_polygon_tc_summary_by_class.csv", index=False)
    fuel_summary.to_csv(OUTPUT_DIR / "powerplant_polygon_tc_summary_by_fuel.csv", index=False)

    plot_dataset_summary(dataset_summary)
    plot_class_summary(class_summary)
    plot_fuel_summary(fuel_summary)

    print("\nDataset summary:")
    print(sorted_plot_labels(dataset_summary).to_string(index=False))
    print("\nTop exposed fuel rows:")
    print(fuel_summary.head(25).to_string(index=False))
    print("\nSaved outputs in:", OUTPUT_DIR)


if __name__ == "__main__":
    main()
