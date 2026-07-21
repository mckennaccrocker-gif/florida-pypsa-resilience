"""
Direct plant/substation impacts from OpenGIRA IBTrACS storms on the newest grid.

This complements the SNAIL transmission-line event damage outputs in:
    data/Exposure/florida_new_grid_tc_event_damage_snail

For polygon assets, rasterio.mask is used to clip every OpenGIRA event wind
band to the asset footprint. The output tables are also shaped so the existing
PyPSA hazard scenario runner can use them for operational/cascading runs.
"""

from __future__ import annotations

import re
import warnings
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from rasterio.mask import mask


PROJECT_DIR = Path(r"C:\oxford_tc_project")
ELECTRICITY_DIR = PROJECT_DIR / "data" / "Electricity"
COST_DIR = PROJECT_DIR / "data" / "Cost"
LINE_DAMAGE_DIR = PROJECT_DIR / "data" / "Exposure" / "florida_new_grid_tc_event_damage_snail"
OUTPUT_DIR = PROJECT_DIR / "data" / "Exposure" / "florida_new_grid_ibtracs_asset_impacts"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

WIND_RASTER = LINE_DAMAGE_DIR / "open_gira_ibtracs_florida_event_wind_stack.tif"
POWER_PLANT_POLYGONS = ELECTRICITY_DIR / "florida_osm_powerplant_polygons_with_point_attributes.gpkg"
NEW_GRID_GPKG = (
    ELECTRICITY_DIR
    / "pypsa_florida_network_extended_tie_lines"
    / "qgis"
    / "florida_extended_tie_lines_for_qgis.gpkg"
)
NEW_GRID_DIR = ELECTRICITY_DIR / "pypsa_florida_network_extended_tie_lines"
GENERATOR_TABLE = NEW_GRID_DIR / "generators_with_final_marginal_costs.csv"
BUS_TABLE = NEW_GRID_DIR / "buses_with_osm_substation_polygon_area.csv"

POWER_PLANT_WIND_CURVES = COST_DIR / "nhess_tc_generator_wind_fragility_curves_long.csv"
SUBSTATION_WIND_CURVE = COST_DIR / "nhess_w23_substation_open_area_tc_fragility_curve.csv"
REPLACEMENT_COST_LOOKUP = COST_DIR / "replacement_cost_lookup_table.csv"
LINE_EVENT_DAMAGE = LINE_DAMAGE_DIR / "new_grid_ibtracs_line_event_damage.csv"


THERMAL_FUELS = {"gas", "oil", "cogeneration", "biomass", "waste"}


def normalize_text(value) -> str:
    if pd.isna(value):
        return ""
    return re.sub(r"\s+", " ", str(value).strip().lower())


def parse_event_ids(src: rasterio.io.DatasetReader) -> list[str]:
    event_ids = []
    for band_number in range(1, src.count + 1):
        description = src.descriptions[band_number - 1]
        event_ids.append(description or f"event_{band_number:03d}")
    return event_ids


def load_curve(path: Path, curve_id: str) -> pd.DataFrame:
    curve = pd.read_csv(path)
    curve = curve[curve["curve_id"].astype(str).eq(curve_id)].copy()
    curve["wind_speed_ms"] = pd.to_numeric(curve["wind_speed_ms"], errors="coerce")
    curve["damage_ratio"] = pd.to_numeric(curve["damage_ratio"], errors="coerce")
    curve = curve.dropna(subset=["wind_speed_ms", "damage_ratio"])
    curve["damage_ratio"] = curve["damage_ratio"].clip(0.0, 1.0)
    if curve.empty:
        raise ValueError(f"No usable curve {curve_id} in {path}")
    return curve.sort_values("wind_speed_ms")


def interp_damage(wind_speed: pd.Series, curve: pd.DataFrame) -> pd.Series:
    values = pd.to_numeric(wind_speed, errors="coerce")
    ratios = np.interp(
        values.fillna(curve["wind_speed_ms"].iloc[0]).to_numpy(dtype=float),
        curve["wind_speed_ms"].to_numpy(dtype=float),
        curve["damage_ratio"].to_numpy(dtype=float),
        left=float(curve["damage_ratio"].iloc[0]),
        right=float(curve["damage_ratio"].iloc[-1]),
    )
    return pd.Series(ratios, index=wind_speed.index).where(values.notna(), 0.0).clip(0.0, 1.0)


def plant_curve_id(fuel: str) -> tuple[str, str]:
    fuel = normalize_text(fuel)
    if fuel == "coal":
        return "W1.10", "Power plant, coal"
    if fuel in THERMAL_FUELS:
        return "W1.11", "Power plant, gas/thermal fallback"
    if fuel == "nuclear":
        return "W1.12", "Power plant, nuclear"
    if fuel == "solar":
        return "W1.13", "Power plant, solar panel"
    if fuel in {"wind", "onwind", "offwind"}:
        return "W1.6", "Wind turbine"
    return "", "No matched TC wind curve"


def polygon_band_max_winds(assets: gpd.GeoDataFrame, raster_path: Path) -> pd.DataFrame:
    def centroid_sample(src: rasterio.io.DatasetReader, geom) -> np.ndarray:
        centroid = geom.centroid
        values = next(
            src.sample(
                [(centroid.x, centroid.y)],
                indexes=list(range(1, src.count + 1)),
            )
        )
        values = np.asarray(values, dtype=float)
        values[values == src.nodata] = np.nan
        return values

    records = []
    with rasterio.open(raster_path) as src:
        event_ids = parse_event_ids(src)
        assets = assets.to_crs(src.crs)
        for asset_idx, row in assets.iterrows():
            geom = row.geometry
            try:
                clipped, _ = mask(
                    src,
                    [geom],
                    crop=True,
                    filled=True,
                    nodata=np.nan,
                    indexes=list(range(1, src.count + 1)),
                )
                if clipped.ndim == 2:
                    clipped = clipped[np.newaxis, :, :]
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", category=RuntimeWarning)
                    max_wind = np.nanmax(clipped.reshape((src.count, -1)), axis=1)
                if np.all(~np.isfinite(max_wind)):
                    max_wind = centroid_sample(src, geom)
            except Exception:
                max_wind = centroid_sample(src, geom)
            max_wind = np.atleast_1d(np.asarray(max_wind, dtype=float))

            for band_number, (event_id, wind_ms) in enumerate(zip(event_ids, max_wind), start=1):
                records.append(
                    {
                        "asset_index": asset_idx,
                        "event_id": event_id,
                        "event_year": int(str(event_id)[:4]) if str(event_id)[:4].isdigit() else pd.NA,
                        "band_number": band_number,
                        "max_wind_speed_ms": wind_ms if np.isfinite(wind_ms) else np.nan,
                    }
                )
    return pd.DataFrame(records)


def load_power_plant_polygons() -> gpd.GeoDataFrame:
    plants = gpd.read_file(POWER_PLANT_POLYGONS).to_crs("EPSG:4326")
    plants = plants[plants.geometry.notna() & ~plants.geometry.is_empty].copy()
    plants["plant_asset_id"] = plants["polygon_osm_id"].astype(str)
    plants["capacity_mw"] = pd.to_numeric(plants["capacity_mw"], errors="coerce")
    plants["polygon_area_m2"] = pd.to_numeric(plants["polygon_area_m2"], errors="coerce")
    return plants.reset_index(drop=True)


def load_substation_polygons() -> gpd.GeoDataFrame:
    substations = gpd.read_file(NEW_GRID_GPKG, layer="osm_substation_polygons").to_crs("EPSG:4326")
    substations = substations[substations.geometry.notna() & ~substations.geometry.is_empty].copy()
    substations["substation_asset_id"] = substations["osm_type"].astype(str) + "/" + substations["osm_id"].astype(str)
    substations["area_m2"] = pd.to_numeric(substations["area_m2"], errors="coerce")
    return substations.reset_index(drop=True)


def direct_power_plant_impacts() -> pd.DataFrame:
    plants = load_power_plant_polygons()
    winds = polygon_band_max_winds(plants, WIND_RASTER)
    attrs = plants.drop(columns="geometry").reset_index(names="asset_index")
    impacts = winds.merge(attrs, on="asset_index", how="left")

    curves = {
        curve_id: load_curve(POWER_PLANT_WIND_CURVES, curve_id)
        for curve_id in ["W1.6", "W1.10", "W1.11", "W1.12", "W1.13"]
    }
    curve_info = impacts["primary_fuel"].apply(plant_curve_id)
    impacts["assigned_curve_id"] = [item[0] for item in curve_info]
    impacts["assigned_curve_description"] = [item[1] for item in curve_info]
    impacts["damage_ratio"] = 0.0
    for curve_id, curve in curves.items():
        mask_rows = impacts["assigned_curve_id"].eq(curve_id)
        impacts.loc[mask_rows, "damage_ratio"] = interp_damage(
            impacts.loc[mask_rows, "max_wind_speed_ms"],
            curve,
        ).to_numpy()

    impacts["capacity_loss_mw"] = impacts["capacity_mw"].fillna(0.0) * impacts["damage_ratio"]
    impacts.to_csv(OUTPUT_DIR / "direct_powerplant_polygon_event_damage.csv", index=False)
    return impacts


def direct_substation_impacts() -> tuple[pd.DataFrame, pd.DataFrame]:
    substations = load_substation_polygons()
    winds = polygon_band_max_winds(substations, WIND_RASTER)
    attrs = substations.drop(columns="geometry").reset_index(names="asset_index")
    impacts = winds.merge(attrs, on="asset_index", how="left")

    curve = load_curve(SUBSTATION_WIND_CURVE, "W2.3")
    impacts["assigned_curve_id"] = "W2.3"
    impacts["assigned_curve_description"] = "Substation, open area, moderate damage"
    impacts["damage_ratio"] = interp_damage(impacts["max_wind_speed_ms"], curve)

    buses = pd.read_csv(BUS_TABLE).set_index("name")
    cost_lookup = pd.read_csv(REPLACEMENT_COST_LOOKUP)
    cost_lookup["voltage_kv"] = pd.to_numeric(cost_lookup["voltage_kv"], errors="coerce")
    cost_lookup["substation_cost_usd"] = pd.to_numeric(cost_lookup["substation_cost_usd"], errors="coerce")
    costs = cost_lookup.set_index("voltage_kv")["substation_cost_usd"]

    bus_rows = []
    for row in impacts.itertuples(index=False):
        matched_buses = [part.strip() for part in str(getattr(row, "matched_buses", "")).split(";") if part.strip()]
        for bus in matched_buses:
            if bus not in buses.index:
                continue
            bus_v_nom = pd.to_numeric(pd.Series([buses.loc[bus, "v_nom"]]), errors="coerce").iloc[0]
            nearest_voltage = costs.index[np.argmin(np.abs(costs.index.to_numpy(dtype=float) - float(bus_v_nom)))] if pd.notna(bus_v_nom) else np.nan
            replacement_cost = float(costs.loc[nearest_voltage]) if pd.notna(nearest_voltage) else np.nan
            bus_rows.append(
                {
                    "event_id": row.event_id,
                    "event_year": row.event_year,
                    "band_number": row.band_number,
                    "bus": bus,
                    "substation_asset_id": row.substation_asset_id,
                    "osm_name": getattr(row, "osm_name", ""),
                    "osm_operator": getattr(row, "osm_operator", ""),
                    "bus_substation_name": buses.loc[bus, "substation_name"],
                    "v_nom": bus_v_nom,
                    "area_m2": getattr(row, "area_m2", np.nan),
                    "max_wind_speed_ms": row.max_wind_speed_ms,
                    "damage_ratio": row.damage_ratio,
                    "assigned_curve_id": row.assigned_curve_id,
                    "assigned_curve_description": row.assigned_curve_description,
                    "substation_replacement_cost_usd": replacement_cost,
                    "direct_substation_damage_usd": replacement_cost * row.damage_ratio if np.isfinite(replacement_cost) else np.nan,
                }
            )
    bus_impacts = pd.DataFrame(bus_rows)
    impacts["direct_substation_damage_usd_proxy"] = np.nan
    impacts.to_csv(OUTPUT_DIR / "direct_substation_polygon_event_damage.csv", index=False)
    bus_impacts.to_csv(OUTPUT_DIR / "direct_substation_bus_event_damage.csv", index=False)
    return impacts, bus_impacts


def write_asset_rankings(
    plant_impacts: pd.DataFrame,
    substation_bus_impacts: pd.DataFrame,
) -> None:
    plant_summary = (
        plant_impacts.groupby(["plant_asset_id", "name", "primary_fuel"], dropna=False)
        .agg(
            max_wind_speed_ms=("max_wind_speed_ms", "max"),
            max_damage_ratio=("damage_ratio", "max"),
            cumulative_capacity_loss_mw=("capacity_loss_mw", "sum"),
            events_with_damage=("damage_ratio", lambda s: int((s > 0).sum())),
            capacity_mw=("capacity_mw", "max"),
            polygon_area_m2=("polygon_area_m2", "max"),
            gppd_ids=("gppd_ids", "first"),
        )
        .reset_index()
        .sort_values(["cumulative_capacity_loss_mw", "max_damage_ratio", "max_wind_speed_ms"], ascending=False)
    )
    plant_summary.to_csv(OUTPUT_DIR / "top_directly_impacted_powerplants_by_all_events.csv", index=False)

    sub_summary = (
        substation_bus_impacts.groupby(["bus", "bus_substation_name", "osm_name", "osm_operator"], dropna=False)
        .agg(
            max_wind_speed_ms=("max_wind_speed_ms", "max"),
            max_damage_ratio=("damage_ratio", "max"),
            cumulative_direct_damage_usd=("direct_substation_damage_usd", "sum"),
            events_with_damage=("damage_ratio", lambda s: int((s > 0).sum())),
            v_nom=("v_nom", "max"),
            area_m2=("area_m2", "max"),
        )
        .reset_index()
        .sort_values(["cumulative_direct_damage_usd", "max_damage_ratio", "max_wind_speed_ms"], ascending=False)
    )
    sub_summary.to_csv(OUTPUT_DIR / "top_directly_impacted_substation_buses_by_all_events.csv", index=False)


def write_event_summary_and_manifests(
    plant_impacts: pd.DataFrame,
    substation_bus_impacts: pd.DataFrame,
) -> None:
    line_events = pd.read_csv(LINE_DAMAGE_DIR / "new_grid_ibtracs_event_network_damage.csv")
    plant_events = (
        plant_impacts.groupby("event_id")
        .agg(
            plant_capacity_loss_mw=("capacity_loss_mw", "sum"),
            damaged_plant_polygons=("damage_ratio", lambda s: int((s > 0).sum())),
            max_powerplant_wind_ms=("max_wind_speed_ms", "max"),
        )
        .reset_index()
    )
    sub_events = (
        substation_bus_impacts.groupby("event_id")
        .agg(
            substation_direct_damage_usd=("direct_substation_damage_usd", "sum"),
            damaged_substation_buses=("damage_ratio", lambda s: int((s > 0).sum())),
            max_substation_wind_ms=("max_wind_speed_ms", "max"),
        )
        .reset_index()
    )
    summary = (
        line_events.merge(plant_events, on="event_id", how="left")
        .merge(sub_events, on="event_id", how="left")
        .fillna(0)
    )
    summary["total_direct_damage_usd_proxy"] = (
        summary["total_network_damage_usd"]
        + summary["substation_direct_damage_usd"]
    )
    summary = summary.sort_values("total_direct_damage_usd_proxy", ascending=False)
    summary.to_csv(OUTPUT_DIR / "direct_event_damage_summary_lines_plants_substations.csv", index=False)

    manifest_rows = []
    for row in summary.itertuples(index=False):
        event_id = str(row.event_id)
        manifest_rows.append(
            {
                "scenario_id": f"ibtracs_{event_id}_tc_direct_damage",
                "hazard": "tropical_cyclone",
                "description": f"IBTrACS event {event_id}: line, generator, and substation wind deratings from OpenGIRA event raster",
                "line_damage_path": str(LINE_EVENT_DAMAGE.relative_to(PROJECT_DIR)),
                "line_filter_column": "event_id",
                "line_filter_value": event_id,
                "line_id_column": "asset_line_id",
                "line_id_type": "pypsa_line",
                "line_value_column": "line_max_wind_ms",
                "line_threshold": "",
                "line_mode": "vulnerability_curve",
                "line_damage_fraction_column": "",
                "line_curve_path": "data/Cost/nhess_w63_fpl_overhead_lines_tc_fragility_curve.csv",
                "generator_damage_path": str((OUTPUT_DIR / "direct_powerplant_polygon_event_damage.csv").relative_to(PROJECT_DIR)),
                "generator_filter_column": "event_id",
                "generator_filter_value": event_id,
                "generator_value_column": "max_wind_speed_ms",
                "generator_threshold": "",
                "generator_mode": "vulnerability_curve",
                "generator_damage_fraction_column": "",
                "generator_curve_path": "data/Cost/nhess_tc_generator_wind_fragility_curves_long.csv",
                "generator_match_column": "gppd_ids",
                "bus_damage_path": str((OUTPUT_DIR / "direct_substation_bus_event_damage.csv").relative_to(PROJECT_DIR)),
                "bus_filter_column": "event_id",
                "bus_filter_value": event_id,
                "bus_id_column": "bus",
                "bus_id_type": "pypsa_bus",
                "bus_value_column": "max_wind_speed_ms",
                "bus_mode": "vulnerability_curve",
                "bus_curve_path": "data/Cost/nhess_w23_substation_open_area_tc_fragility_curve.csv",
            }
        )
    manifest = pd.DataFrame(manifest_rows)
    manifest.to_csv(OUTPUT_DIR / "pypsa_cascading_manifest_all_ibtracs_events.csv", index=False)
    manifest.head(20).to_csv(OUTPUT_DIR / "pypsa_cascading_manifest_top20_direct_events.csv", index=False)


def main() -> None:
    print("Computing direct power plant polygon impacts...")
    plant_impacts = direct_power_plant_impacts()
    print("Computing direct substation polygon/bus impacts...")
    _substation_polygons, substation_bus_impacts = direct_substation_impacts()
    print("Writing rankings and PyPSA cascading manifests...")
    write_asset_rankings(plant_impacts, substation_bus_impacts)
    write_event_summary_and_manifests(plant_impacts, substation_bus_impacts)
    print(f"Saved outputs to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
