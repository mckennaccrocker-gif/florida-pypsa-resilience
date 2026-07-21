"""
Run calibrated Florida PyPSA hazard scenarios.

Each scenario starts from the official calibrated no-hazard baseline assumptions:
  - import slack in every load-bearing connected island
  - line capacity multiplier = 2.0
  - load shedding at every bus
  - import slack cheaper than load shedding

Scenario definitions are read from a CSV manifest. The default manifest is:
    data/Electricity/pypsa_florida_network/calibrated_hazard_scenario_manifest.csv

One output folder is written per scenario.
"""

from __future__ import annotations

import argparse
import math
import re
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd

from run_florida_pypsa_baseline_validation import (
    FINAL_GENERATORS,
    PYPSA_DIR,
    dispatch_by_carrier,
    load_latest_network,
    save_import_slack,
    save_line_loading,
    save_load_shedding,
    select_snapshots,
    solve_dispatch_in_chunks,
    write_baseline_summary,
)
from run_florida_pypsa_load_shedding_dispatch import (
    DISPATCH_TOLERANCE_MW,
    add_import_slack_generators,
    add_standard_load_shedding,
)


PROJECT_DIR = Path(r"C:\oxford_tc_project")
ELECTRICITY_DIR = PROJECT_DIR / "data" / "Electricity"
EXPOSURE_DIR = PROJECT_DIR / "data" / "Exposure"

CALIBRATED_BASELINE_DIR = PYPSA_DIR / "baseline_calibrated_no_hazard"
DEFAULT_MANIFEST = PYPSA_DIR / "calibrated_hazard_scenario_manifest.csv"
DEFAULT_OUTPUT_DIR = PYPSA_DIR / "calibrated_hazard_scenarios"
LINE_CROSSWALK_FILE = ELECTRICITY_DIR / "florida_lines_with_s_nom.csv"
FLOOD_LINE_CURVE_FILE = (
    PROJECT_DIR / "data" / "Cost" / "nhess_f62_distribution_elevated_crossings_flood_vulnerability_curve.csv"
)
FLOOD_POWER_PLANT_CURVE_FILE = (
    PROJECT_DIR / "data" / "Cost" / "nhess_f11_f14_power_plant_flood_vulnerability_curves_long.csv"
)
FLOOD_SUBSTATION_CURVE_FILE = (
    PROJECT_DIR / "data" / "Cost" / "nhess_f21_f23_substation_flood_vulnerability_curves_long.csv"
)
FLOOD_DISTRIBUTION_LINE_CURVE_FILE = (
    PROJECT_DIR / "data" / "Cost" / "nhess_f51_f61_f62_distribution_line_flood_vulnerability_curves_long.csv"
)
TC_W63_OVERHEAD_LINE_CURVE_FILE = (
    PROJECT_DIR / "data" / "Cost" / "nhess_w63_fpl_overhead_lines_tc_fragility_curve.csv"
)
TC_POWER_PLANT_CURVE_FILE = (
    PROJECT_DIR / "data" / "Cost" / "nhess_tc_generator_wind_fragility_curves_long.csv"
)
TC_W23_SUBSTATION_CURVE_FILE = (
    PROJECT_DIR / "data" / "Cost" / "nhess_w23_substation_open_area_tc_fragility_curve.csv"
)

LINE_CAPACITY_MULTIPLIER = 2.0
THERMAL_GENERATOR_CARRIERS = {
    "biomass",
    "coal",
    "cogeneration",
    "gas",
    "nuclear",
    "oil",
    "waste",
}


DEFAULT_SCENARIOS = [
    {
        "scenario_id": "tc_storm_rp100_wind_ge_40ms",
        "hazard": "tropical_cyclone",
        "description": "STORM RP100 TC wind: remove lines and generators exposed to >=40 m/s",
        "line_damage_path": "data/Exposure/lines_tropical_cyclone_exposure.gpkg",
        "line_filter_column": "",
        "line_filter_value": "",
        "line_id_column": "ID",
        "line_id_type": "hifld_id",
        "line_value_column": "tc_wind_speed_max_ms",
        "line_threshold": 40.0,
        "line_mode": "remove",
        "line_damage_fraction_column": "",
        "generator_damage_path": "data/Exposure/powerplant_polygon_tc_exposure/powerplant_polygon_tc_exposure.csv",
        "generator_filter_column": "dataset",
        "generator_filter_value": "storm_rp100",
        "generator_value_column": "max_wind_speed_ms",
        "generator_threshold": 40.0,
        "generator_mode": "remove",
        "generator_damage_fraction_column": "",
        "generator_match_column": "gppd_ids",
        "line_curve_path": "",
        "generator_curve_path": "",
    },
    {
        "scenario_id": "flood_jrc_rp100_depth_ge_1m",
        "hazard": "flood",
        "description": "JRC RP100 flood: SNAIL line/raster exposure; remove lines and generators with max depth >=1 m",
        "line_damage_path": "data/Exposure/line_flood_exposure_with_ids/flood_line_exposure_by_line_return_period.csv",
        "line_filter_column": "return_period",
        "line_filter_value": "100",
        "line_id_column": "florida_line_id",
        "line_id_type": "florida_line_id",
        "line_value_column": "max_flood_depth_m",
        "line_threshold": 1.0,
        "line_mode": "remove",
        "line_damage_fraction_column": "",
        "generator_damage_path": "data/Exposure/powerplant_polygon_flood_exposure/powerplant_polygon_flood_exposure.csv",
        "generator_filter_column": "return_period",
        "generator_filter_value": "100",
        "generator_value_column": "max_flood_depth_m",
        "generator_threshold": 1.0,
        "generator_mode": "remove",
        "generator_damage_fraction_column": "",
        "generator_match_column": "gppd_ids",
        "line_curve_path": "",
        "generator_curve_path": "",
    },
    {
        "scenario_id": "tc_storm_rp100_w63_gradual",
        "hazard": "tropical_cyclone",
        "description": "STORM RP100 TC wind: lines use NHESS W6.3 FPL overhead-line fragility curve and generators use NHESS TC generator wind fragility curves",
        "line_damage_path": "data/Exposure/lines_tropical_cyclone_exposure.gpkg",
        "line_filter_column": "",
        "line_filter_value": "",
        "line_id_column": "ID",
        "line_id_type": "hifld_id",
        "line_value_column": "tc_wind_speed_max_ms",
        "line_threshold": "",
        "line_mode": "vulnerability_curve",
        "line_damage_fraction_column": "",
        "line_curve_path": "data/Cost/nhess_w63_fpl_overhead_lines_tc_fragility_curve.csv",
        "generator_damage_path": "data/Exposure/powerplant_polygon_tc_exposure/powerplant_polygon_tc_exposure.csv",
        "generator_filter_column": "dataset",
        "generator_filter_value": "storm_rp100",
        "generator_value_column": "max_wind_speed_ms",
        "generator_threshold": "",
        "generator_mode": "vulnerability_curve",
        "generator_damage_fraction_column": "",
        "generator_curve_path": "data/Cost/nhess_tc_generator_wind_fragility_curves_long.csv",
        "generator_match_column": "gppd_ids",
    },
    {
        "scenario_id": "flood_jrc_rp100_f62_gradual",
        "hazard": "flood",
        "description": "JRC RP100 flood: SNAIL line/raster exposure; transmission lines use F6.2 and generators use NHESS F1.1-F1.4 power-plant flood vulnerability curves; F2.1-F2.3 substation curves are available when bus flood exposure is supplied",
        "line_damage_path": "data/Exposure/line_flood_exposure_with_ids/flood_line_exposure_by_line_return_period.csv",
        "line_filter_column": "return_period",
        "line_filter_value": "100",
        "line_id_column": "florida_line_id",
        "line_id_type": "florida_line_id",
        "line_value_column": "max_flood_depth_m",
        "line_threshold": "",
        "line_mode": "vulnerability_curve",
        "line_damage_fraction_column": "",
        "line_curve_path": "data/Cost/nhess_f62_distribution_elevated_crossings_flood_vulnerability_curve.csv",
        "generator_damage_path": "data/Exposure/powerplant_polygon_flood_exposure/powerplant_polygon_flood_exposure.csv",
        "generator_filter_column": "return_period",
        "generator_filter_value": "100",
        "generator_value_column": "max_flood_depth_m",
        "generator_threshold": "",
        "generator_mode": "vulnerability_curve",
        "generator_damage_fraction_column": "",
        "generator_curve_path": "data/Cost/nhess_f11_f14_power_plant_flood_vulnerability_curves_long.csv",
        "generator_match_column": "gppd_ids",
    },
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run calibrated Florida PyPSA hazard scenarios.")
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--network-dir", type=Path, default=PYPSA_DIR)
    parser.add_argument("--baseline-dir", type=Path, default=CALIBRATED_BASELINE_DIR)
    parser.add_argument("--line-capacity-multiplier", type=float, default=LINE_CAPACITY_MULTIPLIER)
    parser.add_argument("--solver", default="highs")
    parser.add_argument("--highs-method", default="ipm")
    parser.add_argument("--start", default="2025-01-01 00:00:00")
    parser.add_argument("--periods", type=int, default=24)
    parser.add_argument("--chunk-size", type=int, default=24)
    parser.add_argument("--scenario-id", action="append", default=None)
    parser.add_argument("--create-default-manifest", action="store_true")
    return parser.parse_args()


def is_blank(value) -> bool:
    return pd.isna(value) or str(value).strip() == ""


def as_float(value, default: float | None = None) -> float | None:
    if is_blank(value):
        return default
    return float(value)


def resolve_path(value) -> Path | None:
    if is_blank(value):
        return None
    path = Path(str(value))
    if not path.is_absolute():
        path = PROJECT_DIR / path
    return path


def safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_")


def normalize_text(value) -> str:
    if pd.isna(value):
        return ""
    return re.sub(r"\s+", " ", str(value).strip().lower())


def split_multi_id(value) -> list[str]:
    if pd.isna(value):
        return []
    return [part.strip() for part in re.split(r"[;,|]", str(value)) if part.strip()]


def create_default_manifest(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(DEFAULT_SCENARIOS).to_csv(path, index=False)
    print("Saved default calibrated hazard scenario manifest:", path)


def read_table(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    if path.suffix.lower() == ".gpkg":
        return gpd.read_file(path, ignore_geometry=True)
    return pd.read_csv(path)


def load_manifest(path: Path, create_if_missing: bool) -> pd.DataFrame:
    if create_if_missing or not path.exists():
        create_default_manifest(path)
    manifest = pd.read_csv(path)
    if "scenario_id" not in manifest.columns:
        raise ValueError(f"{path} must include scenario_id.")
    return manifest


def calibrated_import_table(baseline_dir: Path) -> pd.DataFrame:
    preferred = baseline_dir / "import_bus_selection.csv"
    legacy = baseline_dir / "island_import_bus_selection.csv"
    path = preferred if preferred.exists() else legacy
    if not path.exists():
        raise FileNotFoundError(path)
    imports = pd.read_csv(path)
    if "import_bus" in imports.columns:
        imports["calibrated_import_bus"] = imports["import_bus"].astype(str)
    elif "chosen_import_bus" in imports.columns:
        imports["calibrated_import_bus"] = imports["chosen_import_bus"].astype(str)
    else:
        raise ValueError(f"{path} must include import_bus or chosen_import_bus.")
    imports["calibrated_import_source_file"] = str(path)
    return imports


def calibrated_import_buses(baseline_dir: Path) -> pd.Index:
    imports = calibrated_import_table(baseline_dir)
    return pd.Index(imports["calibrated_import_bus"].dropna().astype(str).unique())


def apply_calibrated_import_caps(
    network,
    import_generators: pd.Index,
    baseline_dir: Path,
    output_dir: Path,
) -> pd.DataFrame:
    imports = calibrated_import_table(baseline_dir)
    if "import_p_nom_mw" not in imports.columns:
        imports["import_p_nom_mw"] = pd.NA
    imports["import_p_nom_mw"] = pd.to_numeric(imports["import_p_nom_mw"], errors="coerce")
    for _, row in imports.dropna(subset=["import_p_nom_mw"]).iterrows():
        bus = str(row["calibrated_import_bus"])
        generator = f"import_slack_{bus}"
        if generator in import_generators:
            network.generators.loc[generator, "p_nom"] = float(row["import_p_nom_mw"])
    imports.to_csv(output_dir / "calibrated_import_bus_selection_used.csv", index=False)
    return imports


def load_calibrated_network(network_dir: Path, line_capacity_multiplier: float):
    network = load_latest_network(network_dir)
    network.lines["s_nom"] = network.lines["s_nom"].astype(float) * line_capacity_multiplier
    return network


def filter_damage_table(table: pd.DataFrame, filter_column, filter_value) -> pd.DataFrame:
    if is_blank(filter_column):
        return table
    if str(filter_column) not in table.columns:
        raise ValueError(f"Damage table missing filter column {filter_column!r}.")
    column = str(filter_column)
    target = str(filter_value)
    numeric_target = pd.to_numeric(pd.Series([target]), errors="coerce").iloc[0]
    if not pd.isna(numeric_target):
        values = pd.to_numeric(table[column], errors="coerce")
        return table[values.eq(float(numeric_target))].copy()
    return table[table[column].astype(str).eq(target)].copy()


def filter_by_threshold(table: pd.DataFrame, value_column, threshold) -> pd.DataFrame:
    if is_blank(value_column) or threshold is None:
        return table
    column = str(value_column)
    if column not in table.columns:
        raise ValueError(f"Damage table missing value column {column!r}.")
    values = pd.to_numeric(table[column], errors="coerce")
    return table[values.ge(float(threshold))].assign(_damage_value=values).copy()


def load_vulnerability_curve(path: Path) -> tuple[pd.DataFrame, str, str]:
    if not path.exists():
        raise FileNotFoundError(path)
    curve = pd.read_csv(path)
    intensity_candidates = [
        "flood_depth_m",
        "wind_speed_ms",
        "hazard_intensity",
        "intensity",
    ]
    damage_candidates = [
        "nhess_f62_damage_ratio",
        "damage_ratio",
        "vulnerability",
    ]
    intensity_col = next((col for col in intensity_candidates if col in curve.columns), None)
    damage_col = next((col for col in damage_candidates if col in curve.columns), None)
    if intensity_col is None or damage_col is None:
        raise ValueError(
            f"Could not identify intensity/damage columns in {path}. "
            f"Columns are: {list(curve.columns)}"
        )
    curve = curve[[intensity_col, damage_col]].copy()
    curve[intensity_col] = pd.to_numeric(curve[intensity_col], errors="coerce")
    curve[damage_col] = pd.to_numeric(curve[damage_col], errors="coerce")
    curve = curve.dropna().sort_values(intensity_col).drop_duplicates(intensity_col)
    curve[damage_col] = curve[damage_col].clip(0.0, 1.0)
    if curve.empty:
        raise ValueError(f"Vulnerability curve has no valid numeric points: {path}")
    return curve, intensity_col, damage_col


def load_power_plant_flood_curves(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    curves = pd.read_csv(path)
    required = {"curve_id", "asset_description", "flood_depth_m", "damage_ratio"}
    missing = required.difference(curves.columns)
    if missing:
        raise ValueError(f"{path} missing columns: {sorted(missing)}")
    curves = curves[list(required) + [col for col in curves.columns if col not in required]].copy()
    curves["flood_depth_m"] = pd.to_numeric(curves["flood_depth_m"], errors="coerce")
    curves["damage_ratio"] = pd.to_numeric(curves["damage_ratio"], errors="coerce")
    curves = curves.dropna(subset=["curve_id", "flood_depth_m", "damage_ratio"])
    curves["damage_ratio"] = curves["damage_ratio"].clip(0.0, 1.0)
    if curves.empty:
        raise ValueError(f"Power-plant flood curves have no valid numeric points: {path}")
    return curves.sort_values(["curve_id", "flood_depth_m"])


def load_substation_flood_curves(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    curves = pd.read_csv(path)
    required = {"curve_id", "asset_description", "flood_depth_m", "damage_ratio"}
    missing = required.difference(curves.columns)
    if missing:
        raise ValueError(f"{path} missing columns: {sorted(missing)}")
    curves = curves[list(required) + [col for col in curves.columns if col not in required]].copy()
    curves["flood_depth_m"] = pd.to_numeric(curves["flood_depth_m"], errors="coerce")
    curves["damage_ratio"] = pd.to_numeric(curves["damage_ratio"], errors="coerce")
    curves = curves.dropna(subset=["curve_id", "flood_depth_m", "damage_ratio"])
    curves["damage_ratio"] = curves["damage_ratio"].clip(0.0, 1.0)
    if curves.empty:
        raise ValueError(f"Substation flood curves have no valid numeric points: {path}")
    return curves.sort_values(["curve_id", "flood_depth_m"])


def load_substation_wind_curves(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    curves = pd.read_csv(path)
    required = {"curve_id", "infrastructure_description", "wind_speed_ms", "damage_ratio"}
    missing = required.difference(curves.columns)
    if missing:
        raise ValueError(f"{path} missing columns: {sorted(missing)}")
    curves = curves.copy()
    curves["wind_speed_ms"] = pd.to_numeric(curves["wind_speed_ms"], errors="coerce")
    curves["damage_ratio"] = pd.to_numeric(curves["damage_ratio"], errors="coerce")
    curves = curves.dropna(subset=["curve_id", "wind_speed_ms", "damage_ratio"])
    curves["damage_ratio"] = curves["damage_ratio"].clip(0.0, 1.0)
    if curves.empty:
        raise ValueError(f"Substation wind curves have no valid numeric points: {path}")
    return curves.sort_values(["curve_id", "wind_speed_ms"])


def load_power_plant_wind_curves(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    curves = pd.read_csv(path)
    required = {
        "curve_id",
        "infrastructure_description",
        "additional_characteristics",
        "damage_state",
        "wind_speed_ms",
        "damage_ratio",
    }
    missing = required.difference(curves.columns)
    if missing:
        raise ValueError(f"{path} missing columns: {sorted(missing)}")
    curves = curves.copy()
    curves["wind_speed_ms"] = pd.to_numeric(curves["wind_speed_ms"], errors="coerce")
    curves["damage_ratio"] = pd.to_numeric(curves["damage_ratio"], errors="coerce")
    curves = curves.dropna(subset=["curve_id", "wind_speed_ms", "damage_ratio"])
    curves["damage_ratio"] = curves["damage_ratio"].clip(0.0, 1.0)
    if curves.empty:
        raise ValueError(f"Power-plant wind curves have no valid numeric points: {path}")
    return curves.sort_values(["curve_id", "wind_speed_ms"])


def interpolate_damage_ratio(intensity: pd.Series, curve: pd.DataFrame, intensity_col: str, damage_col: str) -> pd.Series:
    values = pd.to_numeric(intensity, errors="coerce")
    ratios = np.interp(
        values.fillna(curve[intensity_col].iloc[0]).to_numpy(dtype=float),
        curve[intensity_col].to_numpy(dtype=float),
        curve[damage_col].to_numpy(dtype=float),
        left=float(curve[damage_col].iloc[0]),
        right=float(curve[damage_col].iloc[-1]),
    )
    return pd.Series(ratios, index=intensity.index).where(values.notna(), 0.0).clip(0.0, 1.0)


def assigned_substation_flood_curve(bus_row: pd.Series) -> tuple[str, str]:
    v_nom = pd.to_numeric(pd.Series([bus_row.get("v_nom")]), errors="coerce").iloc[0]
    if pd.isna(v_nom):
        return "F2.2", "Medium Voltage Substation; missing v_nom fallback"
    if float(v_nom) < 115.0:
        return "F2.1", "Low Voltage Substation"
    if float(v_nom) < 230.0:
        return "F2.2", "Medium Voltage Substation"
    return "F2.3", "High Voltage Substation"


def interpolate_substation_flood_damage_ratios(
    buses: pd.DataFrame,
    intensity: pd.Series,
    curves: pd.DataFrame,
) -> pd.DataFrame:
    curve_lookup = {
        curve_id: group.sort_values("flood_depth_m").copy()
        for curve_id, group in curves.groupby("curve_id")
    }
    rows = []
    for bus, bus_row in buses.iterrows():
        value = pd.to_numeric(pd.Series([intensity.get(bus)]), errors="coerce").iloc[0]
        curve_id, curve_description = assigned_substation_flood_curve(bus_row)
        curve = curve_lookup.get(curve_id)
        if curve is None or curve.empty or pd.isna(value):
            damage_ratio = 0.0
        else:
            damage_ratio = float(
                np.interp(
                    float(value),
                    curve["flood_depth_m"].to_numpy(dtype=float),
                    curve["damage_ratio"].to_numpy(dtype=float),
                    left=float(curve["damage_ratio"].iloc[0]),
                    right=float(curve["damage_ratio"].iloc[-1]),
                )
            )
        rows.append(
            {
                "bus": bus,
                "hazard_intensity": value,
                "damage_ratio": max(0.0, min(1.0, damage_ratio)),
                "assigned_curve_id": curve_id,
                "assigned_curve_description": curve_description,
            }
        )
    return pd.DataFrame(rows).set_index("bus")


def interpolate_substation_wind_damage_ratios(
    buses: pd.DataFrame,
    intensity: pd.Series,
    curves: pd.DataFrame,
) -> pd.DataFrame:
    curve = curves[curves["curve_id"].astype(str).eq("W2.3")].sort_values("wind_speed_ms").copy()
    if curve.empty:
        raise ValueError("TC substation curve file does not include W2.3.")
    rows = []
    for bus, _bus_row in buses.iterrows():
        value = pd.to_numeric(pd.Series([intensity.get(bus)]), errors="coerce").iloc[0]
        if pd.isna(value):
            damage_ratio = 0.0
        else:
            damage_ratio = float(
                np.interp(
                    float(value),
                    curve["wind_speed_ms"].to_numpy(dtype=float),
                    curve["damage_ratio"].to_numpy(dtype=float),
                    left=float(curve["damage_ratio"].iloc[0]),
                    right=float(curve["damage_ratio"].iloc[-1]),
                )
            )
        rows.append(
            {
                "bus": bus,
                "hazard_intensity": value,
                "damage_ratio": max(0.0, min(1.0, damage_ratio)),
                "assigned_curve_id": "W2.3",
                "assigned_curve_description": "Substation, open area, moderate damage",
            }
        )
    return pd.DataFrame(rows).set_index("bus")


def assigned_power_plant_wind_curve(generator_row: pd.Series) -> tuple[str | None, str]:
    carrier = normalize_text(generator_row.get("carrier", ""))
    if carrier == "coal":
        return "W1.10", "Power plant, coal"
    if carrier in {"gas", "oil", "cogeneration", "biomass", "waste"}:
        return "W1.11", "Power plant, gas; thermal fallback for gas/oil/cogeneration/biomass/waste"
    if carrier == "nuclear":
        return "W1.12", "Power plant, nuclear"
    if carrier == "solar":
        return "W1.13", "Power plant, solar panel"
    if carrier in {"wind", "onwind", "offwind", "offshore wind", "landbasedwind"}:
        return "W1.6", "Wind turbine, 3.3-MW capacity, 100-m hub height"
    return None, f"No NHESS TC generator wind curve assigned for carrier {carrier!r}"


def interpolate_power_plant_wind_damage_ratios(
    generators: pd.DataFrame,
    intensity: pd.Series,
    curves: pd.DataFrame,
) -> pd.DataFrame:
    curve_lookup = {
        curve_id: group.sort_values("wind_speed_ms").copy()
        for curve_id, group in curves.groupby("curve_id")
    }
    rows = []
    for generator, generator_row in generators.iterrows():
        value = pd.to_numeric(pd.Series([intensity.get(generator)]), errors="coerce").iloc[0]
        curve_id, curve_description = assigned_power_plant_wind_curve(generator_row)
        curve = curve_lookup.get(curve_id) if curve_id is not None else None
        if curve is None or curve.empty or pd.isna(value):
            damage_ratio = 0.0
        else:
            damage_ratio = float(
                np.interp(
                    float(value),
                    curve["wind_speed_ms"].to_numpy(dtype=float),
                    curve["damage_ratio"].to_numpy(dtype=float),
                    left=float(curve["damage_ratio"].iloc[0]),
                    right=float(curve["damage_ratio"].iloc[-1]),
                )
            )
        rows.append(
            {
                "generator": generator,
                "hazard_intensity": value,
                "damage_ratio": max(0.0, min(1.0, damage_ratio)),
                "assigned_curve_id": curve_id or "",
                "assigned_curve_description": curve_description,
            }
        )
    return pd.DataFrame(rows).set_index("generator")


def assigned_power_plant_flood_curve(generator_row: pd.Series) -> tuple[str, str]:
    carrier = normalize_text(generator_row.get("carrier", ""))
    p_nom = pd.to_numeric(pd.Series([generator_row.get("p_nom")]), errors="coerce").iloc[0]

    if carrier in THERMAL_GENERATOR_CARRIERS:
        return "F1.4", "Thermal plant"
    if pd.isna(p_nom):
        return "F1.1", "Small power plants, capacity <100 MW; missing p_nom fallback"
    if float(p_nom) < 100:
        return "F1.1", "Small power plants, capacity <100 MW"
    if float(p_nom) <= 500:
        return "F1.2", "Medium power plants, capacity 100-500 MW"
    return "F1.3", "Large power plants, >500 MW"


def interpolate_power_plant_flood_damage_ratios(
    generators: pd.DataFrame,
    intensity: pd.Series,
    curves: pd.DataFrame,
) -> pd.DataFrame:
    curve_lookup = {
        curve_id: group.sort_values("flood_depth_m").copy()
        for curve_id, group in curves.groupby("curve_id")
    }
    rows = []
    for generator, generator_row in generators.iterrows():
        value = pd.to_numeric(pd.Series([intensity.get(generator)]), errors="coerce").iloc[0]
        curve_id, curve_description = assigned_power_plant_flood_curve(generator_row)
        curve = curve_lookup.get(curve_id)
        if curve is None or curve.empty or pd.isna(value):
            damage_ratio = 0.0
        else:
            damage_ratio = float(
                np.interp(
                    float(value),
                    curve["flood_depth_m"].to_numpy(dtype=float),
                    curve["damage_ratio"].to_numpy(dtype=float),
                    left=float(curve["damage_ratio"].iloc[0]),
                    right=float(curve["damage_ratio"].iloc[-1]),
                )
            )
        rows.append(
            {
                "generator": generator,
                "hazard_intensity": value,
                "damage_ratio": max(0.0, min(1.0, damage_ratio)),
                "assigned_curve_id": curve_id,
                "assigned_curve_description": curve_description,
            }
        )
    return pd.DataFrame(rows).set_index("generator")


def document_vulnerability_curve(
    scenario_id: str,
    asset_type: str,
    curve_path: Path,
    curve: pd.DataFrame,
    intensity_col: str,
    damage_col: str,
    output_dir: Path,
) -> None:
    curve_copy = output_dir / f"{asset_type}_vulnerability_curve_points.csv"
    curve.to_csv(curve_copy, index=False)
    notes_path = output_dir / f"{asset_type}_vulnerability_curve_documentation.txt"
    notes = [
        f"Scenario: {scenario_id}",
        f"Asset type: {asset_type}",
        f"Curve source file: {curve_path}",
        f"Copied curve points: {curve_copy.name}",
        f"Intensity column: {intensity_col}",
        f"Damage ratio column: {damage_col}",
        f"Number of curve points: {len(curve)}",
        f"Minimum intensity: {curve[intensity_col].min()}",
        f"Maximum intensity: {curve[intensity_col].max()}",
        f"Minimum damage ratio: {curve[damage_col].min()}",
        f"Maximum damage ratio: {curve[damage_col].max()}",
        "Interpolation method: numpy.interp linear interpolation between the exact CSV curve points.",
        "Values below/above the curve domain use the first/last CSV damage-ratio value.",
        "No curve values are estimated beyond this interpolation rule.",
    ]
    notes_path.write_text("\n".join(notes) + "\n", encoding="utf-8")


def resolve_curve_path(scenario: pd.Series, asset_type: str) -> Path:
    configured = resolve_path(scenario.get(f"{asset_type}_curve_path"))
    if configured is not None:
        return configured
    hazard = str(scenario.get("hazard", "")).strip().lower()
    if hazard == "flood":
        return FLOOD_LINE_CURVE_FILE
    if hazard in {"tc", "tropical_cyclone", "tropical cyclone"}:
        return TC_W63_OVERHEAD_LINE_CURVE_FILE
    raise ValueError(f"No vulnerability curve configured for scenario hazard {hazard!r}.")


def line_intensity_by_pypsa_line(
    network,
    table: pd.DataFrame,
    id_column: str,
    id_type: str,
    value_column: str,
) -> pd.Series:
    if id_column not in table.columns:
        raise ValueError(f"Line damage table missing ID column {id_column!r}.")
    if value_column not in table.columns:
        raise ValueError(f"Line damage table missing value column {value_column!r}.")

    id_type = str(id_type or "").strip().lower()
    working = table[[id_column, value_column]].copy()
    working[value_column] = pd.to_numeric(working[value_column], errors="coerce")
    working = working.dropna(subset=[value_column])

    if id_type in {"hifld", "hifld_id", "id"}:
        crosswalk = pd.read_csv(LINE_CROSSWALK_FILE)
        if not {"florida_line_id", "ID"}.issubset(crosswalk.columns):
            raise ValueError(f"{LINE_CROSSWALK_FILE} must include florida_line_id and ID.")
        working[id_column] = pd.to_numeric(working[id_column], errors="coerce")
        crosswalk["ID"] = pd.to_numeric(crosswalk["ID"], errors="coerce")
        crosswalk["florida_line_id"] = pd.to_numeric(crosswalk["florida_line_id"], errors="coerce")
        working = working.merge(
            crosswalk[["ID", "florida_line_id"]],
            left_on=id_column,
            right_on="ID",
            how="inner",
        )
        intensity_by_source_id = working.groupby("florida_line_id")[value_column].max()
        source_ids = pd.to_numeric(network.lines["source_edge_id"], errors="coerce")
        return source_ids.map(intensity_by_source_id)

    if id_type in {"florida_line_id", "source_edge_id", "internal"}:
        working[id_column] = pd.to_numeric(working[id_column], errors="coerce")
        intensity_by_source_id = working.groupby(id_column)[value_column].max()
        source_ids = pd.to_numeric(network.lines["source_edge_id"], errors="coerce")
        return source_ids.map(intensity_by_source_id)

    if id_type in {"pypsa_line", "line", "name"}:
        return working.groupby(working[id_column].astype(str))[value_column].max().reindex(network.lines.index)

    raise ValueError(f"Unsupported line_id_type: {id_type!r}")


def line_ids_to_pypsa_lines(network, damaged: pd.DataFrame, id_column: str, id_type: str) -> pd.Index:
    if id_column not in damaged.columns:
        raise ValueError(f"Line damage table missing ID column {id_column!r}.")
    id_type = str(id_type or "").strip().lower()
    ids = pd.to_numeric(damaged[id_column], errors="coerce").dropna()

    if id_type in {"hifld", "hifld_id", "id"}:
        crosswalk = pd.read_csv(LINE_CROSSWALK_FILE)
        if not {"florida_line_id", "ID"}.issubset(crosswalk.columns):
            raise ValueError(f"{LINE_CROSSWALK_FILE} must include florida_line_id and ID.")
        matched_internal = crosswalk[pd.to_numeric(crosswalk["ID"], errors="coerce").isin(ids)]["florida_line_id"]
        source_ids = pd.to_numeric(network.lines["source_edge_id"], errors="coerce")
        return pd.Index(network.lines.index[source_ids.isin(matched_internal)])

    if id_type in {"florida_line_id", "source_edge_id", "internal"}:
        source_ids = pd.to_numeric(network.lines["source_edge_id"], errors="coerce")
        return pd.Index(network.lines.index[source_ids.isin(ids)])

    if id_type in {"pypsa_line", "line", "name"}:
        return pd.Index(damaged[id_column].astype(str)).intersection(network.lines.index)

    raise ValueError(f"Unsupported line_id_type: {id_type!r}")


def bus_intensity_by_pypsa_bus(
    network,
    table: pd.DataFrame,
    id_column: str,
    id_type: str,
    value_column: str,
) -> pd.Series:
    if id_column not in table.columns:
        raise ValueError(f"Bus damage table missing ID column {id_column!r}.")
    if value_column not in table.columns:
        raise ValueError(f"Bus damage table missing value column {value_column!r}.")

    id_type = str(id_type or "").strip().lower()
    working = table[[id_column, value_column]].copy()
    working[value_column] = pd.to_numeric(working[value_column], errors="coerce")
    working = working.dropna(subset=[value_column])
    intensity = pd.Series(index=network.buses.index, dtype=float)

    if id_type in {"pypsa_bus", "bus", "name"}:
        by_bus = working.groupby(working[id_column].astype(str))[value_column].max()
        return by_bus.reindex(network.buses.index)

    if id_type in {"source_node_id", "node_id", "hifld_node_id"}:
        if "source_node_id" not in network.buses.columns:
            raise ValueError("network.buses is missing source_node_id for bus flood matching.")
        working[id_column] = pd.to_numeric(working[id_column], errors="coerce")
        by_source_node = working.groupby(id_column)[value_column].max()
        source_node_ids = pd.to_numeric(network.buses["source_node_id"], errors="coerce")
        return source_node_ids.map(by_source_node)

    if id_type in {"substation_name", "station_name"}:
        if "substation_name" not in network.buses.columns:
            raise ValueError("network.buses is missing substation_name for bus flood matching.")
        by_name = working.assign(_match=working[id_column].map(normalize_text)).groupby("_match")[value_column].max()
        return network.buses["substation_name"].map(normalize_text).map(by_name)

    raise ValueError(f"Unsupported bus_id_type: {id_type!r}")


def apply_bus_substation_damage(network, scenario: pd.Series, output_dir: Path) -> pd.DataFrame:
    path = resolve_path(scenario.get("bus_damage_path"))
    if path is None:
        pd.DataFrame().to_csv(output_dir / "bus_substation_deratings.csv", index=False)
        return pd.DataFrame()

    table = read_table(path)
    table = filter_damage_table(
        table,
        scenario.get("bus_filter_column"),
        scenario.get("bus_filter_value"),
    )
    mode = str(scenario.get("bus_mode", "vulnerability_curve")).strip().lower()
    if mode != "vulnerability_curve":
        raise ValueError("bus_mode currently supports only vulnerability_curve.")

    value_column = str(scenario.get("bus_value_column"))
    hazard = str(scenario.get("hazard", "")).strip().lower()
    if hazard == "flood":
        curve_path = resolve_path(scenario.get("bus_curve_path")) or FLOOD_SUBSTATION_CURVE_FILE
        curves = load_substation_flood_curves(curve_path)
        assignments = None
        curve_intensity_column = "flood_depth_m"
        curve_family_note = "NHESS F2.1-F2.3 substation flood vulnerability curves from user-provided Notes  (1).pdf"
        assignment_notes = [
            "- v_nom <115 kV -> F2.1 Low Voltage Substation",
            "- 115 <= v_nom <230 kV -> F2.2 Medium Voltage Substation",
            "- v_nom >=230 kV -> F2.3 High Voltage Substation",
        ]
    elif hazard in {"tc", "tropical_cyclone", "tropical cyclone"}:
        curve_path = resolve_path(scenario.get("bus_curve_path")) or TC_W23_SUBSTATION_CURVE_FILE
        curves = load_substation_wind_curves(curve_path)
        assignments = None
        curve_intensity_column = "wind_speed_ms"
        curve_family_note = "NHESS W2.3 substation open-area tropical-cyclone wind fragility curve from user-provided Notes  (6).pdf"
        assignment_notes = [
            "- all PyPSA buses/substations -> W2.3 Substation, open area, moderate damage",
        ]
    else:
        raise ValueError(f"Unsupported hazard for bus/substation damage: {hazard!r}")
    curves.to_csv(output_dir / "bus_vulnerability_curve_points.csv", index=False)

    intensity = bus_intensity_by_pypsa_bus(
        network,
        table,
        str(scenario.get("bus_id_column")),
        str(scenario.get("bus_id_type")),
        value_column,
    )
    if hazard == "flood":
        assignments = interpolate_substation_flood_damage_ratios(network.buses, intensity, curves)
    else:
        assignments = interpolate_substation_wind_damage_ratios(network.buses, intensity, curves)
    damage_ratio = assignments["damage_ratio"].reindex(network.buses.index).fillna(0.0)
    bus_ids = pd.Index(damage_ratio.index[damage_ratio.gt(0.0)])
    if len(bus_ids) == 0:
        pd.DataFrame().to_csv(output_dir / "bus_substation_deratings.csv", index=False)
        return pd.DataFrame()

    line_endpoint_damage = pd.concat(
        [
            network.lines["bus0"].map(damage_ratio).fillna(0.0),
            network.lines["bus1"].map(damage_ratio).fillna(0.0),
        ],
        axis=1,
    ).max(axis=1)
    line_ids = pd.Index(line_endpoint_damage.index[line_endpoint_damage.gt(0.0)])
    original_line_s_nom = network.lines.loc[line_ids, "s_nom"].astype(float)
    reduced_line_s_nom = original_line_s_nom * (1.0 - line_endpoint_damage.loc[line_ids])
    network.lines.loc[line_ids, "s_nom"] = reduced_line_s_nom.to_numpy()

    generator_bus_damage = network.generators["bus"].map(damage_ratio).fillna(0.0)
    generator_ids = pd.Index(generator_bus_damage.index[generator_bus_damage.gt(0.0)])
    generator_ids = generator_ids.difference(
        network.generators.index[network.generators["carrier"].isin(["import_slack", "load_shedding"])]
    )
    original_generator_p_nom = network.generators.loc[generator_ids, "p_nom"].astype(float)
    reduced_generator_p_nom = original_generator_p_nom * (1.0 - generator_bus_damage.loc[generator_ids])
    network.generators.loc[generator_ids, "p_nom"] = reduced_generator_p_nom.to_numpy()

    report = network.buses.loc[bus_ids, ["v_nom", "x", "y", "substation_name", "source_node_id"]].copy()
    report.insert(0, "bus", report.index)
    report["damage_mode"] = mode
    report["curve_file"] = str(curve_path)
    report["curve_intensity_column"] = curve_intensity_column
    report["curve_damage_ratio_column"] = "damage_ratio"
    report["hazard_intensity"] = assignments["hazard_intensity"].reindex(bus_ids).to_numpy()
    report["hazard_intensity_column"] = value_column
    report["damage_ratio"] = damage_ratio.loc[bus_ids].to_numpy()
    report["assigned_curve_id"] = assignments["assigned_curve_id"].reindex(bus_ids).to_numpy()
    report["assigned_curve_description"] = assignments["assigned_curve_description"].reindex(bus_ids).to_numpy()
    report["incident_lines_derated"] = report["bus"].map(
        lambda bus: int(((network.lines["bus0"] == bus) | (network.lines["bus1"] == bus)).sum())
    )
    report["connected_generators_derated"] = report["bus"].map(
        lambda bus: int((network.generators["bus"] == bus).sum())
    )
    report["scenario_id"] = scenario["scenario_id"]
    report.to_csv(output_dir / "bus_substation_deratings.csv", index=False)

    line_report = pd.DataFrame(
        {
            "line": line_ids,
            "substation_endpoint_damage_ratio": line_endpoint_damage.loc[line_ids].to_numpy(),
            "s_nom_before_substation_derating_mva": original_line_s_nom.to_numpy(),
            "s_nom_after_substation_derating_mva": reduced_line_s_nom.to_numpy(),
            "capacity_loss_from_substation_derating_mva": (
                original_line_s_nom - reduced_line_s_nom
            ).to_numpy(),
            "scenario_id": scenario["scenario_id"],
        }
    )
    line_report.to_csv(output_dir / "line_substation_dependency_deratings.csv", index=False)

    generator_report = pd.DataFrame(
        {
            "generator": generator_ids,
            "bus_substation_damage_ratio": generator_bus_damage.loc[generator_ids].to_numpy(),
            "p_nom_before_substation_derating_mw": original_generator_p_nom.to_numpy(),
            "p_nom_after_substation_derating_mw": reduced_generator_p_nom.to_numpy(),
            "capacity_loss_from_substation_derating_mw": (
                original_generator_p_nom - reduced_generator_p_nom
            ).to_numpy(),
            "scenario_id": scenario["scenario_id"],
        }
    )
    generator_report.to_csv(output_dir / "generator_substation_dependency_deratings.csv", index=False)

    notes = [
        f"Scenario: {scenario['scenario_id']}",
        "Asset type: PyPSA buses/substations",
        f"Curve source file: {curve_path}",
        f"Curve family: {curve_family_note}",
        "Assignment rule:",
        *assignment_notes,
        "Application rule:",
        "- bus/substation damage additionally derates connected generators and incident lines",
        "- incident lines use the larger damage ratio of their two endpoint buses",
        "- this is a substation-dependency derating and is reported separately from direct line/generator hazard derating",
        "Interpolation method: numpy.interp linear interpolation within each assigned curve.",
        "Values below/above each curve domain use the first/last CSV damage-ratio value.",
        "No curve values are estimated beyond this interpolation rule.",
    ]
    (output_dir / "bus_vulnerability_curve_documentation.txt").write_text(
        "\n".join(notes) + "\n",
        encoding="utf-8",
    )
    return report


def apply_line_damage(network, scenario: pd.Series, output_dir: Path) -> pd.DataFrame:
    path = resolve_path(scenario.get("line_damage_path"))
    if path is None:
        report = pd.DataFrame(
            [{"warning": "No line_damage_path configured; no line damage applied."}]
        )
        report.to_csv(output_dir / "damaged_lines.csv", index=False)
        return pd.DataFrame()

    table = read_table(path)
    table = filter_damage_table(
        table,
        scenario.get("line_filter_column"),
        scenario.get("line_filter_value"),
    )
    mode = str(scenario.get("line_mode", "remove")).strip().lower()

    if mode == "vulnerability_curve":
        value_column = str(scenario.get("line_value_column"))
        curve_path = resolve_curve_path(scenario, "line")
        curve, curve_intensity_col, curve_damage_col = load_vulnerability_curve(curve_path)
        document_vulnerability_curve(
            str(scenario["scenario_id"]),
            "line",
            curve_path,
            curve,
            curve_intensity_col,
            curve_damage_col,
            output_dir,
        )
        intensity = line_intensity_by_pypsa_line(
            network,
            table,
            str(scenario.get("line_id_column")),
            str(scenario.get("line_id_type")),
            value_column,
        )
        damage_ratio = interpolate_damage_ratio(
            intensity,
            curve,
            curve_intensity_col,
            curve_damage_col,
        )
        line_ids = pd.Index(damage_ratio.index[damage_ratio.gt(0.0)])
        if len(line_ids) == 0:
            pd.DataFrame().to_csv(output_dir / "damaged_lines.csv", index=False)
            return pd.DataFrame()

        original_s_nom = network.lines.loc[line_ids, "s_nom"].astype(float)
        reduced_s_nom = original_s_nom * (1.0 - damage_ratio.loc[line_ids])
        network.lines.loc[line_ids, "s_nom"] = reduced_s_nom.to_numpy()

        report = network.lines.loc[
            line_ids, ["bus0", "bus1", "v_nom", "s_nom", "length", "source_edge_id"]
        ].copy()
        report = report.rename(columns={"s_nom": "reduced_s_nom_mva"})
        report.insert(0, "line", report.index)
        report["damage_mode"] = mode
        report["curve_file"] = str(curve_path)
        report["curve_intensity_column"] = curve_intensity_col
        report["curve_damage_ratio_column"] = curve_damage_col
        report["hazard_intensity"] = intensity.loc[line_ids].to_numpy()
        report["hazard_intensity_column"] = value_column
        report["damage_ratio"] = damage_ratio.loc[line_ids].to_numpy()
        report["original_s_nom_mva"] = original_s_nom.to_numpy()
        report["reduced_s_nom_mva"] = reduced_s_nom.to_numpy()
        report["capacity_loss_mva"] = report["original_s_nom_mva"] - report["reduced_s_nom_mva"]
        report["capacity_multiplier_after_damage"] = 1.0 - report["damage_ratio"]
        report["scenario_id"] = scenario["scenario_id"]
        report.to_csv(output_dir / "damaged_lines.csv", index=False)
        report.to_csv(output_dir / "line_capacity_deratings.csv", index=False)
        return report

    table = filter_by_threshold(
        table,
        scenario.get("line_value_column"),
        as_float(scenario.get("line_threshold")),
    )
    if table.empty:
        pd.DataFrame().to_csv(output_dir / "damaged_lines.csv", index=False)
        return pd.DataFrame()

    line_ids = line_ids_to_pypsa_lines(
        network,
        table,
        str(scenario.get("line_id_column")),
        str(scenario.get("line_id_type")),
    )

    report = network.lines.loc[line_ids, ["bus0", "bus1", "v_nom", "s_nom", "length", "source_edge_id"]].copy()
    report = report.reset_index(names="line")
    report["damage_mode"] = mode

    if mode == "remove":
        network.lines.loc[line_ids, "active"] = False
        report["capacity_multiplier_after_damage"] = 0.0
    elif mode in {"reduce", "reduce_fraction"}:
        frac_col = str(scenario.get("line_damage_fraction_column", "")).strip()
        if not frac_col or frac_col not in table.columns:
            raise ValueError("line_mode=reduce requires line_damage_fraction_column.")
        fraction_by_id = pd.to_numeric(table[frac_col], errors="coerce").clip(0, 1)
        damaged = table.assign(_damage_fraction=fraction_by_id)
        # Conservative default for duplicate segments/records: use max damage fraction per line id.
        id_column = str(scenario.get("line_id_column"))
        damage_by_id = damaged.groupby(id_column)["_damage_fraction"].max()
        source_ids = pd.to_numeric(network.lines.loc[line_ids, "source_edge_id"], errors="coerce")
        multipliers = 1.0 - source_ids.map(damage_by_id).fillna(0.0).to_numpy()
        network.lines.loc[line_ids, "s_nom"] = network.lines.loc[line_ids, "s_nom"].to_numpy() * multipliers
        report["capacity_multiplier_after_damage"] = multipliers
    else:
        raise ValueError(f"Unsupported line_mode: {mode!r}")

    report["scenario_id"] = scenario["scenario_id"]
    report.to_csv(output_dir / "damaged_lines.csv", index=False)
    return report


def damaged_generator_names(network, damaged: pd.DataFrame, match_column: str) -> pd.Index:
    match_column = (match_column or "gppd_ids").strip()
    generators = network.generators.copy()
    original = generators[
        ~generators["carrier"].isin(["import_slack", "load_shedding"])
    ].copy()

    matched: set[str] = set()
    if match_column == "gppd_ids" and "gppd_ids" in damaged.columns and "gppd_idnr" in original.columns:
        damage_ids = {
            item
            for value in damaged["gppd_ids"]
            for item in split_multi_id(value)
        }
        matched.update(original.index[original["gppd_idnr"].astype(str).isin(damage_ids)])

    if match_column in damaged.columns:
        damage_names = {normalize_text(value) for value in damaged[match_column]}
        for col in ["name", "source_name"]:
            if col in original.columns:
                matched.update(original.index[original[col].map(normalize_text).isin(damage_names)])

    if "name" in damaged.columns:
        damage_names = {normalize_text(value) for value in damaged["name"]}
        for col in ["source_name", "name"]:
            if col in original.columns:
                matched.update(original.index[original[col].map(normalize_text).isin(damage_names)])

    return pd.Index(sorted(matched))


def generator_intensity_by_pypsa_generator(
    network,
    table: pd.DataFrame,
    match_column: str,
    value_column: str,
) -> pd.Series:
    if value_column not in table.columns:
        raise ValueError(f"Generator damage table missing value column {value_column!r}.")

    match_column = (match_column or "gppd_ids").strip()
    working = table.copy()
    working[value_column] = pd.to_numeric(working[value_column], errors="coerce")
    working = working.dropna(subset=[value_column])

    generators = network.generators.copy()
    original = generators[
        ~generators["carrier"].isin(["import_slack", "load_shedding"])
    ].copy()
    intensity = pd.Series(index=original.index, dtype=float)

    if match_column == "gppd_ids" and "gppd_ids" in working.columns and "gppd_idnr" in original.columns:
        rows = []
        for _, row in working.iterrows():
            for gppd_id in split_multi_id(row["gppd_ids"]):
                rows.append({"gppd_idnr": gppd_id, value_column: row[value_column]})
        if rows:
            by_gppd = pd.DataFrame(rows).groupby("gppd_idnr")[value_column].max()
            mapped = original["gppd_idnr"].astype(str).map(by_gppd)
            intensity = intensity.combine_first(mapped)

    if match_column in working.columns:
        by_name = working.assign(_match=working[match_column].map(normalize_text)).groupby("_match")[value_column].max()
        for col in ["name", "source_name"]:
            if col in original.columns:
                mapped = original[col].map(normalize_text).map(by_name)
                intensity = intensity.combine_first(mapped)

    if "name" in working.columns:
        by_name = working.assign(_match=working["name"].map(normalize_text)).groupby("_match")[value_column].max()
        for col in ["source_name", "name"]:
            if col in original.columns:
                mapped = original[col].map(normalize_text).map(by_name)
                intensity = intensity.combine_first(mapped)

    return intensity.reindex(network.generators.index)


def apply_generator_damage(network, scenario: pd.Series, output_dir: Path) -> pd.DataFrame:
    path = resolve_path(scenario.get("generator_damage_path"))
    if path is None:
        report = pd.DataFrame(
            [{"warning": "No generator_damage_path configured; no generator damage applied."}]
        )
        report.to_csv(output_dir / "damaged_generators.csv", index=False)
        return pd.DataFrame()

    table = read_table(path)
    table = filter_damage_table(
        table,
        scenario.get("generator_filter_column"),
        scenario.get("generator_filter_value"),
    )
    mode = str(scenario.get("generator_mode", "remove")).strip().lower()

    if mode == "vulnerability_curve":
        value_column = str(scenario.get("generator_value_column"))
        curve_path = resolve_curve_path(scenario, "generator")
        intensity = generator_intensity_by_pypsa_generator(
            network,
            table,
            str(scenario.get("generator_match_column", "gppd_ids")),
            value_column,
        )
        hazard = str(scenario.get("hazard", "")).strip().lower()
        using_power_plant_flood_curves = hazard == "flood" and curve_path.name == FLOOD_POWER_PLANT_CURVE_FILE.name
        using_power_plant_wind_curves = hazard in {
            "tc",
            "tropical_cyclone",
            "tropical cyclone",
        } and curve_path.name == TC_POWER_PLANT_CURVE_FILE.name

        curve_intensity_col = "flood_depth_m" if using_power_plant_flood_curves else ""
        curve_damage_col = "damage_ratio" if using_power_plant_flood_curves else ""
        if using_power_plant_wind_curves:
            curve_intensity_col = "wind_speed_ms"
            curve_damage_col = "damage_ratio"
        assignments = pd.DataFrame(index=network.generators.index)

        if using_power_plant_flood_curves:
            curves = load_power_plant_flood_curves(curve_path)
            curves.to_csv(output_dir / "generator_vulnerability_curve_points.csv", index=False)
            original_generators = network.generators[
                ~network.generators["carrier"].isin(["import_slack", "load_shedding"])
            ].copy()
            assignments = interpolate_power_plant_flood_damage_ratios(
                original_generators,
                intensity,
                curves,
            )
            damage_ratio = assignments["damage_ratio"].reindex(network.generators.index).fillna(0.0)
            notes = [
                f"Scenario: {scenario['scenario_id']}",
                "Asset type: generator",
                f"Curve source file: {curve_path}",
                "Copied curve points: generator_vulnerability_curve_points.csv",
                "Curve family: NHESS F1.1-F1.4 power-plant flood vulnerability curves from user-provided Notes .pdf",
                "Assignment rule:",
                "- thermal carriers (gas, coal, oil, nuclear, biomass, waste, cogeneration) -> F1.4 thermal plant",
                "- non-thermal p_nom <100 MW -> F1.1 small power plant",
                "- non-thermal p_nom 100-500 MW -> F1.2 medium power plant",
                "- non-thermal p_nom >500 MW -> F1.3 large power plant",
                "Intensity column: flood_depth_m",
                "Damage ratio column: damage_ratio",
                "Interpolation method: numpy.interp linear interpolation within each assigned curve.",
                "Values below/above each curve domain use the first/last CSV damage-ratio value.",
                "No curve values are estimated beyond this interpolation rule.",
            ]
            (output_dir / "generator_vulnerability_curve_documentation.txt").write_text(
                "\n".join(notes) + "\n",
                encoding="utf-8",
            )
        elif using_power_plant_wind_curves:
            curves = load_power_plant_wind_curves(curve_path)
            curves.to_csv(output_dir / "generator_vulnerability_curve_points.csv", index=False)
            original_generators = network.generators[
                ~network.generators["carrier"].isin(["import_slack", "load_shedding"])
            ].copy()
            assignments = interpolate_power_plant_wind_damage_ratios(
                original_generators,
                intensity,
                curves,
            )
            damage_ratio = assignments["damage_ratio"].reindex(network.generators.index).fillna(0.0)
            notes = [
                f"Scenario: {scenario['scenario_id']}",
                "Asset type: generator",
                f"Curve source file: {curve_path}",
                "Copied curve points: generator_vulnerability_curve_points.csv",
                "Curve family: NHESS TC generator wind fragility curves: W1.10-W1.13 from user-provided Notes  (5).pdf plus W1.6 from user-provided Notes  (4).pdf.",
                "Assignment rule:",
                "- coal -> W1.10 power plant coal",
                "- gas/oil/cogeneration/biomass/waste -> W1.11 power plant gas thermal fallback",
                "- nuclear -> W1.12 power plant nuclear",
                "- solar -> W1.13 power plant solar panel",
                "- wind -> W1.6 wind turbine, 3.3-MW capacity, 100-m hub height",
                "- carriers without a matching TC generator curve receive damage_ratio=0 and are explicitly labelled in the asset report",
                "Intensity column: wind_speed_ms",
                "Damage ratio column: damage_ratio",
                "Interpretation: fragility probability is used as the generator derating damage ratio for this first-pass operational scenario.",
                "Interpolation method: numpy.interp linear interpolation within each assigned curve.",
                "Values below/above each curve domain use the first/last CSV damage-ratio value.",
                "No curve values are estimated beyond this interpolation rule.",
            ]
            (output_dir / "generator_vulnerability_curve_documentation.txt").write_text(
                "\n".join(notes) + "\n",
                encoding="utf-8",
            )
        else:
            curve, curve_intensity_col, curve_damage_col = load_vulnerability_curve(curve_path)
            document_vulnerability_curve(
                str(scenario["scenario_id"]),
                "generator",
                curve_path,
                curve,
                curve_intensity_col,
                curve_damage_col,
                output_dir,
            )
            damage_ratio = interpolate_damage_ratio(
                intensity,
                curve,
                curve_intensity_col,
                curve_damage_col,
            )
        generator_ids = pd.Index(damage_ratio.index[damage_ratio.gt(0.0)])
        if len(generator_ids) == 0:
            pd.DataFrame().to_csv(output_dir / "damaged_generators.csv", index=False)
            return pd.DataFrame()

        original_p_nom = network.generators.loc[generator_ids, "p_nom"].astype(float)
        reduced_p_nom = original_p_nom * (1.0 - damage_ratio.loc[generator_ids])
        network.generators.loc[generator_ids, "p_nom"] = reduced_p_nom.to_numpy()

        report = network.generators.loc[
            generator_ids,
            ["bus", "carrier", "p_nom", "marginal_cost", "source_name"],
        ].copy()
        report = report.rename(columns={"p_nom": "reduced_p_nom_mw"})
        report.insert(0, "generator", report.index)
        report["damage_mode"] = mode
        report["curve_file"] = str(curve_path)
        report["curve_intensity_column"] = curve_intensity_col
        report["curve_damage_ratio_column"] = curve_damage_col
        if using_power_plant_flood_curves or using_power_plant_wind_curves:
            report["assigned_curve_id"] = assignments["assigned_curve_id"].reindex(generator_ids).to_numpy()
            report["assigned_curve_description"] = assignments["assigned_curve_description"].reindex(generator_ids).to_numpy()
        report["hazard_intensity"] = intensity.loc[generator_ids].to_numpy()
        report["hazard_intensity_column"] = value_column
        report["damage_ratio"] = damage_ratio.loc[generator_ids].to_numpy()
        report["original_p_nom_mw"] = original_p_nom.to_numpy()
        report["reduced_p_nom_mw"] = reduced_p_nom.to_numpy()
        report["capacity_loss_mw"] = report["original_p_nom_mw"] - report["reduced_p_nom_mw"]
        report["capacity_multiplier_after_damage"] = 1.0 - report["damage_ratio"]
        report["scenario_id"] = scenario["scenario_id"]
        report.to_csv(output_dir / "damaged_generators.csv", index=False)
        report.to_csv(output_dir / "generator_capacity_deratings.csv", index=False)
        return report

    table = filter_by_threshold(
        table,
        scenario.get("generator_value_column"),
        as_float(scenario.get("generator_threshold")),
    )
    if table.empty:
        pd.DataFrame().to_csv(output_dir / "damaged_generators.csv", index=False)
        return pd.DataFrame()

    generator_ids = damaged_generator_names(
        network,
        table,
        str(scenario.get("generator_match_column", "gppd_ids")),
    )

    report = network.generators.loc[
        generator_ids,
        ["bus", "carrier", "p_nom", "marginal_cost", "source_name"],
    ].copy()
    report = report.reset_index(names="generator")
    report["damage_mode"] = mode

    if mode == "remove":
        network.generators.loc[generator_ids, "p_nom"] = 0.0
        report["p_nom_after_damage"] = 0.0
    elif mode in {"reduce", "reduce_fraction"}:
        frac_col = str(scenario.get("generator_damage_fraction_column", "")).strip()
        if not frac_col or frac_col not in table.columns:
            raise ValueError("generator_mode=reduce requires generator_damage_fraction_column.")
        # If multiple polygon records match one generator, use max damage fraction.
        damage_fraction = pd.to_numeric(table[frac_col], errors="coerce").max()
        if math.isnan(float(damage_fraction)):
            damage_fraction = 0.0
        multiplier = max(0.0, 1.0 - min(1.0, float(damage_fraction)))
        network.generators.loc[generator_ids, "p_nom"] = network.generators.loc[generator_ids, "p_nom"] * multiplier
        report["p_nom_after_damage"] = report["p_nom"] * multiplier
    else:
        raise ValueError(f"Unsupported generator_mode: {mode!r}")

    report["scenario_id"] = scenario["scenario_id"]
    report.to_csv(output_dir / "damaged_generators.csv", index=False)
    return report


def system_cost_from_dispatch(network, snapshots: pd.Index) -> float:
    dispatch = network.generators_t.p.reindex(snapshots).clip(lower=0.0)
    costs = network.generators["marginal_cost"].reindex(dispatch.columns).fillna(0.0)
    weights = pd.Series(1.0, index=snapshots)
    if hasattr(network, "snapshot_weightings") and "generators" in network.snapshot_weightings:
        weights = network.snapshot_weightings["generators"].reindex(snapshots).fillna(1.0)
    return float(dispatch.multiply(costs, axis=1).multiply(weights, axis=0).sum().sum())


def write_incremental_summary(
    scenario_id: str,
    scenario_summary: pd.Series,
    baseline_summary: pd.Series,
    damaged_lines: pd.DataFrame,
    damaged_generators: pd.DataFrame,
    output_dir: Path,
) -> pd.DataFrame:
    baseline_load_shed = float(baseline_summary.get("total_load_shed_mwh", 0.0))
    scenario_load_shed = float(scenario_summary.get("total_load_shed_mwh", 0.0))
    baseline_import = float(baseline_summary.get("total_import_slack_mwh", 0.0))
    scenario_import = float(scenario_summary.get("total_import_slack_mwh", 0.0))
    baseline_cost = pd.to_numeric(pd.Series([baseline_summary.get("total_system_cost_usd", pd.NA)]), errors="coerce").iloc[0]
    scenario_cost = float(scenario_summary.get("total_system_cost_usd", 0.0))
    incremental_cost = scenario_cost - float(baseline_cost) if pd.notna(baseline_cost) else pd.NA
    damaged_line_capacity = 0.0
    line_capacity_loss = 0.0
    if "original_s_nom_mva" in damaged_lines:
        damaged_line_capacity = float(damaged_lines["original_s_nom_mva"].sum())
    elif "s_nom" in damaged_lines:
        damaged_line_capacity = float(damaged_lines["s_nom"].sum())
    if "capacity_loss_mva" in damaged_lines:
        line_capacity_loss = float(damaged_lines["capacity_loss_mva"].sum())
    elif "s_nom" in damaged_lines:
        line_capacity_loss = float(damaged_lines["s_nom"].sum())

    damaged_generator_capacity = 0.0
    generator_capacity_loss = 0.0
    if "original_p_nom_mw" in damaged_generators:
        damaged_generator_capacity = float(damaged_generators["original_p_nom_mw"].sum())
    elif "p_nom" in damaged_generators:
        damaged_generator_capacity = float(damaged_generators["p_nom"].sum())
    if "capacity_loss_mw" in damaged_generators:
        generator_capacity_loss = float(damaged_generators["capacity_loss_mw"].sum())
    elif "p_nom" in damaged_generators:
        generator_capacity_loss = float(damaged_generators["p_nom"].sum())

    incremental = pd.DataFrame(
        [
            {
                "scenario_id": scenario_id,
                "damaged_lines": len(damaged_lines),
                "damaged_generators": len(damaged_generators),
                "damaged_line_capacity_mva": damaged_line_capacity,
                "line_capacity_loss_mva": line_capacity_loss,
                "damaged_generator_capacity_mw": damaged_generator_capacity,
                "generator_capacity_loss_mw": generator_capacity_loss,
                "baseline_load_shed_mwh": baseline_load_shed,
                "scenario_load_shed_mwh": scenario_load_shed,
                "incremental_load_shed_mwh": scenario_load_shed - baseline_load_shed,
                "baseline_import_slack_mwh": baseline_import,
                "scenario_import_slack_mwh": scenario_import,
                "incremental_import_slack_mwh": scenario_import - baseline_import,
                "baseline_system_cost_usd": baseline_cost,
                "scenario_system_cost_usd": scenario_cost,
                "incremental_system_cost_usd": incremental_cost,
            }
        ]
    )
    incremental.to_csv(output_dir / "incremental_vs_calibrated_baseline.csv", index=False)
    return incremental


def run_scenario(scenario: pd.Series, args: argparse.Namespace) -> pd.Series:
    scenario_id = str(scenario["scenario_id"])
    output_dir = args.output_dir / safe_name(scenario_id)
    output_dir.mkdir(parents=True, exist_ok=True)

    network = load_calibrated_network(args.network_dir, args.line_capacity_multiplier)
    snapshots = select_snapshots(network, args.start, args.periods, all_snapshots=False)

    damaged_lines = apply_line_damage(network, scenario, output_dir)
    damaged_generators = apply_generator_damage(network, scenario, output_dir)
    damaged_buses = apply_bus_substation_damage(network, scenario, output_dir)

    import_buses = calibrated_import_buses(args.baseline_dir)
    missing_import_buses = import_buses.difference(network.buses.index)
    if len(missing_import_buses) > 0:
        raise ValueError(f"Calibrated import buses missing from network: {missing_import_buses[:10].tolist()}")
    import_generators = add_import_slack_generators(network, import_buses)
    apply_calibrated_import_caps(network, import_generators, args.baseline_dir, output_dir)
    load_shedding_generators = add_standard_load_shedding(network)

    status, condition = solve_dispatch_in_chunks(
        network,
        snapshots,
        args.solver,
        args.highs_method,
        args.chunk_size,
    )

    dispatch_hourly, generation_by_carrier = dispatch_by_carrier(network, snapshots, output_dir)
    load_shedding_hourly, load_shedding_by_bus = save_load_shedding(
        network,
        load_shedding_generators,
        snapshots,
        output_dir,
    )
    import_hourly, _import_by_bus = save_import_slack(
        network,
        import_generators,
        snapshots,
        output_dir,
    )
    line_loading = save_line_loading(network, snapshots, output_dir)
    scenario_summary = write_baseline_summary(
        network,
        snapshots,
        status,
        condition,
        generation_by_carrier,
        load_shedding_hourly,
        load_shedding_by_bus,
        import_hourly,
        line_loading,
        output_dir,
    ).iloc[0]

    total_system_cost = system_cost_from_dispatch(network, snapshots)
    summary_path = output_dir / "baseline_summary.csv"
    summary_df = pd.read_csv(summary_path)
    summary_df["scenario_id"] = scenario_id
    summary_df["total_system_cost_usd"] = total_system_cost
    summary_df["damaged_lines"] = len(damaged_lines)
    summary_df["damaged_generators"] = len(damaged_generators)
    summary_df["damaged_buses"] = len(damaged_buses)
    summary_df.to_csv(output_dir / "scenario_summary.csv", index=False)
    summary_path.unlink(missing_ok=True)

    rename_outputs = {
        "baseline_dispatch_by_carrier.csv": "dispatch_by_carrier.csv",
        "baseline_generation_by_carrier_summary.csv": "generation_by_carrier.csv",
        "baseline_import_slack.csv": "import_slack.csv",
        "baseline_import_slack_by_bus.csv": "import_slack_by_bus.csv",
        "baseline_load_shedding.csv": "load_shedding.csv",
        "baseline_load_shedding_by_bus.csv": "load_shedding_by_bus.csv",
        "baseline_line_loading.csv": "line_loading.csv",
    }
    for old_name, new_name in rename_outputs.items():
        old_path = output_dir / old_name
        if old_path.exists():
            old_path.replace(output_dir / new_name)

    line_loading.head(50).to_csv(output_dir / "congested_corridors.csv", index=False)
    scenario.to_frame().T.to_csv(output_dir / "scenario_definition.csv", index=False)

    baseline_summary = pd.read_csv(args.baseline_dir / "baseline_summary.csv").iloc[0]
    baseline_summary = baseline_summary.copy()
    incremental = write_incremental_summary(
        scenario_id,
        summary_df.iloc[0],
        baseline_summary,
        damaged_lines,
        damaged_generators,
        output_dir,
    )

    print("\nScenario complete:", scenario_id)
    print(summary_df.T.to_string(header=False))
    print("\nIncremental vs calibrated baseline:")
    print(incremental.to_string(index=False, float_format=lambda x: f"{x:,.3f}"))
    return summary_df.iloc[0]


def main() -> None:
    args = parse_args()
    manifest = load_manifest(args.manifest, args.create_default_manifest)
    if args.scenario_id:
        manifest = manifest[manifest["scenario_id"].astype(str).isin(args.scenario_id)].copy()
    if manifest.empty:
        raise ValueError("No scenarios selected.")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    scenario_summaries = []
    for _, scenario in manifest.iterrows():
        scenario_summaries.append(run_scenario(scenario, args))

    summary = pd.DataFrame(scenario_summaries)
    summary.to_csv(args.output_dir / "all_scenario_summary.csv", index=False)
    print("\nSaved all scenario summary:", args.output_dir / "all_scenario_summary.csv")


if __name__ == "__main__":
    main()
