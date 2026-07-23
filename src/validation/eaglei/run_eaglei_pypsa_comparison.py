"""
Compare EAGLE-I Hurricane Ian county outages with Florida PyPSA hazard outputs.

The script is intentionally conservative. It will not fabricate county-level
PyPSA results from statewide outputs, and it will stop before validation
statistics if a bus-to-county crosswalk cannot be created or supplied.
"""

from __future__ import annotations

import argparse
import logging
import re
from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
from shapely.geometry import Point


PROJECT_DIR = Path(r"C:\oxford_tc_project")
DEFAULT_VALIDATION_DIR = PROJECT_DIR / "data" / "Electricity" / "eaglei_florida_validation"
DEFAULT_OBSERVED_TARGETS = DEFAULT_VALIDATION_DIR / "outputs" / "hurricane_ian_2022" / "ian_observed_validation_targets.csv"
DEFAULT_PYPSA_ROOT = PROJECT_DIR / "data" / "Electricity"
DEFAULT_IAN_SCENARIO = (
    DEFAULT_PYPSA_ROOT
    / "pypsa_florida_network_extended_tie_lines"
    / "ibtracs_cascading_top5_direct_events"
    / "ibtracs_2022266N12294_tc_direct_damage"
)
DEFAULT_BUS_FILE = DEFAULT_PYPSA_ROOT / "pypsa_florida_network_extended_tie_lines" / "buses.csv"
DEFAULT_LOAD_FILE = DEFAULT_PYPSA_ROOT / "pypsa_florida_network_extended_tie_lines" / "loads.csv"
DEFAULT_LOAD_P_SET = DEFAULT_PYPSA_ROOT / "pypsa_florida_network_extended_tie_lines" / "loads-p_set.csv"
DEFAULT_OUTPUT_DIR = DEFAULT_VALIDATION_DIR / "pypsa_comparison" / "outputs" / "hurricane_ian_2022"
DEFAULT_LOG_DIR = DEFAULT_VALIDATION_DIR / "pypsa_comparison" / "logs"

SEVERITY_CLASSES = ["Low", "Moderate", "High", "Severe"]


def setup_logging(log_dir: Path) -> logging.Logger:
    log_dir.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("eaglei_pypsa_comparison")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    stream = logging.StreamHandler()
    stream.setFormatter(formatter)
    file_handler = logging.FileHandler(log_dir / "run_eaglei_pypsa_comparison.log", mode="w", encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(stream)
    logger.addHandler(file_handler)
    return logger


def snake_case(value: str) -> str:
    value = re.sub(r"[^0-9a-zA-Z]+", "_", value.strip())
    return re.sub(r"_+", "_", value).strip("_").lower()


def zero_pad_fips(series: pd.Series) -> pd.Series:
    return (
        series.astype("string")
        .str.replace(r"\.0$", "", regex=True)
        .str.extract(r"(\d+)", expand=False)
        .str.zfill(5)
    )


def standardize_columns(df: pd.DataFrame | gpd.GeoDataFrame) -> pd.DataFrame | gpd.GeoDataFrame:
    df = df.copy()
    df.columns = [snake_case(col) for col in df.columns]
    return df


def csv_columns_and_rows(path: Path) -> tuple[list[str], int]:
    sample = pd.read_csv(path, nrows=0)
    try:
        row_count = sum(1 for _ in path.open("r", encoding="utf-8", errors="ignore")) - 1
    except Exception:
        row_count = -1
    return sample.columns.tolist(), row_count


def classify_file_resolution(path: Path, columns: list[str]) -> str:
    lower = {col.lower() for col in columns}
    name = path.name.lower()
    if {"bus", "snapshot"}.issubset(lower):
        return "bus-by-snapshot"
    if "bus" in lower:
        return "bus-level"
    if "county_fips" in lower or "county" in lower:
        return "county-level"
    if "snapshot" in lower or "total_load_shed_mwh" in lower or "total_demand_mwh" in lower:
        return "statewide/time or scenario-level"
    if "scenario" in name or "definition" in name or "summary" in name:
        return "scenario metadata"
    return "other"


def infer_scenario(path: Path) -> str:
    text = str(path).lower()
    if "2022266n12294" in text or "ian" in text:
        return "Hurricane Ian / IBTrACS 2022266N12294 candidate"
    if "tc_" in text or "tropical" in text or "wind" in text:
        return "tropical-cyclone scenario candidate"
    if "flood" in text:
        return "flood scenario candidate"
    if "baseline" in text:
        return "baseline/no-hazard run"
    return "unknown"


def inventory_pypsa_outputs(pypsa_root: Path, scenario_dir: Path, output_dir: Path, logger: logging.Logger) -> pd.DataFrame:
    relevant_terms = re.compile(
        r"(ian|2022266|2022|hurricane|tropical|tc|storm|load.?shed|unserved|demand.?shed|buses|loads|snapshots|scenario|manifest)",
        re.I,
    )
    candidates: list[Path] = []
    if scenario_dir.exists():
        candidates.extend(scenario_dir.glob("*.csv"))
    for path in pypsa_root.rglob("*.csv"):
        if "eaglei_florida_validation" in {part.lower() for part in path.parts}:
            continue
        if relevant_terms.search(str(path)):
            candidates.append(path)
    seen = set()
    rows = []
    for path in candidates:
        if path in seen or not path.exists():
            continue
        seen.add(path)
        try:
            columns, row_count = csv_columns_and_rows(path)
        except Exception as exc:
            logger.warning("Could not inspect %s: %s", path, exc)
            continue
        rows.append(
            {
                "path": str(path.resolve()),
                "file_type": path.suffix.lower().lstrip("."),
                "important_columns": ", ".join(columns[:30]) + (" ..." if len(columns) > 30 else ""),
                "row_count": row_count,
                "scenario_represented": infer_scenario(path),
                "resolution": classify_file_resolution(path, columns),
            }
        )
    inventory = pd.DataFrame(rows).sort_values(["scenario_represented", "resolution", "path"])
    inventory.to_csv(output_dir / "pypsa_candidate_file_inventory.csv", index=False)
    return inventory


def inspect_selected_scenario(scenario_dir: Path, output_dir: Path) -> pd.DataFrame:
    checks = [
        ("scenario_definition", scenario_dir / "scenario_definition.csv"),
        ("scenario_summary", scenario_dir / "scenario_summary.csv"),
        ("load_shedding_statewide_by_snapshot", scenario_dir / "load_shedding.csv"),
        ("load_shedding_by_bus_total", scenario_dir / "load_shedding_by_bus.csv"),
        ("line_loading", scenario_dir / "line_loading.csv"),
        ("damaged_lines", scenario_dir / "damaged_lines.csv"),
        ("bus_substation_deratings", scenario_dir / "bus_substation_deratings.csv"),
    ]
    rows = []
    for label, path in checks:
        if not path.exists():
            rows.append({"dataset": label, "path": str(path), "exists": False})
            continue
        df = pd.read_csv(path, nrows=5)
        row_count = sum(1 for _ in path.open("r", encoding="utf-8", errors="ignore")) - 1
        rows.append(
            {
                "dataset": label,
                "path": str(path.resolve()),
                "exists": True,
                "rows": row_count,
                "columns": ", ".join(df.columns),
                "resolution": classify_file_resolution(path, df.columns.tolist()),
            }
        )
    out = pd.DataFrame(rows)
    out.to_csv(output_dir / "selected_pypsa_scenario_resolution.csv", index=False)
    return out


def load_buses(bus_file: Path) -> gpd.GeoDataFrame:
    buses = pd.read_csv(bus_file)
    buses = standardize_columns(buses)
    required = {"name", "x", "y"}
    missing = required - set(buses.columns)
    if missing:
        raise ValueError(f"Bus file missing required columns: {sorted(missing)}")
    geometry = [Point(xy) for xy in zip(pd.to_numeric(buses["x"], errors="coerce"), pd.to_numeric(buses["y"], errors="coerce"))]
    return gpd.GeoDataFrame(buses, geometry=geometry, crs="EPSG:4326")


def load_bus_peak_demand(load_file: Path, load_p_set: Path | None, output_dir: Path, logger: logging.Logger) -> pd.DataFrame:
    loads = standardize_columns(pd.read_csv(load_file))
    if "bus" not in loads.columns or "name" not in loads.columns:
        raise ValueError("loads.csv must contain name and bus columns.")
    loads["static_p_set"] = pd.to_numeric(loads.get("p_set", 0), errors="coerce").fillna(0)
    if load_p_set is None or not load_p_set.exists():
        logger.warning("No loads-p_set.csv supplied; using static p_set from loads.csv.")
        by_bus = loads.groupby("bus", as_index=False)["static_p_set"].sum().rename(columns={"static_p_set": "assigned_demand_mw"})
        by_bus.to_csv(output_dir / "bus_assigned_demand_static_only.csv", index=False)
        return by_bus

    logger.info("Reading load time series header and calculating peak demand by load column.")
    # This file is wide but manageable for one pass. It is not modified.
    load_ts = pd.read_csv(load_p_set)
    load_cols = [col for col in load_ts.columns if col != "snapshot"]
    peak_by_load = load_ts[load_cols].max(axis=0).rename("assigned_demand_mw").reset_index().rename(columns={"index": "name"})
    by_load = loads[["name", "bus"]].merge(peak_by_load, on="name", how="left")
    by_bus = by_load.groupby("bus", as_index=False)["assigned_demand_mw"].sum()
    by_bus.to_csv(output_dir / "bus_assigned_peak_demand_mw.csv", index=False)
    return by_bus


def find_county_boundary(data_root: Path) -> Path | None:
    candidates = []
    for suffix in ("*.shp", "*.gpkg", "*.geojson", "*.json", "*.parquet"):
        candidates.extend(data_root.rglob(suffix))
    county_candidates = [p for p in candidates if re.search(r"(county|counties|tl_|cb_)", p.name, re.I)]
    # Avoid returning the known state boundary file as if it were counties.
    county_candidates = [p for p in county_candidates if "state" not in p.name.lower()]
    return county_candidates[0] if county_candidates else None


def build_bus_to_county_crosswalk(
    buses: gpd.GeoDataFrame,
    bus_demand: pd.DataFrame,
    county_boundaries: Path | None,
    output_dir: Path,
    logger: logging.Logger,
) -> tuple[pd.DataFrame | None, str | None]:
    buses = buses.merge(bus_demand, left_on="name", right_on="bus", how="left")
    buses["assigned_demand_mw"] = pd.to_numeric(buses["assigned_demand_mw"], errors="coerce").fillna(0)

    if county_boundaries is None:
        county_boundaries = find_county_boundary(PROJECT_DIR / "data")
    if county_boundaries is None:
        reason = (
            "No local Florida/U.S. county boundary file was found. A county boundary dataset with GEOID/county FIPS is "
            "required to aggregate PyPSA bus results to counties. No county-level PyPSA validation statistics were calculated."
        )
        logger.warning(reason)
        buses.drop(columns="geometry").to_csv(output_dir / "bus_to_county_crosswalk_unmatched_buses.csv", index=False)
        return None, reason

    try:
        counties = gpd.read_file(county_boundaries).to_crs("EPSG:4326")
    except Exception as exc:
        reason = f"Could not read county boundary file {county_boundaries}: {exc}"
        logger.warning(reason)
        return None, reason
    counties = standardize_columns(counties)
    fips_col = next((col for col in ["geoid", "county_fips", "fips", "fips_code"] if col in counties.columns), None)
    if fips_col is None:
        reason = f"County boundary file {county_boundaries} does not have a recognized FIPS/GEOID column."
        logger.warning(reason)
        return None, reason
    counties["county_fips"] = zero_pad_fips(counties[fips_col])
    fl_counties = counties[counties["county_fips"].str.startswith("12")].copy()
    if fl_counties.empty:
        reason = f"County boundary file {county_boundaries} contains no detected Florida counties."
        logger.warning(reason)
        return None, reason

    joined = gpd.sjoin(buses, fl_counties[["county_fips", "geometry"]], how="left", predicate="intersects")
    name_cols = [col for col in fl_counties.columns if col in {"name", "namelsad", "county", "county_name"}]
    if name_cols:
        joined = joined.merge(fl_counties[["county_fips", name_cols[0]]].drop_duplicates(), on="county_fips", how="left")
        joined = joined.rename(columns={name_cols[0]: "county_name"})
    else:
        joined["county_name"] = pd.NA
    joined.drop(columns="geometry").to_csv(output_dir / "bus_to_county_crosswalk.csv", index=False)
    joined.to_file(output_dir / "bus_to_county_crosswalk.gpkg", layer="bus_to_county", driver="GPKG")
    unmatched = joined[joined["county_fips"].isna()].drop(columns="geometry")
    unmatched.to_csv(output_dir / "bus_to_county_unmatched_buses.csv", index=False)
    return pd.DataFrame(joined.drop(columns="geometry")), None


def create_static_pypsa_county_results(
    scenario_dir: Path,
    crosswalk: pd.DataFrame,
    output_dir: Path,
    scenario_name: str,
) -> pd.DataFrame:
    shed = pd.read_csv(scenario_dir / "load_shedding_by_bus.csv")
    shed = standardize_columns(shed)
    bus = crosswalk.copy()
    bus_id_col = next((col for col in ["name", "bus", "name_x"] if col in bus.columns), None)
    if bus_id_col is None:
        raise ValueError("Bus-to-county crosswalk does not contain a recognizable bus ID column.")
    if "county_name" not in bus.columns:
        county_name_col = next((col for col in ["name_y", "namelsad", "county"] if col in bus.columns), None)
        bus["county_name"] = bus[county_name_col] if county_name_col is not None else pd.NA
    county = bus.merge(shed, left_on=bus_id_col, right_on="bus", how="left", suffixes=("", "_shed"))
    county["total_load_shed_mwh"] = pd.to_numeric(county["total_load_shed_mwh"], errors="coerce").fillna(0)
    county["max_hourly_load_shed_mw"] = pd.to_numeric(county["max_hourly_load_shed_mw"], errors="coerce").fillna(0)
    county["assigned_demand_mw"] = pd.to_numeric(county["assigned_demand_mw"], errors="coerce").fillna(0)
    county = county[county["county_fips"].notna()].copy()
    grouped = (
        county.groupby(["county_fips", "county_name"], dropna=False)
        .agg(
            pypsa_peak_load_shed_mw=("max_hourly_load_shed_mw", "sum"),
            pypsa_total_load_shed_mwh=("total_load_shed_mwh", "sum"),
            total_county_demand_mw=("assigned_demand_mw", "sum"),
        )
        .reset_index()
        .rename(columns={"county_name": "county"})
    )
    grouped["pypsa_peak_fraction_demand_shed"] = np.where(
        grouped["total_county_demand_mw"] > 0,
        grouped["pypsa_peak_load_shed_mw"] / grouped["total_county_demand_mw"],
        np.nan,
    )
    grouped["pypsa_peak_fraction_demand_shed_percent"] = grouped["pypsa_peak_fraction_demand_shed"] * 100
    grouped["pypsa_peak_timestamp"] = pd.NA
    grouped["pypsa_hours_above_1_percent"] = pd.NA
    grouped["pypsa_hours_above_5_percent"] = pd.NA
    grouped["pypsa_hours_above_10_percent"] = pd.NA
    grouped["simulation_coverage_warning"] = "static bus-total only; no bus-by-snapshot load shedding saved"
    grouped["scenario_name"] = scenario_name
    grouped.to_csv(output_dir / "pypsa_county_static_results.csv", index=False)
    grouped.to_csv(output_dir / "pypsa_county_metrics.csv", index=False)
    return grouped


def severity_class(percent: pd.Series) -> pd.Series:
    bins = [-np.inf, 1, 5, 20, np.inf]
    return pd.cut(percent, bins=bins, labels=SEVERITY_CLASSES, right=False)


def safe_corr(x: pd.Series, y: pd.Series, method: str) -> float:
    valid = pd.concat([x, y], axis=1).replace([np.inf, -np.inf], np.nan).dropna()
    if len(valid) < 3 or valid.iloc[:, 0].nunique() < 2 or valid.iloc[:, 1].nunique() < 2:
        return np.nan
    if method == "pearson":
        return float(pearsonr(valid.iloc[:, 0], valid.iloc[:, 1]).statistic)
    return float(spearmanr(valid.iloc[:, 0], valid.iloc[:, 1]).statistic)


def compare_county_results(observed_targets: Path, pypsa_county: pd.DataFrame, output_dir: Path, timing_valid: bool) -> pd.DataFrame:
    obs = standardize_columns(pd.read_csv(observed_targets))
    py = standardize_columns(pypsa_county)
    obs["county_fips"] = zero_pad_fips(obs["county_fips"])
    py["county_fips"] = zero_pad_fips(py["county_fips"])
    comp = obs.merge(py, on="county_fips", how="outer", suffixes=("_observed", "_pypsa"))
    if "county_observed" in comp.columns:
        comp["county"] = comp["county_observed"].fillna(comp.get("county_pypsa"))
    comp["matched_status"] = np.where(
        comp["observed_peak_outage_percent"].notna() & comp["pypsa_peak_fraction_demand_shed_percent"].notna(),
        "matched",
        np.where(comp["observed_peak_outage_percent"].notna(), "observed_only", "pypsa_only"),
    )
    for source, col in [
        ("observed_peak_normalized", "observed_peak_outage_percent"),
        ("pypsa_peak_normalized", "pypsa_peak_fraction_demand_shed_percent"),
        ("observed_duration_normalized", "observed_customer_outage_hours"),
        ("pypsa_duration_normalized", "pypsa_total_load_shed_mwh"),
    ]:
        max_value = pd.to_numeric(comp[col], errors="coerce").max()
        comp[source] = pd.to_numeric(comp[col], errors="coerce") / max_value if pd.notna(max_value) and max_value != 0 else np.nan
    comp.to_csv(output_dir / "ian_eaglei_pypsa_county_comparison.csv", index=False)

    matched = comp[comp["matched_status"].eq("matched")].copy()
    stats_rows = []
    if matched.empty:
        stats_rows.append({"metric": "status", "value": "not_run_no_matched_counties"})
    else:
        diff = matched["pypsa_peak_fraction_demand_shed_percent"] - matched["observed_peak_outage_percent"]
        stats_rows.extend(
            [
                {"metric": "matched_counties", "value": len(matched)},
                {"metric": "peak_pearson", "value": safe_corr(matched["observed_peak_outage_percent"], matched["pypsa_peak_fraction_demand_shed_percent"], "pearson")},
                {"metric": "peak_spearman", "value": safe_corr(matched["observed_peak_outage_percent"], matched["pypsa_peak_fraction_demand_shed_percent"], "spearman")},
                {"metric": "peak_mae_percentage_points", "value": diff.abs().mean()},
                {"metric": "peak_rmse_percentage_points", "value": float(np.sqrt((diff**2).mean()))},
                {"metric": "mean_bias_pypsa_minus_observed_pp", "value": diff.mean()},
                {"metric": "normalized_peak_pearson", "value": safe_corr(matched["observed_peak_normalized"], matched["pypsa_peak_normalized"], "pearson")},
                {"metric": "normalized_peak_spearman", "value": safe_corr(matched["observed_peak_normalized"], matched["pypsa_peak_normalized"], "spearman")},
                {"metric": "duration_raw_pearson_relative_only", "value": safe_corr(matched["observed_customer_outage_hours"], matched["pypsa_total_load_shed_mwh"], "pearson")},
                {"metric": "duration_raw_spearman_relative_only", "value": safe_corr(matched["observed_customer_outage_hours"], matched["pypsa_total_load_shed_mwh"], "spearman")},
                {"metric": "duration_normalized_pearson", "value": safe_corr(matched["observed_duration_normalized"], matched["pypsa_duration_normalized"], "pearson")},
                {"metric": "duration_normalized_spearman", "value": safe_corr(matched["observed_duration_normalized"], matched["pypsa_duration_normalized"], "spearman")},
                {"metric": "timing_comparison_valid", "value": timing_valid},
            ]
        )

        matched["observed_rank"] = matched["observed_peak_outage_percent"].rank(ascending=False, method="min")
        matched["pypsa_rank"] = matched["pypsa_peak_fraction_demand_shed_percent"].rank(ascending=False, method="min")
        matched["rank_abs_difference"] = (matched["observed_rank"] - matched["pypsa_rank"]).abs()
        matched["normalized_peak_difference"] = matched["pypsa_peak_normalized"] - matched["observed_peak_normalized"]
        top5_obs = set(matched.nlargest(5, "observed_peak_outage_percent")["county_fips"])
        top5_py = set(matched.nlargest(5, "pypsa_peak_fraction_demand_shed_percent")["county_fips"])
        top10_obs = set(matched.nlargest(10, "observed_peak_outage_percent")["county_fips"])
        top10_py = set(matched.nlargest(10, "pypsa_peak_fraction_demand_shed_percent")["county_fips"])
        stats_rows.extend(
            [
                {"metric": "top5_overlap_count", "value": len(top5_obs & top5_py)},
                {"metric": "top10_overlap_count", "value": len(top10_obs & top10_py)},
                {"metric": "top10_jaccard", "value": len(top10_obs & top10_py) / len(top10_obs | top10_py) if top10_obs | top10_py else np.nan},
                {"metric": "mean_absolute_rank_difference", "value": matched["rank_abs_difference"].mean()},
            ]
        )
        overlap = pd.DataFrame(
            [
                {"set": "top5", "overlap_count": len(top5_obs & top5_py), "observed_fips": sorted(top5_obs), "pypsa_fips": sorted(top5_py)},
                {"set": "top10", "overlap_count": len(top10_obs & top10_py), "observed_fips": sorted(top10_obs), "pypsa_fips": sorted(top10_py)},
            ]
        )
        overlap.to_csv(output_dir / "ian_top_county_overlap.csv", index=False)
        matched.sort_values("rank_abs_difference", ascending=False).to_csv(output_dir / "ian_county_rank_comparison.csv", index=False)
        comp_classes = matched.copy()
        comp_classes["observed_class"] = severity_class(comp_classes["observed_peak_outage_percent"])
        comp_classes["pypsa_class"] = severity_class(comp_classes["pypsa_peak_fraction_demand_shed_percent"])
        confusion = pd.crosstab(comp_classes["observed_class"], comp_classes["pypsa_class"], dropna=False).reindex(index=SEVERITY_CLASSES, columns=SEVERITY_CLASSES, fill_value=0)
        confusion.to_csv(output_dir / "ian_severity_confusion_matrix.csv")
        comp.to_csv(output_dir / "ian_eaglei_pypsa_county_comparison.csv", index=False)
    stats = pd.DataFrame(stats_rows)
    stats.to_csv(output_dir / "ian_validation_statistics.csv", index=False)
    return stats


def write_blocked_outputs(output_dir: Path, inventory: pd.DataFrame, resolution: pd.DataFrame, reason: str, scenario_name: str) -> None:
    report = f"""# EAGLE-I / PyPSA Comparison Preparation

Selected PyPSA scenario: `{scenario_name}`

The selected scenario is Hurricane Ian-specific if it is `ibtracs_2022266N12294_tc_direct_damage`.

## Status
County-level validation statistics were **not calculated**.

Reason:
{reason}

## What Was Found
- PyPSA scenario metadata and load-shedding outputs were inventoried.
- The selected Ian scenario saves statewide load shedding by snapshot and total load shedding by bus.
- It does not save bus-by-snapshot load shedding.
- Scenario timestamps are synthetic model snapshots, not historical Ian timestamps, so timing comparison is not valid without a rerun using historical aligned snapshots.

## Required Next Input
Provide a Florida/U.S. county boundary file with county FIPS/GEOID, or provide a verified bus-to-county crosswalk. Then rerun this script using `--county-boundaries` or `--bus-county-crosswalk`.
"""
    (output_dir / "validation_interpretation.md").write_text(report, encoding="utf-8")


def create_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Compare EAGLE-I observed county outages with PyPSA county load-shed outputs.")
    parser.add_argument("--observed-targets", type=Path, default=DEFAULT_OBSERVED_TARGETS)
    parser.add_argument("--pypsa-results", type=Path, default=DEFAULT_IAN_SCENARIO, help="PyPSA scenario directory or county results file.")
    parser.add_argument("--pypsa-root", type=Path, default=DEFAULT_PYPSA_ROOT)
    parser.add_argument("--bus-file", type=Path, default=DEFAULT_BUS_FILE)
    parser.add_argument("--load-file", type=Path, default=DEFAULT_LOAD_FILE)
    parser.add_argument("--load-p-set", type=Path, default=DEFAULT_LOAD_P_SET)
    parser.add_argument("--county-boundaries", type=Path, default=None)
    parser.add_argument("--bus-county-crosswalk", type=Path, default=None)
    parser.add_argument("--scenario-name", default="ibtracs_2022266N12294_tc_direct_damage")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser


def main() -> None:
    args = create_argument_parser().parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    logger = setup_logging(DEFAULT_LOG_DIR)
    logger.info("Starting EAGLE-I / PyPSA comparison preparation.")

    inventory = inventory_pypsa_outputs(args.pypsa_root, args.pypsa_results, args.output_dir, logger)
    resolution = inspect_selected_scenario(args.pypsa_results, args.output_dir) if args.pypsa_results.is_dir() else pd.DataFrame()

    scenario_specific = args.pypsa_results.is_dir() and (args.pypsa_results / "scenario_definition.csv").exists()
    if scenario_specific:
        scenario_def = pd.read_csv(args.pypsa_results / "scenario_definition.csv")
        is_ian_specific = scenario_def.to_string().lower().find("2022266n12294") >= 0
    else:
        is_ian_specific = False

    buses = load_buses(args.bus_file)
    bus_demand = load_bus_peak_demand(args.load_file, args.load_p_set, args.output_dir, logger)

    if args.bus_county_crosswalk and args.bus_county_crosswalk.exists():
        crosswalk = pd.read_csv(args.bus_county_crosswalk)
        crosswalk_failure_reason = None
    else:
        crosswalk, crosswalk_failure_reason = build_bus_to_county_crosswalk(
            buses=buses,
            bus_demand=bus_demand,
            county_boundaries=args.county_boundaries,
            output_dir=args.output_dir,
            logger=logger,
        )

    timing_valid = False
    if crosswalk is None:
        write_blocked_outputs(args.output_dir, inventory, resolution, crosswalk_failure_reason or "Unknown crosswalk failure.", args.scenario_name)
        summary = pd.DataFrame(
            [
                {
                    "pypsa_scenario_used": args.scenario_name,
                    "is_truly_hurricane_ian_specific": is_ian_specific,
                    "comparison_status": "blocked_before_validation_statistics",
                    "matched_counties": 0,
                    "pearson_correlation": np.nan,
                    "spearman_correlation": np.nan,
                    "top10_overlap": np.nan,
                    "mean_bias": np.nan,
                    "rmse": np.nan,
                    "timing_comparison_valid": timing_valid,
                    "blocking_reason": crosswalk_failure_reason,
                }
            ]
        )
        summary.to_csv(args.output_dir / "comparison_run_summary.csv", index=False)
        print(summary.to_string(index=False))
        return

    if not args.pypsa_results.is_dir() or not (args.pypsa_results / "load_shedding_by_bus.csv").exists():
        reason = "Selected PyPSA input is not a scenario directory with load_shedding_by_bus.csv."
        write_blocked_outputs(args.output_dir, inventory, resolution, reason, args.scenario_name)
        raise FileNotFoundError(reason)

    pypsa_county = create_static_pypsa_county_results(args.pypsa_results, crosswalk, args.output_dir, args.scenario_name)
    stats = compare_county_results(args.observed_targets, pypsa_county, args.output_dir, timing_valid=timing_valid)
    summary = pd.DataFrame(
        [
            {
                "pypsa_scenario_used": args.scenario_name,
                "is_truly_hurricane_ian_specific": is_ian_specific,
                "comparison_status": "statistics_calculated_static_bus_total_only",
                "matched_counties": stats.loc[stats["metric"].eq("matched_counties"), "value"].iloc[0] if (stats["metric"].eq("matched_counties")).any() else np.nan,
                "pearson_correlation": stats.loc[stats["metric"].eq("peak_pearson"), "value"].iloc[0] if (stats["metric"].eq("peak_pearson")).any() else np.nan,
                "spearman_correlation": stats.loc[stats["metric"].eq("peak_spearman"), "value"].iloc[0] if (stats["metric"].eq("peak_spearman")).any() else np.nan,
                "top10_overlap": stats.loc[stats["metric"].eq("top10_overlap_count"), "value"].iloc[0] if (stats["metric"].eq("top10_overlap_count")).any() else np.nan,
                "mean_bias": stats.loc[stats["metric"].eq("mean_bias_pypsa_minus_observed_pp"), "value"].iloc[0] if (stats["metric"].eq("mean_bias_pypsa_minus_observed_pp")).any() else np.nan,
                "rmse": stats.loc[stats["metric"].eq("peak_rmse_percentage_points"), "value"].iloc[0] if (stats["metric"].eq("peak_rmse_percentage_points")).any() else np.nan,
                "timing_comparison_valid": timing_valid,
            }
        ]
    )
    summary.to_csv(args.output_dir / "comparison_run_summary.csv", index=False)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
