"""
Reusable EAGLE-I Florida historical-outage validation workflow.

The workflow reads large yearly EAGLE-I outage files in chunks, filters to a
single event window, calculates county and statewide outage metrics, and exports
figures/tables for comparison with PyPSA hazard-model outputs.
"""

from __future__ import annotations

import argparse
import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import geopandas as gpd
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import ListedColormap
from matplotlib.ticker import FuncFormatter


PROJECT_DIR = Path(r"C:\oxford_tc_project")
DEFAULT_INPUT_DIR = PROJECT_DIR / "data" / "24237376"
DEFAULT_PROJECT_DIR = PROJECT_DIR / "data" / "Electricity" / "eaglei_florida_validation"
FLORIDA_FIPS_PREFIX = "12"
FIFTEEN_MINUTES_HOURS = 0.25


@dataclass
class WorkflowPaths:
    project_dir: Path
    scripts_dir: Path
    outputs_dir: Path
    event_dir: Path
    logs_dir: Path


def snake_case(value: str) -> str:
    value = re.sub(r"[^0-9a-zA-Z]+", "_", value.strip())
    value = re.sub(r"_+", "_", value).strip("_")
    return value.lower()


def setup_logging(logs_dir: Path, event_name: str) -> logging.Logger:
    logs_dir.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("eaglei_validation")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    file_handler = logging.FileHandler(logs_dir / f"{event_name}.log", mode="w", encoding="utf-8")
    file_handler.setFormatter(formatter)
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)
    return logger


def build_paths(project_dir: Path, event_name: str) -> WorkflowPaths:
    paths = WorkflowPaths(
        project_dir=project_dir,
        scripts_dir=project_dir / "scripts",
        outputs_dir=project_dir / "outputs",
        event_dir=project_dir / "outputs" / event_name,
        logs_dir=project_dir / "logs",
    )
    paths.scripts_dir.mkdir(parents=True, exist_ok=True)
    paths.event_dir.mkdir(parents=True, exist_ok=True)
    paths.logs_dir.mkdir(parents=True, exist_ok=True)
    return paths


def standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [snake_case(col) for col in df.columns]
    if "sum" in df.columns and "customers_out" not in df.columns:
        df = df.rename(columns={"sum": "customers_out"})
    if "county_fips" in df.columns and "fips_code" not in df.columns:
        df = df.rename(columns={"county_fips": "fips_code"})
    if "customers" in df.columns and "modeled_customers" not in df.columns:
        df = df.rename(columns={"customers": "modeled_customers"})
    return df


def inspect_headers(input_dir: Path, logger: logging.Logger) -> pd.DataFrame:
    rows = []
    for path in sorted(input_dir.glob("*.csv")):
        sample = pd.read_csv(path, nrows=5)
        standardized = standardize_columns(sample)
        row = {
            "file": path.name,
            "original_columns": ", ".join(sample.columns),
            "standardized_columns": ", ".join(standardized.columns),
            "row_sample_count": len(sample),
        }
        rows.append(row)
        logger.info("Header %s: %s", path.name, row["standardized_columns"])
    return pd.DataFrame(rows)


def zero_pad_fips(series: pd.Series) -> pd.Series:
    return (
        series.astype("string")
        .str.replace(r"\.0$", "", regex=True)
        .str.extract(r"(\d+)", expand=False)
        .str.zfill(5)
    )


def parse_utc(series: pd.Series) -> pd.Series:
    return pd.to_datetime(series, errors="coerce", utc=True)


def load_florida_event_outages(
    outage_file: Path,
    start: pd.Timestamp,
    end: pd.Timestamp,
    state: str,
    chunksize: int,
    logger: logging.Logger,
) -> tuple[pd.DataFrame, dict[str, object]]:
    chunks = []
    florida_rows_total = 0
    event_rows_before_clean = 0
    missing_by_column: dict[str, int] = {}
    negative_customers = 0
    max_customers_out = np.nan
    required = {"fips_code", "county", "state", "customers_out", "run_start_time"}

    for i, chunk in enumerate(pd.read_csv(outage_file, chunksize=chunksize, low_memory=False), start=1):
        chunk = standardize_columns(chunk)
        missing_required = required - set(chunk.columns)
        if missing_required:
            raise ValueError(f"{outage_file.name} missing required standardized columns: {sorted(missing_required)}")

        chunk["state"] = chunk["state"].astype("string")
        chunk["county_fips"] = zero_pad_fips(chunk["fips_code"])
        state_mask = chunk["state"].str.casefold().eq(state.casefold()) | chunk["county_fips"].str.startswith(FLORIDA_FIPS_PREFIX)
        florida = chunk.loc[state_mask, ["county_fips", "county", "state", "customers_out", "run_start_time"]].copy()
        florida_rows_total += len(florida)
        if florida.empty:
            continue

        florida["run_start_time"] = parse_utc(florida["run_start_time"])
        florida["customers_out"] = pd.to_numeric(florida["customers_out"], errors="coerce")
        negative_customers += int((florida["customers_out"] < 0).sum())
        chunk_max = florida["customers_out"].max(skipna=True)
        if pd.notna(chunk_max):
            max_customers_out = chunk_max if pd.isna(max_customers_out) else max(max_customers_out, chunk_max)
        for col, count in florida.isna().sum().items():
            missing_by_column[col] = missing_by_column.get(col, 0) + int(count)

        event = florida[(florida["run_start_time"] >= start) & (florida["run_start_time"] < end)].copy()
        event_rows_before_clean += len(event)
        if not event.empty:
            chunks.append(event)
        if i % 25 == 0:
            logger.info("Processed %s chunks; Florida rows seen so far: %s", i, florida_rows_total)

    if chunks:
        event_data = pd.concat(chunks, ignore_index=True)
    else:
        event_data = pd.DataFrame(columns=["county_fips", "county", "state", "customers_out", "run_start_time"])

    exact_duplicates = int(event_data.duplicated().sum())
    event_data = event_data.drop_duplicates().copy()
    duplicate_county_time = int(event_data.duplicated(["county_fips", "run_start_time"], keep=False).sum())
    if duplicate_county_time:
        logger.warning(
            "Found %s non-exact duplicate county/timestamp rows. No utility column exists, so customers_out is aggregated by max to avoid double-counting.",
            duplicate_county_time,
        )
        event_data = (
            event_data.sort_values(["county_fips", "run_start_time"])
            .groupby(["county_fips", "run_start_time"], as_index=False)
            .agg(
                county=("county", "first"),
                state=("state", "first"),
                customers_out=("customers_out", "max"),
            )
        )

    event_data = event_data.sort_values(["county_fips", "run_start_time"]).reset_index(drop=True)
    quality = {
        "original_florida_rows": florida_rows_total,
        "rows_inside_event_window_before_cleaning": event_rows_before_clean,
        "rows_inside_event_window_after_cleaning": len(event_data),
        "exact_duplicate_rows_removed": exact_duplicates,
        "duplicate_county_timestamp_rows_before_aggregation": duplicate_county_time,
        "negative_customers_out_values": negative_customers,
        "maximum_customers_out_value": max_customers_out,
        "missing_values_by_column": missing_by_column,
    }
    return event_data, quality


def load_mcc(input_dir: Path, logger: logging.Logger) -> pd.DataFrame:
    mcc = pd.read_csv(input_dir / "MCC.csv")
    logger.info("MCC original columns: %s", ", ".join(mcc.columns))
    mcc = standardize_columns(mcc)
    if "fips_code" not in mcc.columns:
        if "county_fips" in mcc.columns:
            mcc = mcc.rename(columns={"county_fips": "fips_code"})
        else:
            raise ValueError("MCC.csv needs County_FIPS/county_fips/fips_code column.")
    if "modeled_customers" not in mcc.columns:
        raise ValueError("MCC.csv needs Customers/customers/modeled_customers column.")
    mcc["county_fips"] = zero_pad_fips(mcc["fips_code"])
    mcc["modeled_customers"] = pd.to_numeric(mcc["modeled_customers"], errors="coerce")
    return mcc[["county_fips", "modeled_customers"]].drop_duplicates("county_fips")


def join_modeled_customers(event_data: pd.DataFrame, mcc: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    joined = event_data.merge(mcc, on="county_fips", how="left")
    unmatched = sorted(joined.loc[joined["modeled_customers"].isna(), "county_fips"].dropna().unique().tolist())
    joined["outage_fraction"] = joined["customers_out"] / joined["modeled_customers"]
    joined["outage_percent"] = 100 * joined["outage_fraction"]
    joined["customers_exceed_modeled"] = joined["customers_out"] > joined["modeled_customers"]
    joined["outage_percent_plot"] = joined["outage_percent"].clip(lower=0, upper=100)
    return joined, unmatched


def complete_county_time_grid(joined: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    timestamps = pd.date_range(start=start, end=end, freq="15min", inclusive="left")
    counties = (
        joined[["county_fips", "county", "state", "modeled_customers"]]
        .drop_duplicates("county_fips")
        .sort_values("county_fips")
        .reset_index(drop=True)
    )
    grid = pd.MultiIndex.from_product([counties["county_fips"], timestamps], names=["county_fips", "run_start_time"]).to_frame(index=False)
    grid = grid.merge(counties, on="county_fips", how="left")
    value_cols = ["customers_out", "outage_fraction", "outage_percent", "outage_percent_plot", "customers_exceed_modeled"]
    full = grid.merge(joined[["county_fips", "run_start_time", *value_cols]], on=["county_fips", "run_start_time"], how="left")
    full["observation_status"] = np.where(full["customers_out"].notna(), "observed", "missing_unknown")

    filled_groups = []
    for _fips, county_df in full.groupby("county_fips", sort=False):
        county_df = county_df.sort_values("run_start_time").copy()
        is_missing = county_df["customers_out"].isna().to_numpy()
        run_id = (pd.Series(is_missing).ne(pd.Series(is_missing).shift(fill_value=False))).cumsum().to_numpy()
        for rid in np.unique(run_id[is_missing]):
            idx = np.where((run_id == rid) & is_missing)[0]
            if len(idx) <= 4 and idx[0] > 0 and idx[-1] < len(county_df) - 1:
                before_ok = pd.notna(county_df.iloc[idx[0] - 1]["customers_out"])
                after_ok = pd.notna(county_df.iloc[idx[-1] + 1]["customers_out"])
                if before_ok and after_ok:
                    fill_index = county_df.index[idx]
                    county_df.loc[fill_index, "customers_out"] = 0.0
                    county_df.loc[fill_index, "outage_fraction"] = 0.0
                    county_df.loc[fill_index, "outage_percent"] = 0.0
                    county_df.loc[fill_index, "outage_percent_plot"] = 0.0
                    county_df.loc[fill_index, "customers_exceed_modeled"] = False
                    county_df.loc[fill_index, "observation_status"] = "imputed_zero"
        filled_groups.append(county_df)
    full = pd.concat(filled_groups, ignore_index=True)
    return full.sort_values(["county_fips", "run_start_time"]).reset_index(drop=True)


def first_time_above(df: pd.DataFrame, threshold: float) -> pd.Timestamp | pd.NaT:
    above = df.loc[df["outage_percent"] > threshold, "run_start_time"]
    return above.min() if not above.empty else pd.NaT


def last_time_above(df: pd.DataFrame, threshold: float) -> pd.Timestamp | pd.NaT:
    above = df.loc[df["outage_percent"] > threshold, "run_start_time"]
    return above.max() if not above.empty else pd.NaT


def recovery_time_from_peak(df: pd.DataFrame, threshold: float) -> float:
    valid = df[df["customers_out"].notna()].copy()
    if valid.empty:
        return np.nan
    peak_idx = valid["customers_out"].idxmax()
    peak_time = valid.loc[peak_idx, "run_start_time"]
    after = valid[valid["run_start_time"] >= peak_time]
    below = after[after["outage_percent"] < threshold]
    if below.empty:
        return np.nan
    return (below.iloc[0]["run_start_time"] - peak_time).total_seconds() / 3600


def calculate_county_metrics(full: pd.DataFrame, expected_count: int) -> pd.DataFrame:
    rows = []
    for county_fips, df in full.groupby("county_fips"):
        valid = df[df["customers_out"].notna()].copy()
        county = df["county"].dropna().iloc[0] if df["county"].notna().any() else ""
        modeled = df["modeled_customers"].dropna().iloc[0] if df["modeled_customers"].notna().any() else np.nan
        if valid.empty:
            peak_customers = np.nan
            peak_percent = np.nan
            peak_time = pd.NaT
        else:
            peak_row = valid.loc[valid["customers_out"].idxmax()]
            peak_customers = peak_row["customers_out"]
            peak_percent = peak_row["outage_percent"]
            peak_time = peak_row["run_start_time"]
        observed_count = int((df["observation_status"] == "observed").sum())
        missing_unknown = int((df["observation_status"] == "missing_unknown").sum())
        rows.append(
            {
                "county_fips": county_fips,
                "county": county,
                "modeled_customers": modeled,
                "peak_customers_out": peak_customers,
                "peak_outage_percent": peak_percent,
                "timestamp_of_peak": peak_time,
                "first_timestamp_above_1_percent": first_time_above(valid, 1),
                "first_timestamp_above_5_percent": first_time_above(valid, 5),
                "last_timestamp_above_5_percent": last_time_above(valid, 5),
                "hours_above_1_percent": float((valid["outage_percent"] > 1).sum()) * FIFTEEN_MINUTES_HOURS,
                "hours_above_5_percent": float((valid["outage_percent"] > 5).sum()) * FIFTEEN_MINUTES_HOURS,
                "hours_above_10_percent": float((valid["outage_percent"] > 10).sum()) * FIFTEEN_MINUTES_HOURS,
                "customer_outage_hours": float(valid["customers_out"].sum()) * FIFTEEN_MINUTES_HOURS,
                "recovery_time_from_peak_to_below_5_percent": recovery_time_from_peak(valid, 5),
                "percentage_of_expected_intervals_observed": observed_count / expected_count * 100 if expected_count else np.nan,
                "missing_data_warning": missing_unknown > 0,
                "missing_unknown_intervals": missing_unknown,
            }
        )
    return pd.DataFrame(rows).sort_values("peak_customers_out", ascending=False)


def calculate_statewide(full: pd.DataFrame, county_metrics: pd.DataFrame, expected_count: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    total_denominator = county_metrics["modeled_customers"].sum()
    rows = []
    for ts, df in full.groupby("run_start_time"):
        valid = df[df["customers_out"].notna()]
        rows.append(
            {
                "run_start_time": ts,
                "total_florida_customers_out": valid["customers_out"].sum(),
                "total_modeled_florida_customers_represented": total_denominator,
                "statewide_outage_percent": valid["customers_out"].sum() / total_denominator * 100 if total_denominator else np.nan,
                "counties_above_1_percent": int((valid["outage_percent"] > 1).sum()),
                "counties_above_5_percent": int((valid["outage_percent"] > 5).sum()),
                "counties_above_10_percent": int((valid["outage_percent"] > 10).sum()),
                "observed_or_imputed_counties": int(len(valid)),
                "missing_unknown_counties": int((df["observation_status"] == "missing_unknown").sum()),
                "coverage_rate_counties": len(valid) / len(df) if len(df) else np.nan,
            }
        )
    statewide = pd.DataFrame(rows).sort_values("run_start_time")
    peak_row = statewide.loc[statewide["total_florida_customers_out"].idxmax()]
    after_peak = statewide[statewide["run_start_time"] >= peak_row["run_start_time"]]
    below_1 = after_peak[after_peak["statewide_outage_percent"] < 1]
    below_5 = after_peak[after_peak["statewide_outage_percent"] < 5]
    metrics = pd.DataFrame(
        [
            {
                "peak_statewide_customers_out": peak_row["total_florida_customers_out"],
                "peak_statewide_outage_percent": peak_row["statewide_outage_percent"],
                "timestamp_of_statewide_peak": peak_row["run_start_time"],
                "statewide_customer_outage_hours": statewide["total_florida_customers_out"].sum() * FIFTEEN_MINUTES_HOURS,
                "duration_above_1_percent_statewide_hours": (statewide["statewide_outage_percent"] > 1).sum() * FIFTEEN_MINUTES_HOURS,
                "duration_above_5_percent_statewide_hours": (statewide["statewide_outage_percent"] > 5).sum() * FIFTEEN_MINUTES_HOURS,
                "recovery_time_from_peak_to_below_1_percent_hours": (
                    (below_1.iloc[0]["run_start_time"] - peak_row["run_start_time"]).total_seconds() / 3600 if not below_1.empty else np.nan
                ),
                "recovery_time_from_peak_to_below_5_percent_hours": (
                    (below_5.iloc[0]["run_start_time"] - peak_row["run_start_time"]).total_seconds() / 3600 if not below_5.empty else np.nan
                ),
                "maximum_counties_simultaneously_above_5_percent": statewide["counties_above_5_percent"].max(),
                "observation_coverage_rate": statewide["coverage_rate_counties"].mean(),
                "expected_15min_timestamp_count": expected_count,
                "actual_timestamp_count": statewide["run_start_time"].nunique(),
            }
        ]
    )
    return statewide, metrics


def calculate_temporal_gaps(event_data: pd.DataFrame) -> pd.DataFrame:
    timestamps = pd.Series(sorted(event_data["run_start_time"].dropna().unique()))
    if timestamps.empty:
        return pd.DataFrame(columns=["gap_start", "gap_end", "gap_hours"])
    diffs = timestamps.diff()
    gap_rows = []
    for idx, diff in diffs.items():
        if pd.notna(diff) and diff > pd.Timedelta(minutes=15):
            gap_rows.append(
                {
                    "gap_start": timestamps.iloc[idx - 1],
                    "gap_end": timestamps.iloc[idx],
                    "gap_hours": diff.total_seconds() / 3600,
                }
            )
    return pd.DataFrame(gap_rows)


def write_data_quality_summary(
    event_data: pd.DataFrame,
    full: pd.DataFrame,
    quality: dict[str, object],
    expected_count: int,
    event_dir: Path,
) -> None:
    gaps = calculate_temporal_gaps(event_data)
    missing_cols = quality.get("missing_values_by_column", {})
    rows = [
        {"metric": "original_number_of_florida_rows", "value": quality["original_florida_rows"]},
        {"metric": "rows_inside_event_window_before_cleaning", "value": quality["rows_inside_event_window_before_cleaning"]},
        {"metric": "rows_inside_event_window_after_cleaning", "value": quality["rows_inside_event_window_after_cleaning"]},
        {"metric": "number_of_unique_counties", "value": event_data["county_fips"].nunique()},
        {"metric": "earliest_timestamp", "value": event_data["run_start_time"].min()},
        {"metric": "latest_timestamp", "value": event_data["run_start_time"].max()},
        {"metric": "exact_duplicate_rows_removed", "value": quality["exact_duplicate_rows_removed"]},
        {"metric": "duplicate_county_timestamp_rows_before_aggregation", "value": quality["duplicate_county_timestamp_rows_before_aggregation"]},
        {"metric": "negative_customers_out_values", "value": quality["negative_customers_out_values"]},
        {"metric": "maximum_customers_out_value", "value": quality["maximum_customers_out_value"]},
        {"metric": "expected_15min_timestamp_count", "value": expected_count},
        {"metric": "actual_timestamp_count", "value": event_data["run_start_time"].nunique()},
        {"metric": "major_temporal_gap_count", "value": len(gaps)},
        {"metric": "missing_unknown_county_time_records", "value": int((full["observation_status"] == "missing_unknown").sum())},
        {"metric": "imputed_zero_county_time_records", "value": int((full["observation_status"] == "imputed_zero").sum())},
    ]
    for col, count in missing_cols.items():
        rows.append({"metric": f"missing_values_{col}", "value": count})
    pd.DataFrame(rows).to_csv(event_dir / "data_quality_summary.csv", index=False)
    gaps.to_csv(event_dir / "major_temporal_gaps.csv", index=False)


def customers_formatter(x: float, _pos: int) -> str:
    if abs(x) >= 1_000_000:
        return f"{x / 1_000_000:.1f}M"
    if abs(x) >= 1_000:
        return f"{x / 1_000:.0f}k"
    return f"{x:.0f}"


def save_fig(fig: plt.Figure, event_dir: Path, stem: str, pdf: bool = True) -> None:
    fig.savefig(event_dir / f"{stem}.png", dpi=300, bbox_inches="tight")
    if pdf:
        fig.savefig(event_dir / f"{stem}.pdf", bbox_inches="tight")
    plt.close(fig)


def plot_figures(event_dir: Path, statewide: pd.DataFrame, statewide_metrics: pd.DataFrame, county_metrics: pd.DataFrame, full: pd.DataFrame) -> None:
    plt.rcParams.update({"font.size": 10.5, "figure.facecolor": "white", "axes.facecolor": "white"})
    peak_time = statewide_metrics.iloc[0]["timestamp_of_statewide_peak"]
    peak_value = statewide_metrics.iloc[0]["peak_statewide_customers_out"]

    fig, ax = plt.subplots(figsize=(11, 5))
    ax.plot(statewide["run_start_time"], statewide["total_florida_customers_out"], color="#1D4ED8", linewidth=2.2)
    ax.scatter([peak_time], [peak_value], color="#DC2626", zorder=5, label=f"Peak: {peak_value:,.0f}")
    ax.set_title("Hurricane Ian: Statewide Customers Without Power", weight="bold", loc="left")
    ax.set_ylabel("Customers without power")
    ax.set_xlabel("UTC date and time")
    ax.yaxis.set_major_formatter(FuncFormatter(customers_formatter))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %d\n%H:%M", tz=statewide["run_start_time"].dt.tz))
    ax.grid(color="#D0D7DE", alpha=0.7)
    ax.legend()
    save_fig(fig, event_dir, "figure_01_statewide_customers_out")

    fig, ax = plt.subplots(figsize=(11, 5))
    ax.plot(statewide["run_start_time"], statewide["statewide_outage_percent"], color="#7C3AED", linewidth=2.2)
    ax.scatter([peak_time], [statewide_metrics.iloc[0]["peak_statewide_outage_percent"]], color="#DC2626", zorder=5)
    ax.set_title("Hurricane Ian: Statewide Outage Percentage", weight="bold", loc="left")
    ax.set_ylabel("Statewide outage (%)")
    ax.set_xlabel("UTC date and time")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %d\n%H:%M", tz=statewide["run_start_time"].dt.tz))
    ax.grid(color="#D0D7DE", alpha=0.7)
    save_fig(fig, event_dir, "figure_01b_statewide_outage_percent")

    top_peak = county_metrics.nlargest(15, "peak_customers_out").sort_values("peak_customers_out")
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.barh(top_peak["county"], top_peak["peak_customers_out"], color="#2563EB")
    ax.set_title("Top Florida Counties by Peak Customers Out", weight="bold", loc="left")
    ax.set_xlabel("Peak customers without power")
    ax.xaxis.set_major_formatter(FuncFormatter(customers_formatter))
    ax.grid(axis="x", color="#D0D7DE", alpha=0.7)
    save_fig(fig, event_dir, "figure_02_top_counties_peak_customers")

    top_pct = county_metrics.nlargest(15, "peak_outage_percent").sort_values("peak_outage_percent")
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.barh(top_pct["county"], top_pct["peak_outage_percent"], color="#EA580C")
    ax.set_title("Top Florida Counties by Peak Outage Percentage", weight="bold", loc="left")
    ax.set_xlabel("Peak county outage (%)")
    ax.grid(axis="x", color="#D0D7DE", alpha=0.7)
    save_fig(fig, event_dir, "figure_03_top_counties_peak_percent")

    top5_fips = county_metrics.nlargest(5, "peak_customers_out")["county_fips"].tolist()
    fig, ax = plt.subplots(figsize=(11, 5.5))
    for fips in top5_fips:
        county_df = full[(full["county_fips"] == fips) & full["outage_percent"].notna()].copy()
        label = county_df["county"].iloc[0] if not county_df.empty else fips
        ax.plot(county_df["run_start_time"], county_df["outage_percent"], linewidth=2, label=label)
    ax.set_title("Top Five County Outage Time Series", weight="bold", loc="left")
    ax.set_ylabel("County outage (%)")
    ax.set_xlabel("UTC date and time")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %d\n%H:%M", tz=statewide["run_start_time"].dt.tz))
    ax.grid(color="#D0D7DE", alpha=0.7)
    ax.legend(ncol=2)
    save_fig(fig, event_dir, "figure_04_top_five_county_timeseries")

    heat_fips = county_metrics.nlargest(25, "peak_outage_percent")["county_fips"].tolist()
    heat = full[full["county_fips"].isin(heat_fips)].copy()
    order = county_metrics[county_metrics["county_fips"].isin(heat_fips)].sort_values("peak_outage_percent", ascending=False)
    pivot = heat.pivot(index="county_fips", columns="run_start_time", values="outage_percent_plot").reindex(order["county_fips"])
    masked = np.ma.masked_invalid(pivot.to_numpy(dtype=float))
    cmap = plt.cm.magma.copy()
    cmap.set_bad("#C7CDD6")
    fig, ax = plt.subplots(figsize=(12, 8))
    im = ax.imshow(masked, aspect="auto", interpolation="nearest", cmap=cmap, vmin=0, vmax=100)
    ax.set_title("County-Time Outage Heatmap: Top 25 Counties by Peak Percentage", weight="bold", loc="left")
    ax.set_yticks(np.arange(len(order)))
    ax.set_yticklabels(order["county"])
    xtick_idx = np.linspace(0, len(pivot.columns) - 1, 8, dtype=int)
    ax.set_xticks(xtick_idx)
    ax.set_xticklabels([pd.Timestamp(pivot.columns[i]).strftime("%b %d\n%H:%M") for i in xtick_idx])
    ax.set_xlabel("UTC date and time")
    cbar = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02)
    cbar.set_label("Outage percentage, capped at 100% for plotting")
    ax.legend(handles=[plt.Rectangle((0, 0), 1, 1, color="#C7CDD6", label="missing_unknown")], loc="lower right")
    save_fig(fig, event_dir, "figure_05_county_outage_heatmap")

    recovery = statewide[statewide["run_start_time"] >= peak_time].copy()
    recovery["hours_since_peak"] = (recovery["run_start_time"] - peak_time).dt.total_seconds() / 3600
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(recovery["hours_since_peak"], recovery["total_florida_customers_out"], color="#16A34A", linewidth=2.2)
    ax.set_title("Statewide Recovery After Peak Outages", weight="bold", loc="left")
    ax.set_xlabel("Hours since statewide outage peak")
    ax.set_ylabel("Customers without power")
    ax.yaxis.set_major_formatter(FuncFormatter(customers_formatter))
    ax.grid(color="#D0D7DE", alpha=0.7)
    save_fig(fig, event_dir, "figure_06_statewide_recovery")

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(recovery["hours_since_peak"], recovery["total_florida_customers_out"] / peak_value * 100, color="#16A34A", linewidth=2.2)
    ax.set_title("Normalized Statewide Recovery After Peak Outages", weight="bold", loc="left")
    ax.set_xlabel("Hours since statewide outage peak")
    ax.set_ylabel("Customers out (% of peak)")
    ax.grid(color="#D0D7DE", alpha=0.7)
    save_fig(fig, event_dir, "figure_06b_normalized_recovery")

    top_hours = county_metrics.nlargest(15, "customer_outage_hours").sort_values("customer_outage_hours")
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.barh(top_hours["county"], top_hours["customer_outage_hours"], color="#9333EA")
    ax.set_title("Top Counties by Customer-Outage Hours", weight="bold", loc="left")
    ax.set_xlabel("Customer-outage hours")
    ax.xaxis.set_major_formatter(FuncFormatter(customers_formatter))
    ax.grid(axis="x", color="#D0D7DE", alpha=0.7)
    save_fig(fig, event_dir, "figure_07_customer_outage_hours")


def find_county_boundary(data_root: Path) -> Path | None:
    patterns = ["*.gpkg", "*.shp", "*.geojson", "*.json", "*.parquet"]
    candidates = []
    for pattern in patterns:
        candidates.extend(data_root.rglob(pattern))
    ranked = [p for p in candidates if re.search(r"county|counties|cb_|tl_", p.name, re.I)]
    return ranked[0] if ranked else None


def try_county_map(county_metrics: pd.DataFrame, event_dir: Path, logger: logging.Logger) -> None:
    boundary = find_county_boundary(PROJECT_DIR / "data")
    if boundary is None:
        message = (
            "No local county boundary file found. To create figure_08_florida_peak_outage_map.png, "
            "add a county boundary dataset with a county FIPS/GEOID field, such as a Census county shapefile."
        )
        logger.warning(message)
        (event_dir / "figure_08_map_skipped.txt").write_text(message, encoding="utf-8")
        return
    try:
        counties = gpd.read_file(boundary).to_crs("EPSG:4326")
    except Exception as exc:
        message = f"Found possible boundary file {boundary}, but could not read it: {exc}"
        logger.warning(message)
        (event_dir / "figure_08_map_skipped.txt").write_text(message, encoding="utf-8")
        return
    counties = standardize_columns(counties)
    fips_col = next((col for col in ["geoid", "county_fips", "fips", "fips_code"] if col in counties.columns), None)
    if fips_col is None:
        message = f"Found {boundary}, but no county FIPS/GEOID field was recognized."
        logger.warning(message)
        (event_dir / "figure_08_map_skipped.txt").write_text(message, encoding="utf-8")
        return
    counties["county_fips"] = zero_pad_fips(counties[fips_col])
    fl = counties[counties["county_fips"].str.startswith(FLORIDA_FIPS_PREFIX)].copy()
    if fl.empty:
        message = f"Found {boundary}, but no Florida counties were detected."
        logger.warning(message)
        (event_dir / "figure_08_map_skipped.txt").write_text(message, encoding="utf-8")
        return
    mapped = fl.merge(county_metrics, on="county_fips", how="left")
    mapped.to_file(event_dir / "ian_county_metrics.gpkg", layer="county_metrics", driver="GPKG")
    fig, ax = plt.subplots(figsize=(7, 8))
    mapped.plot(
        ax=ax,
        column="peak_outage_percent",
        cmap="OrRd",
        legend=True,
        missing_kwds={"color": "#D1D5DB", "label": "Missing EAGLE-I data"},
        edgecolor="white",
        linewidth=0.35,
    )
    ax.set_title("Florida Peak County Outage Percentage", weight="bold", loc="left")
    ax.axis("off")
    save_fig(fig, event_dir, "figure_08_florida_peak_outage_map")


def create_validation_tables(county_metrics: pd.DataFrame, event_dir: Path) -> None:
    validation = county_metrics[
        [
            "county_fips",
            "county",
            "peak_customers_out",
            "peak_outage_percent",
            "customer_outage_hours",
            "timestamp_of_peak",
            "hours_above_5_percent",
            "modeled_customers",
            "percentage_of_expected_intervals_observed",
        ]
    ].rename(
        columns={
            "peak_customers_out": "observed_peak_customers_out",
            "peak_outage_percent": "observed_peak_outage_percent",
            "customer_outage_hours": "observed_customer_outage_hours",
            "timestamp_of_peak": "observed_peak_timestamp",
            "hours_above_5_percent": "observed_hours_above_5_percent",
            "percentage_of_expected_intervals_observed": "observation_coverage_rate",
        }
    )
    validation.to_csv(event_dir / "ian_observed_validation_targets.csv", index=False)
    template = county_metrics[["county_fips", "county"]].copy()
    template["pypsa_peak_load_shed_mw"] = np.nan
    template["pypsa_total_load_shed_mwh"] = np.nan
    template["pypsa_peak_fraction_demand_shed"] = np.nan
    template["pypsa_peak_timestamp"] = pd.NaT
    template.to_csv(event_dir / "pypsa_county_results_template.csv", index=False)


def compare_with_pypsa_placeholder(observed: pd.DataFrame, pypsa_results: pd.DataFrame | None = None) -> pd.DataFrame:
    """Placeholder for future comparison once real county-level PyPSA outputs exist."""
    if pypsa_results is None:
        return pd.DataFrame(
            [
                {
                    "status": "not_run",
                    "reason": "No real PyPSA county results file was supplied. Statistics are intentionally not calculated.",
                    "planned_metrics": (
                        "Pearson correlation; Spearman rank correlation; MAE; RMSE; top-ten overlap; "
                        "spatial ranking comparison; peak timing difference"
                    ),
                }
            ]
        )
    raise NotImplementedError("Supply real PyPSA county results before enabling comparison statistics.")


def write_readme(paths: WorkflowPaths, args: argparse.Namespace) -> None:
    readme = f"""# EAGLE-I Florida Historical-Outage Validation

This workflow processes EAGLE-I county-level customer outage observations for Florida and prepares them for validation against the Florida PyPSA hazard model.

## What EAGLE-I Measures
EAGLE-I reports observed customers without power by county at roughly 15-minute intervals. These are customer outage observations, not transmission-line outages and not MWh of unserved energy.

## Inputs
- Yearly EAGLE-I outage CSV: `eaglei_outages_<year>.csv`
- `MCC.csv`: modeled total electric customers per county, joined by five-digit county FIPS.
- `coverage_history.csv` and `DQI.csv`: retained as source documentation/quality context, but not required for the Hurricane Ian county metrics.

## Hurricane Ian Event Window
Start: `{args.start}` UTC
End: `{args.end}` UTC

The end timestamp is treated as exclusive when building the 15-minute grid.

## Missing Data Treatment
Missing county-time rows are not automatically treated as zero. A complete county-by-15-minute grid is created with `observation_status`:
- `observed`: record exists in EAGLE-I.
- `imputed_zero`: a short internal gap, no more than one hour, bounded by valid observations before and after.
- `missing_unknown`: longer or unbounded missing gaps.

Metrics use observed plus conservatively imputed-zero records. Missing unknown intervals are flagged in county metrics.

## Outage Percentages
County outage percentage is:

`outage_percent = 100 * customers_out / modeled_customers`

Values above 100% are not silently capped. The original value is preserved and flagged; `outage_percent_plot` is capped only for visualization.

## Customer-Outage Hours
Customer-outage hours are calculated by time integration:

`sum(customers_out at valid 15-minute intervals) * 0.25 hours`

This is not the same as multiplying the peak outage by total event duration.

## Rerun Example
```powershell
python scripts/run_eaglei_florida_validation.py --year 2022 --event-name hurricane_ian_2022 --start "2022-09-26 00:00:00" --end "2022-10-10 00:00:00" --state Florida
```

## Why This Is Not a Direct PyPSA Match
PyPSA reports modeled load shedding in MW/MWh by buses or network elements. EAGLE-I reports customer counts without power by county. The workflow is intended to validate broad spatial patterns, relative severity, timing, rankings, and recovery behavior rather than require a perfect one-to-one match.
"""
    (paths.project_dir / "README.md").write_text(readme, encoding="utf-8")


def create_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run EAGLE-I Florida outage validation workflow.")
    parser.add_argument("--year", type=int, default=2022)
    parser.add_argument("--event-name", default="hurricane_ian_2022")
    parser.add_argument("--start", default="2022-09-26 00:00:00")
    parser.add_argument("--end", default="2022-10-10 00:00:00")
    parser.add_argument("--state", default="Florida")
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument("--project-dir", type=Path, default=DEFAULT_PROJECT_DIR)
    parser.add_argument("--chunksize", type=int, default=1_000_000)
    return parser


def main() -> None:
    parser = create_argument_parser()
    args = parser.parse_args()
    paths = build_paths(args.project_dir, args.event_name)
    logger = setup_logging(paths.logs_dir, args.event_name)

    start = pd.Timestamp(args.start, tz="UTC")
    end = pd.Timestamp(args.end, tz="UTC")
    expected_timestamps = pd.date_range(start=start, end=end, freq="15min", inclusive="left")
    expected_count = len(expected_timestamps)
    outage_file = args.input_dir / f"eaglei_outages_{args.year}.csv"
    if not outage_file.exists():
        raise FileNotFoundError(outage_file)

    logger.info("Inspecting headers in %s", args.input_dir)
    header_summary = inspect_headers(args.input_dir, logger)
    header_summary.to_csv(paths.event_dir / "input_header_summary.csv", index=False)

    logger.info("Loading Florida outage rows from %s in chunks", outage_file.name)
    event_data, quality = load_florida_event_outages(outage_file, start, end, args.state, args.chunksize, logger)
    event_data.to_csv(paths.event_dir / "ian_florida_outages_clean.csv", index=False)

    logger.info("Joining modeled county customer totals from MCC.csv")
    mcc = load_mcc(args.input_dir, logger)
    joined, unmatched = join_modeled_customers(event_data, mcc)
    joined.to_csv(paths.event_dir / "ian_county_outage_timeseries.csv", index=False)
    pd.DataFrame({"unmatched_county_fips": unmatched}).to_csv(paths.event_dir / "unmatched_counties.csv", index=False)

    logger.info("Building complete county-time grid and conservative zero imputations")
    full = complete_county_time_grid(joined, start, end)
    full.to_csv(paths.event_dir / "ian_county_outage_timeseries_complete_grid.csv", index=False)

    write_data_quality_summary(event_data, full, quality, expected_count, paths.event_dir)
    logger.info("Calculating county and statewide metrics")
    county_metrics = calculate_county_metrics(full, expected_count)
    county_metrics.to_csv(paths.event_dir / "ian_county_metrics.csv", index=False)
    statewide, statewide_metrics = calculate_statewide(full, county_metrics, expected_count)
    statewide.to_csv(paths.event_dir / "ian_statewide_timeseries.csv", index=False)
    statewide_metrics.to_csv(paths.event_dir / "ian_statewide_metrics.csv", index=False)

    logger.info("Creating figures")
    plot_figures(paths.event_dir, statewide, statewide_metrics, county_metrics, full)
    try_county_map(county_metrics, paths.event_dir, logger)
    create_validation_tables(county_metrics, paths.event_dir)
    placeholder = compare_with_pypsa_placeholder(pd.read_csv(paths.event_dir / "ian_observed_validation_targets.csv"))
    placeholder.to_csv(paths.event_dir / "pypsa_comparison_placeholder.csv", index=False)
    write_readme(paths, args)

    top_peak = county_metrics.nlargest(5, "peak_customers_out")[["county", "peak_customers_out"]]
    top_pct = county_metrics.nlargest(5, "peak_outage_percent")[["county", "peak_outage_percent"]]
    summary_lines = [
        f"Florida counties matched: {county_metrics['county_fips'].nunique()}",
        f"Unmatched county FIPS: {unmatched if unmatched else 'none'}",
        f"Statewide peak customers out: {statewide_metrics.iloc[0]['peak_statewide_customers_out']:,.0f}",
        f"Statewide peak outage percentage: {statewide_metrics.iloc[0]['peak_statewide_outage_percent']:.2f}%",
        f"Peak timestamp: {statewide_metrics.iloc[0]['timestamp_of_statewide_peak']}",
        f"Statewide customer-outage hours: {statewide_metrics.iloc[0]['statewide_customer_outage_hours']:,.0f}",
        "Top five counties by peak customers out:",
        top_peak.to_string(index=False),
        "Top five counties by peak outage percentage:",
        top_pct.to_string(index=False),
        f"Outputs: {paths.event_dir}",
        f"Warnings: impossible percentages={int(joined['customers_exceed_modeled'].sum())}, missing_unknown_records={int((full['observation_status'] == 'missing_unknown').sum())}",
    ]
    final_summary = "\n".join(summary_lines)
    (paths.event_dir / "final_summary.txt").write_text(final_summary, encoding="utf-8")
    print(final_summary)


if __name__ == "__main__":
    main()
