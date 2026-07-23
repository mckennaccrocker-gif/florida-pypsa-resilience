"""Create final EAGLE-I versus PyPSA Hurricane Ian validation figures.

The figures compare EAGLE-I county customer outages with PyPSA county demand
shedding. These are not physically identical metrics; the plots are intended to
evaluate broad spatial severity and county rankings.
"""

from __future__ import annotations

import argparse
import logging
import math
from pathlib import Path
from typing import Iterable

import geopandas as gpd
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import BoundaryNorm, ListedColormap, TwoSlopeNorm
from scipy.stats import pearsonr, spearmanr

matplotlib.use("Agg")

PROJECT_DIR = Path(__file__).resolve().parents[4]
DEFAULT_BASE = PROJECT_DIR / "data" / "Electricity" / "eaglei_florida_validation" / "pypsa_comparison"
DEFAULT_RESULTS = DEFAULT_BASE / "outputs" / "hurricane_ian_2022"
DEFAULT_OBS_TS = PROJECT_DIR / "data" / "Electricity" / "eaglei_florida_validation" / "outputs" / "hurricane_ian_2022" / "ian_statewide_timeseries.csv"
DEFAULT_COUNTIES = PROJECT_DIR / "data" / "Boundaries" / "tl_us_county.zip"
DEFAULT_OUTPUT = DEFAULT_BASE / "final_figures"

OBS_COL = "observed_peak_outage_percent"
MOD_COL = "pypsa_peak_fraction_demand_shed_percent"
OBS_NORM = "observed_peak_normalized"
MOD_NORM = "pypsa_peak_normalized"
OBS_DUR = "observed_customer_outage_hours"
MOD_DUR = "pypsa_total_load_shed_mwh"

SEVERITY_CLASSES = ["Minimal", "Moderate", "High", "Severe"]
SEVERITY_COLORS = {
    "Minimal": "#5b8fd9",
    "Moderate": "#f2c14e",
    "High": "#f28e2b",
    "Severe": "#c93434",
}


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--comparison-csv", type=Path, default=None)
    parser.add_argument("--validation-statistics-csv", type=Path, default=None)
    parser.add_argument("--county-boundary-file", type=Path, default=DEFAULT_COUNTIES)
    parser.add_argument("--observed-statewide-time-series", type=Path, default=DEFAULT_OBS_TS)
    parser.add_argument("--modeled-statewide-time-series", type=Path, default=None)
    parser.add_argument("--scenario-name", type=str, default=None)
    parser.add_argument("--output-folder", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--comparison-root", type=Path, default=DEFAULT_BASE)
    return parser.parse_args()


def setup_logging(output: Path) -> None:
    """Configure console and file logging."""
    output.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(output / "create_final_validation_graphs.log", mode="w", encoding="utf-8"),
        ],
    )


def find_file(root: Path, candidates: Iterable[str]) -> Path | None:
    """Find the first matching file by filename under a root directory."""
    if not root.exists():
        return None
    candidate_names = {name.lower() for name in candidates}
    matches = sorted(p for p in root.rglob("*") if p.is_file() and p.name.lower() in candidate_names)
    return matches[0] if matches else None


def zero_pad_fips(series: pd.Series) -> pd.Series:
    """Return five-character county FIPS strings."""
    return series.astype(str).str.replace(r"\.0$", "", regex=True).str.zfill(5)


def numeric(series: pd.Series) -> pd.Series:
    """Convert a series to numeric values."""
    return pd.to_numeric(series, errors="coerce")


def safe_corr(x: pd.Series, y: pd.Series, method: str) -> float:
    """Calculate a correlation, returning NaN when insufficient variation exists."""
    valid = pd.concat([numeric(x), numeric(y)], axis=1).replace([np.inf, -np.inf], np.nan).dropna()
    if len(valid) < 3 or valid.iloc[:, 0].nunique() < 2 or valid.iloc[:, 1].nunique() < 2:
        return float("nan")
    if method == "pearson":
        return float(pearsonr(valid.iloc[:, 0], valid.iloc[:, 1]).statistic)
    return float(spearmanr(valid.iloc[:, 0], valid.iloc[:, 1]).statistic)


def load_inputs(args: argparse.Namespace) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, str], list[str], dict[str, Path | None]]:
    """Load comparison data, statistics, and resolved source paths."""
    comparison_csv = args.comparison_csv or find_file(args.comparison_root, ["ian_eaglei_pypsa_county_comparison.csv"])
    stats_csv = args.validation_statistics_csv or find_file(args.comparison_root, ["ian_validation_statistics.csv"])
    if comparison_csv is None:
        raise FileNotFoundError("Could not locate ian_eaglei_pypsa_county_comparison.csv under comparison root.")
    if stats_csv is None:
        raise FileNotFoundError("Could not locate ian_validation_statistics.csv under comparison root.")

    logging.info("Comparison CSV: %s", comparison_csv)
    logging.info("Validation statistics CSV: %s", stats_csv)
    logging.info("County boundary file: %s", args.county_boundary_file)
    logging.info("Observed statewide time series: %s", args.observed_statewide_time_series)
    logging.info("Modeled statewide time series: %s", args.modeled_statewide_time_series)

    comp = pd.read_csv(comparison_csv)
    stats_df = pd.read_csv(stats_csv)
    stats = dict(zip(stats_df["metric"].astype(str), stats_df["value"].astype(str)))
    source_paths = {
        "comparison_csv": comparison_csv,
        "validation_statistics_csv": stats_csv,
        "county_boundary_file": args.county_boundary_file if args.county_boundary_file and args.county_boundary_file.exists() else None,
        "observed_statewide_time_series": args.observed_statewide_time_series
        if args.observed_statewide_time_series and args.observed_statewide_time_series.exists()
        else None,
        "modeled_statewide_time_series": args.modeled_statewide_time_series
        if args.modeled_statewide_time_series and args.modeled_statewide_time_series.exists()
        else None,
    }
    warnings = inspect_and_prepare(comp)
    return comp, stats_df, stats, warnings, source_paths


def inspect_and_prepare(df: pd.DataFrame) -> list[str]:
    """Inspect and standardize the comparison table in place."""
    required = ["county_fips", "county", OBS_COL, MOD_COL, OBS_DUR, MOD_DUR]
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"Comparison table is missing required columns: {missing}")
    warnings: list[str] = []
    logging.info("Comparison table shape: %s", df.shape)
    logging.info("Comparison table columns and dtypes:\n%s", df.dtypes.to_string())

    df["county_fips"] = zero_pad_fips(df["county_fips"])
    for col in [OBS_COL, MOD_COL, OBS_DUR, MOD_DUR, OBS_NORM, MOD_NORM, "observed_duration_normalized", "pypsa_duration_normalized"]:
        if col in df.columns:
            df[col] = numeric(df[col])

    if OBS_NORM not in df.columns:
        mx = df[OBS_COL].max()
        df[OBS_NORM] = df[OBS_COL] / mx if pd.notna(mx) and mx != 0 else np.nan
    if MOD_NORM not in df.columns:
        mx = df[MOD_COL].max()
        df[MOD_NORM] = df[MOD_COL] / mx if pd.notna(mx) and mx != 0 else np.nan
    if "observed_duration_normalized" not in df.columns:
        mx = df[OBS_DUR].max()
        df["observed_duration_normalized"] = df[OBS_DUR] / mx if pd.notna(mx) and mx != 0 else np.nan
    if "pypsa_duration_normalized" not in df.columns:
        mx = df[MOD_DUR].max()
        df["pypsa_duration_normalized"] = df[MOD_DUR] / mx if pd.notna(mx) and mx != 0 else np.nan

    duplicate_count = int(df["county_fips"].duplicated().sum())
    if duplicate_count:
        warnings.append(f"{duplicate_count} duplicate county_fips rows found.")
    for col in [OBS_COL, MOD_COL]:
        missing_count = int(df[col].isna().sum())
        inf_count = int(np.isinf(df[col].replace([np.inf, -np.inf], np.nan)).sum())
        out_count = int(((df[col] < 0) | (df[col] > 100)).sum())
        if missing_count:
            warnings.append(f"{col}: {missing_count} missing values.")
        if inf_count:
            warnings.append(f"{col}: {inf_count} infinite values.")
        if out_count:
            examples = ", ".join(df.loc[(df[col] < 0) | (df[col] > 100), "county"].astype(str).head(8).tolist())
            warnings.append(f"{col}: {out_count} values outside 0-100 percentage points; examples: {examples}. Values are flagged, not capped.")

    df["residual_percentage_points"] = df[MOD_COL] - df[OBS_COL]
    df["absolute_difference_percentage_points"] = df["residual_percentage_points"].abs()
    df["observed_severity_rank"] = df[OBS_COL].rank(method="min", ascending=False)
    df["modeled_severity_rank"] = df[MOD_COL].rank(method="min", ascending=False)
    df["absolute_rank_difference"] = (df["modeled_severity_rank"] - df["observed_severity_rank"]).abs()
    df["observed_severity_class"] = severity_class(df[OBS_COL])
    df["modeled_severity_class"] = severity_class(df[MOD_COL])
    return warnings


def severity_class(values: pd.Series) -> pd.Series:
    """Classify percentage severity."""
    bins = [-np.inf, 1, 5, 20, np.inf]
    return pd.cut(values, bins=bins, labels=SEVERITY_CLASSES, right=False)


def matched(df: pd.DataFrame) -> pd.DataFrame:
    """Return rows with both observed and modeled peak values."""
    return df[df[OBS_COL].notna() & df[MOD_COL].notna()].copy()


def save_figure(fig: plt.Figure, output: Path, stem: str, produced: list[str]) -> None:
    """Save a figure as PNG and PDF."""
    png = output / f"{stem}.png"
    pdf = output / f"{stem}.pdf"
    fig.savefig(png, dpi=300, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)
    produced.extend([str(png), str(pdf)])


def add_footer(ax: plt.Axes, text: str) -> None:
    """Add a small explanatory footer to an axis."""
    ax.text(0, -0.16, text, transform=ax.transAxes, ha="left", va="top", fontsize=9, color="#555555")


def annotate_points(ax: plt.Axes, df: pd.DataFrame, x: str, y: str, label_by: str, n: int) -> None:
    """Annotate selected county points."""
    sub = df.nlargest(n, label_by)
    for _, row in sub.iterrows():
        ax.annotate(
            str(row["county"]),
            (row[x], row[y]),
            xytext=(5, 4),
            textcoords="offset points",
            fontsize=8,
            color="#222222",
        )


def scenario_name(df: pd.DataFrame, requested: str | None) -> str:
    """Resolve the scenario name for captions."""
    if requested:
        return requested
    if "scenario_name" in df.columns and df["scenario_name"].notna().any():
        return str(df["scenario_name"].dropna().iloc[0])
    return "unknown PyPSA scenario"


def figure_1_scatter(df: pd.DataFrame, stats: dict[str, str], scenario: str, output: Path, produced: list[str]) -> None:
    """Create observed-versus-modeled peak severity scatter."""
    data = matched(df)
    lim = max(1, float(np.nanmax([data[OBS_COL].max(), data[MOD_COL].max()])))
    fig, ax = plt.subplots(figsize=(9, 8))
    ax.scatter(data[OBS_COL], data[MOD_COL], s=54, color="#2f6c9f", edgecolor="white", linewidth=0.8, alpha=0.9)
    ax.plot([0, lim], [0, lim], linestyle="--", color="#444444", linewidth=1.4, label="1:1 reference")
    annotate_points(ax, data, OBS_COL, MOD_COL, "absolute_difference_percentage_points", 8)
    ax.set_xlim(0, lim * 1.05)
    ax.set_ylim(0, lim * 1.05)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("Observed outage: EAGLE-I peak customers without power (%)")
    ax.set_ylabel("Modeled load shedding: PyPSA peak county demand not served (%)")
    ax.set_title("County Peak Severity: Observed Outages vs. Modeled Load Shedding", fontweight="bold")
    subtitle = (
        f"Scenario: {scenario} | n={len(data)} | Pearson={float(stats.get('peak_pearson', np.nan)):.2f}, "
        f"Spearman={float(stats.get('peak_spearman', np.nan)):.2f}, MAE={float(stats.get('peak_mae_percentage_points', np.nan)):.1f} pp, "
        f"RMSE={float(stats.get('peak_rmse_percentage_points', np.nan)):.1f} pp"
    )
    ax.text(0, 1.02, subtitle, transform=ax.transAxes, ha="left", fontsize=10)
    ax.legend(loc="upper left")
    ax.grid(True, alpha=0.25)
    add_footer(ax, "Magnitude comparison is cautious: EAGLE-I includes distribution outages, while PyPSA models transmission-level demand not served.")
    save_figure(fig, output, "01_observed_vs_modeled_scatter", produced)


def figure_2_normalized(df: pd.DataFrame, output: Path, produced: list[str]) -> None:
    """Create normalized spatial severity scatter."""
    data = matched(df)
    data["normalized_abs_diff"] = (data[MOD_NORM] - data[OBS_NORM]).abs()
    fig, ax = plt.subplots(figsize=(8.5, 8))
    ax.scatter(data[OBS_NORM], data[MOD_NORM], s=54, color="#5d76a9", edgecolor="white", linewidth=0.8)
    ax.plot([0, 1], [0, 1], linestyle="--", color="#444444", linewidth=1.3)
    annotate_points(ax, data, OBS_NORM, MOD_NORM, "normalized_abs_diff", 8)
    ax.set_xlim(0, 1.05)
    ax.set_ylim(0, 1.05)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("Observed outage severity normalized to county maximum")
    ax.set_ylabel("Modeled load-shedding severity normalized to county maximum")
    ax.set_title("Normalized County Severity Shows Weak Spatial Agreement", fontweight="bold")
    ax.text(
        0,
        1.02,
        f"Pearson={safe_corr(data[OBS_NORM], data[MOD_NORM], 'pearson'):.2f}; "
        f"Spearman={safe_corr(data[OBS_NORM], data[MOD_NORM], 'spearman'):.2f}",
        transform=ax.transAxes,
        ha="left",
        fontsize=10,
    )
    ax.grid(True, alpha=0.25)
    add_footer(ax, "Normalization tests relative county severity rather than direct outage and load-shedding magnitudes.")
    save_figure(fig, output, "02_normalized_spatial_severity", produced)


def figure_3_top15(df: pd.DataFrame, scenario: str, output: Path, produced: list[str]) -> None:
    """Create top observed counties grouped bar chart."""
    data = matched(df).nlargest(15, OBS_COL).sort_values(OBS_COL, ascending=True)
    y = np.arange(len(data))
    height = 0.38
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.barh(y - height / 2, data[OBS_COL], height=height, color="#324f7b", label="Observed outage")
    ax.barh(y + height / 2, data[MOD_COL], height=height, color="#d95f02", label="Modeled load shedding")
    ax.set_yticks(y)
    ax.set_yticklabels(data["county"])
    ax.set_xlabel("Peak county severity (%)")
    ax.set_title("Highest Observed-Outage Counties Are Mostly Not Matched by PyPSA Load Shedding", fontweight="bold")
    ax.text(0, 1.02, f"Scenario: {scenario}", transform=ax.transAxes, ha="left", fontsize=10)
    xmax = max(data[OBS_COL].max(), data[MOD_COL].max())
    for yi, obs, mod in zip(y, data[OBS_COL], data[MOD_COL]):
        ax.text(obs + xmax * 0.01, yi - height / 2, f"{obs:.1f}%", va="center", fontsize=8)
        if mod > 0:
            ax.text(mod + xmax * 0.01, yi + height / 2, f"{mod:.1f}%", va="center", fontsize=8)
    ax.legend(loc="lower right")
    ax.grid(axis="x", alpha=0.25)
    save_figure(fig, output, "03_top15_observed_vs_modeled", produced)


def figure_4_rank(df: pd.DataFrame, stats: dict[str, str], output: Path, produced: list[str]) -> None:
    """Create county rank agreement graph."""
    data = matched(df)
    max_rank = int(max(data["observed_severity_rank"].max(), data["modeled_severity_rank"].max()))
    fig, ax = plt.subplots(figsize=(8.8, 8))
    ax.scatter(data["observed_severity_rank"], data["modeled_severity_rank"], s=48, color="#4377a9", edgecolor="white", linewidth=0.8)
    ax.plot([1, max_rank], [1, max_rank], linestyle="--", color="#444444", linewidth=1.3)
    annotate_points(ax, data, "observed_severity_rank", "modeled_severity_rank", "absolute_rank_difference", 10)
    ax.set_xlim(0, max_rank + 2)
    ax.set_ylim(max_rank + 2, 0)
    ax.set_xlabel("Observed severity rank (1 = highest outage)")
    ax.set_ylabel("Modeled severity rank (1 = highest load shedding)")
    ax.set_title("County Severity Rankings Have Low Agreement", fontweight="bold")
    ax.text(
        0,
        1.02,
        f"Spearman={float(stats.get('peak_spearman', np.nan)):.2f}; "
        f"top-5 overlap={stats.get('top5_overlap_count', 'NA')}; "
        f"top-10 overlap={stats.get('top10_overlap_count', stats.get('top10_overlap', 'NA'))}; "
        f"top-10 Jaccard={float(stats.get('top10_jaccard', np.nan)):.2f}",
        transform=ax.transAxes,
        ha="left",
        fontsize=10,
    )
    ax.grid(True, alpha=0.25)
    save_figure(fig, output, "04_county_rank_agreement", produced)


def figure_5_residuals(df: pd.DataFrame, output: Path, produced: list[str]) -> None:
    """Create largest residual bar chart."""
    data = matched(df).nlargest(20, "absolute_difference_percentage_points").sort_values("residual_percentage_points")
    colors = np.where(data["residual_percentage_points"] >= 0, "#d95f02", "#386cb0")
    fig, ax = plt.subplots(figsize=(10, 8.5))
    ax.barh(data["county"], data["residual_percentage_points"], color=colors)
    ax.axvline(0, color="#333333", linewidth=1)
    for y, value in enumerate(data["residual_percentage_points"]):
        ha = "left" if value >= 0 else "right"
        offset = 1.0 if value >= 0 else -1.0
        ax.text(value + offset, y, f"{value:+.1f} pp", va="center", ha=ha, fontsize=8)
    ax.set_xlabel("Modeled load shedding minus observed outage (percentage points)")
    ax.set_title("Largest County Differences Between Observed Outages and Modeled Load Shedding", fontweight="bold")
    ax.text(0, 1.02, "Negative values mean PyPSA is lower than EAGLE-I; positive values mean PyPSA is higher.", transform=ax.transAxes, ha="left", fontsize=10)
    ax.grid(axis="x", alpha=0.25)
    save_figure(fig, output, "05_largest_model_data_differences", produced)


def figure_6_bland_altman(df: pd.DataFrame, output: Path, produced: list[str]) -> None:
    """Create Bland-Altman-style agreement diagnostic."""
    data = matched(df).copy()
    data["mean_severity"] = (data[OBS_COL] + data[MOD_COL]) / 2
    data["difference"] = data[MOD_COL] - data[OBS_COL]
    mean_bias = data["difference"].mean()
    sd = data["difference"].std(ddof=1)
    fig, ax = plt.subplots(figsize=(9, 7))
    ax.scatter(data["mean_severity"], data["difference"], s=52, color="#5c7aa8", edgecolor="white", linewidth=0.8)
    for val, label, color in [
        (mean_bias, "Mean bias", "#222222"),
        (mean_bias + 1.96 * sd, "+1.96 SD", "#b23a48"),
        (mean_bias - 1.96 * sd, "-1.96 SD", "#b23a48"),
    ]:
        ax.axhline(val, color=color, linestyle="--", linewidth=1.2)
        ax.text(ax.get_xlim()[1], val, f" {label}: {val:.1f}", va="center", fontsize=8, color=color)
    annotate_points(ax, data.assign(abs_difference=data["difference"].abs()), "mean_severity", "difference", "abs_difference", 8)
    ax.set_xlabel("Mean of observed outage and modeled load shedding (%)")
    ax.set_ylabel("Modeled minus observed (percentage points)")
    ax.set_title("Agreement Diagnostic Shows Systematic Underprediction of Customer-Outage Severity", fontweight="bold")
    ax.grid(True, alpha=0.25)
    add_footer(ax, "Diagnostic only: customer outages and demand shedding are not interchangeable physical quantities.")
    save_figure(fig, output, "06_agreement_diagnostic", produced)


def plot_duration_scatter(df: pd.DataFrame, x: str, y: str, stem: str, title: str, xlabel: str, ylabel: str, output: Path, produced: list[str]) -> None:
    """Create a cumulative severity scatter plot."""
    data = matched(df).copy()
    data["duration_abs_diff"] = (numeric(data[y]) - numeric(data[x])).abs()
    positive = (data[x] > 0) & (data[y] > 0)
    spans_orders = positive.any() and (data.loc[positive, [x, y]].max().max() / max(data.loc[positive, [x, y]].min().min(), 1e-9) > 1000)
    fig, ax = plt.subplots(figsize=(9, 7.5))
    plot_data = data.loc[positive].copy() if spans_orders else data
    ax.scatter(plot_data[x], plot_data[y], s=52, color="#4b78a8", edgecolor="white", linewidth=0.8)
    if spans_orders:
        ax.set_xscale("log")
        ax.set_yscale("log")
        note = "Zero values are excluded from the log-scale panel."
    else:
        note = ""
    annotate_points(ax, plot_data.assign(duration_abs_diff=(numeric(plot_data[y]) - numeric(plot_data[x])).abs()), x, y, "duration_abs_diff", 8)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontweight="bold")
    ax.text(0, 1.02, f"Pearson={safe_corr(plot_data[x], plot_data[y], 'pearson'):.2f}; Spearman={safe_corr(plot_data[x], plot_data[y], 'spearman'):.2f}", transform=ax.transAxes, ha="left", fontsize=10)
    ax.grid(True, alpha=0.25, which="both")
    add_footer(ax, f"Customer-outage hours and MWh have different units and are compared only as relative county-severity indicators. {note}".strip())
    save_figure(fig, output, stem, produced)


def figure_8_classes(df: pd.DataFrame, output: Path, produced: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Create severity confusion matrix and class-count chart."""
    data = matched(df)
    matrix = pd.crosstab(data["observed_severity_class"], data["modeled_severity_class"]).reindex(index=SEVERITY_CLASSES, columns=SEVERITY_CLASSES, fill_value=0)
    row_sum = matrix.sum(axis=1).replace(0, np.nan)
    fig, ax = plt.subplots(figsize=(7.8, 6.8))
    im = ax.imshow(matrix.values, cmap="Blues")
    ax.set_xticks(np.arange(len(SEVERITY_CLASSES)), labels=SEVERITY_CLASSES)
    ax.set_yticks(np.arange(len(SEVERITY_CLASSES)), labels=SEVERITY_CLASSES)
    ax.set_xlabel("Modeled load-shedding class")
    ax.set_ylabel("Observed outage class")
    exact = np.trace(matrix.values) / matrix.values.sum() if matrix.values.sum() else np.nan
    within_one = 0
    for i in range(len(SEVERITY_CLASSES)):
        for j in range(len(SEVERITY_CLASSES)):
            if abs(i - j) <= 1:
                within_one += matrix.values[i, j]
    within_one_share = within_one / matrix.values.sum() if matrix.values.sum() else np.nan
    ax.set_title("Severity-Class Agreement Between EAGLE-I and PyPSA", fontweight="bold")
    ax.text(0, 1.06, f"Exact agreement={exact:.0%}; within one class={within_one_share:.0%}", transform=ax.transAxes, ha="left", fontsize=10)
    for i in range(len(SEVERITY_CLASSES)):
        for j in range(len(SEVERITY_CLASSES)):
            count = matrix.values[i, j]
            pct = count / row_sum.iloc[i] if pd.notna(row_sum.iloc[i]) else 0
            ax.text(j, i, f"{count}\n{pct:.0%}", ha="center", va="center", fontsize=10)
    fig.colorbar(im, ax=ax, shrink=0.8, label="County count")
    save_figure(fig, output, "08_severity_confusion_matrix", produced)

    counts = pd.DataFrame({
        "class": SEVERITY_CLASSES,
        "Observed outage": data["observed_severity_class"].value_counts().reindex(SEVERITY_CLASSES, fill_value=0).values,
        "Modeled load shedding": data["modeled_severity_class"].value_counts().reindex(SEVERITY_CLASSES, fill_value=0).values,
    })
    x = np.arange(len(SEVERITY_CLASSES))
    fig, ax = plt.subplots(figsize=(8.5, 6))
    ax.bar(x - 0.18, counts["Observed outage"], width=0.36, label="Observed outage", color="#324f7b")
    ax.bar(x + 0.18, counts["Modeled load shedding"], width=0.36, label="Modeled load shedding", color="#d95f02")
    ax.set_xticks(x, SEVERITY_CLASSES)
    ax.set_ylabel("Number of counties")
    ax.set_title("Observed Ian Outages Are More Widespread Than Modeled Load Shedding", fontweight="bold")
    ax.legend()
    ax.grid(axis="y", alpha=0.25)
    save_figure(fig, output, "08b_severity_class_counts", produced)
    return matrix, counts


def load_counties(path: Path | None) -> gpd.GeoDataFrame | None:
    """Load Florida county geometries if available."""
    if path is None or not path.exists():
        return None
    counties = gpd.read_file(path).to_crs("EPSG:4326")
    counties.columns = [c.lower() for c in counties.columns]
    fips_col = next((c for c in ["geoid", "county_fips", "fips", "fips_code"] if c in counties.columns), None)
    if fips_col is None:
        return None
    counties["county_fips"] = zero_pad_fips(counties[fips_col])
    return counties[counties["county_fips"].str.startswith("12")].copy()


def figure_9_maps(df: pd.DataFrame, counties: gpd.GeoDataFrame | None, output: Path, produced: list[str], skipped: list[str]) -> None:
    """Create side-by-side county maps and residual map."""
    if counties is None or counties.empty:
        skipped.append("Figure 9 county maps skipped: no Florida county geometry was available.")
        return
    data = matched(df)[["county_fips", "county", OBS_COL, MOD_COL, "residual_percentage_points"]]
    gdf = counties.merge(data, on="county_fips", how="left")
    vmax = max(100, math.ceil(np.nanmax([data[OBS_COL].max(), data[MOD_COL].max()]) / 10) * 10)
    breaks = [0, 1, 5, 20, 50, 100, vmax]
    if breaks[-1] == breaks[-2]:
        breaks[-1] += 1
    cmap = ListedColormap(["#edf8fb", "#b3cde3", "#8c96c6", "#fdb863", "#e34a33", "#7f0000"][: len(breaks) - 1])
    norm = BoundaryNorm(breaks, cmap.N)

    fig, axes = plt.subplots(1, 2, figsize=(12.5, 8), constrained_layout=True)
    for ax, col, title in [
        (axes[0], OBS_COL, "A. Observed outage"),
        (axes[1], MOD_COL, "B. Modeled load shedding"),
    ]:
        gdf.plot(column=col, ax=ax, cmap=cmap, norm=norm, edgecolor="#777777", linewidth=0.35, missing_kwds={"color": "#eeeeee", "label": "Missing"})
        ax.set_title(title, fontweight="bold")
        ax.set_axis_off()
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=axes, shrink=0.75, location="right")
    cbar.set_label("Peak county severity (%)")
    fig.suptitle("Florida County Severity: EAGLE-I Observed Outages vs. PyPSA Modeled Load Shedding", fontsize=15, fontweight="bold")
    save_figure(fig, output, "09_observed_modeled_florida_maps", produced)

    limit = float(np.nanmax(np.abs(data["residual_percentage_points"])))
    limit = max(limit, 1)
    fig, ax = plt.subplots(figsize=(8.5, 8))
    gdf.plot(
        column="residual_percentage_points",
        ax=ax,
        cmap="RdBu_r",
        norm=TwoSlopeNorm(vmin=-limit, vcenter=0, vmax=limit),
        edgecolor="#777777",
        linewidth=0.35,
        legend=True,
        legend_kwds={"label": "Modeled minus observed (percentage points)", "shrink": 0.75},
        missing_kwds={"color": "#eeeeee", "label": "Missing"},
    )
    ax.set_title("Where PyPSA Load Shedding Is Higher or Lower Than EAGLE-I Outages", fontweight="bold")
    ax.set_axis_off()
    save_figure(fig, output, "09b_florida_residual_map", produced)


def draw_dashboard(df: pd.DataFrame, stats: dict[str, str], counties: gpd.GeoDataFrame | None, scenario: str, output: Path, produced: list[str]) -> None:
    """Create a 2-by-2 validation dashboard."""
    data = matched(df)
    fig, axes = plt.subplots(2, 2, figsize=(15, 11))
    ax = axes[0, 0]
    lim = max(1, float(np.nanmax([data[OBS_COL].max(), data[MOD_COL].max()])))
    ax.scatter(data[OBS_COL], data[MOD_COL], s=35, color="#2f6c9f", edgecolor="white", linewidth=0.6)
    ax.plot([0, lim], [0, lim], "--", color="#444444", linewidth=1)
    ax.set_xlim(0, lim * 1.04)
    ax.set_ylim(0, lim * 1.04)
    ax.set_title("A. Peak severity scatter", fontweight="bold")
    ax.set_xlabel("Observed outage (%)")
    ax.set_ylabel("Modeled load shedding (%)")
    ax.grid(True, alpha=0.2)

    top = data.nlargest(10, OBS_COL).sort_values(OBS_COL)
    y = np.arange(len(top))
    ax = axes[0, 1]
    ax.barh(y - 0.18, top[OBS_COL], height=0.36, color="#324f7b", label="Observed")
    ax.barh(y + 0.18, top[MOD_COL], height=0.36, color="#d95f02", label="Modeled")
    ax.set_yticks(y, top["county"])
    ax.set_title("B. Top observed counties", fontweight="bold")
    ax.set_xlabel("Peak severity (%)")
    ax.legend(fontsize=8)
    ax.grid(axis="x", alpha=0.2)

    ax = axes[1, 0]
    max_rank = int(max(data["observed_severity_rank"].max(), data["modeled_severity_rank"].max()))
    ax.scatter(data["observed_severity_rank"], data["modeled_severity_rank"], s=35, color="#4377a9", edgecolor="white", linewidth=0.6)
    ax.plot([1, max_rank], [1, max_rank], "--", color="#444444", linewidth=1)
    ax.set_xlim(0, max_rank + 2)
    ax.set_ylim(max_rank + 2, 0)
    ax.set_title("C. County rank agreement", fontweight="bold")
    ax.set_xlabel("Observed rank")
    ax.set_ylabel("Modeled rank")
    ax.grid(True, alpha=0.2)

    ax = axes[1, 1]
    if counties is not None and not counties.empty:
        gdf = counties.merge(data[["county_fips", "residual_percentage_points"]], on="county_fips", how="left")
        limit = max(1, float(np.nanmax(np.abs(data["residual_percentage_points"]))))
        gdf.plot(column="residual_percentage_points", ax=ax, cmap="RdBu_r", norm=TwoSlopeNorm(vmin=-limit, vcenter=0, vmax=limit), edgecolor="#777777", linewidth=0.3)
        ax.set_axis_off()
        ax.set_title("D. Residual map", fontweight="bold")
    else:
        resid = data.nlargest(10, "absolute_difference_percentage_points").sort_values("residual_percentage_points")
        ax.barh(resid["county"], resid["residual_percentage_points"], color=np.where(resid["residual_percentage_points"] >= 0, "#d95f02", "#386cb0"))
        ax.axvline(0, color="#333333", linewidth=1)
        ax.set_title("D. Largest residuals", fontweight="bold")
        ax.set_xlabel("Modeled minus observed (pp)")
    footer = (
        f"Scenario: {scenario} | n={len(data)} | Pearson={float(stats.get('peak_pearson', np.nan)):.2f} | "
        f"Spearman={float(stats.get('peak_spearman', np.nan)):.2f} | RMSE={float(stats.get('peak_rmse_percentage_points', np.nan)):.1f} pp | "
        f"Top-10 overlap={stats.get('top10_overlap_count', stats.get('top10_overlap', 'NA'))}. "
        "EAGLE-I measures customer outages, while PyPSA represents transmission-level demand not served."
    )
    fig.suptitle("Hurricane Ian Validation Summary", fontsize=17, fontweight="bold")
    fig.text(0.02, 0.015, footer, ha="left", fontsize=10, color="#444444")
    fig.tight_layout(rect=[0, 0.04, 1, 0.95])
    save_figure(fig, output, "11_validation_summary_dashboard", produced)


def write_summary(
    output: Path,
    df: pd.DataFrame,
    stats: dict[str, str],
    warnings: list[str],
    produced: list[str],
    skipped: list[str],
    source_paths: dict[str, Path | None],
    scenario: str,
) -> None:
    """Write final graph data and Markdown summary."""
    data = matched(df).copy()
    data.to_csv(output / "final_graph_data.csv", index=False)
    over = data.nlargest(8, "residual_percentage_points")[["county", "residual_percentage_points"]]
    under = data.nsmallest(8, "residual_percentage_points")[["county", "residual_percentage_points"]]
    unmatched = df.loc[~df["county_fips"].isin(data["county_fips"]), ["county_fips", "county"]].drop_duplicates()
    lines = [
        "# Final EAGLE-I / PyPSA Hurricane Ian Validation Figures",
        "",
        "## Source files used",
        *[f"- {key}: `{value}`" for key, value in source_paths.items()],
        "",
        "## Scenario",
        f"- PyPSA scenario used: `{scenario}`",
        f"- Ian-specific scenario: `{str(scenario).startswith('ibtracs_2022266N12294')}`",
        "- Timing comparison valid: False. The PyPSA county comparison is static bus-total only and does not save bus-by-snapshot county load shedding.",
        "",
        "## County matching",
        f"- Observed counties: {int(df[OBS_COL].notna().sum())}",
        f"- Modeled counties: {int(df[MOD_COL].notna().sum())}",
        f"- Matched counties: {len(data)}",
        f"- Unmatched counties: {', '.join(unmatched['county'].astype(str).tolist()) if len(unmatched) else 'none'}",
        "",
        "## Validation statistics",
        f"- Pearson correlation: {stats.get('peak_pearson', 'NA')}",
        f"- Spearman correlation: {stats.get('peak_spearman', 'NA')}",
        f"- MAE: {stats.get('peak_mae_percentage_points', 'NA')} percentage points",
        f"- RMSE: {stats.get('peak_rmse_percentage_points', 'NA')} percentage points",
        f"- Mean bias: {stats.get('mean_bias_pypsa_minus_observed_pp', 'NA')} percentage points",
        f"- Top-five overlap: {stats.get('top5_overlap_count', 'NA')}",
        f"- Top-ten overlap: {stats.get('top10_overlap_count', stats.get('top10_overlap', 'NA'))}",
        "",
        "## Largest overpredictions",
        markdown_table(over),
        "",
        "## Largest underpredictions",
        markdown_table(under),
        "",
        "## Data-quality warnings",
        *(f"- {w}" for w in warnings),
        "",
        "## Graphs produced",
        *(f"- `{p}`" for p in produced),
        "",
        "## Graphs skipped and why",
        *(f"- {s}" for s in skipped),
        "",
        "## Recommended use",
        "- Thesis main text: Figures 1, 9, 3, and 4.",
        "- Appendix: Figures 5, 6, 8, and 7.",
        "- Advisor presentation: Figures 11 and 9.",
        "",
        "## Interpretation caution",
        "EAGLE-I measures customer outages, while PyPSA represents transmission-level demand not served. The comparison evaluates broad spatial severity and rankings, not a one-to-one physical equivalence.",
    ]
    (output / "final_graph_summary.md").write_text("\n".join(lines), encoding="utf-8")


def markdown_table(df: pd.DataFrame) -> str:
    """Render a small DataFrame as a simple Markdown table without optional dependencies."""
    if df.empty:
        return "none"
    formatted = df.copy()
    for col in formatted.columns:
        if pd.api.types.is_numeric_dtype(formatted[col]):
            formatted[col] = formatted[col].map(lambda value: f"{value:.2f}" if pd.notna(value) else "")
        else:
            formatted[col] = formatted[col].fillna("").astype(str)
    header = "| " + " | ".join(formatted.columns) + " |"
    sep = "| " + " | ".join(["---"] * len(formatted.columns)) + " |"
    rows = ["| " + " | ".join(map(str, row)) + " |" for row in formatted.to_numpy()]
    return "\n".join([header, sep, *rows])


def verify_pngs(produced: list[str], output: Path) -> None:
    """Verify that PNG files can be opened."""
    from PIL import Image

    rows = []
    for file in produced:
        path = Path(file)
        if path.suffix.lower() != ".png":
            continue
        with Image.open(path) as img:
            img.verify()
            rows.append({"file": str(path), "status": "opens", "width_px": img.width, "height_px": img.height})
    pd.DataFrame(rows).to_csv(output / "figure_open_check.csv", index=False)


def main() -> None:
    """Run the final validation graph workflow."""
    args = parse_args()
    setup_logging(args.output_folder)
    comp, _stats_df, stats, warnings, source_paths = load_inputs(args)
    scenario = scenario_name(comp, args.scenario_name)
    output = args.output_folder
    output.mkdir(parents=True, exist_ok=True)
    produced: list[str] = []
    skipped: list[str] = []
    counties = load_counties(source_paths["county_boundary_file"])

    figure_1_scatter(comp, stats, scenario, output, produced)
    figure_2_normalized(comp, output, produced)
    figure_3_top15(comp, scenario, output, produced)
    figure_4_rank(comp, stats, output, produced)
    figure_5_residuals(comp, output, produced)
    figure_6_bland_altman(comp, output, produced)
    plot_duration_scatter(
        comp,
        OBS_DUR,
        MOD_DUR,
        "07_cumulative_severity_raw",
        "Cumulative County Severity Has Weak Relative Agreement",
        "Observed customer-outage hours",
        "Modeled load shed (MWh)",
        output,
        produced,
    )
    plot_duration_scatter(
        comp,
        "observed_duration_normalized",
        "pypsa_duration_normalized",
        "07b_cumulative_severity_normalized",
        "Normalized Cumulative Severity Compares Relative County Burden",
        "Observed customer-outage hours normalized",
        "Modeled load-shed MWh normalized",
        output,
        produced,
    )
    matrix, counts = figure_8_classes(comp, output, produced)
    matrix.to_csv(output / "08_severity_confusion_matrix_data.csv")
    counts.to_csv(output / "08b_severity_class_counts_data.csv", index=False)
    figure_9_maps(comp, counties, output, produced, skipped)
    skipped.append(
        "Figure 10 statewide time-series comparison skipped: PyPSA comparison is static bus-total only, uses synthetic/nonhistorical timestamps, and does not contain aligned Hurricane Ian county/statewide time-varying load shedding."
    )
    draw_dashboard(comp, stats, counties, scenario, output, produced)
    write_summary(output, comp, stats, warnings, produced, skipped, source_paths, scenario)
    verify_pngs(produced, output)
    logging.info("Created %s figure files in %s", len(produced), output)
    for path in produced:
        print(path)


if __name__ == "__main__":
    main()
