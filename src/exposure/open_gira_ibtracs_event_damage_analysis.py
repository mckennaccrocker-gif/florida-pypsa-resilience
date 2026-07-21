"""
Event-by-event tropical cyclone wind damage analysis for Florida transmission lines.

This script uses the open-gira IBTrACS wind footprint NetCDF one event at a time.
For each historical event band, it samples wind speed along each Florida overhead
transmission line, takes the maximum wind speed experienced by that line, applies
the same TC wind damage-ratio curve used elsewhere in this project, and converts
damage ratios into dollar damage using voltage-based line replacement costs.

The workflow follows the advisor-requested event logic:
    event wind footprint -> transmission line exposure -> damage ratio -> dollars

Outputs are saved in:
    data/Exposure/florida_open_gira_tc_event_damage/
"""

from __future__ import annotations

import re
from math import erf, log, sqrt
from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rasterio


PROJECT_DIR = Path(r"C:\oxford_tc_project")
ELECTRICITY_DIR = PROJECT_DIR / "data" / "Electricity"
OUTPUT_DIR = PROJECT_DIR / "data" / "Exposure" / "florida_open_gira_tc_event_damage"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

WIND_NETCDF = PROJECT_DIR / "open_gira_outputs" / "max_wind_field_FLORIDA_IBTrACS_0.nc"
FLORIDA_LINES = ELECTRICITY_DIR / "florida_transmission_lines.gpkg"

STORAGE_CRS = "EPSG:4326"
PROJECTED_CRS = "EPSG:3086"  # Florida Albers, useful for line length and sampling spacing.

# Sampling interval along each transmission line. Endpoints are always included too.
LINE_SAMPLE_SPACING_M = 5_000

# Wind analysis focuses on overhead transmission lines that are in service.
KEEP_ONLY_IN_SERVICE = True

# Replacement cost assumptions reused from the earlier TC damage workflow.
STANDARD_VOLTAGES = np.array([69, 115, 161, 230, 345, 500, 765])
LINE_COST_PER_MILE_USD = {
    69: 1_500_000,
    115: 2_500_000,
    161: 2_750_000,
    230: 3_000_000,
    345: 3_050_000,
    500: 3_600_000,
    765: 5_900_000,
}


def normal_cdf(value: float) -> float:
    """Normal CDF using the standard library, avoiding an extra scipy dependency."""
    return 0.5 * (1 + erf(value / sqrt(2)))


def baseline_expected_damage_ratio(wind_speed_ms: float) -> float:
    """
    TC wind damage-ratio curve used in the previous project scripts.

    This is the baseline lognormal expected damage ratio curve with four damage
    states. It maps wind speed in m/s to a damage ratio between 0 and 1.
    """
    if pd.isna(wind_speed_ms) or wind_speed_ms <= 0:
        return 0.0

    theta_values = np.array([30.0, 42.0, 55.0, 67.0])
    damage_ratios = np.array([0.05, 0.20, 0.50, 1.00])
    beta = 0.25

    exceedance = np.array(
        [
            normal_cdf(log(float(wind_speed_ms) / theta) / beta)
            for theta in theta_values
        ]
    )
    state_probabilities = exceedance - np.append(exceedance[1:], 0.0)
    state_probabilities = np.clip(state_probabilities, 0, 1)
    return float(np.sum(state_probabilities * damage_ratios))


def parse_voltage_kv(value) -> float:
    """Extract a numeric voltage in kV from HIFLD VOLTAGE or VOLT_CLASS fields."""
    if pd.isna(value):
        return np.nan

    text = str(value).upper()
    if "DC" in text and not re.search(r"\d", text):
        return np.nan

    numbers = [float(match) for match in re.findall(r"\d+(?:\.\d+)?", text)]
    if not numbers:
        return np.nan

    # Values above 1000 are assumed to be volts and converted to kV.
    voltages_kv = [number / 1000 if number >= 1000 else number for number in numbers]
    voltages_kv = [number for number in voltages_kv if number > 0]
    return max(voltages_kv) if voltages_kv else np.nan


def snap_voltage_to_cost_class(voltage_kv: float) -> float:
    """Map an observed voltage to the nearest available replacement-cost class."""
    if pd.isna(voltage_kv):
        return np.nan
    if voltage_kv >= STANDARD_VOLTAGES.max():
        return int(STANDARD_VOLTAGES.max())
    nearest_index = int(np.argmin(np.abs(STANDARD_VOLTAGES - voltage_kv)))
    return int(STANDARD_VOLTAGES[nearest_index])


def parse_event_ids(src: rasterio.io.DatasetReader) -> list[str]:
    """
    Pull event IDs from the NetCDF metadata.

    Rasterio exposes the open-gira NetCDF event dimension as bands, with event IDs
    stored in the NETCDF_DIM_event_id_VALUES tag.
    """
    tag_value = src.tags().get("NETCDF_DIM_event_id_VALUES", "")
    match = re.search(r"\{(.+)\}", tag_value)
    if not match:
        return [f"event_{i:03d}" for i in range(1, src.count + 1)]

    event_ids = [part.strip() for part in match.group(1).split(",") if part.strip()]
    if len(event_ids) != src.count:
        return [f"event_{i:03d}" for i in range(1, src.count + 1)]
    return event_ids


def load_florida_overhead_lines() -> gpd.GeoDataFrame:
    """Load Florida HIFLD transmission lines and keep overhead/in-service lines."""
    if not FLORIDA_LINES.exists():
        raise FileNotFoundError(
            f"Missing {FLORIDA_LINES}. Run build_florida_transmission_network.py first."
        )

    lines = gpd.read_file(FLORIDA_LINES).to_crs(STORAGE_CRS)
    lines = lines.reset_index(drop=True)

    # Keep only overhead lines for tropical cyclone wind damage.
    type_text = lines.get("TYPE", pd.Series("", index=lines.index)).fillna("").str.upper()
    lines = lines[type_text.str.contains("OVERHEAD", na=False)].copy()

    if KEEP_ONLY_IN_SERVICE and "STATUS" in lines.columns:
        status_text = lines["STATUS"].fillna("").str.upper().str.strip()
        lines = lines[status_text.eq("IN SERVICE")].copy()

    lines = lines.reset_index(drop=True)
    if "florida_line_id" not in lines.columns:
        lines["florida_line_id"] = np.arange(len(lines))

    # Compute line length and replacement cost in a projected CRS.
    projected = lines.to_crs(PROJECTED_CRS)
    lines["length_m"] = projected.length
    lines["length_km"] = lines["length_m"] / 1000
    lines["length_miles"] = lines["length_km"] / 1.609344

    voltage_source = "VOLT_CLASS" if "VOLT_CLASS" in lines.columns else "VOLTAGE"
    lines["voltage_kv_raw"] = lines[voltage_source].apply(parse_voltage_kv)
    lines["voltage_kv_cost_class"] = lines["voltage_kv_raw"].apply(
        snap_voltage_to_cost_class
    )
    lines["line_cost_per_mile_usd"] = lines["voltage_kv_cost_class"].map(
        LINE_COST_PER_MILE_USD
    )
    lines["replacement_cost_usd"] = (
        lines["length_miles"] * lines["line_cost_per_mile_usd"]
    )

    return lines


def build_line_sample_points(lines: gpd.GeoDataFrame) -> tuple[list[tuple[float, float]], np.ndarray]:
    """
    Create regularly spaced sample points along each line.

    The points are created in Florida Albers so spacing is in meters, then converted
    back to EPSG:4326 for raster sampling. Each point remembers which line it came
    from, so we can take the maximum sampled wind speed per line for each event.
    """
    projected = lines.to_crs(PROJECTED_CRS)
    sample_points = []
    sample_line_positions = []

    for line_position, geom in enumerate(projected.geometry):
        if geom is None or geom.is_empty:
            continue

        # Multipart lines are rare after clipping, but this keeps the script robust.
        parts = list(geom.geoms) if geom.geom_type == "MultiLineString" else [geom]
        for part in parts:
            length = float(part.length)
            if length <= 0:
                continue

            distances = list(np.arange(0, length, LINE_SAMPLE_SPACING_M))
            if not distances or distances[-1] < length:
                distances.append(length)

            for distance in distances:
                sample_points.append(part.interpolate(distance))
                sample_line_positions.append(line_position)

    sample_gdf = gpd.GeoDataFrame(
        {"line_position": sample_line_positions},
        geometry=sample_points,
        crs=PROJECTED_CRS,
    ).to_crs(STORAGE_CRS)

    coords = [(point.x, point.y) for point in sample_gdf.geometry]
    return coords, sample_gdf["line_position"].to_numpy()


def sample_event_band_by_line(
    band_data: np.ndarray,
    sample_rows: np.ndarray,
    sample_cols: np.ndarray,
    sample_line_positions: np.ndarray,
    n_lines: int,
) -> np.ndarray:
    """Read one event array at all line sample cells and return max wind per line."""
    sample_values = band_data[sample_rows, sample_cols]
    samples = pd.DataFrame(
        {
            "line_position": sample_line_positions,
            "wind_speed_ms": sample_values,
        }
    )
    per_line = samples.groupby("line_position")["wind_speed_ms"].max()

    max_wind = np.full(n_lines, np.nan, dtype="float64")
    max_wind[per_line.index.to_numpy(dtype=int)] = per_line.to_numpy(dtype="float64")
    return max_wind


def raster_cell_indices_for_samples(
    src: rasterio.io.DatasetReader,
    coords: list[tuple[float, float]],
    sample_line_positions: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Convert sample point coordinates to raster row/column indices once.

    The Florida raster is small, so reading all event bands into memory and using
    row/column indexing is much faster than calling rasterio.sample repeatedly.
    """
    rows = []
    cols = []
    kept_line_positions = []

    for (x, y), line_position in zip(coords, sample_line_positions):
        row, col = src.index(x, y)
        if 0 <= row < src.height and 0 <= col < src.width:
            rows.append(row)
            cols.append(col)
            kept_line_positions.append(line_position)

    return (
        np.asarray(rows, dtype=int),
        np.asarray(cols, dtype=int),
        np.asarray(kept_line_positions, dtype=int),
    )


def damage_ratio_array(wind_speeds: np.ndarray) -> np.ndarray:
    """Apply the TC damage-ratio function to a numpy array."""
    return np.array([baseline_expected_damage_ratio(value) for value in wind_speeds])


def run_event_damage_analysis() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Process every IBTrACS event band and save event/line-level damage outputs."""
    if not WIND_NETCDF.exists():
        raise FileNotFoundError(f"Missing wind NetCDF: {WIND_NETCDF}")

    lines = load_florida_overhead_lines()
    coords, sample_line_positions = build_line_sample_points(lines)

    print(f"Loaded Florida overhead in-service lines: {len(lines):,}")
    print(f"Generated line sample points: {len(coords):,}")

    line_static = lines[
        [
            "florida_line_id",
            "TYPE",
            "STATUS",
            "OWNER",
            "VOLTAGE",
            "VOLT_CLASS",
            "SUB_1",
            "SUB_2",
            "length_km",
            "length_miles",
            "voltage_kv_raw",
            "voltage_kv_cost_class",
            "line_cost_per_mile_usd",
            "replacement_cost_usd",
        ]
    ].copy()

    event_records = []
    line_event_records = []

    with rasterio.open(WIND_NETCDF) as src:
        event_ids = parse_event_ids(src)
        print(f"Open-gira event bands found: {src.count:,}")
        sample_rows, sample_cols, sample_line_positions = raster_cell_indices_for_samples(
            src, coords, sample_line_positions
        )
        wind_cube = src.read(masked=True).filled(np.nan).astype("float64")
        print(f"Sample points inside raster: {len(sample_rows):,}")

        for band_number, event_id in enumerate(event_ids, start=1):
            max_wind_by_line = sample_event_band_by_line(
                wind_cube[band_number - 1],
                sample_rows,
                sample_cols,
                sample_line_positions,
                len(lines),
            )
            damage_ratio = damage_ratio_array(max_wind_by_line)
            damage_usd = (
                line_static["replacement_cost_usd"].fillna(0).to_numpy()
                * damage_ratio
            )

            valid_wind = max_wind_by_line[np.isfinite(max_wind_by_line)]
            event_year = int(str(event_id)[:4]) if re.match(r"^\d{4}", str(event_id)) else np.nan

            event_records.append(
                {
                    "event_id": event_id,
                    "event_year": event_year,
                    "band_number": band_number,
                    "lines_sampled": int(np.isfinite(max_wind_by_line).sum()),
                    "lines_with_wind_ge_25ms": int(np.nansum(max_wind_by_line >= 25)),
                    "mean_line_max_wind_ms": float(np.nanmean(max_wind_by_line)),
                    "max_line_wind_ms": float(np.nanmax(max_wind_by_line)),
                    "total_replacement_cost_usd": float(
                        line_static["replacement_cost_usd"].fillna(0).sum()
                    ),
                    "total_network_damage_usd": float(np.nansum(damage_usd)),
                    "mean_damage_ratio": float(np.nanmean(damage_ratio)),
                    "replacement_cost_weighted_damage_ratio": float(
                        np.nansum(damage_usd)
                        / line_static["replacement_cost_usd"].fillna(0).sum()
                    ),
                }
            )

            line_event = line_static.copy()
            line_event.insert(0, "event_id", event_id)
            line_event.insert(1, "event_year", event_year)
            line_event["line_max_wind_ms"] = max_wind_by_line
            line_event["damage_ratio"] = damage_ratio
            line_event["damage_usd"] = damage_usd
            line_event_records.append(line_event.drop(columns=["geometry"], errors="ignore"))

            if band_number % 25 == 0 or band_number == src.count:
                print(f"Processed {band_number:,}/{src.count:,} events")

    event_damage = pd.DataFrame(event_records).sort_values(
        "total_network_damage_usd", ascending=False
    )
    line_event_damage = pd.concat(line_event_records, ignore_index=True)

    event_damage_path = OUTPUT_DIR / "open_gira_ibtracs_event_network_damage.csv"
    line_event_path = OUTPUT_DIR / "open_gira_ibtracs_line_event_damage.csv"
    event_damage.to_csv(event_damage_path, index=False)
    line_event_damage.to_csv(line_event_path, index=False)

    print(f"Saved event damage summary: {event_damage_path}")
    print(f"Saved line-event damage table: {line_event_path}")

    return event_damage, line_event_damage


def make_event_damage_graphs(event_damage: pd.DataFrame) -> None:
    """Create simple graphs for the historical event damage distribution."""
    if event_damage.empty:
        return

    top_events = event_damage.head(20).sort_values("total_network_damage_usd")

    fig, ax = plt.subplots(figsize=(10, 7))
    bars = ax.barh(
        top_events["event_id"],
        top_events["total_network_damage_usd"] / 1e9,
        color="#1f77b4",
    )
    for bar, value in zip(bars, top_events["total_network_damage_usd"] / 1e9):
        ax.text(
            value,
            bar.get_y() + bar.get_height() / 2,
            f" ${value:.2f}B",
            va="center",
            fontsize=8,
        )
    ax.set_xlabel("Total transmission line damage (billion USD)")
    ax.set_ylabel("IBTrACS event ID")
    ax.set_title("Top 20 Historical IBTrACS Events by Florida Transmission Line Damage")
    ax.grid(axis="x", alpha=0.25)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "open_gira_ibtracs_top20_event_damage.png", dpi=300)
    plt.close()

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.hist(event_damage["total_network_damage_usd"] / 1e9, bins=30, color="#4c78a8")
    ax.set_xlabel("Total transmission line damage (billion USD)")
    ax.set_ylabel("Number of historical events")
    ax.set_title("Distribution of Historical IBTrACS Event Damages")
    ax.grid(axis="y", alpha=0.25)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "open_gira_ibtracs_event_damage_histogram.png", dpi=300)
    plt.close()

    exceedance = event_damage.sort_values("total_network_damage_usd", ascending=False).copy()
    exceedance["rank"] = np.arange(1, len(exceedance) + 1)
    exceedance["empirical_exceedance_probability"] = exceedance["rank"] / (
        len(exceedance) + 1
    )

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(
        exceedance["empirical_exceedance_probability"],
        exceedance["total_network_damage_usd"] / 1e9,
        marker="o",
        markersize=3,
        linewidth=1.2,
    )
    ax.set_xlabel("Empirical exceedance probability among historical events")
    ax.set_ylabel("Total transmission line damage (billion USD)")
    ax.set_title("Historical Event Damage Exceedance Curve")
    ax.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "open_gira_ibtracs_event_damage_exceedance_curve.png", dpi=300)
    plt.close()

    yearly = (
        event_damage.dropna(subset=["event_year"])
        .groupby("event_year", as_index=False)["total_network_damage_usd"]
        .sum()
        .sort_values("event_year")
    )
    if not yearly.empty:
        fig, ax = plt.subplots(figsize=(11, 5))
        ax.bar(yearly["event_year"].astype(int), yearly["total_network_damage_usd"] / 1e9)
        ax.set_xlabel("Year")
        ax.set_ylabel("Total event damage in year (billion USD)")
        ax.set_title("Florida Transmission Line TC Damage by Historical Event Year")
        ax.grid(axis="y", alpha=0.25)
        plt.tight_layout()
        plt.savefig(OUTPUT_DIR / "open_gira_ibtracs_event_damage_by_year.png", dpi=300)
        plt.close()


def main() -> None:
    event_damage, line_event_damage = run_event_damage_analysis()
    make_event_damage_graphs(event_damage)

    print("\nTop 10 damaging events:")
    print(
        event_damage[
            [
                "event_id",
                "event_year",
                "max_line_wind_ms",
                "lines_with_wind_ge_25ms",
                "total_network_damage_usd",
            ]
        ]
        .head(10)
        .to_string(index=False)
    )

    print("\nDamage outputs saved in:", OUTPUT_DIR)


if __name__ == "__main__":
    main()
