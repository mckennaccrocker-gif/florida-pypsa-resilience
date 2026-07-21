from pathlib import Path
import re

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rasterio
from pyproj import Transformer
from shapely.geometry import LineString


# -----------------------------
# Paths
# -----------------------------
PROJECT = Path(r"C:\oxford_tc_project")
ELECTRICITY = PROJECT / "data" / "Electricty"
HAZARDS_FLOOD = PROJECT / "data" / "Hazards" / "Flood"
COST = PROJECT / "data" / "Cost"
EXPOSURE = PROJECT / "data" / "Exposure"
COST.mkdir(parents=True, exist_ok=True)
EXPOSURE.mkdir(parents=True, exist_ok=True)

lines_file = ELECTRICITY / "TransmissionLines2.gpkg"
replacement_cost_lookup_file = COST / "replacement_cost_lookup_table.csv"

# Edit this file to use a different flood vulnerability curve. It must contain
# flood depth in meters and a damage ratio between 0 and 1.
preferred_vulnerability_curve_file = COST / "flood_vulnerability_curve.csv"
fallback_vulnerability_curve_file = COST / "nhess_f63_energy_assets_diked_areas_curve.csv"


# -----------------------------
# Helpers
# -----------------------------
def return_period_from_path(path):
    match = re.search(r"RP(\d+)", path.name, flags=re.IGNORECASE)
    if match:
        return int(match.group(1))
    return None


def discover_flood_rasters():
    rasters_by_return_period = {}

    for path in HAZARDS_FLOOD.glob("*RP*"):
        if path.suffix.lower() not in [".vrt", ".tif", ".tiff"]:
            continue

        return_period = return_period_from_path(path)
        if return_period is None:
            continue

        existing = rasters_by_return_period.get(return_period)
        if existing is None or "USA_assets" in path.name:
            rasters_by_return_period[return_period] = path

    return sorted(rasters_by_return_period.items(), key=lambda item: item[0])


def parse_voltage_kv(value):
    if pd.isna(value):
        return np.nan

    text = str(value).upper()
    if "DC" in text and not re.search(r"\d", text):
        return np.nan

    numbers = [
        float(match)
        for match in re.findall(r"\d+(?:\.\d+)?", text)
    ]

    if not numbers:
        return np.nan

    voltages_kv = [
        number / 1000 if number >= 1000 else number
        for number in numbers
        if number > 0
    ]

    if not voltages_kv:
        return np.nan

    return max(voltages_kv)


def snap_voltage_to_cost_class(voltage_kv, standard_voltages):
    if pd.isna(voltage_kv):
        return np.nan

    standard_voltages = np.array(standard_voltages)

    if voltage_kv >= standard_voltages.max():
        return int(standard_voltages.max())

    nearest_index = np.argmin(np.abs(standard_voltages - voltage_kv))
    return int(standard_voltages[nearest_index])


def filter_lines_by_type(lines, target):
    type_text = lines["TYPE"].fillna("").astype(str)
    return lines[type_text.str.contains(target, case=False, na=False)].copy()


def load_vulnerability_curve():
    curve_file = (
        preferred_vulnerability_curve_file
        if preferred_vulnerability_curve_file.exists()
        else fallback_vulnerability_curve_file
    )

    curve = pd.read_csv(curve_file)

    depth_candidates = ["flood_depth_m", "depth_m", "depth"]
    ratio_candidates = [
        "damage_ratio",
        "nhess_f63_damage_ratio",
        "flood_damage_ratio",
        "vulnerability",
    ]

    depth_column = next(
        (column for column in depth_candidates if column in curve.columns),
        None,
    )
    ratio_column = next(
        (column for column in ratio_candidates if column in curve.columns),
        None,
    )

    if depth_column is None or ratio_column is None:
        raise ValueError(
            f"Could not identify depth/damage ratio columns in {curve_file}. "
            f"Columns found: {list(curve.columns)}"
        )

    curve = curve[[depth_column, ratio_column]].rename(
        columns={
            depth_column: "flood_depth_m",
            ratio_column: "damage_ratio",
        }
    )
    curve = curve.dropna().sort_values("flood_depth_m")
    curve["damage_ratio"] = curve["damage_ratio"].clip(0, 1)

    print("Using flood vulnerability curve:", curve_file)
    return curve, curve_file


def damage_ratio_from_depth(depth, vulnerability_curve):
    if pd.isna(depth):
        return 0.0

    depth = max(float(depth), 0.0)

    return float(
        np.interp(
            depth,
            vulnerability_curve["flood_depth_m"],
            vulnerability_curve["damage_ratio"],
            left=vulnerability_curve["damage_ratio"].iloc[0],
            right=vulnerability_curve["damage_ratio"].iloc[-1],
        )
    )


def clean_depth(value, nodata):
    if value is None or np.isnan(value):
        return np.nan
    if nodata is not None and value == nodata:
        return 0.0
    return max(float(value), 0.0)


def sample_line_segments(line, raster, transformer, spacing_m=5000):
    if line is None or line.is_empty:
        return []

    line_length = line.length
    if line_length == 0:
        return []

    n_segments = max(1, int(np.ceil(line_length / spacing_m)))
    distances = np.linspace(0, line_length, n_segments + 1)
    points = [line.interpolate(distance) for distance in distances]

    segments = []
    midpoint_coords = []

    for start, end in zip(points[:-1], points[1:]):
        segment = LineString([start, end])
        midpoint = segment.interpolate(0.5, normalized=True)
        x, y = transformer.transform(midpoint.x, midpoint.y)
        segments.append(segment)
        midpoint_coords.append((x, y))

    sampled_values = [value[0] for value in raster.sample(midpoint_coords)]

    return [
        {
            "flood_depth_m": clean_depth(value, raster.nodata),
            "length_km": segment.length / 1000,
        }
        for segment, value in zip(segments, sampled_values)
    ]


def calculate_ead(asset_damage):
    ead_rows = []

    for line_id, group in asset_damage.groupby("line_id"):
        curve = group.sort_values("aep")
        ead = np.trapezoid(curve["damage_usd"], curve["aep"])
        ead_rows.append(
            {
                "line_id": line_id,
                "ead_usd": ead,
            }
        )

    return pd.DataFrame(ead_rows)


def plot_damage_probability_curve(network_damage, output_path):
    curve = network_damage.sort_values("aep")

    x = curve["aep"].to_numpy()
    y = curve["damage_usd"].to_numpy()

    x_smooth = np.linspace(x.min(), x.max(), 300)
    y_smooth = np.interp(x_smooth, x, y)

    ead = np.trapezoid(y, x)

    plt.figure(figsize=(8, 5))
    plt.fill_between(
        x_smooth,
        y_smooth,
        color="#9ecae1",
        alpha=0.5,
        label=f"EAD = ${ead:,.0f}/year",
    )
    plt.plot(
        x_smooth,
        y_smooth,
        color="#08519c",
        linewidth=2,
    )
    plt.scatter(x, y, color="#08306b", zorder=3)

    for _, row in curve.iterrows():
        plt.text(
            row["aep"],
            row["damage_usd"],
            f"RP{int(row['return_period'])}",
            ha="center",
            va="bottom",
            fontsize=8,
        )

    plt.xlabel("Annual exceedance probability (AEP)")
    plt.ylabel("Expected damage (USD)")
    plt.title("Expected Annual Damage (EAD) Calculation")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


def plot_return_period_damage(network_damage, output_path):
    curve = network_damage.sort_values("return_period")

    plt.figure(figsize=(8, 5))
    plt.plot(
        curve["return_period"],
        curve["damage_usd"],
        marker="o",
        linewidth=2,
        color="#08519c",
    )

    for _, row in curve.iterrows():
        plt.text(
            row["return_period"],
            row["damage_usd"],
            f"${row['damage_usd'] / 1e6:.1f}M",
            ha="center",
            va="bottom",
            fontsize=8,
        )

    plt.xlabel("Return period (years)")
    plt.ylabel("Expected damage (USD)")
    plt.title("Flood Expected Damage by Return Period")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


print("Checking files...")
for path in [lines_file, replacement_cost_lookup_file, fallback_vulnerability_curve_file]:
    print(path, "exists?", path.exists())

flood_rasters = discover_flood_rasters()
print("\nFlood return-period rasters found:")
for return_period, path in flood_rasters:
    print(f"RP{return_period}:", path)

if not flood_rasters:
    raise FileNotFoundError("No flood return-period rasters found.")

vulnerability_curve, vulnerability_curve_file = load_vulnerability_curve()

cost_lookup = pd.read_csv(replacement_cost_lookup_file)
standard_voltages = cost_lookup["voltage_kv"].to_numpy()
cost_per_mile = dict(
    zip(
        cost_lookup["voltage_kv"].astype(int),
        cost_lookup["line_cost_per_mile_usd"],
    )
)

lines = gpd.read_file(lines_file).reset_index(drop=True)
lines["line_id"] = lines.index
lines = filter_lines_by_type(lines, "UNDERGROUND")
lines["voltage_class"] = np.where(
    lines["VOLT_CLASS"].notna(),
    lines["VOLT_CLASS"],
    lines["VOLTAGE"],
)
lines["voltage_kv_raw"] = lines["voltage_class"].apply(parse_voltage_kv)
lines["voltage_kv_cost_class"] = lines["voltage_kv_raw"].apply(
    lambda value: snap_voltage_to_cost_class(value, standard_voltages)
)
lines["line_cost_per_mile_usd"] = lines["voltage_kv_cost_class"].map(cost_per_mile)

lines_metric = lines.to_crs("EPSG:5070").copy()
lines_metric["length_km"] = lines_metric.geometry.length / 1000
lines_metric["length_miles"] = lines_metric.geometry.length / 1609.344
lines_metric["replacement_cost_usd"] = (
    lines_metric["length_miles"]
    * lines_metric["line_cost_per_mile_usd"]
)

print("\nUnderground transmission lines:", len(lines_metric))
print(
    "Lines with replacement cost:",
    lines_metric["replacement_cost_usd"].notna().sum(),
)

damage_rows = []

for return_period, flood_raster in flood_rasters:
    aep = 1 / return_period
    print(f"\nProcessing RP{return_period} (AEP={aep:.6f})")

    with rasterio.open(flood_raster) as src:
        transformer = Transformer.from_crs(lines_metric.crs, src.crs, always_xy=True)

        for line in lines_metric.itertuples():
            line_segments = sample_line_segments(
                line.geometry,
                src,
                transformer,
                spacing_m=5000,
            )

            segment_damage = 0.0
            segment_length = 0.0
            segment_depths = []

            for segment in line_segments:
                depth = segment["flood_depth_m"]
                length_km = segment["length_km"]
                damage_ratio = damage_ratio_from_depth(depth, vulnerability_curve)
                line_cost_per_mile = (
                    0
                    if pd.isna(line.line_cost_per_mile_usd)
                    else line.line_cost_per_mile_usd
                )
                segment_replacement_cost = (
                    length_km
                    / 1.609344
                    * line_cost_per_mile
                )
                segment_damage += segment_replacement_cost * damage_ratio
                segment_length += length_km
                segment_depths.append(depth)

            damage_rows.append(
                {
                    "line_id": line.line_id,
                    "return_period": return_period,
                    "aep": aep,
                    "damage_usd": segment_damage,
                    "replacement_cost_usd": line.replacement_cost_usd,
                    "length_km": line.length_km,
                    "sampled_length_km": segment_length,
                    "mean_flood_depth_m": (
                        float(np.nanmean(segment_depths))
                        if segment_depths
                        else np.nan
                    ),
                    "max_flood_depth_m": (
                        float(np.nanmax(segment_depths))
                        if segment_depths
                        else np.nan
                    ),
                    "voltage_kv_cost_class": line.voltage_kv_cost_class,
                    "line_cost_per_mile_usd": line.line_cost_per_mile_usd,
                }
            )

asset_damage = pd.DataFrame(damage_rows)
asset_damage = asset_damage.sort_values(["line_id", "aep"])

asset_ead = calculate_ead(asset_damage)
asset_ead = lines_metric[
    [
        "line_id",
        "ID",
        "TYPE",
        "OWNER",
        "VOLTAGE",
        "VOLT_CLASS",
        "length_km",
        "replacement_cost_usd",
    ]
].merge(asset_ead, on="line_id", how="left")

network_damage = (
    asset_damage
    .groupby(["return_period", "aep"], as_index=False)
    .agg(
        damage_usd=("damage_usd", "sum"),
        exposed_line_count=("damage_usd", lambda values: int((values > 0).sum())),
        mean_damage_usd=("damage_usd", "mean"),
        max_damage_usd=("damage_usd", "max"),
    )
    .sort_values("aep")
)

total_network_ead = asset_ead["ead_usd"].sum()
network_ead_from_curve = np.trapezoid(
    network_damage.sort_values("aep")["damage_usd"],
    network_damage.sort_values("aep")["aep"],
)

asset_damage_output = COST / "flood_ead_asset_damage_by_return_period.csv"
asset_ead_output = COST / "flood_ead_by_transmission_line.csv"
network_damage_output = COST / "flood_ead_network_damage_by_return_period.csv"
summary_output = COST / "flood_ead_summary.csv"

asset_damage.to_csv(asset_damage_output, index=False)
asset_ead.to_csv(asset_ead_output, index=False)
network_damage.to_csv(network_damage_output, index=False)

summary = pd.DataFrame(
    [
        {
            "vulnerability_curve_file": str(vulnerability_curve_file),
            "line_subset": "UNDERGROUND",
            "integration_method": "trapezoidal_rule",
            "total_network_ead_usd_per_year": total_network_ead,
            "network_ead_from_aggregated_curve_usd_per_year": network_ead_from_curve,
            "return_period_min": network_damage["return_period"].min(),
            "return_period_max": network_damage["return_period"].max(),
        }
    ]
)
summary.to_csv(summary_output, index=False)

damage_probability_plot = EXPOSURE / "flood_ead_damage_probability_curve.png"
return_period_damage_plot = EXPOSURE / "flood_ead_return_period_damage.png"

plot_damage_probability_curve(network_damage, damage_probability_plot)
plot_return_period_damage(network_damage, return_period_damage_plot)

print("\nFlood EAD complete!")
print("Total network EAD:", "${:,.0f}/year".format(total_network_ead))
print("\nSaved outputs:")
for path in [
    asset_damage_output,
    asset_ead_output,
    network_damage_output,
    summary_output,
    damage_probability_plot,
    return_period_damage_plot,
]:
    print(path)
