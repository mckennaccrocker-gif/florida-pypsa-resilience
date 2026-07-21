from pathlib import Path
import re

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rasterio
from shapely.geometry import LineString


PROJECT = Path(r"C:\oxford_tc_project")
ELECTRICITY = PROJECT / "data" / "Electricty"
HAZARDS_TC = PROJECT / "data" / "Hazards" / "Tropical_cyclones"
EXPOSURE = PROJECT / "data" / "Exposure"
EXPOSURE.mkdir(parents=True, exist_ok=True)

NODES_FILE = ELECTRICITY / "Nodes2.gpkg"
LINES_FILE = ELECTRICITY / "TransmissionLines2.gpkg"

TC_BINS = [0, 25, 30, 35, 40, np.inf]
TC_LABELS = ["<25", "25-30", "30-35", "35-40", ">40"]
TC_CLASS_COLORS = {
    "<25": "#d9d9d9",
    "25-30": "#abd9e9",
    "30-35": "#74add1",
    "35-40": "#fdae61",
    ">40": "#d73027",
}


def return_period_from_path(path):
    patterns = [
        r"RP(\d+)",
        r"_(\d+)_YR_RP",
    ]
    for pattern in patterns:
        match = re.search(pattern, path.name, flags=re.IGNORECASE)
        if match:
            return int(match.group(1))
    return None


def discover_tc_rasters():
    rasters_by_return_period = {}

    for path in HAZARDS_TC.glob("STORM_constant_RP*_US_crop.tif"):
        return_period = return_period_from_path(path)
        if return_period is not None:
            rasters_by_return_period[return_period] = path

    for path in HAZARDS_TC.glob("STORM_FIXED_RETURN_PERIODS_constant_*_YR_RP.tif"):
        return_period = return_period_from_path(path)
        if return_period is None:
            continue
        existing = rasters_by_return_period.get(return_period)
        if existing is None:
            rasters_by_return_period[return_period] = path

    return sorted(rasters_by_return_period.items(), key=lambda item: item[0])


def clean_wind(value, nodata):
    if value is None or np.isnan(value):
        return np.nan
    if nodata is not None and value == nodata:
        return np.nan
    return max(float(value), 0.0)


def classify_wind(wind_speed):
    if pd.isna(wind_speed):
        return np.nan
    return pd.cut(
        [wind_speed],
        bins=TC_BINS,
        labels=TC_LABELS,
        include_lowest=True,
    )[0]


def add_percent(summary, value_column):
    total = summary[value_column].sum()
    summary["percent"] = np.where(
        total > 0,
        summary[value_column] / total * 100,
        0,
    )
    return summary


def filter_lines_by_type(lines, target):
    type_text = lines["TYPE"].fillna("").astype(str)
    return lines[type_text.str.contains(target, case=False, na=False)].copy()


def split_line_into_segments(line, spacing_m=5000):
    if line is None or line.is_empty:
        return []

    line_length = line.length
    if line_length == 0:
        return []

    n_segments = max(1, int(np.ceil(line_length / spacing_m)))
    distances = np.linspace(0, line_length, n_segments + 1)
    points = [line.interpolate(distance) for distance in distances]

    segments = []
    for start, end in zip(points[:-1], points[1:]):
        segment = LineString([start, end])
        segments.append(
            {
                "length_km": segment.length / 1000,
                "geometry": segment,
            }
        )

    return segments


def plot_return_period_bars_with_percent(
    summary,
    x_column,
    y_column,
    percent_column,
    ylabel,
    title,
    output_path,
):
    plt.figure(figsize=(8, 5))
    bars = plt.bar(summary[x_column].astype(str), summary[y_column])

    max_value = summary[y_column].max()
    label_offset = max_value * 0.02 if max_value > 0 else 0.1

    for bar, value, percent in zip(
        bars,
        summary[y_column],
        summary[percent_column],
    ):
        if "length" in y_column:
            label = f"{value:,.0f}\n({percent:.1f}%)"
        else:
            label = f"{int(value):,}\n({percent:.1f}%)"
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + label_offset,
            label,
            ha="center",
            va="bottom",
        )

    plt.xlabel("Tropical cyclone return period (years)")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.ylim(0, max_value * 1.22 if max_value > 0 else 1)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


def plot_class_stacked(
    summary,
    index_column,
    class_column,
    value_column,
    ylabel,
    title,
    output_path,
    included_classes=None,
):
    if included_classes is None:
        included_classes = TC_LABELS

    wide = (
        summary
        .pivot_table(
            index=index_column,
            columns=class_column,
            values=value_column,
            aggfunc="sum",
            fill_value=0,
            observed=False,
        )
        .reindex(columns=included_classes, fill_value=0)
        .reset_index()
    )

    x = np.arange(len(wide))
    bottom = np.zeros(len(wide))

    fig, ax = plt.subplots(figsize=(10, 6))

    for tc_class in included_classes:
        values = wide[tc_class].to_numpy()
        ax.bar(
            x,
            values,
            bottom=bottom,
            color=TC_CLASS_COLORS[tc_class],
            label=tc_class,
        )
        bottom += values

    ax.set_xticks(x)
    ax.set_xticklabels(wide[index_column].astype(str))
    ax.set_xlabel("Tropical cyclone return period (years)")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(title="Wind speed (m/s)", bbox_to_anchor=(1.02, 1), loc="upper left")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()


print("Checking files...")
for path in [NODES_FILE, LINES_FILE]:
    print(path, "exists?", path.exists())

tc_rasters = discover_tc_rasters()
print("\nSTORM tropical cyclone return-period rasters found:")
for return_period, path in tc_rasters:
    print(f"RP{return_period}:", path)

if not tc_rasters:
    raise FileNotFoundError(
        f"No STORM return-period rasters found in {HAZARDS_TC}. "
        "Run data/Hazards/Tropical_cyclones/download_storm_return_periods.py first."
    )

nodes = gpd.read_file(NODES_FILE)
nodes = nodes[nodes["country"] == "USA"].copy()

lines = gpd.read_file(LINES_FILE)
lines = filter_lines_by_type(lines, "OVERHEAD")
lines_metric = lines.to_crs("EPSG:5070").copy()

print("\nSplitting overhead lines into 5 km sampling segments...")
line_segments = []
for index, line in enumerate(lines_metric.geometry, start=1):
    line_segments.extend(split_line_into_segments(line, spacing_m=5000))
    if index % 10000 == 0:
        print(f"Prepared {index}/{len(lines_metric)} overhead lines...")

line_segments_metric = gpd.GeoDataFrame(line_segments, crs=lines_metric.crs)
line_segments_metric["midpoint_geometry"] = (
    line_segments_metric.geometry.interpolate(0.5, normalized=True)
)
lines_total_length_km = line_segments_metric["length_km"].sum()

print("\nUSA nodes:", len(nodes))
print("Overhead transmission lines:", len(lines))
print("Overhead line sampling segments:", len(line_segments_metric))

node_summaries = []
line_summaries = []
node_overall = []
line_overall = []

for return_period, tc_raster in tc_rasters:
    print(f"\nProcessing tropical cyclone RP{return_period}: {tc_raster}")

    with rasterio.open(tc_raster) as src:
        raster_nodata = src.nodata
        nodes_rp = nodes.to_crs(src.crs).copy()
        coords = [(geom.x, geom.y) for geom in nodes_rp.geometry]
        values = [value[0] for value in src.sample(coords)]
        nodes_rp["tc_wind_speed_ms"] = [
            clean_wind(value, raster_nodata)
            for value in values
        ]

        midpoint_gdf = gpd.GeoDataFrame(
            line_segments_metric[["length_km"]].copy(),
            geometry=line_segments_metric["midpoint_geometry"],
            crs=line_segments_metric.crs,
        ).to_crs(src.crs)
        midpoint_coords = [(geom.x, geom.y) for geom in midpoint_gdf.geometry]
        line_values = [value[0] for value in src.sample(midpoint_coords)]

    nodes_rp["tc_wind_class"] = nodes_rp["tc_wind_speed_ms"].apply(classify_wind)
    nodes_rp["tc_wind_class"] = pd.Categorical(
        nodes_rp["tc_wind_class"],
        categories=TC_LABELS,
        ordered=True,
    )

    node_summary = (
        nodes_rp
        .dropna(subset=["tc_wind_class"])
        .groupby("tc_wind_class", observed=False)
        .size()
        .reset_index(name="node_count")
    )
    node_summary = add_percent(node_summary, "node_count")
    node_summary["return_period"] = return_period
    node_summaries.append(node_summary)

    node_exposed_count = int((nodes_rp["tc_wind_speed_ms"].fillna(0) >= 25).sum())
    node_overall.append(
        {
            "return_period": return_period,
            "node_count": len(nodes_rp),
            "exposed_node_count_ge_25ms": node_exposed_count,
            "exposed_node_percent_ge_25ms": node_exposed_count / len(nodes_rp) * 100,
            "mean_tc_wind_speed_ms": nodes_rp["tc_wind_speed_ms"].mean(),
            "max_tc_wind_speed_ms": nodes_rp["tc_wind_speed_ms"].max(),
        }
    )

    line_segments_rp = line_segments_metric[["length_km"]].copy()
    line_segments_rp["tc_wind_speed_ms"] = [
        clean_wind(value, raster_nodata)
        for value in line_values
    ]
    line_segments_rp["tc_wind_class"] = line_segments_rp["tc_wind_speed_ms"].apply(
        classify_wind
    )
    line_segments_rp["tc_wind_class"] = pd.Categorical(
        line_segments_rp["tc_wind_class"],
        categories=TC_LABELS,
        ordered=True,
    )

    line_summary = (
        line_segments_rp
        .dropna(subset=["tc_wind_class"])
        .groupby("tc_wind_class", observed=False)["length_km"]
        .sum()
        .reset_index()
    )
    line_summary = add_percent(line_summary, "length_km")
    line_summary["return_period"] = return_period
    line_summaries.append(line_summary)

    line_exposed_length = line_segments_rp.loc[
        line_segments_rp["tc_wind_speed_ms"].fillna(0) >= 25,
        "length_km",
    ].sum()
    line_overall.append(
        {
            "return_period": return_period,
            "overhead_line_length_km": lines_total_length_km,
            "exposed_overhead_line_length_ge_25ms_km": line_exposed_length,
            "exposed_overhead_line_percent_ge_25ms": (
                line_exposed_length / lines_total_length_km * 100
                if lines_total_length_km > 0
                else 0
            ),
            "mean_tc_wind_speed_ms": line_segments_rp["tc_wind_speed_ms"].mean(),
            "max_tc_wind_speed_ms": line_segments_rp["tc_wind_speed_ms"].max(),
        }
    )

node_summary_all = pd.concat(node_summaries, ignore_index=True)
line_summary_all = pd.concat(line_summaries, ignore_index=True)
node_overall = pd.DataFrame(node_overall).sort_values("return_period")
line_overall = pd.DataFrame(line_overall).sort_values("return_period")

node_summary_output = EXPOSURE / "nodes_tc_by_return_period_and_class.csv"
line_summary_output = EXPOSURE / "overhead_lines_tc_by_return_period_and_class.csv"
node_overall_output = EXPOSURE / "nodes_tc_exposure_by_return_period.csv"
line_overall_output = EXPOSURE / "overhead_lines_tc_exposure_by_return_period.csv"

node_summary_all.to_csv(node_summary_output, index=False)
line_summary_all.to_csv(line_summary_output, index=False)
node_overall.to_csv(node_overall_output, index=False)
line_overall.to_csv(line_overall_output, index=False)

plot_return_period_bars_with_percent(
    node_overall,
    "return_period",
    "exposed_node_count_ge_25ms",
    "exposed_node_percent_ge_25ms",
    "TC wind-exposed electricity nodes (>=25 m/s)",
    "Electricity Node Tropical Cyclone Exposure by Return Period",
    EXPOSURE / "nodes_tc_exposure_by_return_period.png",
)

plot_return_period_bars_with_percent(
    line_overall,
    "return_period",
    "exposed_overhead_line_length_ge_25ms_km",
    "exposed_overhead_line_percent_ge_25ms",
    "TC wind-exposed overhead line length (>=25 m/s, km)",
    "Overhead Transmission Line Tropical Cyclone Exposure by Return Period",
    EXPOSURE / "overhead_lines_tc_exposure_by_return_period.png",
)

plot_class_stacked(
    node_summary_all,
    "return_period",
    "tc_wind_class",
    "node_count",
    "Number of electricity nodes",
    "Electricity Nodes by Tropical Cyclone Return Period and Wind Class",
    EXPOSURE / "nodes_tc_wind_classes_by_return_period.png",
    included_classes=TC_LABELS[1:],
)

plot_class_stacked(
    line_summary_all,
    "return_period",
    "tc_wind_class",
    "length_km",
    "Overhead transmission line length (km)",
    "Overhead Lines by Tropical Cyclone Return Period and Wind Class",
    EXPOSURE / "overhead_lines_tc_wind_classes_by_return_period.png",
    included_classes=TC_LABELS[1:],
)

print("\nSaved tropical cyclone return-period exposure outputs:")
for path in [
    node_summary_output,
    line_summary_output,
    node_overall_output,
    line_overall_output,
    EXPOSURE / "nodes_tc_exposure_by_return_period.png",
    EXPOSURE / "overhead_lines_tc_exposure_by_return_period.png",
    EXPOSURE / "nodes_tc_wind_classes_by_return_period.png",
    EXPOSURE / "overhead_lines_tc_wind_classes_by_return_period.png",
]:
    print(path)
