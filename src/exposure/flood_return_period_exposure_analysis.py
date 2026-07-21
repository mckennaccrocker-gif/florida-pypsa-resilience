from pathlib import Path
import re

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rasterio
from pyproj import Transformer
from shapely.geometry import LineString


project = Path(r"C:\oxford_tc_project")
electricity_dir = project / "data" / "Electricty"
hazards_flood_dir = project / "data" / "Hazards" / "Flood"
exposure_dir = project / "data" / "Exposure"
exposure_dir.mkdir(parents=True, exist_ok=True)

nodes_file = electricity_dir / "Nodes2.gpkg"
lines_file = electricity_dir / "TransmissionLines2.gpkg"

flood_class_order = [
    "0 m",
    "0-0.5 m",
    "0.5-1 m",
    "1-2 m",
    "2-5 m",
    ">5 m",
]

flood_class_colors = {
    "0 m": "#d9d9d9",
    "0-0.5 m": "#abd9e9",
    "0.5-1 m": "#74add1",
    "1-2 m": "#4575b4",
    "2-5 m": "#fdae61",
    ">5 m": "#d73027",
}


def return_period_from_path(path):
    match = re.search(r"RP(\d+)", path.name, flags=re.IGNORECASE)
    if match:
        return int(match.group(1))
    return None


def discover_flood_rasters():
    rasters_by_return_period = {}

    for path in hazards_flood_dir.glob("*RP*"):
        if path.suffix.lower() in [".vrt", ".tif", ".tiff"]:
            return_period = return_period_from_path(path)
            if return_period is not None:
                existing = rasters_by_return_period.get(return_period)
                if existing is None or "USA_assets" in path.name:
                    rasters_by_return_period[return_period] = path

    rasters = sorted(rasters_by_return_period.items(), key=lambda item: item[0])
    return rasters


def clean_depth(value, nodata):
    if value is None or np.isnan(value):
        return np.nan
    if nodata is not None and value == nodata:
        return 0.0
    return max(float(value), 0.0)


def classify_flood_depth(depth):
    if pd.isna(depth):
        return np.nan
    if depth == 0:
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


def sample_line_flood_segments(line, raster, to_raster_transformer, spacing_m=5000):
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
        x, y = to_raster_transformer.transform(midpoint.x, midpoint.y)
        segments.append(segment)
        midpoint_coords.append((x, y))

    sampled_values = [value[0] for value in raster.sample(midpoint_coords)]

    return [
        {
            "flood_depth_m": clean_depth(value, raster.nodata),
            "length_km": segment.length / 1000,
            "geometry": segment,
        }
        for segment, value in zip(segments, sampled_values)
    ]


def plot_return_period_bars(summary, x_column, y_column, ylabel, title, output_path):
    plt.figure(figsize=(8, 5))
    bars = plt.bar(summary[x_column].astype(str), summary[y_column])

    max_value = summary[y_column].max()
    label_offset = max_value * 0.02 if max_value > 0 else 0.1

    for bar, value in zip(bars, summary[y_column]):
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + label_offset,
            f"{value:,.0f}",
            ha="center",
            va="bottom",
        )

    plt.xlabel("Flood return period (years)")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.ylim(0, max_value * 1.16 if max_value > 0 else 1)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


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
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + label_offset,
            f"{value:,.0f}\n({percent:.1f}%)",
            ha="center",
            va="bottom",
        )

    plt.xlabel("Flood return period (years)")
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
        included_classes = flood_class_order

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

    for flood_class in included_classes:
        values = wide[flood_class].to_numpy()
        ax.bar(
            x,
            values,
            bottom=bottom,
            color=flood_class_colors[flood_class],
            label=flood_class,
        )
        bottom += values

    ax.set_xticks(x)
    ax.set_xticklabels(wide[index_column].astype(str))
    ax.set_xlabel("Flood return period (years)")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(title="Flood depth", bbox_to_anchor=(1.02, 1), loc="upper left")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()


print("Checking files...")
for path in [nodes_file, lines_file]:
    print(path, "exists?", path.exists())

flood_rasters = discover_flood_rasters()
print("\nFlood return-period rasters found:")
for return_period, path in flood_rasters:
    print(f"RP{return_period}:", path)

if not flood_rasters:
    raise FileNotFoundError(
        f"No flood return-period rasters found in {hazards_flood_dir}. "
        "Expected files such as JRC_RP100_Global.vrt."
    )

nodes = gpd.read_file(nodes_file)
nodes = nodes[nodes["country"] == "USA"].copy()

lines = gpd.read_file(lines_file)
lines = filter_lines_by_type(lines, "UNDERGROUND")
lines_metric = lines.to_crs("EPSG:5070").copy()

print("\nUSA nodes:", len(nodes))
print("Underground transmission lines:", len(lines))

node_summaries = []
line_summaries = []
node_overall = []
line_overall = []

for return_period, flood_raster in flood_rasters:
    print(f"\nProcessing flood RP{return_period}: {flood_raster}")

    with rasterio.open(flood_raster) as src:
        nodes_rp = nodes.to_crs(src.crs).copy()
        coords = [(geom.x, geom.y) for geom in nodes_rp.geometry]
        values = [value[0] for value in src.sample(coords)]
        nodes_rp["flood_depth_m"] = [
            clean_depth(value, src.nodata)
            for value in values
        ]

        transformer = Transformer.from_crs(lines_metric.crs, src.crs, always_xy=True)

        flood_segments = []
        total_lines = len(lines_metric)
        for index, line in enumerate(lines_metric.geometry, start=1):
            flood_segments.extend(
                sample_line_flood_segments(
                    line,
                    src,
                    transformer,
                    spacing_m=5000,
                )
            )
            if index % 500 == 0:
                print(f"Sampled {index}/{total_lines} underground lines...")

    nodes_rp["flood_class"] = nodes_rp["flood_depth_m"].apply(classify_flood_depth)
    nodes_rp["flood_class"] = pd.Categorical(
        nodes_rp["flood_class"],
        categories=flood_class_order,
        ordered=True,
    )

    node_summary = (
        nodes_rp
        .dropna(subset=["flood_class"])
        .groupby("flood_class", observed=False)
        .size()
        .reset_index(name="node_count")
    )
    node_summary = add_percent(node_summary, "node_count")
    node_summary["return_period"] = return_period
    node_summaries.append(node_summary)

    node_exposed_count = int((nodes_rp["flood_depth_m"].fillna(0) > 0.5).sum())
    node_overall.append(
        {
            "return_period": return_period,
            "node_count": len(nodes_rp),
            "exposed_node_count_gt_0_5m": node_exposed_count,
            "exposed_node_percent_gt_0_5m": node_exposed_count / len(nodes_rp) * 100,
            "mean_flood_depth_m": nodes_rp["flood_depth_m"].mean(),
            "max_flood_depth_m": nodes_rp["flood_depth_m"].max(),
        }
    )

    lines_rp = gpd.GeoDataFrame(flood_segments, crs=lines_metric.crs)
    lines_rp["flood_class"] = lines_rp["flood_depth_m"].apply(classify_flood_depth)
    lines_rp["flood_class"] = pd.Categorical(
        lines_rp["flood_class"],
        categories=flood_class_order,
        ordered=True,
    )

    line_summary = (
        lines_rp
        .dropna(subset=["flood_class"])
        .groupby("flood_class", observed=False)["length_km"]
        .sum()
        .reset_index()
    )
    line_summary = add_percent(line_summary, "length_km")
    line_summary["return_period"] = return_period
    line_summaries.append(line_summary)

    line_exposed_length = lines_rp.loc[
        lines_rp["flood_depth_m"].fillna(0) > 0.5,
        "length_km",
    ].sum()
    line_total_length = lines_rp["length_km"].sum()
    line_overall.append(
        {
            "return_period": return_period,
            "underground_line_length_km": line_total_length,
            "exposed_underground_line_length_gt_0_5m_km": line_exposed_length,
            "exposed_underground_line_percent_gt_0_5m": (
                line_exposed_length / line_total_length * 100
                if line_total_length > 0
                else 0
            ),
            "mean_flood_depth_m": lines_rp["flood_depth_m"].mean(),
            "max_flood_depth_m": lines_rp["flood_depth_m"].max(),
        }
    )

node_summary_all = pd.concat(node_summaries, ignore_index=True)
line_summary_all = pd.concat(line_summaries, ignore_index=True)
node_overall = pd.DataFrame(node_overall).sort_values("return_period")
line_overall = pd.DataFrame(line_overall).sort_values("return_period")

node_summary_output = exposure_dir / "nodes_flood_by_return_period_and_class.csv"
line_summary_output = exposure_dir / "underground_lines_flood_by_return_period_and_class.csv"
node_overall_output = exposure_dir / "nodes_flood_exposure_by_return_period.csv"
line_overall_output = exposure_dir / "underground_lines_flood_exposure_by_return_period.csv"

node_summary_all.to_csv(node_summary_output, index=False)
line_summary_all.to_csv(line_summary_output, index=False)
node_overall.to_csv(node_overall_output, index=False)
line_overall.to_csv(line_overall_output, index=False)

plot_return_period_bars_with_percent(
    node_overall,
    "return_period",
    "exposed_node_count_gt_0_5m",
    "exposed_node_percent_gt_0_5m",
    "Flood-exposed electricity nodes (>0.5 m)",
    "Electricity Node Flood Exposure by Return Period",
    exposure_dir / "nodes_flood_exposure_by_return_period.png",
)

plot_return_period_bars_with_percent(
    line_overall,
    "return_period",
    "exposed_underground_line_length_gt_0_5m_km",
    "exposed_underground_line_percent_gt_0_5m",
    "Flood-exposed underground line length (>0.5 m, km)",
    "Underground Transmission Line Flood Exposure by Return Period",
    exposure_dir / "underground_lines_flood_exposure_by_return_period.png",
)

plot_class_stacked(
    node_summary_all,
    "return_period",
    "flood_class",
    "node_count",
    "Number of electricity nodes",
    "Electricity Nodes by Flood Return Period and Depth Class",
    exposure_dir / "nodes_flood_depth_classes_by_return_period.png",
    included_classes=flood_class_order[1:],
)

plot_class_stacked(
    line_summary_all,
    "return_period",
    "flood_class",
    "length_km",
    "Underground transmission line length (km)",
    "Underground Lines by Flood Return Period and Depth Class",
    exposure_dir / "underground_lines_flood_depth_classes_by_return_period.png",
    included_classes=flood_class_order[1:],
)

print("\nSaved return-period exposure outputs:")
for path in [
    node_summary_output,
    line_summary_output,
    node_overall_output,
    line_overall_output,
    exposure_dir / "nodes_flood_exposure_by_return_period.png",
    exposure_dir / "underground_lines_flood_exposure_by_return_period.png",
    exposure_dir / "nodes_flood_depth_classes_by_return_period.png",
    exposure_dir / "underground_lines_flood_depth_classes_by_return_period.png",
]:
    print(path)
