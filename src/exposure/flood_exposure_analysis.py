from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rasterio
from pyproj import Transformer
from shapely.geometry import LineString

project = Path(r"C:\oxford_tc_project")

electricity_dir = project / "data" / "Electricty"  # misspelled folder name
hazards_dir = project / "data" / "Hazards"
exposure_dir = project / "data" / "Exposure"
exposure_dir.mkdir(parents=True, exist_ok=True)

nodes_file = electricity_dir / "Nodes2.gpkg"
lines_file = electricity_dir / "TransmissionLines2.gpkg"

flood_raster = hazards_dir / "Flood" / "JRC_RP100_Global.vrt"

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
fuel_colors = {
    "Solar": "#f4c430",
    "Gas": "#7f7f7f",
    "Hydro": "#2b8cbe",
    "Wind": "#41ab5d",
    "Oil": "#252525",
    "Waste": "#8c6bb1",
    "Coal": "#4d4d4d",
    "Biomass": "#a1d99b",
    "Storage": "#fb6a4a",
    "Geothermal": "#d95f0e",
    "Nuclear": "#31a354",
    "Cogeneration": "#756bb1",
    "Other": "#969696",
    "Petcoke": "#8c510a",
    "Unknown": "#bdbdbd",
}


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
        0
    )
    return summary


def plot_bar_with_percent(summary, class_column, value_column, ylabel, title, output_path):
    plt.figure(figsize=(8, 5))

    bars = plt.bar(
        summary[class_column].astype(str),
        summary[value_column]
    )

    max_value = summary[value_column].max()
    label_offset = max_value * 0.02 if max_value > 0 else 0.1

    for bar, percent in zip(bars, summary["percent"]):
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + label_offset,
            f"{percent:.1f}%",
            ha="center",
            va="bottom"
        )

    plt.xlabel("Flood depth class")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.ylim(0, max_value * 1.12 if max_value > 0 else 1)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


def plot_stacked_fuel_bars(summary_wide, output_path):
    fig, ax = plt.subplots(figsize=(11, 6))

    bottom = np.zeros(len(summary_wide))
    x = np.arange(len(summary_wide))

    fuel_columns = [
        column for column in summary_wide.columns
        if column not in ["flood_class", "total"]
    ]

    for fuel in fuel_columns:
        values = summary_wide[fuel].to_numpy()
        ax.bar(
            x,
            values,
            bottom=bottom,
            label=fuel,
            color=fuel_colors.get(fuel, fuel_colors["Unknown"]),
        )
        bottom += values

    max_total = summary_wide["total"].max()
    label_offset = max_total * 0.02 if max_total > 0 else 0.1

    for index, total in enumerate(summary_wide["total"]):
        ax.text(
            index,
            total + label_offset,
            f"{int(total):,}",
            ha="center",
            va="bottom",
        )

    ax.set_xticks(x)
    ax.set_xticklabels(summary_wide["flood_class"].astype(str))
    ax.set_xlabel("Flood depth class")
    ax.set_ylabel("Number of electricity nodes")
    ax.set_title("Flood-Exposed Electricity Nodes by Energy Type")
    ax.set_ylim(0, max_total * 1.16 if max_total > 0 else 1)
    ax.legend(
        title="Energy type",
        bbox_to_anchor=(1.02, 1),
        loc="upper left",
        frameon=True,
    )

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()


def sample_line_flood_segments(
    line,
    raster,
    to_raster_transformer,
    spacing_m=5000,
    line_type=None,
):
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
            "TYPE": line_type,
            "flood_depth_m": clean_depth(value, raster.nodata),
            "length_km": segment.length / 1000,
            "geometry": segment,
        }
        for segment, value in zip(segments, sampled_values)
    ]

print("Checking files...")
for path in [nodes_file, lines_file, flood_raster]:
    print(path, "exists?", path.exists())


def preview_columns(frame, columns):
    return [column for column in columns if column in frame.columns]


def filter_lines_by_type(lines, target):
    type_text = lines["TYPE"].fillna("").astype(str)
    return lines[type_text.str.contains(target, case=False, na=False)].copy()


nodes = gpd.read_file(nodes_file)
nodes = nodes[nodes["country"] == "USA"].copy()
lines = gpd.read_file(lines_file)
lines = filter_lines_by_type(lines, "UNDERGROUND")

print("\nNodes:", len(nodes))
print("Underground lines:", len(lines))
print("Nodes CRS:", nodes.crs)
print("Lines CRS:", lines.crs)

with rasterio.open(flood_raster) as src:
    print("\nFlood raster loaded!")
    print("Raster CRS:", src.crs)
    print("Raster bounds:", src.bounds)
    print("Raster shape:", src.shape)
    print("Raster nodata:", src.nodata)

# -----------------------------
# Node exposure to flood depth
# -----------------------------

with rasterio.open(flood_raster) as src:
    nodes_flood = nodes.to_crs(src.crs).copy()
    coords = [(geom.x, geom.y) for geom in nodes_flood.geometry]
    values = [value[0] for value in src.sample(coords)]

    nodes_flood["flood_depth_m"] = [
        clean_depth(value, src.nodata)
        for value in values
    ]

nodes_flood["flood_class"] = nodes_flood["flood_depth_m"].apply(classify_flood_depth)
nodes_flood["flood_class"] = pd.Categorical(
    nodes_flood["flood_class"],
    categories=flood_class_order,
    ordered=True
)

print("\nNode flood exposure complete!")
print(
    nodes_flood[
        preview_columns(
            nodes_flood,
            ["type", "name", "country", "primary_fuel", "flood_depth_m", "flood_class"],
        )
    ].head()
)
print("Nodes with flood value:", nodes_flood["flood_depth_m"].notna().sum())
print("Nodes missing flood value:", nodes_flood["flood_depth_m"].isna().sum())

nodes_flood_output = exposure_dir / "nodes_flood_exposure.gpkg"
nodes_flood.to_file(
    nodes_flood_output,
    layer="nodes_flood_exposure",
    driver="GPKG"
)
print("\nSaved:", nodes_flood_output)

node_flood_summary = (
    nodes_flood
    .dropna(subset=["flood_class"])
    .groupby("flood_class", observed=False)
    .size()
    .reset_index(name="node_count")
)
node_flood_summary = add_percent(node_flood_summary, "node_count")

print("\nElectricity nodes by flood depth class:")
print(node_flood_summary)

node_flood_summary.to_csv(
    exposure_dir / "nodes_flood_count_by_class.csv",
    index=False
)

node_flood_chart_output = exposure_dir / "nodes_by_flood_depth_class.png"
plot_bar_with_percent(
    node_flood_summary[node_flood_summary["flood_class"].astype(str) != "0 m"],
    "flood_class",
    "node_count",
    "Number of electricity nodes",
    "Electricity Node Exposure to Flooding by Depth Class",
    node_flood_chart_output
)
print("Saved node flood class chart:", node_flood_chart_output)

# -----------------------------
# Node flood exposure by energy type
# -----------------------------

nodes_flood_by_fuel = nodes_flood.copy()
nodes_flood_by_fuel["primary_fuel"] = nodes_flood_by_fuel["primary_fuel"].fillna("Unknown")

node_fuel_flood_summary = (
    nodes_flood_by_fuel
    .dropna(subset=["flood_class"])
    .groupby(["flood_class", "primary_fuel"], observed=False)
    .size()
    .reset_index(name="node_count")
)
node_fuel_flood_summary["class_total"] = (
    node_fuel_flood_summary
    .groupby("flood_class", observed=False)["node_count"]
    .transform("sum")
)
node_fuel_flood_summary["percent_of_class"] = np.where(
    node_fuel_flood_summary["class_total"] > 0,
    node_fuel_flood_summary["node_count"] / node_fuel_flood_summary["class_total"] * 100,
    0,
)

node_fuel_flood_summary.to_csv(
    exposure_dir / "nodes_flood_count_by_class_and_fuel.csv",
    index=False,
)

node_fuel_flood_plot = (
    node_fuel_flood_summary[
        node_fuel_flood_summary["flood_class"].astype(str) != "0 m"
    ]
    .pivot_table(
        index="flood_class",
        columns="primary_fuel",
        values="node_count",
        aggfunc="sum",
        fill_value=0,
        observed=False,
    )
    .reindex(flood_class_order[1:])
)

fuel_order = (
    node_fuel_flood_plot
    .sum(axis=0)
    .sort_values(ascending=False)
    .index
)
node_fuel_flood_plot = node_fuel_flood_plot[fuel_order]
node_fuel_flood_plot["total"] = node_fuel_flood_plot.sum(axis=1)
node_fuel_flood_plot = node_fuel_flood_plot.reset_index()

node_fuel_flood_wide_csv = exposure_dir / "nodes_flood_count_by_class_and_fuel_wide.csv"
node_fuel_flood_plot.to_csv(node_fuel_flood_wide_csv, index=False)

node_fuel_flood_chart_output = exposure_dir / "nodes_flood_by_depth_class_and_fuel.png"
plot_stacked_fuel_bars(
    node_fuel_flood_plot,
    node_fuel_flood_chart_output,
)
print("\nElectricity nodes by flood depth class and energy type:")
print(node_fuel_flood_plot)
print("Saved node flood energy type chart:", node_fuel_flood_chart_output)

# -----------------------------
# Transmission line exposure to flood depth
# -----------------------------

lines_metric = lines.to_crs("EPSG:5070").copy()

with rasterio.open(flood_raster) as src:
    transformer = Transformer.from_crs(lines_metric.crs, src.crs, always_xy=True)

    flood_segments = []
    total_lines = len(lines_metric)

    print("\nSampling flood depth along transmission lines...")

    for index, line_record in enumerate(lines_metric.itertuples(), start=1):
        flood_segments.extend(
            sample_line_flood_segments(
                line_record.geometry,
                src,
                transformer,
                spacing_m=5000,
                line_type=getattr(line_record, "TYPE", None),
            )
        )

        if index % 5000 == 0:
            print(f"Sampled {index}/{total_lines} lines...")

lines_flood_segments = gpd.GeoDataFrame(
    flood_segments,
    crs=lines_metric.crs
)
lines_flood_segments["flood_class"] = lines_flood_segments["flood_depth_m"].apply(
    classify_flood_depth
)
lines_flood_segments["flood_class"] = pd.Categorical(
    lines_flood_segments["flood_class"],
    categories=flood_class_order,
    ordered=True
)

print("\nTransmission line flood exposure complete!")
print("Flood line segments:", len(lines_flood_segments))
print("Total sampled line length km:", lines_flood_segments["length_km"].sum())
print(lines_flood_segments["flood_depth_m"].describe())

lines_flood_output = exposure_dir / "lines_flood_exposure_segments.gpkg"
lines_flood_segments.to_file(
    lines_flood_output,
    layer="lines_flood_exposure_segments",
    driver="GPKG"
)
print("\nSaved:", lines_flood_output)

line_flood_summary = (
    lines_flood_segments
    .dropna(subset=["flood_class"])
    .groupby("flood_class", observed=False)["length_km"]
    .sum()
    .reset_index()
)
line_flood_summary = add_percent(line_flood_summary, "length_km")

print("\nTransmission line length by flood depth class:")
print(line_flood_summary)

line_flood_summary.to_csv(
    exposure_dir / "lines_flood_length_by_class.csv",
    index=False
)

line_flood_chart_output = exposure_dir / "transmission_length_by_flood_depth_class.png"
plot_bar_with_percent(
    line_flood_summary[line_flood_summary["flood_class"].astype(str) != "0 m"],
    "flood_class",
    "length_km",
    "Transmission line length (km)",
    "Underground Transmission Line Exposure to Flooding by Depth Class",
    line_flood_chart_output
)
print("Saved transmission line flood class chart:", line_flood_chart_output)

# -----------------------------
# Map exposed transmission lines by flood depth class
# -----------------------------

fig, ax = plt.subplots(figsize=(11, 7))

for flood_class in flood_class_order:
    class_lines = lines_flood_segments[
        lines_flood_segments["flood_class"].astype(str) == flood_class
    ]
    if class_lines.empty:
        continue

    class_lines.plot(
        ax=ax,
        color=flood_class_colors[flood_class],
        linewidth=0.45,
        label=flood_class
    )

ax.set_title("Underground Transmission Lines Exposed to Flooding by Depth Class")
ax.set_axis_off()
ax.legend(
    title="Flood depth",
    loc="lower left",
    frameon=True
)

plt.tight_layout()

line_flood_map_output = exposure_dir / "transmission_lines_by_flood_depth_class_map.png"

plt.savefig(
    line_flood_map_output,
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print("Saved transmission line flood exposure map:", line_flood_map_output)
