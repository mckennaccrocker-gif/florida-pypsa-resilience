from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rasterio

project = Path(r"C:\oxford_tc_project")

electricity_dir = project / "data" / "Electricty"   # misspelled folder name
hazards_dir = project / "data" / "Hazards"
exposure_dir = project / "data" / "Exposure"
exposure_dir.mkdir(parents=True, exist_ok=True)

nodes_file = electricity_dir / "Nodes2.gpkg"
lines_file = electricity_dir / "TransmissionLines2.gpkg"

flood_raster = hazards_dir / "Flood" / "JRC_RP100_Global.vrt"
tc_raster = hazards_dir / "Tropical_cyclones" / "STORM_constant_100yr_US_crop.tif"
strong_wind_file = hazards_dir / "Strong_wind" / "Strong_Wind.gpkg"

print("Checking files...")
for path in [nodes_file, lines_file, flood_raster, tc_raster, strong_wind_file]:
    print(path, "exists?", path.exists())


def preview_columns(frame, columns):
    return [column for column in columns if column in frame.columns]


def filter_lines_by_type(lines, target):
    type_text = lines["TYPE"].fillna("").astype(str)
    return lines[type_text.str.contains(target, case=False, na=False)].copy()


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


def plot_stacked_fuel_bars(summary_wide, class_column, xlabel, title, output_path):
    fig, ax = plt.subplots(figsize=(11, 6))

    bottom = np.zeros(len(summary_wide))
    x = np.arange(len(summary_wide))

    fuel_columns = [
        column for column in summary_wide.columns
        if column not in [class_column, "total"]
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
    ax.set_xticklabels(summary_wide[class_column].astype(str))
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Number of electricity nodes")
    ax.set_title(title)
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


nodes = gpd.read_file(nodes_file)
nodes = nodes[nodes["country"] == "USA"].copy()

print("\nNodes loaded!")
print(nodes.head())
print("\nCRS:", nodes.crs)
print("Number of nodes:", len(nodes))

# Load strong wind data
wind = gpd.read_file(strong_wind_file)

print("\nStrong wind loaded!")
print("Wind CRS:", wind.crs)
print("Wind polygons:", len(wind))

# Load transmission lines
lines = gpd.read_file(lines_file)
lines = filter_lines_by_type(lines, "OVERHEAD")

print("\nTransmission lines loaded!")
print("Lines CRS:", lines.crs)
print("Number of overhead lines:", len(lines))

# Reproject nodes to match strong wind polygons
nodes_for_join = nodes.to_crs(wind.crs)

# Spatial join: each node gets the strong wind value from the county polygon it falls inside
nodes_wind = gpd.sjoin(
    nodes_for_join,
    wind[["SWND_AFREQ", "geometry"]],
    how="left",
    predicate="within"
)

# Rename for clarity
nodes_wind = nodes_wind.rename(columns={"SWND_AFREQ": "strong_wind_frequency"})

print("\nNode strong wind exposure complete!")
print(
    nodes_wind[
        preview_columns(
            nodes_wind,
            ["type", "name", "country", "primary_fuel", "strong_wind_frequency"],
        )
    ].head()
)
print("Nodes with wind value:", nodes_wind["strong_wind_frequency"].notna().sum())
print("Nodes missing wind value:", nodes_wind["strong_wind_frequency"].isna().sum())

# Save output
output_file = exposure_dir / "nodes_strong_wind_exposure.gpkg"
nodes_wind.to_file(output_file, layer="nodes_strong_wind_exposure", driver="GPKG")

nodes_wind.to_csv(
    exposure_dir / "nodes_strong_wind_exposure.csv",
    index=False
)

print("\nSaved:", output_file)

# -----------------------------
# Shared strong wind classes
# -----------------------------

bins = [0, 0.98, 2.2, 3.7, 5.35, 9]
labels = ["0-0.98", "0.98-2.2", "2.2-3.7", "3.7-5.35", ">5.35"]

# -----------------------------
# Node exposure by wind class
# -----------------------------

nodes_wind["wind_class"] = pd.cut(
    nodes_wind["strong_wind_frequency"],
    bins=bins,
    labels=labels,
    include_lowest=True
)

node_summary = (
    nodes_wind
    .groupby("wind_class", observed=True)
    .size()
    .reset_index(name="node_count")
)
node_summary["percent"] = (
    node_summary["node_count"]
    / node_summary["node_count"].sum()
    * 100
)

print("\nElectricity nodes by strong wind class:")
print(node_summary)

node_summary.to_csv(
    exposure_dir / "nodes_strong_wind_count_by_class.csv",
    index=False
)

plt.figure(figsize=(8, 5))

bars = plt.bar(
    node_summary["wind_class"].astype(str),
    node_summary["node_count"]
)

label_offset = node_summary["node_count"].max() * 0.02

for bar, percent in zip(bars, node_summary["percent"]):
    plt.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height() + label_offset,
        f"{percent:.1f}%",
        ha="center",
        va="bottom"
    )

plt.xlabel("Strong wind frequency (events/year)")
plt.ylabel("Number of electricity nodes")
plt.title("Electricity Node Exposure to Strong Wind by Class")
plt.ylim(0, node_summary["node_count"].max() * 1.12)

plt.tight_layout()

node_chart_output = exposure_dir / "nodes_by_wind_class.png"

plt.savefig(
    node_chart_output,
    dpi=300
)

plt.close()

print("Saved node wind class chart:", node_chart_output)

# -----------------------------
# Node strong wind exposure by energy type
# -----------------------------

nodes_wind_by_fuel = nodes_wind.copy()
nodes_wind_by_fuel["primary_fuel"] = (
    nodes_wind_by_fuel["primary_fuel"].fillna("Unknown")
)

node_fuel_wind_summary = (
    nodes_wind_by_fuel
    .dropna(subset=["wind_class"])
    .groupby(["wind_class", "primary_fuel"], observed=False)
    .size()
    .reset_index(name="node_count")
)
node_fuel_wind_summary["class_total"] = (
    node_fuel_wind_summary
    .groupby("wind_class", observed=False)["node_count"]
    .transform("sum")
)
node_fuel_wind_summary["percent_of_class"] = np.where(
    node_fuel_wind_summary["class_total"] > 0,
    node_fuel_wind_summary["node_count"] / node_fuel_wind_summary["class_total"] * 100,
    0,
)

node_fuel_wind_summary.to_csv(
    exposure_dir / "nodes_strong_wind_count_by_class_and_fuel.csv",
    index=False,
)

node_fuel_wind_plot = (
    node_fuel_wind_summary
    .pivot_table(
        index="wind_class",
        columns="primary_fuel",
        values="node_count",
        aggfunc="sum",
        fill_value=0,
        observed=False,
    )
    .reindex(labels)
)

fuel_order = (
    node_fuel_wind_plot
    .sum(axis=0)
    .sort_values(ascending=False)
    .index
)
node_fuel_wind_plot = node_fuel_wind_plot[fuel_order]
node_fuel_wind_plot["total"] = node_fuel_wind_plot.sum(axis=1)
node_fuel_wind_plot = node_fuel_wind_plot.reset_index()

node_fuel_wind_wide_csv = exposure_dir / "nodes_strong_wind_count_by_class_and_fuel_wide.csv"
node_fuel_wind_plot.to_csv(node_fuel_wind_wide_csv, index=False)

node_fuel_wind_chart_output = exposure_dir / "nodes_strong_wind_by_class_and_fuel.png"
plot_stacked_fuel_bars(
    node_fuel_wind_plot,
    "wind_class",
    "Strong wind frequency (events/year)",
    "Strong Wind-Exposed Electricity Nodes by Energy Type",
    node_fuel_wind_chart_output,
)

print("\nElectricity nodes by strong wind class and energy type:")
print(node_fuel_wind_plot)
print("Saved node strong wind energy type chart:", node_fuel_wind_chart_output)



# -----------------------------
# Transmission line exposure to strong wind
# Option B: split lines by wind polygons
# -----------------------------

# Reproject lines to match wind polygons
lines_for_join = lines.to_crs(wind.crs)

# Keep only useful wind columns
wind_simple = wind[["SWND_AFREQ", "geometry"]].copy()

print("\nIntersecting transmission lines with strong wind polygons...")

# This splits the lines wherever they intersect county wind polygons
lines_wind_split = gpd.overlay(
    lines_for_join,
    wind_simple,
    how="intersection"
)

# Rename for clarity
lines_wind_split = lines_wind_split.rename(
    columns={"SWND_AFREQ": "strong_wind_frequency"}
)

# Calculate length of each split segment in km using a US projected CRS.
lines_wind_split["length_km"] = (
    lines_wind_split.to_crs("EPSG:5070").geometry.length / 1000
)

print("\nLine strong wind exposure complete!")
print(lines_wind_split[["strong_wind_frequency", "length_km"]].head())
print("Number of split line segments:", len(lines_wind_split))
print("Total exposed line length km:", lines_wind_split["length_km"].sum())

# Save output
lines_output = exposure_dir / "lines_strong_wind_exposure_split.gpkg"

lines_wind_split.to_file(
    lines_output,
    layer="lines_strong_wind_exposure_split",
    driver="GPKG"
)

print("\nSaved:", lines_output)

# Summary statistics
print("\nStrong wind exposure by line segment:")
print(lines_wind_split["strong_wind_frequency"].describe())

lines_wind_split["wind_class"] = pd.cut(
    lines_wind_split["strong_wind_frequency"],
    bins=bins,
    labels=labels,
    include_lowest=True
)

length_by_class = (
    lines_wind_split
    .groupby("wind_class", observed=True)["length_km"]
    .sum()
    .reset_index()
)
length_by_class["percent"] = (
    length_by_class["length_km"]
    / length_by_class["length_km"].sum()
    * 100
)

print("\nTransmission line length by strong wind class:")
print(length_by_class)

length_by_class.to_csv(
    exposure_dir / "lines_strong_wind_length_by_class.csv",
    index=False
)

plt.figure(figsize=(8, 5))

bars = plt.bar(
    length_by_class["wind_class"].astype(str),
    length_by_class["length_km"]
)

label_offset = length_by_class["length_km"].max() * 0.02

for bar, percent in zip(bars, length_by_class["percent"]):
    plt.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height() + label_offset,
        f"{percent:.1f}%",
        ha="center",
        va="bottom"
    )

plt.xlabel("Strong wind frequency (events/year)")
plt.ylabel("Transmission line length (km)")
plt.title("Above-Ground Transmission Line Exposure to Strong Wind")
plt.ylim(0, length_by_class["length_km"].max() * 1.12)

plt.tight_layout()

plt.savefig(
    project / "data" / "Exposure" / "transmission_length_by_wind_class.png",
    dpi=300
)

plt.close()

# -----------------------------
# Map exposed transmission lines by wind class
# -----------------------------

wind_class_colors = {
    "0-0.98": "#2c7bb6",
    "0.98-2.2": "#abd9e9",
    "2.2-3.7": "#ffffbf",
    "3.7-5.35": "#fdae61",
    ">5.35": "#d7191c",
}

fig, ax = plt.subplots(figsize=(11, 7))

for wind_class in labels:
    class_lines = lines_wind_split[
        lines_wind_split["wind_class"].astype(str) == wind_class
    ]
    if class_lines.empty:
        continue

    class_lines.plot(
        ax=ax,
        color=wind_class_colors[wind_class],
        linewidth=0.45,
        label=wind_class
    )

ax.set_title("Above-Ground Transmission Lines Exposed to Strong Wind by Class")
ax.set_axis_off()
ax.legend(
    title="Strong wind frequency\n(events/year)",
    loc="lower left",
    frameon=True
)

plt.tight_layout()

line_map_output = exposure_dir / "transmission_lines_by_wind_class_map.png"

plt.savefig(
    line_map_output,
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print("Saved transmission line exposure map:", line_map_output)
