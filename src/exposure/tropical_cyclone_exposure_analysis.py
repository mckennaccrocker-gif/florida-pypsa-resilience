from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rasterio

project = Path(r"C:\oxford_tc_project")

electricity_dir = project / "data" / "Electricty"  # misspelled folder name
hazards_dir = project / "data" / "Hazards"
exposure_dir = project / "data" / "Exposure"
exposure_dir.mkdir(parents=True, exist_ok=True)

nodes_file = electricity_dir / "Nodes2.gpkg"
lines_file = electricity_dir / "TransmissionLines2.gpkg"

tc_raster = hazards_dir / "Tropical_cyclones" / "STORM_constant_100yr_US_crop.tif"

tc_bins = [0, 25, 30, 35, 40, np.inf]
tc_labels = ["<25", "25-30", "30-35", "35-40", ">40"]
tc_class_colors = {
    "<25": "#2c7bb6",
    "25-30": "#abd9e9",
    "30-35": "#ffffbf",
    "35-40": "#fdae61",
    ">40": "#d7191c",
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

print("Checking files...")
for path in [nodes_file, lines_file, tc_raster]:
    print(path, "exists?", path.exists())


def preview_columns(frame, columns):
    return [column for column in columns if column in frame.columns]


def filter_lines_by_type(lines, target):
    type_text = lines["TYPE"].fillna("").astype(str)
    return lines[type_text.str.contains(target, case=False, na=False)].copy()


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
lines = gpd.read_file(lines_file)
lines = filter_lines_by_type(lines, "OVERHEAD")

print("\nNodes:", len(nodes))
print("Overhead lines:", len(lines))
print("Nodes CRS:", nodes.crs)
print("Lines CRS:", lines.crs)

with rasterio.open(tc_raster) as src:
    print("\nTropical cyclone raster loaded!")
    print("Raster CRS:", src.crs)
    print("Raster bounds:", src.bounds)
    print("Raster shape:", src.shape)

# -----------------------------
# Node exposure to tropical cyclone wind
# -----------------------------

with rasterio.open(tc_raster) as src:
    # Make sure nodes are in same CRS as raster
    nodes_tc = nodes.to_crs(src.crs)

    # Get node coordinates
    coords = [(geom.x, geom.y) for geom in nodes_tc.geometry]

    # Sample raster values at node locations
    values = [val[0] for val in src.sample(coords)]

    # Add values to nodes
    nodes_tc["tc_wind_speed_ms"] = values

    # Replace nodata values with missing values
    if src.nodata is not None:
        nodes_tc.loc[nodes_tc["tc_wind_speed_ms"] == src.nodata, "tc_wind_speed_ms"] = None

print("\nNode tropical cyclone exposure complete!")
print(
    nodes_tc[
        preview_columns(
            nodes_tc,
            ["type", "name", "country", "primary_fuel", "tc_wind_speed_ms"],
        )
    ].head()
)
print("Nodes with TC value:", nodes_tc["tc_wind_speed_ms"].notna().sum())
print("Nodes missing TC value:", nodes_tc["tc_wind_speed_ms"].isna().sum())

print("\nTC wind speed statistics:")
print(nodes_tc["tc_wind_speed_ms"].describe())

nodes_tc["tc_wind_class"] = pd.cut(
    nodes_tc["tc_wind_speed_ms"],
    bins=tc_bins,
    labels=tc_labels,
    include_lowest=True
)

# Save output
nodes_tc_output = exposure_dir / "nodes_tropical_cyclone_exposure.gpkg"

nodes_tc.to_file(
    nodes_tc_output,
    layer="nodes_tropical_cyclone_exposure",
    driver="GPKG"
)

print("\nSaved:", nodes_tc_output)

# -----------------------------
# Electricity nodes by TC wind class
# -----------------------------

node_tc_summary = (
    nodes_tc
    .dropna(subset=["tc_wind_class"])
    .groupby("tc_wind_class", observed=False)
    .size()
    .reset_index(name="node_count")
)
node_tc_summary["percent"] = (
    node_tc_summary["node_count"]
    / node_tc_summary["node_count"].sum()
    * 100
)

print("\nElectricity nodes by tropical cyclone wind class:")
print(node_tc_summary)

node_tc_summary.to_csv(
    exposure_dir / "nodes_tropical_cyclone_count_by_class.csv",
    index=False
)

plt.figure(figsize=(8, 5))

bars = plt.bar(
    node_tc_summary["tc_wind_class"].astype(str),
    node_tc_summary["node_count"]
)

label_offset = node_tc_summary["node_count"].max() * 0.02

for bar, percent in zip(bars, node_tc_summary["percent"]):
    plt.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height() + label_offset,
        f"{percent:.1f}%",
        ha="center",
        va="bottom"
    )

plt.xlabel("Tropical cyclone wind speed (m/s)")
plt.ylabel("Number of electricity nodes")
plt.title("Electricity Node Exposure to Tropical Cyclone Wind by Class")
plt.ylim(0, node_tc_summary["node_count"].max() * 1.12)
plt.tight_layout()

node_tc_chart_output = exposure_dir / "nodes_by_tropical_cyclone_wind_class.png"

plt.savefig(
    node_tc_chart_output,
    dpi=300
)

plt.close()

print("Saved node TC wind class chart:", node_tc_chart_output)

# -----------------------------
# Electricity nodes by TC wind class and energy type
# -----------------------------

nodes_tc_by_fuel = nodes_tc.copy()
nodes_tc_by_fuel["primary_fuel"] = nodes_tc_by_fuel["primary_fuel"].fillna("Unknown")

node_fuel_tc_summary = (
    nodes_tc_by_fuel
    .dropna(subset=["tc_wind_class"])
    .groupby(["tc_wind_class", "primary_fuel"], observed=False)
    .size()
    .reset_index(name="node_count")
)
node_fuel_tc_summary["class_total"] = (
    node_fuel_tc_summary
    .groupby("tc_wind_class", observed=False)["node_count"]
    .transform("sum")
)
node_fuel_tc_summary["percent_of_class"] = np.where(
    node_fuel_tc_summary["class_total"] > 0,
    node_fuel_tc_summary["node_count"] / node_fuel_tc_summary["class_total"] * 100,
    0,
)

node_fuel_tc_summary.to_csv(
    exposure_dir / "nodes_tropical_cyclone_count_by_class_and_fuel.csv",
    index=False,
)

node_fuel_tc_plot = (
    node_fuel_tc_summary
    .pivot_table(
        index="tc_wind_class",
        columns="primary_fuel",
        values="node_count",
        aggfunc="sum",
        fill_value=0,
        observed=False,
    )
    .reindex(tc_labels)
)

fuel_order = (
    node_fuel_tc_plot
    .sum(axis=0)
    .sort_values(ascending=False)
    .index
)
node_fuel_tc_plot = node_fuel_tc_plot[fuel_order]
node_fuel_tc_plot["total"] = node_fuel_tc_plot.sum(axis=1)
node_fuel_tc_plot = node_fuel_tc_plot.reset_index()

node_fuel_tc_wide_csv = exposure_dir / "nodes_tropical_cyclone_count_by_class_and_fuel_wide.csv"
node_fuel_tc_plot.to_csv(node_fuel_tc_wide_csv, index=False)

node_fuel_tc_chart_output = exposure_dir / "nodes_tropical_cyclone_by_class_and_fuel.png"
plot_stacked_fuel_bars(
    node_fuel_tc_plot,
    "tc_wind_class",
    "Tropical cyclone wind speed (m/s)",
    "Tropical Cyclone Wind-Exposed Electricity Nodes by Energy Type",
    node_fuel_tc_chart_output,
)

print("\nElectricity nodes by tropical cyclone wind class and energy type:")
print(node_fuel_tc_plot)
print("Saved node TC wind energy type chart:", node_fuel_tc_chart_output)

# -----------------------------
# Transmission line exposure to tropical cyclone wind
# -----------------------------

def sample_line_max_value(line, raster, n_points=20):
    """Sample points along a line and return the max raster value."""
    if line is None or line.is_empty:
        return np.nan

    distances = np.linspace(0, line.length, n_points)
    points = [line.interpolate(d) for d in distances]
    coords = [(p.x, p.y) for p in points]

    values = [v[0] for v in raster.sample(coords)]

    values = [
        v for v in values
        if raster.nodata is None or v != raster.nodata
    ]

    if len(values) == 0:
        return np.nan

    return max(values)


with rasterio.open(tc_raster) as src:
    lines_tc = lines.to_crs(src.crs).copy()

    print("\nSampling TC wind along transmission lines...")

    lines_tc["tc_wind_speed_max_ms"] = lines_tc.geometry.apply(
        lambda geom: sample_line_max_value(geom, src, n_points=25)
    )

print("\nLine tropical cyclone exposure complete!")
print(lines_tc["tc_wind_speed_max_ms"].describe())

lines_tc["tc_wind_class"] = pd.cut(
    lines_tc["tc_wind_speed_max_ms"],
    bins=tc_bins,
    labels=tc_labels,
    include_lowest=True
)

lines_tc_metric = lines_tc.to_crs("EPSG:5070")
lines_tc["length_km"] = lines_tc_metric.geometry.length / 1000

lines_tc_output = exposure_dir / "lines_tropical_cyclone_exposure.gpkg"

lines_tc.to_file(
    lines_tc_output,
    layer="lines_tropical_cyclone_exposure",
    driver="GPKG"
)

print("\nSaved:", lines_tc_output)

# -----------------------------
# Transmission line length by TC wind class
# -----------------------------

line_tc_summary = (
    lines_tc
    .dropna(subset=["tc_wind_class"])
    .groupby("tc_wind_class", observed=False)["length_km"]
    .sum()
    .reset_index()
)
line_tc_summary["percent"] = (
    line_tc_summary["length_km"]
    / line_tc_summary["length_km"].sum()
    * 100
)

print("\nTransmission line length by tropical cyclone wind class:")
print(line_tc_summary)

line_tc_summary.to_csv(
    exposure_dir / "lines_tropical_cyclone_length_by_class.csv",
    index=False
)

plt.figure(figsize=(8, 5))

bars = plt.bar(
    line_tc_summary["tc_wind_class"].astype(str),
    line_tc_summary["length_km"]
)

label_offset = line_tc_summary["length_km"].max() * 0.02

for bar, percent in zip(bars, line_tc_summary["percent"]):
    plt.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height() + label_offset,
        f"{percent:.1f}%",
        ha="center",
        va="bottom"
    )

plt.xlabel("Tropical cyclone wind speed (m/s)")
plt.ylabel("Transmission line length (km)")
plt.title("Above-Ground Transmission Line Exposure to Tropical Cyclone Wind")
plt.ylim(0, line_tc_summary["length_km"].max() * 1.12)
plt.tight_layout()

line_tc_chart_output = exposure_dir / "transmission_length_by_tropical_cyclone_wind_class.png"

plt.savefig(
    line_tc_chart_output,
    dpi=300
)

plt.close()

print("Saved transmission line TC wind class chart:", line_tc_chart_output)

# -----------------------------
# Map exposed transmission lines by TC wind class
# -----------------------------

fig, ax = plt.subplots(figsize=(11, 7))

for tc_class in tc_labels:
    class_lines = lines_tc[
        lines_tc["tc_wind_class"].astype(str) == tc_class
    ]
    if class_lines.empty:
        continue

    class_lines.plot(
        ax=ax,
        color=tc_class_colors[tc_class],
        linewidth=0.45,
        label=tc_class
    )

ax.set_title("Above-Ground Transmission Lines Exposed to Tropical Cyclone Wind by Class")
ax.set_axis_off()
ax.legend(
    title="TC wind speed\n(m/s)",
    loc="lower left",
    frameon=True
)

plt.tight_layout()

line_tc_map_output = exposure_dir / "transmission_lines_by_tropical_cyclone_wind_class_map.png"

plt.savefig(
    line_tc_map_output,
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print("Saved transmission line TC exposure map:", line_tc_map_output)
