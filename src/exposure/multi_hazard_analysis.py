from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
import pandas as pd


project = Path(r"C:\oxford_tc_project")
exposure_dir = project / "data" / "Exposure"

nodes_wind_file = exposure_dir / "nodes_strong_wind_exposure.gpkg"
nodes_tc_file = exposure_dir / "nodes_tropical_cyclone_exposure.gpkg"
nodes_flood_file = exposure_dir / "nodes_flood_exposure.gpkg"

lines_wind_summary_file = exposure_dir / "lines_strong_wind_length_by_class.csv"
lines_tc_file = exposure_dir / "lines_tropical_cyclone_exposure.gpkg"
lines_flood_summary_file = exposure_dir / "lines_flood_length_by_class.csv"

print("Checking files...")
for path in [
    nodes_wind_file,
    nodes_tc_file,
    nodes_flood_file,
    lines_wind_summary_file,
    lines_tc_file,
    lines_flood_summary_file,
]:
    print(path, "exists?", path.exists())


def add_percent(summary, count_column):
    total = summary[count_column].sum()
    summary["percent"] = summary[count_column] / total * 100
    return summary


def plot_bar_with_percent(summary, x_column, y_column, ylabel, title, output_path):
    plt.figure(figsize=(8, 5))

    bars = plt.bar(summary[x_column].astype(str), summary[y_column])
    max_value = summary[y_column].max()
    label_offset = max_value * 0.02 if max_value > 0 else 0.1

    for bar, percent in zip(bars, summary["percent"]):
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + label_offset,
            f"{percent:.1f}%",
            ha="center",
            va="bottom",
        )

    plt.ylabel(ylabel)
    plt.title(title)
    plt.ylim(0, max_value * 1.12 if max_value > 0 else 1)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


def preview_columns(frame, columns):
    return [column for column in columns if column in frame.columns]


nodes_wind = gpd.read_file(nodes_wind_file).reset_index(drop=True)
nodes_tc = gpd.read_file(nodes_tc_file).reset_index(drop=True)
nodes_flood = gpd.read_file(nodes_flood_file).reset_index(drop=True)

if not (len(nodes_wind) == len(nodes_tc) == len(nodes_flood)):
    raise ValueError("Node exposure files do not have the same number of rows.")

multi_nodes = nodes_flood.copy()
multi_nodes["strong_wind_frequency"] = nodes_wind["strong_wind_frequency"]
multi_nodes["tc_wind_speed_ms"] = nodes_tc["tc_wind_speed_ms"]

multi_nodes["flood_exposed"] = multi_nodes["flood_depth_m"].fillna(0) > 0.5
multi_nodes["tc_exposed"] = multi_nodes["tc_wind_speed_ms"].fillna(0) > 33
multi_nodes["wind_exposed"] = multi_nodes["strong_wind_frequency"].fillna(0) > 3.7

multi_nodes["hazard_count"] = (
    multi_nodes[["flood_exposed", "tc_exposed", "wind_exposed"]]
    .sum(axis=1)
    .astype(int)
)

print("\nMulti-hazard node dataset complete!")
print(
    multi_nodes[
        preview_columns(
            multi_nodes,
            [
                "type",
                "name",
                "country",
                "primary_fuel",
                "flood_depth_m",
                "tc_wind_speed_ms",
                "strong_wind_frequency",
                "hazard_count",
            ],
        )
    ].head()
)

multi_nodes_output = exposure_dir / "nodes_multi_hazard_exposure.gpkg"
multi_nodes.to_file(
    multi_nodes_output,
    layer="nodes_multi_hazard_exposure",
    driver="GPKG",
)
print("\nSaved:", multi_nodes_output)

multi_nodes_csv = exposure_dir / "nodes_multi_hazard_exposure.csv"
multi_nodes.drop(columns="geometry").to_csv(multi_nodes_csv, index=False)
print("Saved:", multi_nodes_csv)

# -----------------------------
# Figure 1: Percentage of nodes exposed by hazard
# -----------------------------

node_hazard_summary = pd.DataFrame(
    {
        "hazard": ["Flood", "Tropical Cyclone", "Strong Wind"],
        "node_count": [
            int(multi_nodes["flood_exposed"].sum()),
            int(multi_nodes["tc_exposed"].sum()),
            int(multi_nodes["wind_exposed"].sum()),
        ],
    }
)
node_hazard_summary["percent"] = (
    node_hazard_summary["node_count"]
    / len(multi_nodes)
    * 100
)

print("\nPercentage of nodes exposed by hazard:")
print(node_hazard_summary)

node_hazard_summary.to_csv(
    exposure_dir / "nodes_exposed_by_hazard.csv",
    index=False,
)

plt.figure(figsize=(8, 5))
bars = plt.bar(node_hazard_summary["hazard"], node_hazard_summary["percent"])

for bar, percent in zip(bars, node_hazard_summary["percent"]):
    plt.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height() + 1,
        f"{percent:.1f}%",
        ha="center",
        va="bottom",
    )

plt.ylabel("Exposed nodes (%)")
plt.title("Percentage of Electricity Nodes Exposed by Hazard")
plt.ylim(0, max(5, node_hazard_summary["percent"].max() * 1.2))
plt.tight_layout()

nodes_by_hazard_output = exposure_dir / "nodes_exposed_by_hazard.png"
plt.savefig(nodes_by_hazard_output, dpi=300)
plt.close()
print("Saved node hazard comparison chart:", nodes_by_hazard_output)

# -----------------------------
# Figure 2: Multi-hazard hotspot count
# -----------------------------

hazard_count_summary = (
    multi_nodes
    .groupby("hazard_count")
    .size()
    .reindex([0, 1, 2, 3], fill_value=0)
    .reset_index(name="node_count")
)
hazard_count_summary = add_percent(hazard_count_summary, "node_count")

print("\nNodes by number of hazards:")
print(hazard_count_summary)

hazard_count_summary.to_csv(
    exposure_dir / "nodes_by_hazard_count.csv",
    index=False,
)

hazard_count_chart_output = exposure_dir / "nodes_by_hazard_count.png"
plot_bar_with_percent(
    hazard_count_summary,
    "hazard_count",
    "node_count",
    "Number of electricity nodes",
    "Electricity Nodes by Number of Hazards",
    hazard_count_chart_output,
)
print("Saved multi-hazard count chart:", hazard_count_chart_output)

# -----------------------------
# Figure 3: Multi-hazard hotspot map
# -----------------------------

hazard_count_colors = {
    0: "#bdbdbd",
    1: "#ffff99",
    2: "#fdae61",
    3: "#d7191c",
}

fig, ax = plt.subplots(figsize=(11, 7))

for hazard_count in [0, 1, 2, 3]:
    class_nodes = multi_nodes[multi_nodes["hazard_count"] == hazard_count]
    if class_nodes.empty:
        continue

    class_nodes.plot(
        ax=ax,
        color=hazard_count_colors[hazard_count],
        markersize=4,
        alpha=0.85,
        label=f"{hazard_count} hazards",
    )

ax.set_title("Electricity Node Multi-Hazard Hotspots")
ax.set_axis_off()
ax.legend(
    title="Hazard count",
    loc="lower left",
    frameon=True,
)
plt.tight_layout()

hazard_count_map_output = exposure_dir / "nodes_multi_hazard_hotspot_map.png"
plt.savefig(
    hazard_count_map_output,
    dpi=300,
    bbox_inches="tight",
)
plt.close()
print("Saved multi-hazard hotspot map:", hazard_count_map_output)

# -----------------------------
# Figure 4: Hazard comparison for transmission-line exposure
# -----------------------------

line_flood_summary = pd.read_csv(lines_flood_summary_file)
line_wind_summary = pd.read_csv(lines_wind_summary_file)
lines_tc = gpd.read_file(lines_tc_file)

flood_line_length_km = line_flood_summary.loc[
    line_flood_summary["flood_class"].isin(["0.5-1 m", "1-2 m", "2-5 m", ">5 m"]),
    "length_km",
].sum()

strong_wind_line_length_km = line_wind_summary.loc[
    line_wind_summary["wind_class"].isin(["3.7-5.35", ">5.35"]),
    "length_km",
].sum()

tc_line_length_km = lines_tc.loc[
    lines_tc["tc_wind_speed_max_ms"].fillna(0) > 33,
    "length_km",
].sum()

line_hazard_summary = pd.DataFrame(
    {
        "hazard": ["Flood", "Strong Wind", "Tropical Cyclone"],
        "exposed_line_length_km": [
            flood_line_length_km,
            strong_wind_line_length_km,
            tc_line_length_km,
        ],
    }
)
line_hazard_summary["percent"] = (
    line_hazard_summary["exposed_line_length_km"]
    / line_hazard_summary["exposed_line_length_km"].sum()
    * 100
)

print("\nExposed transmission-line length by hazard:")
print(line_hazard_summary)

line_hazard_summary.to_csv(
    exposure_dir / "transmission_line_exposure_by_hazard.csv",
    index=False,
)

plt.figure(figsize=(8, 5))
bars = plt.bar(
    line_hazard_summary["hazard"],
    line_hazard_summary["exposed_line_length_km"],
)

max_length = line_hazard_summary["exposed_line_length_km"].max()
label_offset = max_length * 0.02

for bar, length_km in zip(bars, line_hazard_summary["exposed_line_length_km"]):
    plt.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height() + label_offset,
        f"{length_km:,.0f} km",
        ha="center",
        va="bottom",
    )

plt.ylabel("Exposed transmission line length (km)")
plt.title("Transmission Line Exposure by Hazard")
plt.ylim(0, max_length * 1.12)
plt.tight_layout()

line_hazard_chart_output = exposure_dir / "transmission_line_exposure_by_hazard.png"
plt.savefig(line_hazard_chart_output, dpi=300)
plt.close()
print("Saved transmission-line hazard comparison chart:", line_hazard_chart_output)
