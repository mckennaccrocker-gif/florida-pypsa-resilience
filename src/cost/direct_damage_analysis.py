from pathlib import Path
import re

import geopandas as gpd
import numpy as np
import pandas as pd


# -----------------------------
# Paths
# -----------------------------
PROJECT = Path(r"C:\oxford_tc_project")
ELECTRICITY = PROJECT / "data" / "Electricty"  # keeping your folder spelling
COST = PROJECT / "data" / "Cost"
COST.mkdir(parents=True, exist_ok=True)

lines_path = ELECTRICITY / "TransmissionLines2.gpkg"
nodes_path = ELECTRICITY / "Nodes2.gpkg"


# -----------------------------
# Cost table from literature
# -----------------------------
standard_voltages = np.array([69, 115, 161, 230, 345, 500, 765])

line_cost_per_mile_usd = {
    69: 1_500_000,
    115: 2_500_000,
    161: 2_750_000,
    230: 3_000_000,
    345: 3_050_000,
    500: 3_600_000,
    765: 5_900_000,
}

substation_cost_usd = {
    69: 8_000_000,
    115: 15_000_000,
    161: 20_000_000,
    230: 37_500_000,
    345: 75_000_000,
    500: 150_000_000,
    765: 300_000_000,
}

cost_table = pd.DataFrame(
    {
        "voltage_kv": standard_voltages,
        "line_cost_per_mile_usd": [
            line_cost_per_mile_usd[int(voltage)]
            for voltage in standard_voltages
        ],
        "substation_cost_usd": [
            substation_cost_usd[int(voltage)]
            for voltage in standard_voltages
        ],
    }
)

cost_table.to_csv(COST / "replacement_cost_lookup_table.csv", index=False)


# -----------------------------
# Helpers
# -----------------------------
def parse_voltage_kv(value):
    """Return the highest listed voltage in kV.

    The source data mixes volts (115000), kV (345), and strings with multiple
    voltages. Values >= 1000 are treated as volts; smaller values are kV.
    """
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


def snap_voltage_to_cost_class(voltage_kv):
    if pd.isna(voltage_kv):
        return np.nan

    if voltage_kv >= standard_voltages.max():
        return int(standard_voltages.max())

    nearest_index = np.argmin(np.abs(standard_voltages - voltage_kv))
    return int(standard_voltages[nearest_index])


def assign_voltage_costs(gdf):
    gdf = gdf.copy()
    if "voltage_class" not in gdf.columns:
        if "VOLT_CLASS" in gdf.columns:
            gdf["voltage_class"] = gdf["VOLT_CLASS"]
        elif "VOLTAGE" in gdf.columns:
            gdf["voltage_class"] = gdf["VOLTAGE"]
        else:
            gdf["voltage_class"] = np.nan
    gdf["voltage_kv_raw"] = gdf["voltage_class"].apply(parse_voltage_kv)
    gdf["voltage_kv_cost_class"] = gdf["voltage_kv_raw"].apply(
        snap_voltage_to_cost_class
    )
    gdf["line_cost_per_mile_usd"] = gdf["voltage_kv_cost_class"].map(
        line_cost_per_mile_usd
    )
    gdf["substation_cost_usd"] = gdf["voltage_kv_cost_class"].map(
        substation_cost_usd
    )
    return gdf


def preview_columns(frame, columns):
    return [column for column in columns if column in frame.columns]


# -----------------------------
# Load electricity assets
# -----------------------------
lines = gpd.read_file(lines_path).reset_index(drop=True)
nodes = gpd.read_file(nodes_path).reset_index(drop=True)

lines["line_id"] = lines.index
nodes["node_id"] = nodes.index

print("Lines loaded:", len(lines))
print("Nodes loaded:", len(nodes))


# -----------------------------
# Assign transmission line replacement costs
# -----------------------------
lines_cost = assign_voltage_costs(lines)

lines_metric = lines_cost.to_crs("EPSG:5070")
lines_cost["length_km"] = lines_metric.geometry.length / 1000
lines_cost["length_miles"] = lines_metric.geometry.length / 1609.344

lines_cost["replacement_cost_usd"] = (
    lines_cost["length_miles"]
    * lines_cost["line_cost_per_mile_usd"]
)

print("\nTransmission line voltage cost classes:")
print(lines_cost["voltage_kv_cost_class"].value_counts(dropna=False).sort_index())

print("\nTransmission line replacement cost check:")
print(
    lines_cost[
        preview_columns(
            lines_cost,
            [
                "line_id",
                "ID",
                "voltage_class",
                "VOLTAGE",
                "VOLT_CLASS",
                "voltage_kv_raw",
                "voltage_kv_cost_class",
                "length_miles",
                "line_cost_per_mile_usd",
                "replacement_cost_usd",
            ],
        )
    ].head(12)
)

print("\nLines with assigned replacement cost:", lines_cost["replacement_cost_usd"].notna().sum())
print("Lines missing replacement cost:", lines_cost["replacement_cost_usd"].isna().sum())
print(
    "Total estimated line replacement value:",
    "${:,.0f}".format(lines_cost["replacement_cost_usd"].sum()),
)

lines_cost_output = COST / "transmission_lines_replacement_costs.gpkg"
lines_cost.to_file(
    lines_cost_output,
    layer="transmission_lines_replacement_costs",
    driver="GPKG",
)

lines_cost_csv = COST / "transmission_lines_replacement_costs.csv"
lines_cost.drop(columns="geometry").to_csv(lines_cost_csv, index=False)

print("\nSaved:", lines_cost_output)
print("Saved:", lines_cost_csv)


# -----------------------------
# Assign node/substation replacement cost fields where voltage exists
# -----------------------------
nodes_cost = assign_voltage_costs(nodes)

print("\nNode voltage cost classes:")
print(nodes_cost["voltage_kv_cost_class"].value_counts(dropna=False).sort_index())

print("\nNodes with assigned substation cost:", nodes_cost["substation_cost_usd"].notna().sum())
print("Nodes missing substation cost:", nodes_cost["substation_cost_usd"].isna().sum())

nodes_cost_output = COST / "nodes_replacement_costs.gpkg"
nodes_cost.to_file(
    nodes_cost_output,
    layer="nodes_replacement_costs",
    driver="GPKG",
)

nodes_cost_csv = COST / "nodes_replacement_costs.csv"
nodes_cost.drop(columns="geometry").to_csv(nodes_cost_csv, index=False)

print("\nSaved:", nodes_cost_output)
print("Saved:", nodes_cost_csv)


# -----------------------------
# Summary tables for reporting and QA
# -----------------------------
line_cost_summary = (
    lines_cost
    .groupby("voltage_kv_cost_class", dropna=False)
    .agg(
        line_count=("line_id", "count"),
        length_km=("length_km", "sum"),
        length_miles=("length_miles", "sum"),
        replacement_cost_usd=("replacement_cost_usd", "sum"),
    )
    .reset_index()
)

line_cost_summary.to_csv(
    COST / "transmission_line_replacement_cost_summary_by_voltage.csv",
    index=False,
)

print("\nReplacement cost summary by voltage class:")
print(line_cost_summary)
