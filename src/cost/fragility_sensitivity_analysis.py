from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from math import erf, log, sqrt


# -----------------------------
# Paths
# -----------------------------
PROJECT = Path(r"C:\oxford_tc_project")
EXPOSURE = PROJECT / "data" / "Exposure"
COST = PROJECT / "data" / "Cost"

nhess_curve_workbook = (
    COST / "Table_D2_Hazard_Fragility_and_Vulnerability_Curves_V1.1.0.xlsx"
)
baseline_flood_damage_file = COST / "flood_transmission_line_segment_damage.gpkg"
baseline_tc_damage_file = COST / "tc_wind_transmission_line_damage.gpkg"
line_costs_file = COST / "transmission_lines_replacement_costs.gpkg"

print("Checking files...")
for path in [
    nhess_curve_workbook,
    baseline_flood_damage_file,
    baseline_tc_damage_file,
    line_costs_file,
]:
    print(path, "exists?", path.exists())


def normal_cdf(value):
    return 0.5 * (1 + erf(value / sqrt(2)))


def exceedance_probability(intensity, theta, beta):
    if pd.isna(intensity) or intensity <= 0:
        return 0.0
    return normal_cdf(log(intensity / theta) / beta)


def baseline_expected_damage_ratio(intensity, theta_values, beta, damage_ratios):
    exceedance = np.array([
        exceedance_probability(intensity, theta, beta)
        for theta in theta_values
    ])
    state_probabilities = exceedance - np.append(exceedance[1:], 0.0)
    state_probabilities = np.clip(state_probabilities, 0, 1)
    return float(np.sum(state_probabilities * np.array(damage_ratios)))


# -----------------------------
# Load NHESS F6.3 vulnerability curve
# -----------------------------
curve_sheet = pd.read_excel(
    nhess_curve_workbook,
    sheet_name="F_Vuln_Depth",
    header=None,
)

curve_id_row = curve_sheet.iloc[0]
f63_columns = [
    column
    for column, value in curve_id_row.items()
    if str(value).strip() == "F6.3"
]

if len(f63_columns) != 1:
    raise ValueError(f"Expected one F6.3 column, found {f63_columns}")

f63_column = f63_columns[0]

f63_curve = pd.DataFrame(
    {
        "flood_depth_m": pd.to_numeric(curve_sheet.iloc[5:, 0], errors="coerce"),
        "nhess_f63_damage_ratio": pd.to_numeric(
            curve_sheet.iloc[5:, f63_column],
            errors="coerce",
        ),
    }
).dropna()

f63_curve = f63_curve.sort_values("flood_depth_m")
f63_curve["nhess_f63_damage_ratio"] = f63_curve["nhess_f63_damage_ratio"].clip(0, 1)

f63_curve_output = COST / "nhess_f63_energy_assets_diked_areas_curve.csv"
f63_curve.to_csv(f63_curve_output, index=False)

print("\nNHESS F6.3 curve loaded:")
print(f63_curve.head())
print(f63_curve.tail())
print("Saved:", f63_curve_output)


def interpolate_f63_damage_ratio(depth):
    if pd.isna(depth):
        return 0.0

    depth = max(float(depth), 0.0)

    return float(
        np.interp(
            depth,
            f63_curve["flood_depth_m"],
            f63_curve["nhess_f63_damage_ratio"],
            left=f63_curve["nhess_f63_damage_ratio"].iloc[0],
            right=f63_curve["nhess_f63_damage_ratio"].iloc[-1],
        )
    )


# -----------------------------
# Apply F6.3 to flood-exposed transmission line segments
# -----------------------------
baseline_flood_damage = gpd.read_file(baseline_flood_damage_file)

nhess_flood_damage = baseline_flood_damage.copy()
nhess_flood_damage["nhess_f63_damage_ratio"] = nhess_flood_damage[
    "flood_depth_m"
].apply(interpolate_f63_damage_ratio)
nhess_flood_damage["nhess_f63_damage_usd"] = (
    nhess_flood_damage["replacement_cost_usd"].fillna(0)
    * nhess_flood_damage["nhess_f63_damage_ratio"]
)

nhess_segment_output = COST / "flood_damage_nhess_f63_segments.gpkg"
nhess_flood_damage.to_file(
    nhess_segment_output,
    layer="flood_damage_nhess_f63_segments",
    driver="GPKG",
)
nhess_flood_damage.drop(columns="geometry").to_csv(
    COST / "flood_damage_nhess_f63_segments.csv",
    index=False,
)

print("\nNHESS F6.3 segment damage complete!")
print(
    "Total NHESS F6.3 flood damage:",
    "${:,.0f}".format(nhess_flood_damage["nhess_f63_damage_usd"].sum()),
)
print("Saved:", nhess_segment_output)


# -----------------------------
# Aggregate to transmission-line level for easier comparison
# -----------------------------
line_costs = gpd.read_file(line_costs_file)[
    [
        "line_id",
        "voltage_class",
        "voltage_kv_cost_class",
        "line_cost_per_mile_usd",
        "length_km",
        "replacement_cost_usd",
        "geometry",
    ]
].copy()

line_summary = (
    nhess_flood_damage
    .groupby("line_id", dropna=False)
    .agg(
        sampled_length_km=("length_km", "sum"),
        mean_flood_depth_m=("flood_depth_m", "mean"),
        max_flood_depth_m=("flood_depth_m", "max"),
        baseline_flood_damage_usd=("flood_damage_usd", "sum"),
        nhess_f63_flood_damage_usd=("nhess_f63_damage_usd", "sum"),
        mean_baseline_damage_ratio=("flood_damage_ratio", "mean"),
        mean_nhess_f63_damage_ratio=("nhess_f63_damage_ratio", "mean"),
    )
    .reset_index()
)

line_summary = line_costs.merge(line_summary, on="line_id", how="left")
line_summary["damage_difference_usd"] = (
    line_summary["nhess_f63_flood_damage_usd"].fillna(0)
    - line_summary["baseline_flood_damage_usd"].fillna(0)
)
line_summary["damage_ratio_change"] = (
    line_summary["mean_nhess_f63_damage_ratio"].fillna(0)
    - line_summary["mean_baseline_damage_ratio"].fillna(0)
)

line_summary_gdf = gpd.GeoDataFrame(
    line_summary,
    geometry="geometry",
    crs=line_costs.crs,
)

line_summary_output = COST / "flood_damage_nhess_f63_by_line.gpkg"
line_summary_gdf.to_file(
    line_summary_output,
    layer="flood_damage_nhess_f63_by_line",
    driver="GPKG",
)
line_summary_gdf.drop(columns="geometry").to_csv(
    COST / "flood_damage_nhess_f63_by_line.csv",
    index=False,
)

print("Saved line-level NHESS comparison:", line_summary_output)


# -----------------------------
# Summary tables and comparison figures
# -----------------------------
baseline_total = nhess_flood_damage["flood_damage_usd"].sum()
nhess_total = nhess_flood_damage["nhess_f63_damage_usd"].sum()

sensitivity_summary = pd.DataFrame(
    {
        "method": [
            "Baseline paper lognormal flood fragility",
            "NHESS F6.3 energy assets in diked areas",
        ],
        "flood_damage_usd": [
            baseline_total,
            nhess_total,
        ],
    }
)
sensitivity_summary["annual_exceedance_probability"] = 0.01
sensitivity_summary["annualized_expected_damage_usd"] = (
    sensitivity_summary["flood_damage_usd"]
    * sensitivity_summary["annual_exceedance_probability"]
)
sensitivity_summary["change_vs_baseline_usd"] = (
    sensitivity_summary["flood_damage_usd"]
    - baseline_total
)
sensitivity_summary["change_vs_baseline_percent"] = (
    sensitivity_summary["change_vs_baseline_usd"]
    / baseline_total
    * 100
)

sensitivity_summary.to_csv(
    COST / "flood_fragility_sensitivity_summary.csv",
    index=False,
)

print("\nFlood fragility sensitivity summary:")
print(sensitivity_summary)

plt.figure(figsize=(8, 5))
bars = plt.bar(
    ["Baseline\nlognormal", "NHESS F6.3"],
    sensitivity_summary["flood_damage_usd"] / 1e9,
)
max_value = (sensitivity_summary["flood_damage_usd"] / 1e9).max()

for bar, value in zip(bars, sensitivity_summary["flood_damage_usd"] / 1e9):
    plt.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height() + max_value * 0.02,
        f"${value:.1f}B",
        ha="center",
        va="bottom",
    )

plt.ylabel("Expected direct flood damage (billion USD)")
plt.title("Flood Damage Sensitivity to Fragility Curve Choice")
plt.ylim(0, max_value * 1.15 if max_value > 0 else 1)
plt.tight_layout()
plt.savefig(COST / "flood_damage_sensitivity_by_curve.png", dpi=300)
plt.close()

plt.figure(figsize=(8, 5))
depth_grid = np.linspace(0.01, 6, 300)

plt.plot(
    f63_curve["flood_depth_m"],
    f63_curve["nhess_f63_damage_ratio"],
    label="NHESS F6.3",
    linewidth=2,
)
plt.plot(
    depth_grid,
    [
        baseline_expected_damage_ratio(
            depth,
            theta_values=[0.5, 1.0, 1.5, 3.0],
            beta=0.40,
            damage_ratios=[0.05, 0.20, 0.50, 1.00],
        )
        for depth in depth_grid
    ],
    label="Baseline lognormal",
    linewidth=2,
    linestyle="--",
)

plt.xlabel("Flood depth (m)")
plt.ylabel("Damage ratio")
plt.title("Flood Damage Ratio Curves Used in Sensitivity Test")
plt.ylim(0, 1.05)
plt.legend()
plt.tight_layout()
plt.savefig(COST / "flood_fragility_curve_sensitivity_comparison.png", dpi=300)
plt.close()

voltage_summary = (
    line_summary_gdf
    .groupby("voltage_kv_cost_class", dropna=False)
    .agg(
        baseline_flood_damage_usd=("baseline_flood_damage_usd", "sum"),
        nhess_f63_flood_damage_usd=("nhess_f63_flood_damage_usd", "sum"),
    )
    .reset_index()
    .dropna(subset=["voltage_kv_cost_class"])
)
voltage_summary["voltage_kv_cost_class"] = (
    voltage_summary["voltage_kv_cost_class"].astype(int).astype(str)
)
voltage_summary.to_csv(
    COST / "flood_fragility_sensitivity_by_voltage_class.csv",
    index=False,
)

voltage_plot = voltage_summary.set_index("voltage_kv_cost_class")[
    ["baseline_flood_damage_usd", "nhess_f63_flood_damage_usd"]
] / 1e9
voltage_plot = voltage_plot.rename(
    columns={
        "baseline_flood_damage_usd": "Baseline lognormal",
        "nhess_f63_flood_damage_usd": "NHESS F6.3",
    }
)

ax = voltage_plot.plot(kind="bar", figsize=(9, 5))
ax.set_xlabel("Voltage cost class (kV)")
ax.set_ylabel("Expected direct flood damage (billion USD)")
ax.set_title("Flood Fragility Sensitivity by Voltage Class")
ax.legend(title="Curve")
plt.tight_layout()
plt.savefig(COST / "flood_fragility_sensitivity_by_voltage_class.png", dpi=300)
plt.close()

# -----------------------------
# Tropical cyclone wind sensitivity: NHESS W3.10
# -----------------------------
wind_sheet = pd.read_excel(
    nhess_curve_workbook,
    sheet_name="W_Vuln_V10m",
    header=None,
)

w310_columns = [
    column
    for column, value in wind_sheet.iloc[0].items()
    if str(value).strip() == "W3.10"
]

if len(w310_columns) != 1:
    raise ValueError(f"Expected one W3.10 column, found {w310_columns}")

w310_column = w310_columns[0]

w310_curve = pd.DataFrame(
    {
        "wind_speed_ms": pd.to_numeric(wind_sheet.iloc[5:, 0], errors="coerce"),
        "nhess_w310_damage_ratio": pd.to_numeric(
            wind_sheet.iloc[5:, w310_column],
            errors="coerce",
        ),
    }
).dropna()

w310_curve = w310_curve.sort_values("wind_speed_ms")
w310_curve["nhess_w310_damage_ratio"] = w310_curve[
    "nhess_w310_damage_ratio"
].clip(0, 1)

w310_curve_output = COST / "nhess_w310_power_tower_160kmh_urban_curve.csv"
w310_curve.to_csv(w310_curve_output, index=False)

print("\nNHESS W3.10 curve loaded:")
print(w310_curve.head())
print(w310_curve.tail())
print("Saved:", w310_curve_output)


def interpolate_w310_damage_ratio(wind_speed_ms):
    if pd.isna(wind_speed_ms):
        return 0.0

    wind_speed_ms = max(float(wind_speed_ms), 0.0)

    return float(
        np.interp(
            wind_speed_ms,
            w310_curve["wind_speed_ms"],
            w310_curve["nhess_w310_damage_ratio"],
            left=w310_curve["nhess_w310_damage_ratio"].iloc[0],
            right=w310_curve["nhess_w310_damage_ratio"].iloc[-1],
        )
    )


baseline_tc_damage = gpd.read_file(baseline_tc_damage_file)

tc_wind_column = None
for candidate in ["tc_wind_speed_max_ms", "wind_speed_ms", "tc_wind_speed_ms"]:
    if candidate in baseline_tc_damage.columns:
        tc_wind_column = candidate
        break

if tc_wind_column is None:
    raise ValueError(
        "Could not find a TC wind speed column. Expected one of "
        "tc_wind_speed_max_ms, wind_speed_ms, or tc_wind_speed_ms."
    )

nhess_tc_damage = baseline_tc_damage.copy()
nhess_tc_damage["wind_speed_ms_for_nhess_w310"] = nhess_tc_damage[tc_wind_column]
nhess_tc_damage["nhess_w310_damage_ratio"] = nhess_tc_damage[
    "wind_speed_ms_for_nhess_w310"
].apply(interpolate_w310_damage_ratio)
nhess_tc_damage["nhess_w310_damage_usd"] = (
    nhess_tc_damage["replacement_cost_usd"].fillna(0)
    * nhess_tc_damage["nhess_w310_damage_ratio"]
)

nhess_tc_output = COST / "tc_wind_damage_nhess_w310_lines.gpkg"
nhess_tc_damage.to_file(
    nhess_tc_output,
    layer="tc_wind_damage_nhess_w310_lines",
    driver="GPKG",
)
nhess_tc_damage.drop(columns="geometry").to_csv(
    COST / "tc_wind_damage_nhess_w310_lines.csv",
    index=False,
)

baseline_tc_total = nhess_tc_damage["tc_damage_usd"].sum()
nhess_tc_total = nhess_tc_damage["nhess_w310_damage_usd"].sum()

tc_sensitivity_summary = pd.DataFrame(
    {
        "method": [
            "Baseline paper lognormal TC wind fragility",
            "NHESS W3.10 power tower, design speed 160 km/h, urban terrain",
        ],
        "tc_wind_damage_usd": [
            baseline_tc_total,
            nhess_tc_total,
        ],
    }
)
tc_sensitivity_summary["annual_exceedance_probability"] = 0.01
tc_sensitivity_summary["annualized_expected_damage_usd"] = (
    tc_sensitivity_summary["tc_wind_damage_usd"]
    * tc_sensitivity_summary["annual_exceedance_probability"]
)
tc_sensitivity_summary["change_vs_baseline_usd"] = (
    tc_sensitivity_summary["tc_wind_damage_usd"]
    - baseline_tc_total
)
tc_sensitivity_summary["change_vs_baseline_percent"] = (
    tc_sensitivity_summary["change_vs_baseline_usd"]
    / baseline_tc_total
    * 100
)

tc_sensitivity_summary.to_csv(
    COST / "tc_wind_fragility_sensitivity_summary.csv",
    index=False,
)

print("\nTC wind fragility sensitivity summary:")
print(tc_sensitivity_summary)
print("Saved:", nhess_tc_output)

plt.figure(figsize=(8, 5))
bars = plt.bar(
    ["Baseline\nlognormal", "NHESS W3.10"],
    tc_sensitivity_summary["tc_wind_damage_usd"] / 1e9,
)
max_value = (tc_sensitivity_summary["tc_wind_damage_usd"] / 1e9).max()

for bar, value in zip(bars, tc_sensitivity_summary["tc_wind_damage_usd"] / 1e9):
    plt.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height() + max_value * 0.02,
        f"${value:.1f}B",
        ha="center",
        va="bottom",
    )

plt.ylabel("Expected direct TC wind damage (billion USD)")
plt.title("TC Wind Damage Sensitivity to Fragility Curve Choice")
plt.ylim(0, max_value * 1.15 if max_value > 0 else 1)
plt.tight_layout()
plt.savefig(COST / "tc_wind_damage_sensitivity_by_curve.png", dpi=300)
plt.close()

plt.figure(figsize=(8, 5))
wind_grid = np.linspace(0.01, 100, 400)

plt.plot(
    w310_curve["wind_speed_ms"],
    w310_curve["nhess_w310_damage_ratio"],
    label="NHESS W3.10",
    linewidth=2,
)
plt.plot(
    wind_grid,
    [
        baseline_expected_damage_ratio(
            wind_speed,
            theta_values=[30.0, 42.0, 55.0, 67.0],
            beta=0.25,
            damage_ratios=[0.05, 0.20, 0.50, 1.00],
        )
        for wind_speed in wind_grid
    ],
    label="Baseline lognormal",
    linewidth=2,
    linestyle="--",
)

plt.xlabel("Tropical cyclone wind speed (m/s)")
plt.ylabel("Damage ratio")
plt.title("TC Wind Damage Ratio Curves Used in Sensitivity Test")
plt.ylim(0, 1.05)
plt.legend()
plt.tight_layout()
plt.savefig(COST / "tc_wind_fragility_curve_sensitivity_comparison.png", dpi=300)
plt.close()

tc_voltage_summary = (
    nhess_tc_damage
    .groupby("voltage_kv_cost_class", dropna=False)
    .agg(
        baseline_tc_damage_usd=("tc_damage_usd", "sum"),
        nhess_w310_damage_usd=("nhess_w310_damage_usd", "sum"),
    )
    .reset_index()
    .dropna(subset=["voltage_kv_cost_class"])
)
tc_voltage_summary["voltage_kv_cost_class"] = (
    tc_voltage_summary["voltage_kv_cost_class"].astype(int).astype(str)
)
tc_voltage_summary.to_csv(
    COST / "tc_wind_fragility_sensitivity_by_voltage_class.csv",
    index=False,
)

tc_voltage_plot = tc_voltage_summary.set_index("voltage_kv_cost_class")[
    ["baseline_tc_damage_usd", "nhess_w310_damage_usd"]
] / 1e9
tc_voltage_plot = tc_voltage_plot.rename(
    columns={
        "baseline_tc_damage_usd": "Baseline lognormal",
        "nhess_w310_damage_usd": "NHESS W3.10",
    }
)

ax = tc_voltage_plot.plot(kind="bar", figsize=(9, 5))
ax.set_xlabel("Voltage cost class (kV)")
ax.set_ylabel("Expected direct TC wind damage (billion USD)")
ax.set_title("TC Wind Fragility Sensitivity by Voltage Class")
ax.legend(title="Curve")
plt.tight_layout()
plt.savefig(COST / "tc_wind_fragility_sensitivity_by_voltage_class.png", dpi=300)
plt.close()

# -----------------------------
# Combined sensitivity comparison
# -----------------------------
combined_sensitivity = pd.DataFrame(
    {
        "hazard": [
            "Flood",
            "Flood",
            "Tropical Cyclone Wind",
            "Tropical Cyclone Wind",
        ],
        "curve_dataset": [
            "Baseline paper",
            "NHESS F6.3",
            "Baseline paper",
            "NHESS W3.10",
        ],
        "direct_damage_usd": [
            baseline_total,
            nhess_total,
            baseline_tc_total,
            nhess_tc_total,
        ],
    }
)
combined_sensitivity["annual_exceedance_probability"] = 0.01
combined_sensitivity["annualized_expected_damage_usd"] = (
    combined_sensitivity["direct_damage_usd"]
    * combined_sensitivity["annual_exceedance_probability"]
)

combined_sensitivity.to_csv(
    COST / "fragility_sensitivity_combined_summary.csv",
    index=False,
)

combined_direct_pivot = (
    combined_sensitivity
    .pivot(index="hazard", columns="curve_dataset", values="direct_damage_usd")
    / 1e9
)
combined_direct_pivot = combined_direct_pivot[
    ["Baseline paper", "NHESS F6.3", "NHESS W3.10"]
].dropna(axis=1, how="all")

ax = combined_direct_pivot.plot(kind="bar", figsize=(9, 5))
ax.set_xlabel("Hazard")
ax.set_ylabel("Expected direct damage (billion USD)")
ax.set_title("Direct Damage Sensitivity to Fragility Dataset")
ax.legend(title="Fragility dataset")
plt.xticks(rotation=0)
plt.tight_layout()
plt.savefig(COST / "combined_fragility_sensitivity_direct_damage.png", dpi=300)
plt.close()

combined_annual_pivot = (
    combined_sensitivity
    .pivot(
        index="hazard",
        columns="curve_dataset",
        values="annualized_expected_damage_usd",
    )
    / 1e6
)
combined_annual_pivot = combined_annual_pivot[
    ["Baseline paper", "NHESS F6.3", "NHESS W3.10"]
].dropna(axis=1, how="all")

ax = combined_annual_pivot.plot(kind="bar", figsize=(9, 5))
ax.set_xlabel("Hazard")
ax.set_ylabel("Annualized expected damage (million USD/year)")
ax.set_title("Annualized Damage Sensitivity to Fragility Dataset")
ax.legend(title="Fragility dataset")
plt.xticks(rotation=0)
plt.tight_layout()
plt.savefig(COST / "combined_fragility_sensitivity_annualized_damage.png", dpi=300)
plt.close()

print("\nSaved sensitivity outputs in:", COST)
