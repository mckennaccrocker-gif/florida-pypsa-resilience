from pathlib import Path
from math import erf, log, sqrt

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# -----------------------------
# Paths
# -----------------------------
PROJECT = Path(r"C:\oxford_tc_project")
EXPOSURE = PROJECT / "data" / "Exposure"
COST = PROJECT / "data" / "Cost"
COST.mkdir(parents=True, exist_ok=True)

line_costs_file = COST / "transmission_lines_replacement_costs.gpkg"
tc_lines_file = EXPOSURE / "lines_tropical_cyclone_exposure.gpkg"
flood_segments_file = EXPOSURE / "lines_flood_exposure_segments.gpkg"

print("Checking files...")
for path in [line_costs_file, tc_lines_file, flood_segments_file]:
    print(path, "exists?", path.exists())


# -----------------------------
# Fragility functions from Section 2.3
# -----------------------------
damage_ratios = np.array([0.05, 0.20, 0.50, 1.00])

flood_fragility = pd.DataFrame(
    {
        "hazard": "Flood",
        "damage_state": ["Slight", "Moderate", "Extensive", "Complete"],
        "theta": [0.50, 1.00, 1.50, 3.00],
        "beta": [0.40, 0.40, 0.40, 0.40],
        "damage_ratio": damage_ratios,
    }
)

tc_fragility = pd.DataFrame(
    {
        "hazard": "Tropical Cyclone Wind",
        "damage_state": ["Slight", "Moderate", "Extensive", "Complete"],
        "theta": [30.0, 42.0, 55.0, 67.0],
        "beta": [0.25, 0.25, 0.25, 0.25],
        "damage_ratio": damage_ratios,
    }
)

fragility_parameters = pd.concat(
    [flood_fragility, tc_fragility],
    ignore_index=True
)
fragility_parameters.to_csv(COST / "fragility_parameters_used.csv", index=False)


def normal_cdf(value):
    return 0.5 * (1 + erf(value / sqrt(2)))


def exceedance_probability(intensity, theta, beta):
    """Lognormal P(DS >= ds_i | IM = h)."""
    if pd.isna(intensity) or intensity <= 0:
        return 0.0
    return normal_cdf(log(intensity / theta) / beta)


def expected_damage_ratio(intensity, fragility_table):
    exceedance = np.array(
        [
            exceedance_probability(intensity, row.theta, row.beta)
            for row in fragility_table.itertuples(index=False)
        ]
    )

    # Convert exceedance probabilities into mutually exclusive damage-state
    # probabilities: P(ds_i) = P(DS >= ds_i) - P(DS >= ds_{i+1}).
    state_probabilities = exceedance - np.append(exceedance[1:], 0.0)
    state_probabilities = np.clip(state_probabilities, 0, 1)

    return float(np.sum(state_probabilities * fragility_table["damage_ratio"].to_numpy()))


def plot_fragility_curve(fragility_table, x_values, xlabel, title, output_path):
    plt.figure(figsize=(8, 5))

    for row in fragility_table.itertuples(index=False):
        probabilities = [
            exceedance_probability(x, row.theta, row.beta)
            for x in x_values
        ]
        plt.plot(x_values, probabilities, label=row.damage_state)

    edr_values = [
        expected_damage_ratio(x, fragility_table)
        for x in x_values
    ]
    plt.plot(
        x_values,
        edr_values,
        color="black",
        linewidth=2,
        linestyle="--",
        label="Expected damage ratio",
    )

    plt.xlabel(xlabel)
    plt.ylabel("Probability / expected damage ratio")
    plt.title(title)
    plt.ylim(0, 1.05)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


plot_fragility_curve(
    flood_fragility,
    np.linspace(0.01, 6, 300),
    "Flood depth (m)",
    "Flood Fragility Curve for Transmission Assets",
    COST / "flood_fragility_curve.png",
)

plot_fragility_curve(
    tc_fragility,
    np.linspace(1, 80, 300),
    "Tropical cyclone wind speed (m/s)",
    "Tropical Cyclone Wind Fragility Curve for Transmission Assets",
    COST / "tc_wind_fragility_curve.png",
)


# -----------------------------
# Load replacement values and exposure data
# -----------------------------
line_costs = gpd.read_file(line_costs_file).reset_index(drop=True)
tc_lines = gpd.read_file(tc_lines_file).reset_index(drop=True)
flood_segments = gpd.read_file(flood_segments_file).reset_index(drop=True)

if "line_id" not in line_costs.columns:
    line_costs["line_id"] = line_costs.index

tc_lines["line_id"] = tc_lines.index

cost_columns = [
    "line_id",
    "voltage_class",
    "voltage_kv_raw",
    "voltage_kv_cost_class",
    "line_cost_per_mile_usd",
    "length_km",
    "length_miles",
    "replacement_cost_usd",
]


# -----------------------------
# Tropical cyclone wind damage
# -----------------------------
tc_damage = tc_lines.merge(
    line_costs[cost_columns],
    on="line_id",
    how="left",
    suffixes=("", "_cost"),
)

tc_damage["tc_damage_ratio"] = tc_damage["tc_wind_speed_max_ms"].apply(
    lambda value: expected_damage_ratio(value, tc_fragility)
)
tc_damage["tc_damage_usd"] = (
    tc_damage["replacement_cost_usd"].fillna(0)
    * tc_damage["tc_damage_ratio"]
)

tc_damage_output = COST / "tc_wind_transmission_line_damage.gpkg"
tc_damage.to_file(
    tc_damage_output,
    layer="tc_wind_transmission_line_damage",
    driver="GPKG",
)
tc_damage.drop(columns="geometry").to_csv(
    COST / "tc_wind_transmission_line_damage.csv",
    index=False,
)

print("\nTropical cyclone wind damage complete!")
print("Total TC direct damage:", "${:,.0f}".format(tc_damage["tc_damage_usd"].sum()))
print("Saved:", tc_damage_output)


# -----------------------------
# Flood damage
# -----------------------------
spacing_m = 5000

line_costs_metric = line_costs.to_crs("EPSG:5070").copy()
line_costs_metric["line_length_m"] = line_costs_metric.geometry.length
line_costs_metric["segment_count"] = line_costs_metric["line_length_m"].apply(
    lambda length: max(1, int(np.ceil(length / spacing_m))) if length > 0 else 0
)

reconstructed_line_ids = np.repeat(
    line_costs_metric["line_id"].to_numpy(),
    line_costs_metric["segment_count"].to_numpy(),
)

if len(reconstructed_line_ids) != len(flood_segments):
    raise ValueError(
        "Could not reconstruct parent line IDs for flood segments. "
        f"Expected {len(flood_segments)} segments but reconstructed "
        f"{len(reconstructed_line_ids)}."
    )

flood_segments["line_id"] = reconstructed_line_ids

flood_damage = flood_segments.merge(
    line_costs[
        [
            "line_id",
            "voltage_class",
            "voltage_kv_raw",
            "voltage_kv_cost_class",
            "line_cost_per_mile_usd",
        ]
    ],
    on="line_id",
    how="left",
)

flood_damage["replacement_cost_usd"] = (
    flood_damage["length_km"]
    / 1.609344
    * flood_damage["line_cost_per_mile_usd"].fillna(0)
)
flood_damage["flood_damage_ratio"] = flood_damage["flood_depth_m"].apply(
    lambda value: expected_damage_ratio(value, flood_fragility)
)
flood_damage["flood_damage_usd"] = (
    flood_damage["replacement_cost_usd"]
    * flood_damage["flood_damage_ratio"]
)

flood_damage_output = COST / "flood_transmission_line_segment_damage.gpkg"
flood_damage.to_file(
    flood_damage_output,
    layer="flood_transmission_line_segment_damage",
    driver="GPKG",
)
flood_damage.drop(columns="geometry").to_csv(
    COST / "flood_transmission_line_segment_damage.csv",
    index=False,
)

print("\nFlood damage complete!")
print("Total flood direct damage:", "${:,.0f}".format(flood_damage["flood_damage_usd"].sum()))
print("Saved:", flood_damage_output)


# -----------------------------
# Summary outputs and charts
# -----------------------------
damage_summary = pd.DataFrame(
    {
        "hazard": ["Flood", "Tropical Cyclone Wind"],
        "direct_damage_usd": [
            flood_damage["flood_damage_usd"].sum(),
            tc_damage["tc_damage_usd"].sum(),
        ],
        "replacement_value_exposed_usd": [
            flood_damage["replacement_cost_usd"].sum(),
            tc_damage["replacement_cost_usd"].fillna(0).sum(),
        ],
    }
)
damage_summary["mean_damage_ratio_over_replacement_value"] = (
    damage_summary["direct_damage_usd"]
    / damage_summary["replacement_value_exposed_usd"]
)
damage_summary["annual_exceedance_probability"] = 0.01
damage_summary["annualized_expected_damage_usd"] = (
    damage_summary["direct_damage_usd"]
    * damage_summary["annual_exceedance_probability"]
)

damage_summary.to_csv(
    COST / "direct_damage_summary_by_hazard.csv",
    index=False,
)

print("\nDirect damage summary:")
print(damage_summary)

plt.figure(figsize=(8, 5))
bars = plt.bar(
    damage_summary["hazard"],
    damage_summary["direct_damage_usd"] / 1e9,
)
max_value = (damage_summary["direct_damage_usd"] / 1e9).max()

for bar, value in zip(bars, damage_summary["direct_damage_usd"] / 1e9):
    plt.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height() + max_value * 0.02,
        f"${value:.1f}B",
        ha="center",
        va="bottom",
    )

plt.ylabel("Expected direct damage (billion USD)")
plt.title("Expected Direct Transmission Line Damage by Hazard")
plt.ylim(0, max_value * 1.15 if max_value > 0 else 1)
plt.tight_layout()
plt.savefig(COST / "direct_damage_by_hazard.png", dpi=300)
plt.close()

plt.figure(figsize=(8, 5))
bars = plt.bar(
    damage_summary["hazard"],
    damage_summary["annualized_expected_damage_usd"] / 1e6,
)
max_value = (damage_summary["annualized_expected_damage_usd"] / 1e6).max()

for bar, value in zip(bars, damage_summary["annualized_expected_damage_usd"] / 1e6):
    plt.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height() + max_value * 0.02,
        f"${value:.0f}M/yr",
        ha="center",
        va="bottom",
    )

plt.ylabel("Annualized expected damage (million USD/year)")
plt.title("Annualized Expected Transmission Line Damage by Hazard")
plt.ylim(0, max_value * 1.15 if max_value > 0 else 1)
plt.tight_layout()
plt.savefig(COST / "annualized_expected_damage_by_hazard.png", dpi=300)
plt.close()


tc_damage_by_voltage = (
    tc_damage
    .groupby("voltage_kv_cost_class", dropna=False)
    .agg(
        line_count=("line_id", "count"),
        replacement_cost_usd=("replacement_cost_usd", "sum"),
        tc_damage_usd=("tc_damage_usd", "sum"),
        mean_tc_wind_speed_ms=("tc_wind_speed_max_ms", "mean"),
        mean_tc_damage_ratio=("tc_damage_ratio", "mean"),
    )
    .reset_index()
)
tc_damage_by_voltage.to_csv(
    COST / "tc_wind_damage_by_voltage_class.csv",
    index=False,
)

flood_damage_by_voltage = (
    flood_damage
    .groupby("voltage_kv_cost_class", dropna=False)
    .agg(
        segment_count=("line_id", "count"),
        length_km=("length_km", "sum"),
        replacement_cost_usd=("replacement_cost_usd", "sum"),
        flood_damage_usd=("flood_damage_usd", "sum"),
        mean_flood_depth_m=("flood_depth_m", "mean"),
        mean_flood_damage_ratio=("flood_damage_ratio", "mean"),
    )
    .reset_index()
)
flood_damage_by_voltage.to_csv(
    COST / "flood_damage_by_voltage_class.csv",
    index=False,
)

damage_by_voltage = pd.concat(
    [
        flood_damage_by_voltage[
            ["voltage_kv_cost_class", "flood_damage_usd"]
        ].rename(columns={"flood_damage_usd": "damage_usd"}).assign(
            hazard="Flood"
        ),
        tc_damage_by_voltage[
            ["voltage_kv_cost_class", "tc_damage_usd"]
        ].rename(columns={"tc_damage_usd": "damage_usd"}).assign(
            hazard="Tropical Cyclone Wind"
        ),
    ],
    ignore_index=True,
)
damage_by_voltage = damage_by_voltage.dropna(subset=["voltage_kv_cost_class"])
damage_by_voltage["voltage_kv_cost_class"] = (
    damage_by_voltage["voltage_kv_cost_class"].astype(int).astype(str)
)
damage_by_voltage.to_csv(
    COST / "damage_by_hazard_and_voltage_class.csv",
    index=False,
)

damage_by_voltage_pivot = (
    damage_by_voltage
    .pivot_table(
        index="voltage_kv_cost_class",
        columns="hazard",
        values="damage_usd",
        aggfunc="sum",
        fill_value=0,
    )
    .sort_index(key=lambda values: values.astype(int))
)

ax = (damage_by_voltage_pivot / 1e9).plot(
    kind="bar",
    figsize=(9, 5),
)
ax.set_xlabel("Voltage cost class (kV)")
ax.set_ylabel("Expected direct damage (billion USD)")
ax.set_title("Expected Direct Damage by Hazard and Voltage Class")
ax.legend(title="Hazard")
plt.tight_layout()
plt.savefig(COST / "direct_damage_by_voltage_class.png", dpi=300)
plt.close()

print("\nSaved summary and chart outputs in:", COST)
