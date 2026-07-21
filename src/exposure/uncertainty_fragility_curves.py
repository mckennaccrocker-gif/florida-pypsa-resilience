from pathlib import Path
from math import erf, log, sqrt

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# -----------------------------
# Paths
# -----------------------------
PROJECT = Path(r"C:\oxford_tc_project")
COST = PROJECT / "data" / "Cost"
EXPOSURE = PROJECT / "data" / "Exposure"
EXPOSURE.mkdir(parents=True, exist_ok=True)

flood_curve_file = COST / "nhess_f63_energy_assets_diked_areas_curve.csv"
tc_curve_file = COST / "nhess_w310_power_tower_160kmh_urban_curve.csv"

flood_output = EXPOSURE / "flood_fragility_uncertainty.png"
tc_output = EXPOSURE / "tc_fragility_uncertainty.png"
flood_curve_choice_output = EXPOSURE / "flood_curve_choice_uncertainty.png"
tc_curve_choice_output = EXPOSURE / "tc_curve_choice_uncertainty.png"


def check_required_files(paths):
    print("Checking files...")
    for path in paths:
        print(path, "exists?", path.exists())
        if not path.exists():
            raise FileNotFoundError(path)


def add_sensitivity_bounds(curve, value_column):
    curve = curve.copy()

    # These bounds are a sensitivity range unless literature-based uncertainty
    # bounds are available.
    curve["lower_sensitivity"] = (curve[value_column] * 0.90).clip(0, 1)
    curve["upper_sensitivity"] = (curve[value_column] * 1.10).clip(0, 1)
    return curve


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


def plot_damage_ratio_uncertainty(
    curve,
    x_column,
    y_column,
    xlabel,
    title,
    best_label,
    output_path,
):
    plt.figure(figsize=(8, 5))

    plt.fill_between(
        curve[x_column],
        curve["lower_sensitivity"],
        curve["upper_sensitivity"],
        color="#9ecae1",
        alpha=0.45,
        label="+/-10% sensitivity range",
    )
    plt.plot(
        curve[x_column],
        curve[y_column],
        color="#08519c",
        linewidth=2.2,
        label=best_label,
    )
    plt.plot(
        curve[x_column],
        curve["lower_sensitivity"],
        color="#3182bd",
        linewidth=1.2,
        linestyle="--",
        label="Lower sensitivity",
    )
    plt.plot(
        curve[x_column],
        curve["upper_sensitivity"],
        color="#3182bd",
        linewidth=1.2,
        linestyle=":",
        label="Upper sensitivity",
    )

    plt.xlabel(xlabel)
    plt.ylabel("Damage ratio")
    plt.title(title)
    plt.ylim(0, 1.05)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


def plot_curve_choice_uncertainty(
    x_values,
    baseline_values,
    nhess_values,
    xlabel,
    title,
    baseline_label,
    nhess_label,
    output_path,
):
    lower = np.minimum(baseline_values, nhess_values)
    upper = np.maximum(baseline_values, nhess_values)

    plt.figure(figsize=(8, 5))
    plt.fill_between(
        x_values,
        lower,
        upper,
        color="#bdbdbd",
        alpha=0.45,
        label="Curve-choice sensitivity range",
    )
    plt.plot(
        x_values,
        baseline_values,
        color="#252525",
        linewidth=2.2,
        linestyle="--",
        label=baseline_label,
    )
    plt.plot(
        x_values,
        nhess_values,
        color="#08519c",
        linewidth=2.2,
        label=nhess_label,
    )

    plt.xlabel(xlabel)
    plt.ylabel("Damage ratio")
    plt.title(title)
    plt.ylim(0, 1.05)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


check_required_files([flood_curve_file, tc_curve_file])


# -----------------------------
# Flood: NHESS F6.3 damage-ratio curve
# -----------------------------
flood_curve = pd.read_csv(flood_curve_file)
flood_curve = flood_curve.sort_values("flood_depth_m")
flood_curve["nhess_f63_damage_ratio"] = flood_curve[
    "nhess_f63_damage_ratio"
].clip(0, 1)
flood_curve = add_sensitivity_bounds(flood_curve, "nhess_f63_damage_ratio")

plot_damage_ratio_uncertainty(
    flood_curve,
    "flood_depth_m",
    "nhess_f63_damage_ratio",
    "Flood depth (m)",
    "Flood Damage Ratio Sensitivity Range",
    "NHESS F6.3 energy assets in diked areas",
    flood_output,
)

flood_depth_grid = np.linspace(
    flood_curve["flood_depth_m"].min(),
    flood_curve["flood_depth_m"].max(),
    400,
)
baseline_flood_values = np.array([
    baseline_expected_damage_ratio(
        depth,
        theta_values=[0.50, 1.00, 1.50, 3.00],
        beta=0.40,
        damage_ratios=[0.05, 0.20, 0.50, 1.00],
    )
    for depth in flood_depth_grid
])
nhess_flood_values = np.interp(
    flood_depth_grid,
    flood_curve["flood_depth_m"],
    flood_curve["nhess_f63_damage_ratio"],
    left=flood_curve["nhess_f63_damage_ratio"].iloc[0],
    right=flood_curve["nhess_f63_damage_ratio"].iloc[-1],
)

plot_curve_choice_uncertainty(
    flood_depth_grid,
    baseline_flood_values,
    nhess_flood_values,
    "Flood depth (m)",
    "Flood Damage Ratio Curve-Choice Sensitivity",
    "Baseline lognormal expected damage ratio",
    "NHESS F6.3 energy assets in diked areas",
    flood_curve_choice_output,
)


# -----------------------------
# Tropical cyclone wind: NHESS W3.10 damage-ratio curve
# -----------------------------
tc_curve = pd.read_csv(tc_curve_file)
tc_curve = tc_curve.sort_values("wind_speed_ms")
tc_curve["nhess_w310_damage_ratio"] = tc_curve[
    "nhess_w310_damage_ratio"
].clip(0, 1)
tc_curve = add_sensitivity_bounds(tc_curve, "nhess_w310_damage_ratio")

plot_damage_ratio_uncertainty(
    tc_curve,
    "wind_speed_ms",
    "nhess_w310_damage_ratio",
    "Tropical cyclone wind speed (m/s)",
    "Tropical Cyclone Wind Damage Ratio Sensitivity Range",
    "NHESS W3.10 power tower, 160 km/h design speed",
    tc_output,
)

wind_speed_grid = np.linspace(
    tc_curve["wind_speed_ms"].min(),
    tc_curve["wind_speed_ms"].max(),
    500,
)
baseline_tc_values = np.array([
    baseline_expected_damage_ratio(
        wind_speed,
        theta_values=[30.0, 42.0, 55.0, 67.0],
        beta=0.25,
        damage_ratios=[0.05, 0.20, 0.50, 1.00],
    )
    for wind_speed in wind_speed_grid
])
nhess_tc_values = np.interp(
    wind_speed_grid,
    tc_curve["wind_speed_ms"],
    tc_curve["nhess_w310_damage_ratio"],
    left=tc_curve["nhess_w310_damage_ratio"].iloc[0],
    right=tc_curve["nhess_w310_damage_ratio"].iloc[-1],
)

plot_curve_choice_uncertainty(
    wind_speed_grid,
    baseline_tc_values,
    nhess_tc_values,
    "Tropical cyclone wind speed (m/s)",
    "Tropical Cyclone Wind Damage Ratio Curve-Choice Sensitivity",
    "Baseline lognormal expected damage ratio",
    "NHESS W3.10 power tower, 160 km/h design speed",
    tc_curve_choice_output,
)


print("\nSaved uncertainty/sensitivity plots:")
print(flood_output)
print(tc_output)
print(flood_curve_choice_output)
print(tc_curve_choice_output)
