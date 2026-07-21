"""
Create first-pass renewable availability profiles for Florida PyPSA generators.

Output:
    data/Electricity/pypsa_florida_network/generators-p_max_pu.csv

Solar method:
    A transparent solar-position/daylight approximation is used because no
    project-specific NSRDB/PyPSA-USA renewable profiles were found. Each solar
    generator receives an hourly profile based on its latitude/longitude,
    local clock time, solar declination, equation of time, and solar zenith.
    Values are clipped to [0, 1] and are exactly zero when the sun is below the
    horizon. This is a defensible first-pass availability profile, not a
    replacement for NSRDB or PyPSA-USA profile generation.

Wind method:
    If wind generators exist, a smooth representative hourly wind profile is
    generated and clipped to [0, 1]. In the current Florida generator table no
    wind generators are expected, so no wind columns are written unless present.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_DIR = Path(r"C:\oxford_tc_project")
ELECTRICITY_DIR = PROJECT_DIR / "data" / "Electricity"
PYPSA_DIR = ELECTRICITY_DIR / "pypsa_florida_network"

LOAD_TIMESERIES = ELECTRICITY_DIR / "eia_florida_hourly_demand" / "eia_florida_hourly_demand_2025_timeseries.csv"
FINAL_GENERATORS = PYPSA_DIR / "generators_with_final_marginal_costs.csv"
BASE_GENERATORS = PYPSA_DIR / "generators.csv"
GENERATOR_LOCATIONS = PYPSA_DIR / "generator_bus_matches_review.csv"
OUTPUT_P_MAX_PU = PYPSA_DIR / "generators-p_max_pu.csv"
OUTPUT_SUMMARY = ELECTRICITY_DIR / "renewable_profiles_summary.csv"

SOLAR_CARRIERS = {"solar", "pv", "utilitypv", "respv", "commpv"}
WIND_CARRIERS = {"wind", "onwind", "offwind", "offshore wind", "landbasedwind"}


def find_existing_profile_files() -> list[Path]:
    patterns = ["*p_max_pu*", "*capacity_factor*", "*renewable_profile*", "*solar_profile*", "*wind_profile*"]
    files: list[Path] = []
    search_roots = [PYPSA_DIR, ELECTRICITY_DIR]
    for pattern in patterns:
        for root in search_roots:
            files.extend(root.glob(pattern))
    return sorted(
        {
            path
            for path in files
            if path.is_file()
            and path != OUTPUT_P_MAX_PU
            and path.name != Path(__file__).name
        }
    )


def load_snapshots() -> pd.DatetimeIndex:
    loads = pd.read_csv(LOAD_TIMESERIES, usecols=["snapshot"])
    snapshots = pd.to_datetime(loads["snapshot"], errors="raise")
    if len(snapshots) != 8760:
        raise ValueError(f"Expected 8,760 load snapshots, found {len(snapshots):,}.")
    if snapshots.duplicated().any():
        raise ValueError("Load snapshots contain duplicates.")
    return pd.DatetimeIndex(snapshots)


def load_generators() -> pd.DataFrame:
    path = FINAL_GENERATORS if FINAL_GENERATORS.exists() else BASE_GENERATORS
    generators = pd.read_csv(path)
    locations = pd.read_csv(GENERATOR_LOCATIONS)
    generators = generators.merge(
        locations[["generator", "longitude", "latitude"]],
        left_on="name",
        right_on="generator",
        how="left",
    ).drop(columns=["generator"])
    generators["carrier_normalized"] = generators["carrier"].astype(str).str.strip().str.lower()
    return generators


def is_dst_eastern_2025(timestamps: pd.DatetimeIndex) -> np.ndarray:
    # 2025 US daylight saving time: 2025-03-09 02:00 to 2025-11-02 02:00.
    start = pd.Timestamp("2025-03-09 02:00:00")
    end = pd.Timestamp("2025-11-02 02:00:00")
    return np.asarray((timestamps >= start) & (timestamps < end))


def solar_capacity_factor(
    timestamps: pd.DatetimeIndex,
    latitude: float,
    longitude: float,
) -> np.ndarray:
    """Approximate hourly solar availability from solar geometry."""
    day_of_year = timestamps.dayofyear.to_numpy(dtype=float)
    hour = (
        timestamps.hour.to_numpy(dtype=float)
        + timestamps.minute.to_numpy(dtype=float) / 60.0
        + 0.5
    )

    gamma = 2.0 * np.pi / 365.0 * (day_of_year - 1.0 + (hour - 12.0) / 24.0)
    equation_of_time_min = 229.18 * (
        0.000075
        + 0.001868 * np.cos(gamma)
        - 0.032077 * np.sin(gamma)
        - 0.014615 * np.cos(2 * gamma)
        - 0.040849 * np.sin(2 * gamma)
    )
    declination_rad = (
        0.006918
        - 0.399912 * np.cos(gamma)
        + 0.070257 * np.sin(gamma)
        - 0.006758 * np.cos(2 * gamma)
        + 0.000907 * np.sin(2 * gamma)
        - 0.002697 * np.cos(3 * gamma)
        + 0.00148 * np.sin(3 * gamma)
    )

    timezone_offset = np.where(is_dst_eastern_2025(timestamps), -4.0, -5.0)
    true_solar_time_min = (hour * 60.0 + equation_of_time_min + 4.0 * longitude - 60.0 * timezone_offset) % 1440.0
    hour_angle_deg = true_solar_time_min / 4.0 - 180.0
    hour_angle_rad = np.deg2rad(hour_angle_deg)
    latitude_rad = np.deg2rad(latitude)

    cos_zenith = (
        np.sin(latitude_rad) * np.sin(declination_rad)
        + np.cos(latitude_rad) * np.cos(declination_rad) * np.cos(hour_angle_rad)
    )
    daylight = np.maximum(cos_zenith, 0.0)

    # A simple PV-like transform: less production at low sun angles, with a
    # modest derate so peak availability remains plausible rather than perfect.
    capacity_factor = 0.92 * np.power(daylight, 1.18)
    capacity_factor[daylight <= 0] = 0.0
    return np.clip(capacity_factor, 0.0, 1.0)


def wind_capacity_factor(timestamps: pd.DatetimeIndex) -> np.ndarray:
    day = timestamps.dayofyear.to_numpy(dtype=float)
    hour = timestamps.hour.to_numpy(dtype=float)
    seasonal = 0.27 + 0.07 * np.cos(2 * np.pi * (day - 15) / 365.0)
    diurnal = 0.03 * np.cos(2 * np.pi * (hour - 2) / 24.0)
    synoptic = 0.04 * np.sin(2 * np.pi * day / 9.0)
    return np.clip(seasonal + diurnal + synoptic, 0.02, 0.65)


def build_profiles() -> tuple[pd.DataFrame, pd.DataFrame]:
    snapshots = load_snapshots()
    generators = load_generators()
    solar = generators[generators["carrier_normalized"].isin(SOLAR_CARRIERS)].copy()
    wind = generators[generators["carrier_normalized"].isin(WIND_CARRIERS)].copy()

    fleet_lat = solar["latitude"].dropna().mean()
    fleet_lon = solar["longitude"].dropna().mean()
    if pd.isna(fleet_lat):
        fleet_lat = 28.2
    if pd.isna(fleet_lon):
        fleet_lon = -82.2

    profile_columns: dict[str, np.ndarray] = {}
    summary_rows = []

    for _, gen in solar.iterrows():
        lat = gen["latitude"] if pd.notna(gen["latitude"]) else fleet_lat
        lon = gen["longitude"] if pd.notna(gen["longitude"]) else fleet_lon
        cf = solar_capacity_factor(snapshots, float(lat), float(lon))
        profile_columns[gen["name"]] = cf
        summary_rows.append(
            {
                "generator": gen["name"],
                "carrier": gen["carrier"],
                "p_nom": gen["p_nom"],
                "latitude": lat,
                "longitude": lon,
                "coordinate_source": "generator_location" if pd.notna(gen["latitude"]) and pd.notna(gen["longitude"]) else "solar_fleet_centroid_fallback",
                "annual_capacity_factor": float(cf.mean()),
                "max_capacity_factor": float(cf.max()),
            }
        )

    if not wind.empty:
        wind_cf = wind_capacity_factor(snapshots)
        for _, gen in wind.iterrows():
            profile_columns[gen["name"]] = wind_cf
            summary_rows.append(
                {
                    "generator": gen["name"],
                    "carrier": gen["carrier"],
                    "p_nom": gen["p_nom"],
                    "latitude": gen["latitude"],
                    "longitude": gen["longitude"],
                    "coordinate_source": "representative_florida_wind_profile",
                    "annual_capacity_factor": float(wind_cf.mean()),
                    "max_capacity_factor": float(wind_cf.max()),
                }
            )

    summary = pd.DataFrame(summary_rows)
    profiles = pd.DataFrame(profile_columns)
    profiles.insert(0, "snapshot", snapshots.strftime("%Y-%m-%d %H:%M:%S"))
    return profiles, summary


def validate_profiles(profiles: pd.DataFrame, summary: pd.DataFrame) -> dict[str, float | int]:
    snapshots = pd.to_datetime(profiles["snapshot"], errors="raise")
    value_cols = [col for col in profiles.columns if col != "snapshot"]
    values = profiles[value_cols]
    if len(profiles) != 8760:
        raise ValueError(f"Expected 8,760 profile rows, found {len(profiles):,}.")
    if values.empty:
        raise ValueError("No renewable generator profile columns were created.")
    if values.min().min() < -1e-12 or values.max().max() > 1 + 1e-12:
        raise ValueError("Profile values must be between 0 and 1.")

    solar_cols = summary.loc[
        summary["carrier"].astype(str).str.strip().str.lower().isin(SOLAR_CARRIERS),
        "generator",
    ].tolist()
    night_hours_with_solar_gt_zero = 0
    if solar_cols:
        # Validate against the actual created profile: "night" means every solar
        # generator has zero output at that timestamp under the solar geometry.
        solar_sum = profiles[solar_cols].sum(axis=1)
        # Also count any physically suspicious tiny values between midnight and
        # 4 AM local clock, where this daylight model should always be zero.
        deep_night = snapshots.dt.hour.isin([0, 1, 2, 3, 4])
        night_hours_with_solar_gt_zero = int((solar_sum[deep_night] > 1e-9).sum())

    return {
        "profile_rows": len(profiles),
        "profile_columns": len(value_cols),
        "solar_generators": int(summary["carrier"].astype(str).str.lower().isin(SOLAR_CARRIERS).sum()),
        "wind_generators": int(summary["carrier"].astype(str).str.lower().isin(WIND_CARRIERS).sum()),
        "average_annual_solar_capacity_factor": float(
            summary.loc[summary["carrier"].astype(str).str.lower().isin(SOLAR_CARRIERS), "annual_capacity_factor"].mean()
        )
        if solar_cols
        else np.nan,
        "max_solar_capacity_factor": float(profiles[solar_cols].max().max()) if solar_cols else np.nan,
        "night_hours_with_solar_gt_zero": night_hours_with_solar_gt_zero,
    }


def main() -> None:
    existing = find_existing_profile_files()
    print("Existing renewable/profile-like files found:")
    if existing:
        for path in existing[:10]:
            print(f"  - {path}")
        if len(existing) > 10:
            print(f"  ... {len(existing) - 10} more")
    else:
        print("  None found.")
    print("\nNo project-specific renewable p_max_pu table was found; creating first-pass profiles.")

    profiles, summary = build_profiles()
    diagnostics = validate_profiles(profiles, summary)
    profiles.to_csv(OUTPUT_P_MAX_PU, index=False)
    summary.to_csv(OUTPUT_SUMMARY, index=False)

    print("\nRenewable profile diagnostics:")
    print(f"Solar generators: {diagnostics['solar_generators']}")
    print(f"Wind generators: {diagnostics['wind_generators']}")
    print(f"Average annual solar capacity factor: {diagnostics['average_annual_solar_capacity_factor']:.4f}")
    print(f"Max solar capacity factor: {diagnostics['max_solar_capacity_factor']:.4f}")
    print(f"Night hours with solar > 0: {diagnostics['night_hours_with_solar_gt_zero']}")
    print(f"Rows: {diagnostics['profile_rows']}")
    print(f"Renewable profile columns: {diagnostics['profile_columns']}")
    print(f"\nSaved PyPSA p_max_pu table: {OUTPUT_P_MAX_PU}")
    print(f"Saved generator profile summary: {OUTPUT_SUMMARY}")


if __name__ == "__main__":
    main()
