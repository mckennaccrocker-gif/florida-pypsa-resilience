# Validation and Diagnostic Scripts

The repository includes more than the final workflow. Several scripts were used to diagnose, calibrate, and improve the Florida PyPSA model before the final hazard and adaptation runs.

## Topology and Island Diagnostics

These scripts helped identify disconnected islands, dangling line endpoints, topology issues, and QGIS review layers:

```text
src/electricity/diagnose_florida_pypsa_topology.py
src/electricity/diagnose_florida_baseline_load_shedding.py
src/electricity/export_island_review_qgis_original_lines.py
src/electricity/apply_island_review_decisions.py
```

## Baseline Calibration

These scripts tested how the model behaved under no-hazard and seasonal/summer-peak conditions:

```text
src/electricity/calibrate_florida_pypsa_baseline.py
src/electricity/run_seasonal_baseline_validation.py
src/electricity/run_summer_peak_multiplier_sweep.py
src/electricity/compare_summer_peak_cleanup_variants.py
```

## Load and Generator Assignment Review

These scripts improved bus-level demand placement and generator-to-bus assignments:

```text
src/electricity/create_local_load_adjusted_network.py
src/electricity/build_county_population_load_profiles.py
src/electricity/review_large_generator_bus_assignments.py
src/electricity/finalize_generator_bus_assignment_review.py
```

## Sensitivity and Stress Tests

These scripts were used for model exploration, not as final thesis conclusions:

```text
src/electricity/run_florida_pypsa_n1_bulk_contingency.py
src/electricity/run_improved_model_sensitivity_analysis.py
src/electricity/run_florida_pypsa_line_outage_scenario.py
src/electricity/validate_s_nom_multiplier.py
```

## EAGLE-I Validation

The project also compared modeled Hurricane Ian load shedding with county-level observed outage patterns from EAGLE-I. Those outputs are useful for interpretation: the transmission-scale PyPSA model captures broad operational constraints, while observed outages include many distribution-level failures.

## Plotting Scripts

Several plotting scripts are included because they produced thesis figures and exploratory diagnostics:

```text
src/electricity/plot_updated_annualized_risk_graphs.py
src/electricity/plot_updated_gradual_return_period_graphs.py
src/exposure/make_hurricane_ian_publication_figures.py
src/exposure/plot_powerplant_polygon_hazard_maps_polished.py
src/exposure/plot_snail_line_distribution_graphs.py
```

These plotting scripts assume local generated CSVs and geospatial files that are not committed to GitHub.
