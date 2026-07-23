# Script Index and Suggested Run Order

This repository preserves the main scripts used to construct and analyze the Florida PyPSA resilience model. The workflow is not a single-click package; it is a documented research pipeline.

For the fuller story of how the data, grid build, hazards, validation, N-1 screen, and flood cost-benefit work connect, start with:

```text
docs/full_workflow_story.md
```

## A. Build the Electricity Network

1. Download or prepare source data.

```text
src/electricity/download_florida_osm_powerplants.py
src/electricity/download_eia_florida_hourly_demand.py
```

2. Build and clean power plant records.

```text
src/electricity/merge_florida_osm_gppd_powerplants.py
src/electricity/match_florida_powerplants_to_osm_polygons.py
src/electricity/finalize_florida_generator_marginal_costs.py
```

3. Build transmission topology.

```text
src/electricity/build_florida_transmission_network.py
src/electricity/create_voltage_layer_transformer_network.py
src/electricity/apply_island_review_decisions.py
src/electricity/build_extended_georgia_alabama_tie_line_network.py
```

4. Assign capacities, generator buses, demand, and solar profiles.

```text
src/electricity/add_slr_s_nom_to_florida_lines.py
src/electricity/finalize_generator_bus_assignment_review.py
src/electricity/build_county_population_load_profiles.py
src/electricity/create_florida_renewable_profiles.py
```

5. Convert cleaned assets into PyPSA.

```text
src/electricity/convert_florida_assets_to_pypsa.py
```

## B. Run Baseline Dispatch

```text
src/electricity/run_florida_pypsa_load_shedding_dispatch.py
src/electricity/run_florida_pypsa_baseline_validation.py
```

These scripts add emergency import slack and load-shedding generators to keep the dispatch problem feasible.

## B2. Run N-1 Bulk-Line Contingency Screen

```text
src/electricity/run_florida_pypsa_n1_bulk_contingency.py
```

This removes one bulk line at a time and checks post-contingency load shedding, import slack, and overloaded lines.

## C. Run Hazard Scenarios

Flood:

```text
src/exposure/build_flood_line_exposure_with_ids.py
src/exposure/flood_powerplant_polygon_exposure.py
src/electricity/run_florida_pypsa_gradual_return_period_suite.py
```

Tropical cyclone wind:

```text
src/exposure/florida_tc_snail_intersection_clean_assets.py
src/exposure/open_gira_ibtracs_event_damage_snail_new_grid.py
src/electricity/run_florida_pypsa_calibrated_hazard_scenarios.py
```

Single-storm Hurricane Ian:

```text
src/exposure/plot_single_ibtracs_storm_direct_and_cascade.py
src/exposure/make_hurricane_ian_publication_figures.py
```

Expected annual risk / exceedance graphing:

```text
src/electricity/analyze_gradual_return_period_results.py
src/electricity/plot_updated_annualized_risk_graphs.py
src/electricity/plot_updated_gradual_return_period_graphs.py
```

## C2. Compare Hurricane Ian With EAGLE-I

```text
src/validation/eaglei/run_eaglei_florida_validation.py
src/validation/eaglei/run_eaglei_pypsa_comparison.py
src/validation/eaglei/create_final_validation_graphs.py
```

This comparison is best used for spatial severity/ranking validation. EAGLE-I reports customers out by county; PyPSA reports modeled transmission-scale unserved demand.

## D. Run Flood Adaptation and Economics

```text
src/adaptation/run_rp100_top5_pilot.py
src/adaptation/run_rp100_top5_full_suite.py
src/adaptation/run_rp100_top5_cost_benefit.py
src/adaptation/create_final_summary_tables.py
```

## E. Interpret Outputs

Small final outputs are committed under:

```text
outputs/summary_tables/
```

Large generated outputs are ignored by git and should be regenerated locally.

For details on which datasets and generated outputs are excluded from GitHub, see:

```text
docs/data_availability.md
```

## F. Diagnostics and Exploratory Scripts

The repository also includes diagnostic, calibration, sensitivity, and plotting scripts that document how the final model was developed. See:

```text
docs/validation_and_diagnostics.md
```
