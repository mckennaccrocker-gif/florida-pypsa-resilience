# Full Workflow Story

This is the plain-English version of the workflow: what data went in, how the grid was built, how hazards were connected to the grid, and what extra analyses were run after that. I wrote it this way because the project has a lot of scripts, and it is much easier to understand when the pieces are in the same order as the actual research story.

## 1. Data Sources

The repo does not include the raw data files. Some are too large for GitHub, and some are better handled carefully for security/licensing reasons. Instead, the repo keeps the code and points to where the original data can be obtained.

Main sources:

- HIFLD electric transmission lines: https://catalog.data.gov/dataset/electric-power-transmission-lines
- HIFLD ArcGIS layer: https://www.arcgis.com/home/item.html?id=7759b0df07274f30a422e86dc11d4761
- OpenStreetMap extracts through Geofabrik: https://download.geofabrik.de/north-america/us.html
- Global Power Plant Database / WRI: https://datasets.wri.org/datasets/global-power-plant-database
- EIA Open Data API and hourly grid monitor data: https://www.eia.gov/opendata/
- NOAA IBTrACS tropical cyclone tracks: https://www.ncei.noaa.gov/products/international-best-track-archive
- STORM synthetic tropical cyclone return-period wind data: https://zenodo.org/records/7438145
- JRC global river flood hazard maps: https://data.jrc.ec.europa.eu/collection/id-0054
- JRC flood maps through Google Earth Engine: https://developers.google.com/earth-engine/datasets/catalog/JRC_CEMS_GLOFAS_FloodHazard_v2_1
- EAGLE-I outage data access / public portal: https://eagle-i.doe.gov/
- EAGLE-I historical outage dataset reference: https://openenergyhub.ornl.gov/explore/dataset/eaglei_outages_2014/
- snkit spatial network toolkit: https://snkit.readthedocs.io/

Locally, these sources were turned into project folders under `C:/oxford_tc_project/`. The expected layout is summarized in `config/paths.example.yml`.

## 2. Building the Grid

The first step was making a PyPSA-ready electricity network for Florida.

### HIFLD lines and buses

The transmission network starts from HIFLD-style transmission-line geometries. The workflow keeps line geometry, voltage, owner/status fields, and endpoint information where available.

The important steps were:

1. filter to usable Florida transmission lines;
2. extract line endpoints;
3. snap endpoints that are really the same electrical location but are slightly misaligned in GIS;
4. create node/bus IDs from those endpoints;
5. review islands and topology problems in QGIS;
6. keep a crosswalk back to the source HIFLD line IDs.

Main scripts:

```text
src/electricity/build_florida_transmission_network.py
src/electricity/apply_island_review_decisions.py
src/electricity/build_topology_repaired_pypsa_network.py
src/electricity/build_gap_repaired_pypsa_network.py
```

### OSM substations and power plants

OSM was used for substation polygons and power plant polygons where those features existed. The substation polygons helped identify physical footprint areas for exposure work. The power plant polygons helped move beyond simple point locations when estimating plant exposure to flood or wind.

Main scripts:

```text
src/electricity/download_florida_osm_powerplants.py
src/electricity/extract_osm_substation_polygons_for_reviewed_grid.py
src/electricity/merge_florida_osm_gppd_powerplants.py
src/electricity/match_florida_powerplants_to_osm_polygons.py
```

### Matching OSM to GPPD-style plants

The plant workflow used OSM polygons together with GPPD-style point records. The point records gave plant attributes like capacity, technology/fuel, and IDs. The OSM polygons gave better exposure geometry where available.

The matching logic used names, locations, distances, and manual review for important cases. The final plant records were then assigned to PyPSA buses, with extra review for large generators because those have the biggest effect on dispatch.

Main scripts:

```text
src/electricity/merge_florida_osm_gppd_powerplants.py
src/electricity/match_florida_powerplants_to_osm_polygons.py
src/electricity/review_large_generator_bus_assignments.py
src/electricity/finalize_generator_bus_assignment_review.py
src/electricity/apply_generator_bus_overrides.py
```

### Extending lines into Georgia and Alabama

The original Florida-only network can create artificial boundary problems, because real power flows do not stop exactly at the state line. To reduce that issue, selected high-voltage/bulk lines that cross into Georgia and Alabama were extended to external endpoints and boundary import buses.

This does not make a full Southeast grid. It is a practical boundary representation so the Florida model has more realistic external tie-line behavior.

Main script:

```text
src/electricity/build_extended_georgia_alabama_tie_line_network.py
```

### Line capacity equation

Transmission capacity in PyPSA is stored as `s_nom` in MVA. Where static line rating/current assumptions were available, the workflow used:

```text
s_nom = sqrt(3) * voltage_kV * amps / 1000
```

Main script:

```text
src/electricity/add_slr_s_nom_to_florida_lines.py
```

## 3. Demand, Costs, and Dispatch Inputs

### Demand by population

Florida hourly demand was downloaded from EIA and then distributed spatially across model buses using county population weighting. The idea was not to pretend every bus has metered demand data, but to avoid placing all load at arbitrary points and to make county-level demand patterns more reasonable.

Main scripts:

```text
src/electricity/download_eia_florida_hourly_demand.py
src/electricity/build_county_population_load_profiles.py
src/electricity/create_local_load_adjusted_network.py
```

### Solar availability

Solar generators need time-varying availability, so the workflow created solar availability profiles for the Florida generator portfolio and connected those profiles to the PyPSA generator time series.

Main script:

```text
src/electricity/create_florida_renewable_profiles.py
```

### Generator marginal costs

Generator marginal costs were built from available fuel, heat-rate, and fallback assumptions. These costs determine dispatch order in PyPSA, so they also affect how the model responds after a hazard damages part of the grid.

Main scripts:

```text
src/electricity/populate_florida_generator_marginal_costs.py
src/electricity/finalize_florida_generator_marginal_costs.py
```

### Emergency slack and load-shedding generators

Two artificial generator types are important:

- emergency import slack generators, which represent costly external/import support at selected boundary buses;
- load-shedding generators, which are very expensive artificial generators at demand buses so PyPSA can stay feasible when demand cannot be served.

Load shedding is the model's estimate of demand not served under the assumptions of that scenario. It is not the same thing as customer outage counts.

Main scripts:

```text
src/electricity/select_florida_boundary_import_buses.py
src/electricity/run_florida_pypsa_load_shedding_dispatch.py
src/electricity/run_boundary_import_baseline.py
```

## 4. Fragility and Vulnerability Curves

The final hazard workflow uses curve points saved locally in `data/Cost/` and documents them in scenario outputs. The repo includes code that applies the curves, but not every source spreadsheet/PDF.

The main curves used were:

- flood transmission lines: F6.2 distribution/elevated-crossing flood vulnerability curve;
- flood power plants: F1.1, F1.2, F1.3, and F1.4 power-plant flood curves assigned by type/capacity;
- flood substations/buses: F2.1, F2.2, and F2.3 substation flood curves assigned by voltage;
- tropical cyclone transmission lines: W6.3 FPL overhead-line wind fragility curve;
- tropical cyclone generators: W1.10 coal, W1.11 gas/thermal fallback, W1.12 nuclear, W1.13 solar, and W1.6 wind turbine curves;
- tropical cyclone substations/buses: W2.3 open-area substation wind curve.

The damage-ratio interpolation is intentionally simple and traceable: `numpy.interp` between the saved curve points, with values below/above the curve domain using the first/last curve value.

Main scripts:

```text
src/electricity/run_florida_pypsa_calibrated_hazard_scenarios.py
src/electricity/run_florida_pypsa_gradual_return_period_suite.py
src/electricity/validate_gradual_damage_scenarios.py
```

## 5. Intersecting Assets With Hazards

### Tropical cyclone wind

The tropical cyclone workflow can use two kinds of wind hazard:

- real historical storm/event data from IBTrACS, processed through OpenGIRA/SNAIL-style rasters and intersections;
- synthetic/return-period wind data from STORM.

The intersection step connects wind intensity to each line, generator, and bus/substation. Then the fragility/vulnerability curves convert wind speed into damage ratios. Those damage ratios are passed into PyPSA as derated line capacity, derated generator capacity, or bus/substation dependency effects.

Main scripts:

```text
src/exposure/open_gira_ibtracs_event_damage_snail_new_grid.py
src/exposure/open_gira_ibtracs_asset_impacts_new_grid.py
src/exposure/florida_tc_snail_intersection_clean_assets.py
src/electricity/run_florida_pypsa_calibrated_hazard_scenarios.py
src/electricity/run_florida_pypsa_gradual_return_period_suite.py
```

### Flood

The flood workflow intersects JRC return-period flood depth rasters/tables with:

- transmission lines;
- power plant points or polygons;
- PyPSA buses/substations.

Flood depth is then converted into damage ratios using the flood curves described above.

Main scripts:

```text
src/exposure/build_flood_line_exposure_with_ids.py
src/exposure/flood_powerplant_polygon_exposure.py
src/exposure/build_pypsa_bus_hazard_exposure_with_snail.py
src/electricity/run_florida_pypsa_gradual_return_period_suite.py
```

## 6. Expected Annual Cost and Exceedance Curves

For return-period scenarios, the model can calculate:

- load shed by return period;
- import slack by return period;
- generation capacity lost by return period;
- system cost by return period;
- expected annual load shedding or expected annual cost by integrating across exceedance probabilities.

This was done for tropical cyclone wind and for flooding. The same idea is also used later in the flood adaptation analysis: compare baseline expected annual load shedding with adapted expected annual load shedding.

Main scripts:

```text
src/electricity/run_florida_pypsa_gradual_return_period_suite.py
src/electricity/analyze_gradual_return_period_results.py
src/electricity/plot_updated_annualized_risk_graphs.py
src/electricity/plot_updated_gradual_return_period_graphs.py
```

## 7. N-1 Contingency Test

The N-1 script removes one bulk transmission line at a time and reruns a practical PyPSA screen. It checks whether losing a single line causes new overloads, more emergency import use, or load shedding.

This is a DC/linear operational screen. It is useful for seeing which bulk lines are operationally important, but it is not a full protection-system/cascading-failure model.

Main script:

```text
src/electricity/run_florida_pypsa_n1_bulk_contingency.py
```

## 8. Single-Storm Hurricane Ian Analysis

For Hurricane Ian, the workflow uses the IBTrACS event id:

```text
2022266N12294
```

The Ian workflow selects one historical storm, applies the direct wind damage to the grid, reruns PyPSA, and makes figures showing both:

- direct damage, meaning the assets intersected by the storm/wind field;
- indirect operational consequences, like changed loading, new bottlenecks, load shedding, import slack, and system-cost changes.

Main scripts:

```text
src/exposure/plot_single_ibtracs_storm_direct_and_cascade.py
src/exposure/make_hurricane_ian_publication_figures.py
src/electricity/summarize_ibtracs_cascading_results.py
```

## 9. EAGLE-I Comparison

The EAGLE-I workflow compares modeled Hurricane Ian county-level PyPSA consequences with observed county customer outage patterns from EAGLE-I.

This comparison needs a careful caveat: EAGLE-I is customers without power, while PyPSA is transmission-scale demand not served. So the comparison is about broad spatial severity, rankings, and where the model does/does not line up, not a perfect one-to-one match.

The final comparison also notes that timing comparison was not valid for that run because the selected PyPSA Ian output was static bus-total output with synthetic model timestamps, not aligned historical time-varying county load shedding.

Main scripts:

```text
src/validation/eaglei/run_eaglei_florida_validation.py
src/validation/eaglei/run_eaglei_pypsa_comparison.py
src/validation/eaglei/create_final_validation_graphs.py
```

Main graph types:

- observed vs modeled county scatter;
- normalized spatial severity;
- top county observed vs modeled bars;
- county rank agreement;
- largest model-data differences;
- severity confusion matrix;
- observed/modeled Florida maps;
- summary dashboard.

## 10. Flood Hardening and Cost-Benefit

The flood adaptation analysis switches from "what happens if a flood damages the grid?" to "what if we harden selected assets?"

The final workflow selected a five-asset package from the RP100 flood results. For each selected asset, the hardening design depth was:

```text
design_depth_m = RP100_flood_depth_m + 0.30
```

For each flood return-period scenario, residual flood depth was:

```text
residual_flood_depth_m = max(0, scenario_flood_depth_m - design_depth_m)
```

Then the flood damage curves were reapplied using residual depth, PyPSA was rerun, and baseline vs adapted load shedding was compared.

The cost-benefit step monetized avoided expected annual load shedding using illustrative Value of Lost Load assumptions. It reports BCR, NPV, payback, break-even capital cost, and sensitivity to cost/VOLL assumptions. Avoided physical repair costs were not included, so the economics are a planning-level service-interruption analysis rather than a full utility business case.

Main scripts:

```text
src/adaptation/run_flood_asset_criticality.py
src/adaptation/run_rp100_top5_pilot.py
src/adaptation/run_rp100_top5_full_suite.py
src/adaptation/run_rp100_top5_cost_benefit.py
src/adaptation/create_final_summary_tables.py
```

Important saved outputs:

```text
outputs/summary_tables/Flood_Adaptation_Key_Findings.md
outputs/summary_tables/full_suite_cost_benefit_results.csv
outputs/summary_tables/full_suite_annualized_risk.csv
outputs/figures/
```

Headline result from the saved summary:

- Baseline EENS: 190,229.9 MWh/year
- Adapted EENS: 186,129.4 MWh/year
- Avoided EENS: 4,100.5 MWh/year
- Risk reduction: 2.16%
- Central BCR: 19.76
- Central NPV: $763.1M

## 11. What Is Not in GitHub

GitHub intentionally does not contain:

- raw HIFLD/OSM/GPPD/EIA/EAGLE-I extracts;
- large PyPSA `.nc` networks;
- JRC flood rasters;
- OpenGIRA/SNAIL intermediate raster outputs;
- QGIS review layers;
- full scenario run folders.

The repo is meant to show the code, the final selected small outputs, and the logic of the workflow. The raw data should be downloaded from the original providers listed above and rebuilt locally.
