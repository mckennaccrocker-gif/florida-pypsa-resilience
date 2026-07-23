# Hazard Workflow

The project evaluates direct physical damage and indirect operational consequences from tropical-cyclone wind and flood hazards.

## Tropical Cyclone Wind

OpenGIRA/SNAIL-style workflows intersect storm wind footprints with electricity assets. For each event, the workflow estimates direct exposure and damage to lines, substations, and plant polygons where available.

The workflow can use historical IBTrACS events or synthetic/return-period storm wind layers. For historical single-storm work, Hurricane Ian is handled as IBTrACS event `2022266N12294`.

Relevant scripts:

```text
src/exposure/florida_tc_open_gira_exposure.py
src/exposure/florida_tc_snail_intersection_clean_assets.py
src/exposure/open_gira_ibtracs_event_damage_snail_new_grid.py
src/exposure/plot_single_ibtracs_storm_direct_and_cascade.py
src/exposure/make_hurricane_ian_publication_figures.py
```

## Flood

Flood exposure is evaluated by intersecting JRC return-period flood depths with transmission lines, generator locations or polygons, and bus/substation assets. Flood depths are converted into damage ratios using the selected flood vulnerability curves.

Relevant scripts:

```text
src/exposure/build_flood_line_exposure_with_ids.py
src/exposure/flood_powerplant_polygon_exposure.py
src/exposure/build_pypsa_bus_hazard_exposure_with_snail.py
src/cost/flood_ead_analysis.py
src/cost/fragility_damage_analysis.py
```

## Curves Used in Hazard Scenarios

The scenario scripts use saved curve CSVs and linear interpolation between the curve points.

- Flood lines: F6.2.
- Flood power plants: F1.1-F1.4.
- Flood substations/buses: F2.1-F2.3.
- TC lines: W6.3.
- TC generators: W1.10-W1.13 and W1.6.
- TC substations/buses: W2.3.

## Hazard Scenarios in PyPSA

Hazard scenario scripts reduce line capacity or generator availability according to damage ratios, then rerun PyPSA dispatch. The model reports load shed, import slack, line loading, generation by carrier, and system cost.

Relevant scripts:

```text
src/electricity/run_florida_pypsa_calibrated_hazard_scenarios.py
src/electricity/run_florida_pypsa_gradual_return_period_suite.py
src/electricity/validate_gradual_damage_scenarios.py
```

## Annualized Risk

Expected annual load shedding or expected annual system cost is computed by integrating loss values across annual exceedance probabilities. In the final flood workflow, the full suite used RP10, RP20, RP50, RP75, RP100, RP200, and RP500.

## EAGLE-I Comparison

The Hurricane Ian validation workflow compares PyPSA county-level modeled consequences with EAGLE-I county outage observations. This is a spatial severity/ranking comparison, not a perfect physical match, because EAGLE-I is customer outages and PyPSA is transmission-scale unserved demand.

Relevant scripts:

```text
src/validation/eaglei/run_eaglei_florida_validation.py
src/validation/eaglei/run_eaglei_pypsa_comparison.py
src/validation/eaglei/create_final_validation_graphs.py
```
