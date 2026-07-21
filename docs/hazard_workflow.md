# Hazard Workflow

The project evaluates direct physical damage and indirect operational consequences from tropical-cyclone wind and flood hazards.

## Tropical Cyclone Wind

OpenGIRA/SNAIL-style workflows intersect storm wind footprints with electricity assets. For each event, the workflow estimates direct exposure and damage to lines, substations, and plant polygons where available.

Relevant scripts:

```text
src/exposure/florida_tc_open_gira_exposure.py
src/exposure/florida_tc_snail_intersection_clean_assets.py
src/exposure/open_gira_ibtracs_event_damage_snail_new_grid.py
src/exposure/plot_single_ibtracs_storm_direct_and_cascade.py
```

## Flood

Flood exposure is evaluated by intersecting return-period flood depths with transmission lines, generator locations or polygons, and bus/substation assets. Flood depths are converted into damage ratios using the F6.3 flood vulnerability relationship.

Relevant scripts:

```text
src/exposure/build_flood_line_exposure_with_ids.py
src/exposure/flood_powerplant_polygon_exposure.py
src/cost/flood_ead_analysis.py
src/cost/fragility_damage_analysis.py
```

## Hazard Scenarios in PyPSA

Hazard scenario scripts reduce line capacity or generator availability according to damage ratios, then rerun PyPSA dispatch. The model reports load shed, import slack, line loading, generation by carrier, and system cost.

Relevant scripts:

```text
src/electricity/run_florida_pypsa_calibrated_hazard_scenarios.py
src/electricity/run_florida_pypsa_gradual_return_period_suite.py
src/electricity/validate_gradual_damage_scenarios.py
```

## Annualized Risk

Expected annual load shedding is computed by integrating loss values across annual exceedance probabilities. In the final flood workflow, the full suite used RP10, RP20, RP50, RP75, RP100, RP200, and RP500.
