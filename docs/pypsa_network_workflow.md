# PyPSA Network Workflow

This document summarizes how the Florida electricity assets were converted into a PyPSA network.

## 1. Transmission Topology

Transmission line geometries were converted into network edges. Each line endpoint became a candidate node. Nearby endpoints were snapped together so that lines meeting at the same physical location became connected in the graph.

The snapping step matters because PyPSA needs every line to connect two buses. If endpoints that should meet are slightly misaligned, the model creates disconnected islands.

Relevant scripts:

```text
src/electricity/build_florida_transmission_network.py
src/electricity/apply_island_review_decisions.py
src/electricity/build_extended_georgia_alabama_tie_line_network.py
```

## 2. Buses, Voltage Layers, and Transfer Components

PyPSA calls network connection points `Bus` components. Transmission endpoints and substations were converted into buses with coordinates and nominal voltages.

Where different voltage levels meet at the same physical substation, the workflow creates voltage-layer buses and transfer/transformer-style components so power can move between voltage layers.

Relevant script:

```text
src/electricity/create_voltage_layer_transformer_network.py
```

## 3. Line Electrical Parameters and Capacity

Lines need resistance, reactance, susceptance, voltage, and capacity. Some parameters were inferred using voltage-class assumptions when direct electrical parameters were unavailable.

Line capacity was assigned using static line ratings where possible:

```text
s_nom = sqrt(3) * voltage_kV * amps / 1000
```

Relevant script:

```text
src/electricity/add_slr_s_nom_to_florida_lines.py
```

## 4. Power Plants and Generator Assignment

Power plants were matched between OSM and GPPD-style plant data. The final plant/generator records were assigned to the nearest suitable bus, with manual review for important large generators.

Relevant scripts:

```text
src/electricity/merge_florida_osm_gppd_powerplants.py
src/electricity/finalize_generator_bus_assignment_review.py
src/electricity/convert_florida_assets_to_pypsa.py
```

## 5. Demand, Renewable Profiles, and Costs

Hourly Florida demand is assigned to buses. Solar availability profiles provide time-varying renewable availability. Generator marginal costs control dispatch order.

Relevant scripts:

```text
src/electricity/download_eia_florida_hourly_demand.py
src/electricity/build_county_population_load_profiles.py
src/electricity/create_florida_renewable_profiles.py
src/electricity/finalize_florida_generator_marginal_costs.py
```

## 6. Emergency Import Slack and Load Shedding

Emergency import slack generators represent external power that can enter the system at selected boundary buses. Load-shedding generators are artificial expensive generators added at demand buses so PyPSA can remain feasible when demand cannot be served.

Relevant scripts:

```text
src/electricity/select_florida_boundary_import_buses.py
src/electricity/run_florida_pypsa_load_shedding_dispatch.py
```

## 7. Baseline Dispatch

The no-hazard baseline is solved before hazard scenarios. This baseline is the reference used to compare load shedding, dispatch, import slack, and system cost.

Relevant script:

```text
src/electricity/run_florida_pypsa_baseline_validation.py
```
