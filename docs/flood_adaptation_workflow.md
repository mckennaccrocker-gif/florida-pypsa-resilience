# Flood Adaptation Workflow

The final adaptation workflow evaluates a five-asset RP100 flood protection package.

## Selected Assets

The selected package includes:

- `gen_27_Fort Myers`
- `gen_21_Sanford`
- `gen_151_Gulf Clean Energy Center`
- `line_2067`
- `line_1978`

## Adaptation Method

Each selected asset is protected to:

```text
design_depth_m = RP100_flood_depth_m + 0.30
```

For each flood scenario:

```text
residual_flood_depth_m = max(0, scenario_flood_depth_m - design_depth_m)
```

The F6.3 flood vulnerability relationship is then applied to residual depth. This means selected assets are protected against the design event but are not made permanently invulnerable.

## Full Return-Period Suite

The final workflow uses all available F6.3 flood return periods:

```text
RP10, RP20, RP50, RP75, RP100, RP200, RP500
```

Relevant scripts:

```text
src/adaptation/run_rp100_top5_pilot.py
src/adaptation/run_rp100_top5_full_suite.py
```

## Cost-Benefit Assessment

The economic analysis uses avoided expected annual load shedding and illustrative Value of Lost Load assumptions. The values are planning-level societal outage benefits, not utility revenue.

Relevant scripts:

```text
src/adaptation/run_rp100_top5_cost_benefit.py
src/adaptation/create_final_summary_tables.py
```

## Final Full-Suite Result

- Baseline EENS: 190,229.9 MWh/year
- Adapted EENS: 186,129.4 MWh/year
- Avoided EENS: 4,100.5 MWh/year
- Risk reduction: 2.16%
- Central BCR: 19.76
- Central NPV: $763.1M

These results should be interpreted as model-estimated and planning-level.
