# Full-Suite Sanity Checks

1. Load shed monotonic with return period: True. Baseline load shedding is broadly increasing if this is true.
2. Irregular dispatch/solver changes: review scenario comparison warnings. Current warnings: [].
3. Avoided load shed is largest at RP20: 13,173.0 MWh.
4. Integration uses all available return periods and trapezoidal AEP integration; no simple summation is used.
5. RP10 share of summed avoided scenario MWh is 9.35%; this is checked because high-AEP points can strongly affect annualized EENS.
6. RP10 is treated as a modeled annual-exceedance loss point because it is part of the existing gradual flood suite.
7. Every scenario uses 24 snapshots, so load shedding is measured over the same event duration.
8. Load shedding is MWh over the scenario window, not MW.
9. Scenario probabilities are integrated in AEP space rather than summed.
10. Outage benefits are calculated once from avoided EENS, not once per snapshot or asset.

Full-suite avoided EENS: 4,100.5 MWh/year.
