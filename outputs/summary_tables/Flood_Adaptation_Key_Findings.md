# Flood Adaptation Key Findings

## Objective

The objective of this workflow was to evaluate a conceptual five-asset flood adaptation package for the Florida PyPSA transmission model. The analysis estimates how much expected annual load shedding could be avoided if the selected assets were protected against RP100 flood depth plus freeboard, and then translates the modeled avoided load shedding into a preliminary planning-level economic assessment.

## Adaptation Approach

The adaptation package protected the same five modeled assets used in the RP100 top-five pilot. For each selected asset, the protection target was defined as the modeled RP100 flood depth plus 0.30 m of freeboard. Damage was recalculated using residual flood depth, meaning that the selected assets were not assumed to become permanently invulnerable and larger flood events could still exceed the design level.

## Five Selected Assets

- `gen_27_Fort Myers` (generator; Fort Myers)
- `gen_21_Sanford` (generator; Sanford)
- `gen_151_Gulf Clean Energy Center` (generator; Gulf Clean Energy Center)
- `line_2067` (line; 2067)
- `line_1978` (line; 1978)

## Main Engineering Assumptions

The analysis used conceptual adaptation categories rather than detailed engineering designs. Generator assets were interpreted as protection of modeled generating facilities and associated electrical equipment. Line assets were interpreted as protection of the modeled exposed line representation and associated endpoint or structure-level equipment where relevant. The cost assumptions remain planning-level ranges and should not be interpreted as site-specific construction estimates.

## Main Flood-Risk Findings

Across the full return-period suite (RP10, RP20, RP50, RP75, RP100, RP200, RP500), baseline expected annual load shedding was 190,229.9 MWh/year and adapted expected annual load shedding was 186,129.4 MWh/year. The package avoided 4,100.5 MWh/year, corresponding to a 2.16% reduction in modeled annual flood load-shedding risk. The largest scenario-level reduction occurred at RP20, where the package avoided 13,173.0 MWh.

## Main Economic Findings

Using the full-suite avoided EENS and the existing illustrative Value of Lost Load assumptions, the central annual avoided outage value was $41.0M per year. Under the central cost and central VOLL case, the benefit-cost ratio was 19.76, the net present value was $763.1M, and the simple payback period was 0.8 years. Even under the low VOLL and high cost case, the BCR was 2.78, suggesting that the package remains potentially cost-effective within the tested sensitivity range.

## Limitations

These findings should be interpreted cautiously. The PyPSA model represents transmission-scale operational consequences and does not capture all distribution-level outages. The VOLL assumptions are illustrative societal outage-value assumptions, not verified Florida-specific utility revenue. The adaptation costs are conceptual planning ranges, and avoided physical repair costs are not included. The result also depends on the modeled flood exposure data, the F6.3 damage relationship, and the annualized-risk integration method.

## Suggested Future Work

Future work should replace the conceptual cost ranges with site-specific engineering estimates, test additional adaptation packages, and compare model-estimated load shedding with observed outage datasets where possible. The adaptation results would also be strengthened by evaluating additional hazard probabilities, considering distribution-network impacts, and testing whether the selected assets remain important under alternative demand, import, and restoration assumptions.
