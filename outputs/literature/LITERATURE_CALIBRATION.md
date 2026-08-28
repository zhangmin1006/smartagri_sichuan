# Literature-derived parameter candidates

Candidate values extracted from published abstracts, each with the sentence it came from.

> **Nothing here is a calibrated value.** Abstract text is noisy: a number near an anchor phrase may be a sample share, a significance level or a different treatment arm. Every row is marked `verified = False`. The ranges below are prior ENVELOPES for sensitivity analysis, and no value should be promoted to evidence grade B until the source has been read.

## Summary

| Parameter | Symbol | Model value | Lit p25 | median | p75 | n values | n sources |
|---|---|---|---|---|---|---|---|
| Farmer relative risk aversion (CRRA) | `c1` | Beta(4.5, 3.0) on [0.10, 0.92], mean 0.60 | nan | nan | nan | 0 | 0 |
| Loss reduction from mechanised/drone operations under flood | `eta_T3_flood` | 0.35 | 0.5 | 0.515 | 0.585 | 4 | 3 |
| Loss reduction from irrigation / fertigation under drought | `eta_T2_drought` | 0.45 | 0.1455 | 0.226 | 0.5 | 17 | 10 |
| Loss reduction from early warning / climate information | `eta_T1` | 0.12 drought / 0.28 flood | nan | nan | nan | 0 | 0 |
| Outsourced field-operation service price | `price_per_mu` | 12 currency units per mu (T3) | 7.45 | 112.5 | 278.0 | 6 | 1 |
| Observed adoption rate of smart / digital agricultural technology | `adopt` | 0.10 - 0.17 simulated | 0.275 | 0.35 | 0.425 | 2 | 2 |
| Peer / social network effect on adoption | `beta_peer` | 0.30 weight | nan | nan | nan | 0 | 0 |

## Loss reduction from mechanised/drone operations under flood  (`eta_T3_flood`)

Model currently uses: **0.35**

| Value | Source | Year | Cites | Sentence |
|---|---|---|---|---|
| 0.5 | Crops that feed the world 10. Past successes and future challenges to the role played by w | 2013 | 1441 | The developing regions (including China and Central Asia) account for roughly 53 % of the total harvested area and 50 % of the production. |
| 0.5 | A Review on the Effect of Soil Compaction and its Management for Sustainable Crop Producti | 2021 | 321 | Soil compaction resulting from heavy machinery traffic caused a significant crop yield reduction of as much as 50% or even more, depending upon the magnitude and the severity of compaction of the soil. |
| 0.75 | Winter Wheat Yield Prediction at County Level and Uncertainty Analysis in Main Wheat-Produ | 2020 | 275 | We further conducted yield prediction and uncertainty analysis based on the two-branch model and obtained the forecast accuracy in one month prior to harvest of 0.75 and 732 kg/ha. |

## Loss reduction from irrigation / fertigation under drought  (`eta_T2_drought`)

Model currently uses: **0.45**

| Value | Source | Year | Cites | Sentence |
|---|---|---|---|---|
| 0.3 | Regulated deficit irrigation for crop production under drought stress. A review | 2015 | 629 | Among these, partial root-zone irrigation is the most popular and effective because many field crops and some woody crops can save irrigation water up to 20 to 30 % without or with a minimal impact on crop yield. |
| 0.5 | Breeding for water-saving and drought-resistance rice (WDR) in China | 2010 | 449 | The breeding target is a high yield potential under irrigation, an acceptable grain quality, and water consumption reduced by about 50% compared with paddy rice. |
| 0.76 | Water savings potentials of irrigation systems: global simulation of processes and linkage | 2015 | 426 | Replacing surface systems by sprinkler or drip systems could, on average across the world's river basins, reduce the non-beneficial consumption at river basin level by 54 and 76 %, respectively, while maintaining the current level of crop yields. |
| 0.226 | Foliar Application of Zinc Oxide Nanoparticles Promotes Drought Stress Tolerance in Eggpla | 2021 | 363 | Under drought stress, supplementation of 50 and 100 ppm ZnO NP improved growth characteristics and increased fruit yield by 12.2% and 22.6%, respectively, compared with fully irrigated plants and nonapplied ZnO NP. |
| 0.61 | Yield, Mineral Composition, Water Relations, and Water Use Efficiency of Grafted Mini-wate | 2008 | 268 | When averaged over year and irrigation rate, the total and marketable yields were higher by 115% and 61% in grafted than in ungrafted plants, respectively. |
| 0.2892 | Review on Drip Irrigation: Impact on Crop Yield, Quality, and Water Productivity in China | 2023 | 243 | When the drip irrigation amount is more (100–120%), drip irrigation significantly increases crop yields by 28.92%, 14.55%, 8.03%, 2.32%, and 5.17% relative to flooding irrigation, border irrigation, furrow irrigation, sprinkler irrigation, and micro-sprinkler irrigation, respectively. |
| 0.75 | Response of yield, quality, water and nitrogen use efficiency of tomato to different level | 2017 | 184 | The irrigation and fertilisation regime of 75% Ep and 250 kg N ha−1 was the best strategy of water and N management for the production of drip-irrigated greenhouse tomato. |
| 0.20600000000000002 | Effects of Soils and Irrigation Volume on Maize Yield, Irrigation Water Productivity, and  | 2019 | 101 | Medium and low irrigation reduced the maize yield by 12.5–21.8% and 13.5–20.6%, respectively, relative to full irrigation, with the greatest decrease in sandy loam. |
| 0.15 | Water-Saving Potential of Subsurface Drip Irrigation For Winter Wheat | 2019 | 92 | Subsurface drip irrigation reduced ET by 26% compared to flood irrigation, and 15% compared to surface drip irrigation, with significant grain yield and biomass formation due to decreased evaporation losses. |
| 0.51 | Impact of water deficit and irrigation management on winter wheat yield in China | 2023 | 72 | Results indicated that the critical irrigation scenario (CI) was pivotal to alleviating water deficit effects during key growth periods, enhancing wheat growth, and leading to a 51%−92% yield increase compared to rain-fed yields in NCP and NW. |

## Outsourced field-operation service price  (`price_per_mu`)

Model currently uses: **12 currency units per mu (T3)**

| Value | Source | Year | Cites | Sentence |
|---|---|---|---|---|
| 5.6 | What Are the Effects of Participation in Production Outsourcing? Evidence from Chinese App | 2018 | 28 | The results showed that, on average, the outsourcing of apple production increased farmers’ apple production technology efficiency by 5.60%, their labor productivity by 2121.48 kg/person, land productivity by 334.50 kg/mu, capital productivity by 0.05 kg/Yuan, and apple sales revenue by 13,300 Yuan. |

## Observed adoption rate of smart / digital agricultural technology  (`adopt`)

Model currently uses: **0.10 - 0.17 simulated**

| Value | Source | Year | Cites | Sentence |
|---|---|---|---|---|
| 0.5 | Setting the Record Straight on Precision Agriculture Adoption | 2019 | 395 | VRT adoption estimates for niche groups of farmers may exceed 50%. |
| 0.2 | Adoption of digital technologies in agriculture—an inventory in a european small-scale far | 2022 | 166 | Results show that Bavarian farmers cannot be described as exceedingly digitalized but show potential adoption rates of 15–20% within the next five years for technologies such as barn robotics, section control, variable-rate applications, and maps from satellite data. |
