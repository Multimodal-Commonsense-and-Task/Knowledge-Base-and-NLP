# Age-Related Scattered Hypofluorescent Spots as an Adverse Prognostic Factor for Polypoidal Choroidal Vasculopathy

Official code for the paper published in *Ophthalmology Science* (2025), Volume 5, Issue 5.

**Authors:** Kai Tzu-iunn Ong, Seo Hee Kim, Seonghee Choi, Eun Jee Chung, Min Kim, Christopher Seungkyu Lee, Jinyoung Yeo, Eun Young Choi

[Paper](https://www.ophthalmologyscience.org/article/S2666-9145(25)00116-2/fulltext)

## Overview

Polypoidal choroidal vasculopathy (PCV) shows wide variability in treatment outcomes, which motivates the search for imaging biomarkers that can predict its course. Age-related scattered hypofluorescent spots on late-phase indocyanine green angiography (ASHS-LIA) are thought to reflect lipid accumulation in Bruch's membrane, but their prognostic role in PCV had not been established.

This study applies machine learning to a small, multi-factor clinical dataset to quantify how much ASHS-LIA contributes to predicting PCV prognosis.

## Method

- **Design:** Retrospective, cross-sectional analysis of PCV patients treated with anti-VEGF therapy at two institutions (Severance Hospital and National Health Insurance Ilsan Hospital) between January 2012 and December 2021.
- **Feature selection:** Principal component analysis (PCA) to reduce dimensionality of the clinical variables.
- **Model:** AdaBoost meta-estimator — classifiers for disease stability and injection demand, and a regressor for time to first remission.
- **Evaluation:** Feature importance measured by mean decrease in impurity, used to rank the prognostic contribution of ASHS-LIA against other clinical variables.

## Key Findings

Among 57 PCV eyes, 31 showed ASHS-LIA and 26 did not. Relative to the non-ASHS-LIA group, eyes with ASHS-LIA:

- less often reached a super-stable state (no recurrence for over 18 months after remission), *P* = 0.03
- took longer to reach first remission, *P* = 0.04
- required more anti-VEGF injections, *P* < 0.001

The AdaBoost models ranked ASHS-LIA as the 3rd, 7th, and 8th most contributory feature for disease stability, injection demand, and time to first remission, respectively. Overall, ASHS-LIA behaves as an adverse prognostic marker in PCV, and the PCA + boosted-tree pipeline offers a workable approach for identifying risk factors in small clinical datasets with many candidate variables.

## Repository Structure