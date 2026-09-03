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

```
.
├── main.py              # CLI — prepare / run / all
├── requirements.txt
├── src/
│   ├── config.py        # paths, task definitions, seed, plot constants
│   ├── data.py          # excel → standardization → (optional) PCA selection → CSV
│   ├── models.py        # AdaBoost fitting + MDI feature-importance table
│   └── plots.py         # horizontal MDI bar chart (ASHS-LIA highlighted)
├── data/                # patient data (not included in this repository)
└── PCV_main_copy.ipynb  # the original notebook
```

## Usage

```bash
pip install -r requirements.txt

# excel → standardization (+PCA) → training CSV
python main.py prepare --task task8

# CSV → AdaBoost → MDI plot
python main.py run --task task8 --save

# both at once
python main.py all --task task8 --save
```

`--task` selects the model: `task1` (disease stability) and `task2` (injection demand)
are AdaBoost classifiers, `task8` (time to first remission) is an AdaBoost regressor.
`--no-pca` skips PCA feature selection, and `--windows 32: 31:57` overrides the row
windows (the default is the three the notebook used).

The patient data (`data/new_hypoF_PCV 220919.xlsx`) is not part of this repository.
Pass a path with `--excel`.

## Notes on the refactor

Porting the notebook **did not change any number** — running the original cells and
running the `src/` modules produce identical results across all three tasks and all
three row windows (fit score, title score, and `feature_importances_`).

Three quirks of the original are **preserved but flagged in the code**:

- **The R2 in the plot title is always computed on the full training set.** Even in the
  cells that fit on a window (`[32:]` and so on), the title score is computed over the
  whole `X_train`, so the number in the title is not that window's performance. The
  window's own score is exposed as `FitResult.fit_score`, and the title value as
  `FitResult.title_score`.
- **`--pca-truncation` is a no-op.** It keys on `"HypoF"`, but the column is renamed to
  `"ASHS-LIA"` immediately after loading, so the condition never holds. See
  `config.PCA_TRUNCATION_SENTINEL`.
- **Standardization runs on the whole dataset, before shuffling and windowing.**

The original `make_classification(...)` calls were removed: their output was never used,
and they touch no global RNG, so the numbers are unaffected. The commented-out
statsmodels GLM/Logit/WLS blocks and the `task3`–`task6` definitions were moved to
`config.TASKS_DISABLED`.
