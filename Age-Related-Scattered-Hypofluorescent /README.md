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
│   ├── config.py        # 경로 · 태스크 정의 · 시드 · 플롯 상수
│   ├── data.py          # 엑셀 로드 → 표준화 → (선택) PCA 피처 선택 → CSV
│   ├── models.py        # AdaBoost 학습 + MDI 피처 중요도 표
│   └── plots.py         # MDI 가로 막대 그래프 (ASHS-LIA 강조)
├── data/                # 환자 데이터 (저장소에 포함되지 않음)
└── PCV_main_copy.ipynb  # 원본 노트북
```

## Usage

```bash
pip install -r requirements.txt

# 엑셀 → 표준화(+PCA) → 학습용 CSV
python main.py prepare --task task8

# CSV → AdaBoost → MDI 플롯
python main.py run --task task8 --save

# 한 번에
python main.py all --task task8 --save
```

`--task` 로 모델이 정해진다 — `task1`(disease stability) · `task2`(injection demand) 는
AdaBoost 분류, `task8`(time to first remission) 은 AdaBoost 회귀다.
`--no-pca` 로 PCA 피처 선택을 건너뛸 수 있고, `--windows 32: 31:57` 로 행 구간을 지정한다
(기본값은 노트북이 쓰던 세 구간).

환자 데이터(`data/new_hypoF_PCV 220919.xlsx`)는 저장소에 포함되지 않는다.
`--excel` 로 경로를 지정할 수 있다.

## Notes on the refactor

노트북을 옮기면서 **수치는 바꾸지 않았다** — 원본 셀을 그대로 실행한 결과와
`src/` 모듈의 결과가 세 태스크 × 세 구간 전부에서 일치하는 것을 확인했다
(fit score · title score · `feature_importances_`).

원본에 있던 아래 세 가지는 재현을 위해 **동작을 유지하되 코드에 표시해 두었다**:

- **플롯 제목의 R2 는 항상 전체 학습셋 기준이다.** 구간(`[32:]` 등)으로 학습한 셀에서도
  제목 숫자는 전체 `X_train` 으로 계산된다. 즉 제목의 값은 그 구간의 성능이 아니다.
  구간 성능은 `FitResult.fit_score`, 제목 값은 `FitResult.title_score` 로 분리해 두었다.
- **`--pca-truncation` 은 무동작이다.** 기준값이 `"HypoF"` 인데 로드 직후 `"ASHS-LIA"` 로
  rename 되므로 조건이 성립하지 않는다. `config.PCA_TRUNCATION_SENTINEL` 참고.
- **표준화가 셔플·구간 분리 이전에 전체 데이터로 수행된다.**

원본의 `make_classification(...)` 호출은 결과에 쓰이지 않아 제거했다 (전역 RNG 에도
영향이 없어 수치가 바뀌지 않는다). 주석 처리돼 있던 statsmodels GLM/Logit/WLS 블록과
`task3~task6` 정의는 `config.TASKS_DISABLED` 에 옮겨 두었다.
