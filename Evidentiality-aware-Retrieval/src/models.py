"""AdaBoost 학습 + MDI(Mean Decrease in Impurity) 피처 중요도 표.

노트북에서 여섯 번 복붙돼 있던 셀(분류 3 · 회귀 3)을 하나로 합친 것이다.
셀 사이의 차이는 (a) 행 구간 (b) 분류/회귀 뿐이라 인자로 뺐다.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd
from sklearn.ensemble import AdaBoostClassifier, AdaBoostRegressor

from . import config
from .data import set_seed

Window = tuple[int | None, int | None] | None


@dataclass
class FitResult:
    model: object
    gini: pd.DataFrame          # 0 인 행을 제거한, 정렬된 MDI 표
    n_features: int             # 0 제거 *이전* 행 수 (노트북의 n_features)
    fit_score: float            # 학습에 쓴 구간에서의 score
    title_score: float          # 플롯 제목에 쓰는 score — 아래 주석 참고
    window: Window = None
    n_samples: int = 0
    dropped_from_mdi: list[str] = field(default_factory=list)


def slice_window(X: pd.DataFrame, y: pd.Series, window: Window):
    if window is None:
        return X, y
    lo, hi = window
    return X[lo:hi], y[lo:hi]


def _mdi_table(model, columns, kind: str) -> tuple[pd.DataFrame, int, list[str]]:
    col = "MDI" if kind == "regression" else "Mean Decrease in Impurity (MDI)"
    gini = pd.DataFrame(model.feature_importances_,
                        columns=["Mean Decrease in Impurity (MDI)"], index=columns)
    if kind == "regression":
        gini = gini.rename(columns={"Mean Decrease in Impurity (MDI)": "MDI"})
    gini = gini.sort_values(by=[col])
    print(gini)

    dropped = []
    for name in config.DROP_FROM_MDI[kind]:
        if name in gini.index:
            gini = gini.drop(index=name)
            dropped.append(name)

    # 노트북은 0 을 걸러내기 *전*의 행 수를 제목에 쓴다.
    n_features = gini.shape[0]
    gini = gini[gini[col] != 0]
    return gini, n_features, dropped


def fit(X: pd.DataFrame, y: pd.Series, kind: str, window: Window = None,
        seed: int = config.SEED) -> FitResult:
    """한 구간에 대해 AdaBoost 를 학습하고 MDI 표를 만든다.

    title_score 주의: 노트북은 제목의 R2 를 **항상 전체 X_train/y_train** 으로 계산한다.
    구간 학습 셀에서도 그렇다. 즉 제목 숫자는 그 구간의 성능이 아니다.
    재현을 위해 그대로 두되, 구간 성능은 fit_score 에 따로 담는다.
    """
    set_seed(seed)
    X_fit, y_fit = slice_window(X, y, window)

    if kind == "binary":
        # n_estimators 는 노트북대로 **전체** 피처 수를 쓴다 (구간과 무관).
        model = AdaBoostClassifier(n_estimators=len(X.columns), random_state=0)
    elif kind == "regression":
        model = AdaBoostRegressor()
    else:
        raise ValueError(f"알 수 없는 kind: {kind}")

    model.fit(X_fit, y_fit)
    fit_score = model.score(X_fit, y_fit)
    print(f"model score on training data: {fit_score}")
    print(model.feature_importances_)

    gini, n_features, dropped = _mdi_table(model, X_fit.columns, kind)
    return FitResult(model=model, gini=gini, n_features=n_features,
                     fit_score=fit_score, title_score=model.score(X, y),
                     window=window, n_samples=len(y_fit), dropped_from_mdi=dropped)


def run_windows(X: pd.DataFrame, y: pd.Series, kind: str,
                windows: list[Window] | None = None,
                seed: int = config.SEED) -> list[FitResult]:
    """노트북의 세 셀에 해당하는 구간들을 순서대로 돌린다."""
    if windows is None:
        windows = config.DEFAULT_WINDOWS[kind]
    results = []
    for w in windows:
        label = "full" if w is None else f"[{w[0]}:{w[1]}]"
        print(f"\n===== {kind} · window {label} =====")
        results.append(fit(X, y, kind, window=w, seed=seed))
    return results
