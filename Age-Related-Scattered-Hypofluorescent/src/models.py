"""AdaBoost fitting plus the MDI (Mean Decrease in Impurity) feature-importance table.

This merges the six near-duplicate cells of the notebook (three classification,
three regression). The only differences between them were (a) the row window and
(b) classification vs regression, so both became arguments.
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
    gini: pd.DataFrame          # sorted MDI table with zero rows removed
    n_features: int             # row count *before* zeros were removed (the notebook's n_features)
    fit_score: float            # score on the window actually fitted
    title_score: float          # score used in the plot title -- see the note below
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

    # The notebook takes the row count *before* filtering out zeros for the title.
    n_features = gini.shape[0]
    gini = gini[gini[col] != 0]
    return gini, n_features, dropped


def fit(X: pd.DataFrame, y: pd.Series, kind: str, window: Window = None,
        seed: int = config.SEED) -> FitResult:
    """Fit AdaBoost on one window and build the MDI table.

    A note on title_score: the notebook always computes the R2 shown in the title on
    the **full** X_train / y_train, even in the cells that fit on a window. So the
    number in the title is not that window's performance. That behaviour is kept for
    reproducibility, with the window's own score exposed separately as fit_score.
    """
    set_seed(seed)
    X_fit, y_fit = slice_window(X, y, window)

    if kind == "binary":
        # n_estimators uses the **full** feature count, as in the notebook,
        # regardless of the window.
        model = AdaBoostClassifier(n_estimators=len(X.columns), random_state=0)
    elif kind == "regression":
        model = AdaBoostRegressor()
    else:
        raise ValueError(f"unknown kind: {kind}")

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
    """Run the windows corresponding to the notebook's three cells, in order."""
    if windows is None:
        windows = config.DEFAULT_WINDOWS[kind]
    results = []
    for w in windows:
        label = "full" if w is None else f"[{w[0]}:{w[1]}]"
        print(f"\n===== {kind} - window {label} =====")
        results.append(fit(X, y, kind, window=w, seed=seed))
    return results
