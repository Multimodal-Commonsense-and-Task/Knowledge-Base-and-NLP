"""MDI 가로 막대 그래프. ASHS-LIA 만 다른 색으로 강조한다."""
from __future__ import annotations

from pathlib import Path

import matplotlib
from matplotlib import pyplot as plt

from . import config
from .models import FitResult


def plot_mdi(result: FitResult, kind: str, out_path: Path | str | None = None,
             show: bool = False):
    """노트북 플롯을 그대로 재현한다 (제목·색·여백·범례까지)."""
    title_name = "AdaBoost Regressor" if kind == "regression" else "AdaBoost Classifier"
    offset = config.FEATURE_COUNT_OFFSET[kind]
    score = round(result.title_score, 3)

    show_legend = kind == "regression"
    ax = result.gini.plot(kind="barh", figsize=(9, 7), fontsize=14,
                          legend=None if not show_legend else True)
    plt.title(f"{title_name}, R2 Score = {score}\n"
              f" (Number of Features = {result.n_features + offset})", fontsize=18)
    if show_legend:
        plt.legend(loc="lower right", fontsize=15)
    plt.axvline(x=0, color=".5")

    colors = [config.COLOR_HIGHLIGHT if label == config.HIGHLIGHT_FEATURE
              else config.COLOR_DEFAULT for label in result.gini.index]
    for bar, color in zip(ax.patches, colors):
        bar.set_color(color)
    plt.subplots_adjust(left=.3)

    if out_path is not None:
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(out_path, dpi=200, bbox_inches="tight")
        print(f"[saved] {out_path}")
    if show:
        plt.show()
    else:
        plt.close()
    return ax


def use_headless_backend() -> None:
    """디스플레이 없는 서버에서 저장만 할 때 호출한다."""
    matplotlib.use("Agg")
