"""엑셀 원본 -> 표준화 -> (선택) PCA 피처 선택 -> 학습용 CSV.

노트북의 `Data: from df to cleaned df` 두 셀에 해당한다.
표준화는 target 을 제외한 전 컬럼에 대해, split 이전에 전체 데이터로 수행한다 —
노트북 그대로다. 누출 관점에서는 문제가 있으나 논문 수치 재현을 위해 유지한다.
"""
from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from . import config


def set_seed(seed: int = config.SEED) -> None:
    """노트북과 동일하게 numpy / random / torch 를 시드한다.

    모델은 sklearn 이라 torch RNG 를 쓰지 않지만, 원본이 시드하고 있어 그대로 둔다.
    torch 가 없는 환경에서도 돌도록 임포트는 선택적으로 처리한다.
    """
    np.random.seed(seed)
    random.seed(seed)
    try:
        import torch
    except ImportError:
        pass
    else:
        torch.manual_seed(seed)


def load_raw(excel_path: Path | str = config.EXCEL_PATH,
             sheet_name: str = config.SHEET_NAME) -> pd.DataFrame:
    excel_path = Path(excel_path)
    if not excel_path.exists():
        raise FileNotFoundError(
            f"원본 엑셀이 없다: {excel_path}\n"
            "환자 데이터는 저장소에 포함되지 않는다. data/ 에 직접 두거나 --excel 로 경로를 줄 것."
        )
    df = pd.read_excel(excel_path, sheet_name=sheet_name)
    return df.rename(columns=config.COLUMN_RENAMES)


def build_standardized(df: pd.DataFrame, task: str) -> pd.DataFrame:
    """target 을 첫 컬럼으로 두고, 나머지를 표준화한 프레임을 만든다."""
    target, drop_cols = config.TASKS[task]

    df_y = df[target]
    df_x = df.drop(columns=drop_cols).drop(columns=target)
    if task in config.SHIFT_TARGET_BY_ONE:
        df_y = df_y - 1

    out = pd.concat([df_y, df_x], axis=1, join="inner")
    out = out.dropna()
    out = out.astype(float)

    feature_cols = out.columns[1:]
    out[feature_cols] = StandardScaler().fit_transform(out[feature_cols])
    return out


def select_features_pca(df_std: pd.DataFrame, task: str,
                        truncate: bool = False, verbose: bool = True) -> pd.DataFrame:
    """pca 패키지의 topfeat 중 type=='best' 인 피처만 남긴다."""
    try:
        from pca import pca
    except ImportError as e:
        raise ImportError(
            "PCA 단계에는 'pca' 패키지가 필요하다: pip install pca\n"
            "PCA 없이 돌리려면 --no-pca 를 쓸 것."
        ) from e

    target = config.TASKS[task][0]
    df_x = df_std.drop(columns=target)

    model = pca()
    out = model.fit_transform(df_x)
    if verbose:
        print(out["topfeat"])

    kept: list[str] = []
    for _, row in out["topfeat"].iterrows():
        if row["type"] != "best":
            continue
        if row["feature"] in kept:
            continue
        kept.append(row["feature"])
        # ⚠ config.PCA_TRUNCATION_SENTINEL 주석 참고 — 원본에서는 이 분기가 발화하지 않는다.
        if truncate and row["feature"] == config.PCA_TRUNCATION_SENTINEL:
            break

    if verbose:
        print(len(kept))
        print(kept)

    df_x_kept = df_x.drop(df_x.columns.difference(kept), axis=1)
    return pd.concat([df_std[target], df_x_kept], axis=1, join="inner")


def prepare(task: str, use_pca: bool = True, truncate: bool = False,
            excel_path: Path | str = config.EXCEL_PATH,
            out_path: Path | None = None, verbose: bool = True) -> Path:
    """엑셀 -> 학습용 CSV. 저장 경로를 돌려준다."""
    set_seed()
    df = load_raw(excel_path)
    if verbose:
        for col in df.columns:
            print(col)

    df_std = build_standardized(df, task)
    if verbose:
        print(df_std.columns)
        print(len(df_std.columns))
        print("----")

    if use_pca:
        df_std = select_features_pca(df_std, task, truncate=truncate, verbose=verbose)
        if verbose:
            print(len(df_std.columns))

    out_path = Path(out_path) if out_path else config.prepared_csv_path(task, use_pca, truncate)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df_std.to_csv(out_path, index=False)
    if verbose:
        print(f"[saved] {out_path}  shape={df_std.shape}")
    return out_path


def load_prepared(csv_path: Path | str, seed: int = config.SEED
                  ) -> tuple[pd.Series, pd.DataFrame]:
    """CSV 를 읽어 셔플하고 (y, X) 로 나눈다. 첫 컬럼이 target 이다."""
    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(f"{csv_path} 가 없다. 먼저 `main.py prepare` 를 돌릴 것.")
    df = pd.read_csv(csv_path)
    shuffled = df.sample(frac=1, random_state=seed).reset_index(drop=True)
    return shuffled.iloc[:, 0], shuffled.iloc[:, 1:]
