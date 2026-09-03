"""Excel -> standardization -> (optionally) PCA feature selection -> training CSV.

Corresponds to the two "Data: from df to cleaned df" cells of the notebook.

Standardization is applied to every column except the target, over the whole dataset
and before any split -- exactly as in the notebook. This leaks in the strict sense,
but it is kept so the published numbers reproduce.
"""
from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from . import config


def set_seed(seed: int = config.SEED) -> None:
    """Seed numpy / random / torch, as the notebook does.

    The models are sklearn and never touch the torch RNG, but the original seeds it,
    so it is kept. The torch import is optional so this runs without torch installed.
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
            f"Source spreadsheet not found: {excel_path}\n"
            "Patient data is not part of this repository. Place it under data/ or "
            "pass a path with --excel."
        )
    df = pd.read_excel(excel_path, sheet_name=sheet_name)
    return df.rename(columns=config.COLUMN_RENAMES)


def build_standardized(df: pd.DataFrame, task: str) -> pd.DataFrame:
    """Put the target in the first column and standardize the rest."""
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
    """Keep only the features the pca package marks as type == 'best' in topfeat."""
    try:
        from pca import pca
    except ImportError as e:
        raise ImportError(
            "The PCA step needs the 'pca' package: pip install pca\n"
            "Use --no-pca to run without it."
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
        # See the note on config.PCA_TRUNCATION_SENTINEL -- this branch never fires
        # in the original.
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
    """Excel -> training CSV. Returns the path it was written to."""
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
    """Read the CSV, shuffle it, and split into (y, X). The target is the first column."""
    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(f"{csv_path} not found. Run `main.py prepare` first.")
    df = pd.read_csv(csv_path)
    shuffled = df.sample(frac=1, random_state=seed).reset_index(drop=True)
    return shuffled.iloc[:, 0], shuffled.iloc[:, 1:]
