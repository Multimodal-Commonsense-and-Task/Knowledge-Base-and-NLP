"""Paths, task definitions, seed, and plot constants.

These were scattered across the top cells of the notebook and are collected here.
Changing any number changes the published results -- keep the defaults as they are.
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
FIG_DIR = ROOT / "figures"

EXCEL_PATH = DATA_DIR / "new_hypoF_PCV 220919.xlsx"
SHEET_NAME = "cleaned_0"

SEED = 2022

# Column names as used in the paper. Note that this rename collides with
# PCA_TRUNCATION_SENTINEL below -- see the comment there.
COLUMN_RENAMES = {
    "HypoF": "ASHS-LIA",
    "basline_logMAR": "baseline_logMAR",
    "largest_polyp": "Largest polyp diameter",
    "number_of_polyps": "Polyp number",
    "lipid": "Lipid exudation",
    "CNV_location": "Polyp location",
    "SRHeme": "SRH",
}

# task -> (target, features to drop alongside it)
TASKS = {
    "task1": ("group", ["injection_demand", "vision_gain", "rec_period"]),
    "task2": ("injection_demand", ["total_inj_n", "group", "vision_gain", "rec_period"]),
    "task8": ("time_to_rem",
              ["group", "rec_period", "total_inj_n", "vision_gain", "injection_demand", "recur"]),
}

# Tasks that were commented out in the notebook. Move one into TASKS to revive it.
TASKS_DISABLED = {
    "task3": ("total_inj_n", ["injection_demand", "group", "vision_gain", "rec_period"]),
    "task4": ("rec_period", ["group", "vision_gain", "injection_demand", "recur"]),
    "task5": ("recur", ["group", "vision_gain", "injection_demand", "rec_period"]),
    "task6": ("vision_gain", ["group", "injection_demand", "rec_period"]),
}

TASK_KIND = {"task1": "binary", "task2": "binary", "task8": "regression"}

# The target of task1 arrives as 1/2 and is shifted down to 0/1 (`df_y = df_y - 1`).
SHIFT_TARGET_BY_ONE = {"task1"}

# NOTE: the notebook keys truncation on "HypoF", but the column is renamed to
# "ASHS-LIA" right after loading, so the condition can never be true -- i.e.
# --pca-truncation was effectively a no-op in the original. The default is kept for
# reproducibility; change it to "ASHS-LIA" to get the intended behaviour.
PCA_TRUNCATION_SENTINEL = "HypoF"

# Index entries excluded from the MDI table. Classification and regression differ,
# exactly as in the notebook.
DROP_FROM_MDI = {
    "binary": ["recur", "fu_period", "time_to_rem"],
    "regression": ["recur", "fu_period"],
}

# Constant added to "(Number of Features = ...)" in the title: +2 for classification,
# +1 for regression, as in the notebook.
FEATURE_COUNT_OFFSET = {"binary": 2, "regression": 1}

HIGHLIGHT_FEATURE = "ASHS-LIA"
COLOR_HIGHLIGHT = "#e01e5a"
COLOR_DEFAULT = "#236AA7"

# Three notebook cells refit the same model on different row windows (a sensitivity
# check). None means the full set. The task1 windows left in the comments are kept here too.
DEFAULT_WINDOWS = {
    "binary": [None, (32, None), (31, 57)],      # the notebook's task2 windows
    "regression": [None, (5, 45), (10, 55)],
}
WINDOWS_TASK1_FROM_COMMENTS = [None, (32, None), (25, 50)]


def prepared_csv_path(task: str, pca: bool, truncation: bool) -> Path:
    """Same filename as the notebook's f'data/{task}_PCA_{_pca}_truncate_{...}.csv'."""
    return DATA_DIR / f"{task}_PCA_{pca}_truncate_{truncation}.csv"
