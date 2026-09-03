"""데이터 경로 · 태스크 정의 · 시드 · 플롯 상수.

노트북 상단 셀에 흩어져 있던 설정을 한곳에 모은 것이다.
숫자를 바꾸면 논문 수치가 바뀐다 — 기본값은 노트북과 동일하게 유지할 것.
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
FIG_DIR = ROOT / "figures"

EXCEL_PATH = DATA_DIR / "new_hypoF_PCV 220919.xlsx"
SHEET_NAME = "cleaned_0"

SEED = 2022

# 논문 표기에 맞춘 컬럼명. 이 rename 이 PCA_TRUNCATION_SENTINEL 과 충돌한다 (아래 주석 참고).
COLUMN_RENAMES = {
    "HypoF": "ASHS-LIA",
    "basline_logMAR": "baseline_logMAR",
    "largest_polyp": "Largest polyp diameter",
    "number_of_polyps": "Polyp number",
    "lipid": "Lipid exudation",
    "CNV_location": "Polyp location",
    "SRHeme": "SRH",
}

# task -> (target, 함께 버릴 피처들)
TASKS = {
    "task1": ("group", ["injection_demand", "vision_gain", "rec_period"]),
    "task2": ("injection_demand", ["total_inj_n", "group", "vision_gain", "rec_period"]),
    "task8": ("time_to_rem",
              ["group", "rec_period", "total_inj_n", "vision_gain", "injection_demand", "recur"]),
}

# 노트북에 주석 처리돼 있던 태스크들. 되살리려면 TASKS 로 옮길 것.
TASKS_DISABLED = {
    "task3": ("total_inj_n", ["injection_demand", "group", "vision_gain", "rec_period"]),
    "task4": ("rec_period", ["group", "vision_gain", "injection_demand", "recur"]),
    "task5": ("recur", ["group", "vision_gain", "injection_demand", "rec_period"]),
    "task6": ("vision_gain", ["group", "injection_demand", "rec_period"]),
}

TASK_KIND = {"task1": "binary", "task2": "binary", "task8": "regression"}

# task1 의 target 은 1/2 로 들어와 있어 0/1 로 내린다 (노트북 `df_y = df_y - 1`).
SHIFT_TARGET_BY_ONE = {"task1"}

# ⚠ 노트북 원본은 truncation 기준을 "HypoF" 로 두는데, load 직후 "ASHS-LIA" 로 rename 되므로
#   이 조건은 절대 참이 되지 않는다 — 즉 --pca-truncation 은 원본에서 사실상 무동작이었다.
#   재현을 위해 기본값을 그대로 두되, 의도대로 쓰려면 "ASHS-LIA" 로 바꿀 것.
PCA_TRUNCATION_SENTINEL = "HypoF"

# MDI 표에서 제외할 인덱스. 분류/회귀가 서로 다르다 (노트북 그대로).
DROP_FROM_MDI = {
    "binary": ["recur", "fu_period", "time_to_rem"],
    "regression": ["recur", "fu_period"],
}

# 제목의 (Number of Features = ...) 에 더하는 상수. 분류 +2, 회귀 +1 (노트북 그대로).
FEATURE_COUNT_OFFSET = {"binary": 2, "regression": 1}

HIGHLIGHT_FEATURE = "ASHS-LIA"
COLOR_HIGHLIGHT = "#e01e5a"
COLOR_DEFAULT = "#236AA7"

# 노트북의 세 셀이 서로 다른 행 구간으로 같은 모델을 반복 학습했다 (민감도 확인).
# None = 전체. 주석에 남아 있던 task1 용 구간도 함께 옮겨 둔다.
DEFAULT_WINDOWS = {
    "binary": [None, (32, None), (31, 57)],      # 노트북 task2 기준
    "regression": [None, (5, 45), (10, 55)],
}
WINDOWS_TASK1_FROM_COMMENTS = [None, (32, None), (25, 50)]


def prepared_csv_path(task: str, pca: bool, truncation: bool) -> Path:
    """노트북의 f'data/{task}_PCA_{_pca}_truncate_{pca_truncation}.csv' 와 동일한 이름."""
    return DATA_DIR / f"{task}_PCA_{pca}_truncate_{truncation}.csv"
