#!/usr/bin/env python3
"""PCV / ASHS-LIA 예후 분석 CLI.

PCV_main_copy.ipynb 를 옮긴 것이다. 노트북의 실행 순서가 그대로 서브커맨드가 된다:

    python main.py prepare --task task8              # 엑셀 -> 표준화(+PCA) -> CSV
    python main.py run     --task task8              # CSV -> AdaBoost -> MDI 플롯
    python main.py all     --task task8              # 위 둘을 이어서

분류/회귀는 --task 로 정해진다 (task1·task2 = binary, task8 = regression).
"""
from __future__ import annotations

import argparse
import sys

from src import config, data, models, plots


def parse_window(text: str):
    """'32:' '31:57' ':50' 'full' 을 (lo, hi) 또는 None 으로."""
    if text.lower() in ("full", "all", "none"):
        return None
    if ":" not in text:
        raise argparse.ArgumentTypeError(f"구간 형식이 아니다: {text!r} (예: 31:57, 32:, full)")
    lo, _, hi = text.partition(":")
    return (int(lo) if lo else None, int(hi) if hi else None)


def cmd_prepare(args) -> int:
    data.prepare(task=args.task, use_pca=not args.no_pca, truncate=args.pca_truncation,
                 excel_path=args.excel, out_path=args.csv, verbose=not args.quiet)
    return 0


def cmd_run(args) -> int:
    kind = config.TASK_KIND[args.task]
    csv_path = args.csv or config.prepared_csv_path(
        args.task, not args.no_pca, args.pca_truncation)
    y_train, X_train = data.load_prepared(csv_path, seed=args.seed)
    print(f"{csv_path} · n={len(y_train)} · features={len(X_train.columns)} · kind={kind}")

    windows = args.windows if args.windows else None
    results = models.run_windows(X_train, y_train, kind, windows=windows, seed=args.seed)

    if args.no_plot:
        return 0
    if args.save:
        plots.use_headless_backend()
    for i, r in enumerate(results):
        out = None
        if args.save:
            tag = "full" if r.window is None else f"{r.window[0]}-{r.window[1]}"
            out = config.FIG_DIR / f"{args.task}_{kind}_{i}_{tag}.png"
        plots.plot_mdi(r, kind, out_path=out, show=args.show)
    return 0


def cmd_all(args) -> int:
    rc = cmd_prepare(args)
    return rc or cmd_run(args)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    def common(sp):
        sp.add_argument("--task", default="task8", choices=sorted(config.TASKS),
                        help="기본 task8 (time_to_rem 회귀)")
        sp.add_argument("--no-pca", action="store_true", help="PCA 피처 선택을 건너뛴다")
        sp.add_argument("--pca-truncation", action="store_true",
                        help="⚠ 원본에서는 무동작이다 — config.PCA_TRUNCATION_SENTINEL 주석 참고")
        sp.add_argument("--csv", default=None, help="학습용 CSV 경로 (기본: data/<task>_PCA_...csv)")
        sp.add_argument("--seed", type=int, default=config.SEED)
        sp.add_argument("--quiet", action="store_true")

    sp = sub.add_parser("prepare", help="엑셀 -> 표준화(+PCA) -> CSV")
    common(sp)
    sp.add_argument("--excel", default=config.EXCEL_PATH)
    sp.set_defaults(func=cmd_prepare)

    sp = sub.add_parser("run", help="CSV -> AdaBoost -> MDI 플롯")
    common(sp)
    sp.add_argument("--windows", type=parse_window, nargs="*", default=None,
                    metavar="LO:HI", help="행 구간들. 기본은 노트북의 세 구간")
    sp.add_argument("--save", action="store_true", help="figures/ 에 PNG 로 저장")
    sp.add_argument("--show", action="store_true", help="창으로 띄운다")
    sp.add_argument("--no-plot", action="store_true")
    sp.set_defaults(func=cmd_run)

    sp = sub.add_parser("all", help="prepare + run")
    common(sp)
    sp.add_argument("--excel", default=config.EXCEL_PATH)
    sp.add_argument("--windows", type=parse_window, nargs="*", default=None, metavar="LO:HI")
    sp.add_argument("--save", action="store_true")
    sp.add_argument("--show", action="store_true")
    sp.add_argument("--no-plot", action="store_true")
    sp.set_defaults(func=cmd_all)
    return p


if __name__ == "__main__":
    args = build_parser().parse_args()
    sys.exit(args.func(args))
