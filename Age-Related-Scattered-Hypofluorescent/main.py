#!/usr/bin/env python3
"""PCV / ASHS-LIA prognosis analysis CLI.

A port of PCV_main_copy.ipynb. The notebook's execution order becomes subcommands:

    python main.py prepare --task task8              # excel -> standardize (+PCA) -> CSV
    python main.py run     --task task8              # CSV -> AdaBoost -> MDI plot
    python main.py all     --task task8              # both, in sequence

--task decides classification vs regression (task1 and task2 are binary, task8 is
a regression).
"""
from __future__ import annotations

import argparse
import sys

from src import config, data, models, plots


def parse_window(text: str):
    """Turn '32:', '31:57', ':50' or 'full' into (lo, hi) or None."""
    if text.lower() in ("full", "all", "none"):
        return None
    if ":" not in text:
        raise argparse.ArgumentTypeError(
            f"not a window: {text!r} (e.g. 31:57, 32:, full)")
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
    print(f"{csv_path} - n={len(y_train)} - features={len(X_train.columns)} - kind={kind}")

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
                        help="defaults to task8 (time_to_rem regression)")
        sp.add_argument("--no-pca", action="store_true", help="skip PCA feature selection")
        sp.add_argument("--pca-truncation", action="store_true",
                        help="NOTE: a no-op in the original; see config.PCA_TRUNCATION_SENTINEL")
        sp.add_argument("--csv", default=None,
                        help="training CSV path (default: data/<task>_PCA_...csv)")
        sp.add_argument("--seed", type=int, default=config.SEED)
        sp.add_argument("--quiet", action="store_true")

    sp = sub.add_parser("prepare", help="excel -> standardize (+PCA) -> CSV")
    common(sp)
    sp.add_argument("--excel", default=config.EXCEL_PATH)
    sp.set_defaults(func=cmd_prepare)

    sp = sub.add_parser("run", help="CSV -> AdaBoost -> MDI plot")
    common(sp)
    sp.add_argument("--windows", type=parse_window, nargs="*", default=None,
                    metavar="LO:HI", help="row windows; defaults to the notebook's three")
    sp.add_argument("--save", action="store_true", help="write PNGs to figures/")
    sp.add_argument("--show", action="store_true", help="open a window")
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
