#!/usr/bin/env python3
"""Python replica of the original MATLAB demo script.

The script runs the selected test pipeline, loads the generated
prediction/ground truth series, computes simple metrics, and
(optionally) plots the forecast curve.
"""
from __future__ import annotations

import argparse
import math
import subprocess
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Tuple


AVAILABLE_MODES: Dict[str, str] = {
    "SQUARE-Mamba": "SQUARE_Mamba",
    "SQUARE_Mamba_Not_Quantum": "SQUARE_Mamba_Not_Quantum",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run SQUARE-Mamba forecast demo without MATLAB.",
    )
    parser.add_argument(
        "--mode",
        choices=AVAILABLE_MODES.keys(),
        default="SQUARE-Mamba",
        help="Select which pretrained model to evaluate.",
    )
    parser.add_argument(
        "--skip-test",
        action="store_true",
        help="Reuse existing result files instead of re-running the test script.",
    )
    parser.add_argument(
        "--no-plot",
        action="store_true",
        help="Skip plotting (useful when matplotlib is unavailable).",
    )
    parser.add_argument(
        "--save-plot",
        type=Path,
        help="Write the plot to this file (even if --no-plot is passed).",
    )
    return parser.parse_args()


def run_test_script(base_path: Path, mode_key: str) -> None:
    script = base_path / "main" / f"test_{mode_key}.py"
    if not script.exists():
        raise FileNotFoundError(f"Missing test script: {script}")

    print(f"[demo] Running {script.name} ...")
    subprocess.run(
        [sys.executable, str(script)],
        cwd=base_path,
        check=True,
    )


def load_series(base_path: Path, mode_key: str) -> Tuple[List[float], List[float]]:
    result_dir = base_path / "main" / "Result" / mode_key
    gt_path = result_dir / "gt_Pooncarie.csv"
    pred_path = result_dir / "prediction_Pooncarie.csv"

    print("hola")
    print(pred_path)
    if not gt_path.exists() or not pred_path.exists():
        raise FileNotFoundError(
            "Result files not found. Run the test script first or use --skip-test only when files already exist."
        )

    def read_column(path: Path) -> List[float]:
        with path.open("r", encoding="utf-8") as handle:
            return [float(line.strip()) for line in handle if line.strip()]

    gt = read_column(gt_path)
    prediction = read_column(pred_path)
    return gt, prediction


def compute_metrics(gt: Iterable[float], prediction: Iterable[float]) -> Dict[str, float]:
    gt_list = list(gt)
    pred_list = list(prediction)
    if len(gt_list) != len(pred_list):
        raise ValueError("Ground truth and prediction must have the same length.")

    n = len(gt_list)
    diffs = [g - p for g, p in zip(gt_list, pred_list)]
    mae = sum(abs(d) for d in diffs) / n
    rmse = math.sqrt(sum(d * d for d in diffs) / n)
    ss_res = sum(d * d for d in diffs)
    mean_gt = sum(gt_list) / n
    ss_tot = sum((g - mean_gt) ** 2 for g in gt_list)
    r2 = 1.0 - ss_res / ss_tot if ss_tot != 0 else float("nan")

    return {"MAE": mae, "RMSE": rmse, "R2": r2}


def plot_curves(
    gt: List[float],
    prediction: List[float],
    metrics: Dict[str, float],
    mode_label: str,
    *,
    save_path: Path | None = None,
    show: bool = True,
) -> None:
    try:
        import matplotlib
        if save_path is not None or not show:
            matplotlib.use("Agg", force=True)
        import matplotlib.pyplot as plt
    except ImportError as exc:  # pragma: no cover - purely defensive
        raise SystemExit(
            "matplotlib is required for plotting. Install it with `pip install matplotlib` or re-run with --no-plot."
        ) from exc

    years = [
        "2007-12",
        "2008-12",
        "2009-12",
        "2010-12",
        "2011-12",
        "2012-12",
        "2013-12",
        "2014-12",
        "2015-12",
        "2016-12",
        "2017-12",
        "2018-12",
        "2019-12",
        "2020-12",
        "2021-12",
        "2022-12",
        "2023-12",
    ]

    x = list(range(1, len(gt) + 1))
    color = (255 / 255, 105 / 255, 0 / 255)

    fig, ax = plt.subplots(figsize=(18.7, 4))
    ax.plot(x, gt, color="black", linewidth=1.5, label="Observed")
    ax.plot(
        x,
        prediction,
        color=color,
        linewidth=2.3,
        linestyle="--",
        marker="o",
        markersize=4,
        markerfacecolor=color,
        markeredgecolor=color,
        markevery=1,
        label=mode_label,
    )

    xticks = list(range(1, len(gt) + 1, 12))
    xtick_labels = years[: len(xticks)]

    ax.set_xlim(1, len(gt))
    ax.set_ylim(-3, 3)
    ax.set_xticks(xticks)
    ax.set_xticklabels(xtick_labels, rotation=0)
    ax.set_ylabel("Drought Forecasting", fontname="DejaVu Serif", fontsize=14)
    ax.set_title(mode_label.replace("_", " "), fontname="DejaVu Serif", fontsize=18)
    ax.grid(True, color=(0, 0, 0), alpha=0.3)
    ax.set_facecolor("white")
    ax.tick_params(labelsize=14)
    ax.legend(loc="upper left", fontsize=11, frameon=True)

    text_x = max(1, min(len(gt) - 5, int(len(gt) * 0.89)))
    text_y = 2.5
    ax.text(text_x, text_y, f"MAE = {metrics['MAE']:.4f}", fontsize=16, fontname="DejaVu Serif")
    ax.text(text_x, text_y - 0.8, f"RMSE = {metrics['RMSE']:.4f}", fontsize=16, fontname="DejaVu Serif")
    ax.text(text_x, text_y - 1.6, f"R² = {metrics['R2']:.4f}", fontsize=16, fontname="DejaVu Serif")

    fig.tight_layout()

    if save_path is not None:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=200)
        print(f"[demo] Plot saved to {save_path}")

    if show:
        plt.show()
    else:
        plt.close(fig)


def main() -> None:
    args = parse_args()
    mode_label = args.mode
    mode_key = AVAILABLE_MODES[mode_label]

    base_path = Path(__file__).resolve().parent

    if not args.skip_test:
        run_test_script(base_path, mode_key)
    else:
        print("[demo] Skipping test script execution as requested.")

    gt, prediction = load_series(base_path, mode_key)
    metrics = compute_metrics(gt, prediction)

    print("[demo] Metrics:")
    for name, value in metrics.items():
        print(f"  {name}: {value:.4f}")

    should_plot = not args.no_plot or args.save_plot is not None
    if should_plot:
        plot_curves(
            gt,
            prediction,
            metrics,
            mode_key,
            save_path=args.save_plot,
            show=not args.no_plot,
        )
    else:
        print("[demo] Plot skipped.")


if __name__ == "__main__":
    main()
