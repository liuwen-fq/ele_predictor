#!/usr/bin/env python3
"""Render a readable conductivity parity/residual summary from prediction JSON.

The prediction result can be several gigabytes because it also stores molecular
explanation fields.  This renderer therefore parses the top-level JSON array a
record at a time and retains only the labelled conductivity target/prediction
pairs needed for the figure.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import re
import subprocess
from typing import Iterator

import matplotlib.pyplot as plt
import numpy as np


LOG_SHIFT = 4.6002
LOG_SCALE = 8.3344


def iter_json_array(path: Path, chunk_size: int = 2**20) -> Iterator[dict]:
    """Yield objects in a large JSON array without loading it all into memory."""
    decoder = json.JSONDecoder()
    buffer = ""

    with path.open() as handle:
        while True:
            start = 0
            while start < len(buffer) and buffer[start] in " \t\r\n,[":
                start += 1
            if start < len(buffer) and buffer[start] == "]":
                return

            try:
                record, end = decoder.raw_decode(buffer, start)
            except json.JSONDecodeError:
                chunk = handle.read(chunk_size)
                if not chunk:
                    if buffer.strip(" \t\r\n,[]"):
                        raise
                    return
                buffer = buffer[start:] + chunk
                continue

            if not isinstance(record, dict):
                raise ValueError("Expected prediction JSON to contain objects")
            yield record
            buffer = buffer[end:]


CONDUCTIVITY_VALUE = re.compile(r'^\s*"conductivity"\s*:\s*([-+0-9.eE]+)')
CONDUCTIVITY_PREDICTION = re.compile(r'^\s*"conductivity_pred"\s*:\s*([-+0-9.eE]+)')


def conductivity_arrays(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Extract labelled conductivity values, converting targets to mS/cm.

    ``rg`` filters the multi-gigabyte file to the three scalar fields of
    interest before Python sees it; this is much faster than deserializing the
    molecular explanation payload in every record.
    """
    targets: list[float] = []
    predictions: list[float] = []
    command = [
        "rg", "--no-heading", "-e", '"conductivity":',
        "-e", '"conductivity_mask": true',
        "-e", '"conductivity_pred":', str(path),
    ]
    current_target: float | None = None
    labelled = False
    with subprocess.Popen(command, stdout=subprocess.PIPE, text=True) as process:
        assert process.stdout is not None
        for line in process.stdout:
            if match := CONDUCTIVITY_VALUE.match(line):
                current_target = float(match.group(1))
                labelled = False
            elif '"conductivity_mask": true' in line:
                labelled = current_target is not None
            elif match := CONDUCTIVITY_PREDICTION.match(line):
                if labelled and current_target is not None:
                    targets.append(math.exp(current_target * LOG_SCALE - LOG_SHIFT))
                    predictions.append(float(match.group(1)))
                labelled = False
        if process.wait() != 0:
            raise RuntimeError(f"Could not scan prediction file: {' '.join(command)}")
    if not targets:
        raise ValueError(f"No labelled conductivity predictions found in {path}")
    return np.asarray(targets), np.asarray(predictions)


def render_figure(targets: np.ndarray, predictions: np.ndarray, output_path: Path) -> None:
    """Render parity and residual panels with a dedicated, readable metrics row."""
    residuals = predictions - targets
    mae = np.mean(np.abs(residuals))
    mse = np.mean(residuals**2)
    rmse = np.sqrt(mse)
    r2 = 1.0 - np.sum(residuals**2) / np.sum((targets - np.mean(targets))**2)

    plt.rcParams.update({
        "font.size": 11,
        "axes.titlesize": 14,
        "axes.labelsize": 12,
        "legend.fontsize": 10,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
    })
    fig = plt.figure(figsize=(12.2, 7.6), constrained_layout=True)
    grid = fig.add_gridspec(2, 2, height_ratios=(1.0, 0.20), wspace=0.045)
    parity_ax = fig.add_subplot(grid[0, 0])
    residual_ax = fig.add_subplot(grid[0, 1])
    table_ax = fig.add_subplot(grid[1, :])

    accent = "#E66101"
    residual_color = "#2C7FB8"
    all_values = np.concatenate((targets, predictions))
    lower = max(float(all_values.min()), 1e-3) / 1.10
    upper = float(all_values.max()) * 1.10

    parity_ax.scatter(
        targets, predictions, s=18, alpha=0.52, color=accent,
        edgecolors="none", rasterized=True, label="ele_predictor",
    )
    parity_ax.plot((lower, upper), (lower, upper), "--", color="#222222", lw=1.4,
                   label="Perfect prediction")
    parity_ax.set(xscale="log", yscale="log", xlim=(lower, upper), ylim=(lower, upper),
                  xlabel="Measured conductivity (mS/cm)",
                  ylabel="Predicted conductivity (mS/cm)",
                  title="Parity plot (log scale)")
    parity_ax.grid(True, which="major", color="#C8C8C8", linewidth=0.7, alpha=0.6)
    parity_ax.grid(True, which="minor", color="#E7E7E7", linewidth=0.5, alpha=0.45)
    parity_ax.legend(loc="upper left", frameon=True, framealpha=0.94, edgecolor="#DDDDDD")
    parity_ax.text(
        0.96, 0.05,
        f"N = {targets.size:,}\nR² = {r2:.3f}\nMAE = {mae:.3f} mS/cm\nRMSE = {rmse:.3f} mS/cm",
        transform=parity_ax.transAxes, ha="right", va="bottom", fontsize=10,
        bbox={"boxstyle": "round,pad=0.38", "facecolor": "white", "edgecolor": "#BBBBBB", "alpha": 0.94},
    )

    residual_ax.scatter(
        targets, residuals, s=18, alpha=0.48, color=residual_color,
        edgecolors="none", rasterized=True,
    )
    residual_ax.axhline(0.0, color="#222222", linestyle="--", linewidth=1.3)
    residual_ax.set(
        xlabel="Measured conductivity (mS/cm)", ylabel="Prediction residual (mS/cm)",
        title="Residuals",
    )
    residual_ax.grid(True, color="#D0D0D0", linewidth=0.7, alpha=0.6)
    residual_ax.text(
        0.03, 0.95, "Residual = prediction − measurement",
        transform=residual_ax.transAxes, ha="left", va="top", fontsize=10,
        bbox={"boxstyle": "round,pad=0.32", "facecolor": "white", "edgecolor": "none", "alpha": 0.88},
    )

    table_ax.axis("off")
    table = table_ax.table(
        cellText=[["ele_predictor", f"{targets.size:,}", f"{r2:.3f}",
                   f"{mae:.4f}", f"{mse:.4f}", f"{rmse:.4f}"]],
        colLabels=["Model", "N", "R²", "MAE (mS/cm)", "MSE", "RMSE (mS/cm)"],
        colWidths=[0.24, 0.10, 0.10, 0.18, 0.16, 0.18],
        cellLoc="center", loc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10.5)
    table.scale(1.0, 1.45)
    for (row, _column), cell in table.get_celld().items():
        cell.set_edgecolor("#D0D0D0")
        if row == 0:
            cell.set_facecolor("#EAF2F8")
            cell.set_text_props(weight="bold")
        else:
            cell.set_facecolor("#FAFAFA")

    fig.suptitle("Conductivity prediction performance", fontsize=16, fontweight="bold")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=260, facecolor="white")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--predictions",
        type=Path,
        default=Path("run/anion_ratio_property_fragment_explanation/results/ft_test_predictions.json"),
    )
    parser.add_argument("--output", type=Path, default=Path("output_clear.png"))
    parser.add_argument(
        "--cache",
        type=Path,
        default=Path("output_conductivity_points.npz"),
        help="Small cache containing only the plotted target/prediction arrays.",
    )
    parser.add_argument(
        "--extract-only",
        action="store_true",
        help="Create or refresh --cache without rendering the figure.",
    )
    args = parser.parse_args()
    if args.extract_only or not args.cache.exists():
        targets, predictions = conductivity_arrays(args.predictions)
        np.savez_compressed(args.cache, targets=targets, predictions=predictions)
        print(f"Saved {args.cache} ({targets.size:,} labelled conductivity samples)")
    else:
        with np.load(args.cache) as cache:
            targets = cache["targets"]
            predictions = cache["predictions"]
    if args.extract_only:
        return
    render_figure(targets, predictions, args.output)
    print(f"Saved {args.output} ({targets.size:,} labelled conductivity samples)")


if __name__ == "__main__":
    main()
