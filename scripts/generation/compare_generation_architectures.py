#!/usr/bin/env python3
"""Compare released Bamboo and three current conditional-generation variants.

This is a cross-experiment-root summary, deliberately separate from
``aggregate_generation_replicates.py``.  It treats seed as the unit of
replication, plots each seed as a dot, and never pools generated formulas as
if they were independent experimental repeats.

Default arms at the shared target (sigma=20 mS/cm, r=0.5):
  * released Bamboo (G0),
  * fixed-Bamboo vocabulary with mechanism auxiliary supervision (G0-MA),
  * random vocabulary with factual mechanism auxiliary supervision (G1-MA),
  * explain-guided vocabulary with the same supervision (G2-MA).
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap, PowerNorm, TwoSlopeNorm
from matplotlib.patches import Rectangle


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_COMPARISON_ROOT = ROOT / "run/generation_mechanism_auxiliary_comparison"
BLUE_TO_RED = LinearSegmentedColormap.from_list(
    "density_blue_to_red",
    [(0.00, "#dce8f3"), (0.18, "#a9c1d8"), (0.30, "#5e8bb5"),
     (0.42, "#d6a7ad"), (0.68, "#b2182b"), (1.00, "#b2182b")], N=256)
BLUE_TO_RED.set_bad("white")
DIFFERENCE_CMAP = LinearSegmentedColormap.from_list(
    "density_difference", ["#2166ac", "#f7f7f7", "#b2182b"], N=256)


@dataclass(frozen=True)
class Arm:
    key: str
    label: str
    group: str
    root: Path
    color: str


ARM_SPECS = (
    # Muted Nature/Set2-style palette: readable in print and less saturated.
    ("Bamboo", "Released Bamboo\n(G0)", "G0", "bamboo-comp", "#8DA0CB"),
    ("G0-MA", "Fixed Bamboo vocab + MA\n(G0-MA)", "G0-MA", "G0-Q-comp", "#66C2A5"),
    ("G1-MA", "Random vocab + MA\n(G1-MA)", "G1-MA", "G1-comp", "#FC8D62"),
    ("G2-MA", "Soft-guided vocab + MA\n(G2-MA)", "G2-MA", "G2-comp", "#C2A5CF"),
)

ERROR_METRICS = (
    ("σ MAE", "kappa_mae_mS_cm", "mS/cm", 1.0),
    ("σ RMSE", "kappa_rmse_mS_cm", "mS/cm", 1.0),
    ("r MAE", "anion_ratio_pred_mae_from_condition", "ratio", 1.0),
    ("r RMSE", "anion_ratio_pred_rmse_from_condition", "ratio", 1.0),
)
JOINT_AND_QUALITY_METRICS = (
    ("Joint normalized MAE", "joint_normalized_mae", "normalized error", 1.0),
    ("Joint normalized RMSE", "joint_normalized_rmse", "normalized error", 1.0),
    ("Trusted formula rate", "trusted_formula_rate", "%", 100.0),
    ("Strict joint-window coverage", "joint_within_kappa_pm2_ratio_pm0.1", "%", 100.0),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--comparison-root", type=Path, default=DEFAULT_COMPARISON_ROOT,
        help="Directory containing bamboo-comp, G0-Q-comp, G1-comp and G2-comp.")
    parser.add_argument(
        "--output-dir", type=Path, default=None,
        help="Defaults to <comparison-root>/four_group_architecture_comparison.")
    parser.add_argument("--target-conductivity", type=float, default=20.0)
    parser.add_argument("--anion-ratio", type=float, default=0.5)
    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def parse_seed(directory: Path) -> str:
    return directory.name.rsplit("_", 1)[-1]


def numeric(row: dict[str, str], key: str) -> float:
    value = row.get(key, "")
    if value in (None, ""):
        raise ValueError(f"missing {key!r}")
    return float(value)


def make_arms(comparison_root: Path) -> tuple[Arm, ...]:
    return tuple(
        Arm(key, label, group, comparison_root / relative_root, color)
        for key, label, group, relative_root, color in ARM_SPECS
    )


def load_arm(arm: Arm, target: float, ratio: float) -> list[dict[str, float | str]]:
    summaries = sorted(arm.root.glob("*_seed_*/generation_experiment_summary.csv"),
                       key=lambda path: int(parse_seed(path.parent)))
    if len(summaries) < 2:
        raise FileNotFoundError(f"{arm.key}: expected at least two seed summaries below {arm.root}")

    records: list[dict[str, float | str]] = []
    for summary_path in summaries:
        rows = read_csv(summary_path)
        matches = [row for row in rows if row.get("group") == arm.group]
        if len(matches) != 1:
            raise ValueError(f"{summary_path}: expected exactly one {arm.group} row, found {len(matches)}")
        row = matches[0]
        if not math.isclose(numeric(row, "target_mS_cm"), target, abs_tol=1e-8):
            raise ValueError(f"{summary_path}: target conductivity does not match {target}")
        if not math.isclose(numeric(row, "anion_ratio_condition"), ratio, abs_tol=1e-8):
            raise ValueError(f"{summary_path}: anion-ratio condition does not match {ratio}")
        record: dict[str, float | str] = {
            "seed": parse_seed(summary_path.parent),
            "arm": arm.key,
            "label": arm.label,
            "group": arm.group,
            "n_sampled": numeric(row, "n_sampled"),
            "n_trusted": numeric(row, "n_trusted"),
        }
        record["trusted_formula_rate"] = 0.0 if record["n_sampled"] == 0 else (
            float(record["n_trusted"]) / float(record["n_sampled"]))
        for _, key, _, _ in ERROR_METRICS + JOINT_AND_QUALITY_METRICS:
            if key != "trusted_formula_rate":
                record[key] = numeric(row, key)
        records.append(record)
    return records


def write_csv(path: Path, rows: Iterable[dict[str, object]], fields: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def summary_rows(records: list[dict[str, float | str]], arms: tuple[Arm, ...]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    metric_specs = ERROR_METRICS + JOINT_AND_QUALITY_METRICS
    for arm in arms:
        arm_records = [row for row in records if row["arm"] == arm.key]
        for metric_name, metric_key, unit, multiplier in metric_specs:
            values = np.array([float(row[metric_key]) * multiplier for row in arm_records])
            rows.append({
                "arm": arm.key,
                "label": arm.label.replace("\n", " "),
                "group": arm.group,
                "metric": metric_name,
                "metric_key": metric_key,
                "unit": unit,
                "n_seeds": len(values),
                "mean": values.mean(),
                "std": values.std(ddof=1) if len(values) > 1 else math.nan,
                "min": values.min(),
                "max": values.max(),
            })
    return rows


def plot_dashboard(
    records: list[dict[str, float | str]], arms: tuple[Arm, ...], metrics: tuple,
    output: Path, title: str,
) -> None:
    figure, axes = plt.subplots(2, 2, figsize=(16.0, 10.8), constrained_layout=True)
    figure.set_facecolor("white")
    x = np.arange(len(arms))
    for axis, (metric_name, metric_key, unit, multiplier) in zip(axes.flat, metrics):
        means, stds = [], []
        for arm in arms:
            values = np.array([
                float(row[metric_key]) * multiplier
                for row in records if row["arm"] == arm.key
            ])
            means.append(values.mean())
            stds.append(values.std(ddof=1) if len(values) > 1 else 0.0)
        bars = axis.bar(
            x, means, yerr=stds, capsize=7, width=0.66,
            color=[arm.color for arm in arms], edgecolor="white", linewidth=1.4,
            error_kw={"elinewidth": 1.7, "ecolor": "#1f2937", "capthick": 1.7},
            zorder=2,
        )
        for index, arm in enumerate(arms):
            values = np.array([
                float(row[metric_key]) * multiplier
                for row in records if row["arm"] == arm.key
            ])
            offsets = np.linspace(-0.11, 0.11, len(values))
            axis.scatter(np.full(len(values), index) + offsets, values, s=55,
                         color="#111827", edgecolors="white", linewidth=.9, zorder=4)
            axis.text(index, means[index] + stds[index] + max(max(means), 1e-6) * .035,
                      f"{means[index]:.3f}\n± {stds[index]:.3f}",
                      ha="center", va="bottom", fontsize=9.5, color="#111827",
                      fontweight="semibold", zorder=5)
        axis.set_title(metric_name, fontsize=14, fontweight="bold", pad=10)
        axis.set_ylabel(unit, fontsize=11)
        axis.set_xticks(x, [arm.label for arm in arms], fontsize=9.5)
        axis.grid(axis="y", color="#cbd5e1", linewidth=.8, alpha=.55, zorder=0)
        axis.spines[["top", "right"]].set_visible(False)
        axis.spines[["left", "bottom"]].set_color("#94a3b8")
        axis.tick_params(axis="x", length=0, pad=8)
        axis.margins(y=.20)
    figure.suptitle(title, fontsize=18, fontweight="bold", color="#0f172a")
    figure.savefig(output, dpi=240, bbox_inches="tight", facecolor=figure.get_facecolor())
    plt.close(figure)


def load_trusted_predictions(arm: Arm) -> list[dict[str, float | str]]:
    """Load raw trusted predictions for visualization only.

    Formula-level values are intentionally not used for error bars or inferential
    statistics.  They are pooled here solely to display each generator's output
    distribution across its three completed sampling runs.
    """
    rows: list[dict[str, float | str]] = []
    generated_paths = sorted(
        arm.root.glob(f"*_seed_*/{arm.group}/generated/generated_formulas.json"),
        key=lambda path: int(parse_seed(path.parents[2])),
    )
    if len(generated_paths) < 2:
        raise FileNotFoundError(f"{arm.key}: no generated-formula files found below {arm.root}")
    for path in generated_paths:
        seed = parse_seed(path.parents[2])
        with path.open(encoding="utf-8") as handle:
            formulas = json.load(handle)
        if not isinstance(formulas, list):
            raise ValueError(f"{path}: expected a JSON list of generated formulas")
        for formula in formulas:
            if formula.get("prediction_status") != "trusted":
                continue
            sigma = formula.get("conductivity_pred_ms_cm")
            ratio = formula.get("anion_ratio_pred")
            if sigma is None or ratio is None:
                continue
            sigma, ratio = float(sigma), float(ratio)
            if math.isfinite(sigma) and math.isfinite(ratio):
                rows.append({"arm": arm.key, "seed": seed, "sigma": sigma, "ratio": ratio})
    if not rows:
        raise ValueError(f"{arm.key}: no trusted finite predictions in {generated_paths}")
    return rows


def target_window_composition_diversity(
    arms: tuple[Arm, ...], target: float, ratio_target: float,
) -> tuple[list[dict[str, float | str]], dict[str, list[np.ndarray]]]:
    """Measure mixture diversity within the strict joint target window.

    The two arms share the Bamboo solvent vocabulary, so these statistics test
    diversity of solvent *mixtures and proportions*, not molecular-scaffold
    novelty.  Each seed remains a separate replicate.
    """

    selected_arms = [arm for arm in arms if arm.key in {"Bamboo", "G0-MA"}]
    with (ROOT / "emb_dict/solvents.json").open(encoding="utf-8") as handle:
        solvent_count = len(json.load(handle))
    rows: list[dict[str, float | str]] = []
    similarities: dict[str, list[np.ndarray]] = {arm.key: [] for arm in selected_arms}
    for arm in selected_arms:
        generated_paths = sorted(
            arm.root.glob(f"*_seed_*/{arm.group}/generated/generated_formulas.json"),
            key=lambda path: int(parse_seed(path.parents[2])),
        )
        for path in generated_paths:
            seed = parse_seed(path.parents[2])
            with path.open(encoding="utf-8") as handle:
                formulas = json.load(handle)
            bows = []
            for formula in formulas:
                if formula.get("prediction_status") != "trusted":
                    continue
                sigma = formula.get("conductivity_pred_ms_cm")
                ratio = formula.get("anion_ratio_pred")
                if sigma is None or ratio is None:
                    continue
                sigma, ratio = float(sigma), float(ratio)
                if (math.isfinite(sigma) and math.isfinite(ratio) and
                        abs(sigma - target) <= 2.0 and
                        abs(ratio - ratio_target) <= .10):
                    bows.append(np.asarray(formula["bow"][:solvent_count], dtype=float))
            if len(bows) < 2:
                raise ValueError(f"{arm.key} seed {seed}: fewer than two target-window formulas")
            matrix = np.asarray(bows)
            active = matrix > .02
            active_set_count = len({tuple(np.flatnonzero(item)) for item in active})
            rounded_composition_count = len({
                tuple(np.rint(item / .02).astype(int)) for item in matrix
            })
            dominant = np.argmax(matrix, axis=1)
            dominant_counts = np.asarray(list(Counter(dominant).values()), dtype=float)
            dominant_probabilities = dominant_counts / dominant_counts.sum()
            effective_dominant_solvents = float(np.exp(
                -np.sum(dominant_probabilities * np.log(dominant_probabilities))))
            unit = matrix / np.maximum(np.linalg.norm(matrix, axis=1, keepdims=True), 1e-12)
            cosine = unit @ unit.T
            pair_similarity = cosine[np.triu_indices(len(cosine), k=1)]
            np.fill_diagonal(cosine, -np.inf)
            nearest_similarity = cosine.max(axis=1)
            similarities[arm.key].append(pair_similarity)
            rows.append({
                "arm": arm.key,
                "seed": seed,
                "target_window_formula_count": len(matrix),
                "unique_active_solvent_sets": active_set_count,
                "active_set_unique_rate": active_set_count / len(matrix),
                "unique_compositions_rounded_2pct": rounded_composition_count,
                "rounded_composition_unique_rate": rounded_composition_count / len(matrix),
                "mean_active_solvents": float(active.sum(axis=1).mean()),
                "effective_dominant_solvents": effective_dominant_solvents,
                "top_dominant_solvent_share": float(dominant_probabilities.max()),
                "median_pairwise_cosine_similarity": float(np.median(pair_similarity)),
                "mean_nearest_neighbor_cosine_similarity": float(nearest_similarity.mean()),
                "nearest_neighbor_cosine_ge_0_99_rate": float(
                    np.mean(nearest_similarity >= .99)),
                "pairwise_cosine_ge_0_90_rate": float(np.mean(pair_similarity >= .90)),
            })
    return rows, similarities


def plot_target_window_composition_diversity(
    rows: list[dict[str, float | str]], similarities: dict[str, list[np.ndarray]],
    output: Path,
) -> None:
    """Plot target coverage beside mixture-composition diversity diagnostics."""

    bamboo_rows = [row for row in rows if row["arm"] == "Bamboo"]
    ma_rows = [row for row in rows if row["arm"] == "G0-MA"]
    if not bamboo_rows or not ma_rows:
        raise ValueError("Bamboo and G0-MA diversity rows are required")
    metrics = (
        ("target_window_formula_count", "target-window\nformulas", "higher"),
        ("mean_active_solvents", "mean active\nsolvents", "higher"),
        ("effective_dominant_solvents", "effective dominant\nsolvents", "higher"),
        ("active_set_unique_rate", "unique active-set\nrate", "higher"),
    )
    bamboo_means = np.asarray([
        np.mean([float(row[key]) for row in bamboo_rows]) for key, _, _ in metrics
    ])
    ma_means = np.asarray([
        np.mean([float(row[key]) for row in ma_rows]) for key, _, _ in metrics
    ])
    ma_stds = np.asarray([
        np.std([float(row[key]) for row in ma_rows], ddof=1) for key, _, _ in metrics
    ])

    figure, axes = plt.subplots(1, 2, figsize=(13.0, 5.3), constrained_layout=True)
    figure.set_facecolor("white")
    axis = axes[0]
    indices = np.arange(len(metrics))
    ratios = ma_means / bamboo_means
    ratio_stds = ma_stds / bamboo_means
    axis.bar(indices, ratios, color="#2a9d8f", width=.62, zorder=2)
    axis.errorbar(indices, ratios, yerr=ratio_stds, fmt="none", color="#1f2937",
                  capsize=4, linewidth=1.2, zorder=3)
    axis.axhline(1.0, color="#64748b", linewidth=1.0, linestyle="--")
    axis.set_xticks(indices, [label for _, label, _ in metrics], fontsize=9)
    axis.set_ylabel("G0-MA / released Bamboo")
    axis.set_title("Target-window coverage and mixture diversity", fontweight="bold")
    axis.grid(axis="y", alpha=.22, zorder=0)
    axis.spines[["top", "right"]].set_visible(False)
    for index, (ratio_value, bamboo_value, ma_value) in enumerate(
            zip(ratios, bamboo_means, ma_means)):
        axis.text(index, ratio_value + ratio_stds[index] + .035,
                  f"{ma_value:.2f} vs {bamboo_value:.2f}",
                  ha="center", va="bottom", fontsize=8.5, color="#0f172a")
    axis.text(.02, .96, "Dashed line = no change", transform=axis.transAxes,
              ha="left", va="top", fontsize=8.5, color="#475569")

    axis = axes[1]
    grid = np.linspace(0.0, 1.0, 251)
    colors = {"Bamboo": "#8DA0CB", "G0-MA": "#2a9d8f"}
    labels = {"Bamboo": "Released Bamboo", "G0-MA": "G0-MA"}
    for arm_key in ("Bamboo", "G0-MA"):
        curves = np.asarray([
            [np.mean(values <= value) for value in grid]
            for values in similarities[arm_key]
        ])
        mean_curve = curves.mean(axis=0)
        axis.plot(grid, mean_curve, color=colors[arm_key], linewidth=2.3,
                  label=labels[arm_key])
        axis.fill_between(grid, curves.min(axis=0), curves.max(axis=0),
                          color=colors[arm_key], alpha=.14, linewidth=0)
    axis.axvline(.90, color="#64748b", linestyle=":", linewidth=1.0)
    axis.set_xlim(0.0, 1.0)
    axis.set_ylim(0.0, 1.0)
    axis.set_xlabel("Pairwise cosine similarity of solvent-mixture BOW")
    axis.set_ylabel("Cumulative fraction of pairs")
    axis.set_title("Target-window mixture similarity", fontweight="bold")
    axis.grid(alpha=.22)
    axis.spines[["top", "right"]].set_visible(False)
    bamboo_median = np.mean([float(row["median_pairwise_cosine_similarity"])
                              for row in bamboo_rows])
    ma_median = np.mean([float(row["median_pairwise_cosine_similarity"])
                          for row in ma_rows])
    axis.text(.03, .06,
              f"median: {bamboo_median:.3f} → {ma_median:.3f}\n"
              "left shift = less similar mixtures",
              transform=axis.transAxes, fontsize=8.8, color="#0f172a",
              bbox={"facecolor": "white", "edgecolor": "#cbd5e1", "alpha": .9})
    axis.legend(frameon=False, loc="lower right")
    figure.savefig(output, dpi=240, bbox_inches="tight", facecolor=figure.get_facecolor())
    plt.close(figure)


def sigma_grid_limit(predictions: list[dict[str, float | str]], target: float) -> tuple[float, float]:
    values = np.array([float(row["sigma"]) for row in predictions])
    low = min(-1.0, float(np.quantile(values, .002)) - 1.0, target - 10.0)
    high = max(40.0, float(np.quantile(values, .998)) + 1.0, target + 10.0)
    return low, high


def plot_sigma_ratio_density(
    predictions: list[dict[str, float | str]], arms: tuple[Arm, ...], output: Path,
    target: float, ratio_target: float,
) -> None:
    """Draw comparable within-trusted-formula sigma/r density panels."""
    _, observed_sigma_high = sigma_grid_limit(predictions, target)
    # Use a fixed left bound and coarser bins, as in the per-experiment
    # density panels.  It makes the four distributions readable on a shared
    # scale without leaving an unnecessarily large empty range.
    sigma_low, sigma_high = 0.0, max(60.0, observed_sigma_high)
    sigma_edges = np.linspace(sigma_low, sigma_high, 31)
    ratio_edges = np.linspace(0.0, 1.0, 26)
    densities: dict[str, np.ndarray] = {}
    for arm in arms:
        arm_rows = [row for row in predictions if row["arm"] == arm.key]
        hist, _, _ = np.histogram2d(
            [float(row["sigma"]) for row in arm_rows],
            [float(row["ratio"]) for row in arm_rows], bins=(sigma_edges, ratio_edges),
        )
        densities[arm.key] = hist / hist.sum()
    vmax = max(float(density.max()) for density in densities.values())
    # One monotonic shared transform keeps cross-group density values
    # comparable while preventing G1's concentrated low-σ mode from washing
    # out the broader G0/G0-MA/G2-MA distributions.
    density_norm = PowerNorm(gamma=.55, vmin=0.0, vmax=vmax)
    figure, axes = plt.subplots(2, 2, figsize=(15.8, 11.4), sharex=True, sharey=True,
                                 constrained_layout=True)
    figure.set_facecolor("white")
    image = None
    for axis, arm in zip(axes.flat, arms):
        arm_rows = [row for row in predictions if row["arm"] == arm.key]
        density = np.ma.masked_less_equal(densities[arm.key].T, 0.0)
        image = axis.pcolormesh(
            sigma_edges, ratio_edges, density, shading="auto", cmap=BLUE_TO_RED,
            norm=density_norm,
        )
        axis.set_facecolor("white")
        axis.axvline(target, color="#dc2626", linestyle="--", linewidth=1.25, zorder=3)
        axis.axhline(ratio_target, color="white", linestyle=":", linewidth=1.35, zorder=3)
        axis.set_title(f"{arm.label.replace(chr(10), ' ')}\ntrusted formulas: {len(arm_rows):,}",
                       fontsize=12, fontweight="bold", pad=9)
        axis.set_xlim(sigma_low, sigma_high)
        axis.set_ylim(0.0, 1.0)
        axis.set_xlabel("predicted σ (mS/cm)")
        axis.set_ylabel("predicted anion ratio r")
        axis.grid(alpha=.18)
        axis.spines[["top", "right"]].set_visible(False)
    assert image is not None
    colorbar = figure.colorbar(image, ax=axes, shrink=.88, pad=.015)
    colorbar.set_label("within-group density (all trusted formulas per bin; power-scaled color)", fontsize=10)
    figure.suptitle(
        "Four-group generated-formula density: conductivity and anion ratio",
        fontsize=18, fontweight="bold", color="#0f172a",
    )
    figure.savefig(output, dpi=240, bbox_inches="tight", facecolor=figure.get_facecolor())
    plt.close(figure)


def seed_mean_density(
    predictions: list[dict[str, float | str]], arm_key: str,
    sigma_edges: np.ndarray, ratio_edges: np.ndarray,
) -> np.ndarray:
    """Return the mean of per-seed normalized sigma/r density maps."""
    seed_densities: list[np.ndarray] = []
    seeds = sorted({str(row["seed"]) for row in predictions if row["arm"] == arm_key}, key=int)
    if not seeds:
        raise ValueError(f"No trusted predictions found for {arm_key}")
    for seed in seeds:
        rows = [row for row in predictions if row["arm"] == arm_key and row["seed"] == seed]
        histogram, _, _ = np.histogram2d(
            [float(row["sigma"]) for row in rows],
            [float(row["ratio"]) for row in rows],
            bins=(sigma_edges, ratio_edges),
        )
        if histogram.sum() == 0:
            raise ValueError(f"No in-range trusted predictions found for {arm_key}, seed {seed}")
        seed_densities.append(histogram / histogram.sum())
    return np.mean(seed_densities, axis=0)


def plot_g0_ma_density_difference(
    predictions: list[dict[str, float | str]], output: Path, target: float, ratio_target: float,
) -> tuple[dict[str, float], list[dict[str, float]]]:
    """Plot seed-averaged Δp = p(G0-MA) - p(released Bamboo) over sigma/r bins."""
    _, observed_sigma_high = sigma_grid_limit(predictions, target)
    sigma_low, sigma_high = 0.0, max(60.0, observed_sigma_high)
    sigma_edges = np.linspace(sigma_low, sigma_high, 31)
    ratio_edges = np.linspace(0.0, 1.0, 26)
    bamboo_density = seed_mean_density(predictions, "Bamboo", sigma_edges, ratio_edges)
    ma_density = seed_mean_density(predictions, "G0-MA", sigma_edges, ratio_edges)
    difference = ma_density - bamboo_density

    # The selected edges align exactly with σ=20±2 and r=0.5±0.1 here.
    sigma_mask = ((sigma_edges[:-1] >= target - 2.0 - 1e-9) &
                  (sigma_edges[1:] <= target + 2.0 + 1e-9))
    ratio_mask = ((ratio_edges[:-1] >= ratio_target - .10 - 1e-9) &
                  (ratio_edges[1:] <= ratio_target + .10 + 1e-9))
    bamboo_window_pct = 100.0 * float(bamboo_density[np.ix_(sigma_mask, ratio_mask)].sum())
    ma_window_pct = 100.0 * float(ma_density[np.ix_(sigma_mask, ratio_mask)].sum())
    window_difference_pct = ma_window_pct - bamboo_window_pct

    difference_pct = 100.0 * difference.T
    limit = max(float(np.abs(difference_pct).max()), .01)
    figure, axis = plt.subplots(figsize=(9.6, 6.7), constrained_layout=True)
    figure.set_facecolor("white")
    image = axis.pcolormesh(
        sigma_edges, ratio_edges, difference_pct, shading="auto", cmap=DIFFERENCE_CMAP,
        norm=TwoSlopeNorm(vmin=-limit, vcenter=0.0, vmax=limit),
    )
    axis.axvline(target, color="#111827", linestyle="--", linewidth=1.25, zorder=3)
    axis.axhline(ratio_target, color="#111827", linestyle=":", linewidth=1.25, zorder=3)
    axis.add_patch(Rectangle(
        (target - 2.0, ratio_target - .10), 4.0, .20, fill=False,
        edgecolor="#111827", linestyle="--", linewidth=1.05, zorder=3,
    ))
    axis.set_xlim(sigma_low, sigma_high)
    axis.set_ylim(0.0, 1.0)
    axis.set_xlabel("predicted σ (mS/cm)")
    axis.set_ylabel("predicted anion ratio r")
    axis.grid(alpha=.18)
    axis.spines[["top", "right"]].set_visible(False)
    axis.set_title(
        "G0-MA density difference relative to released Bamboo\n"
        "Δp = p(G0-MA) − p(G0): red = enriched in G0-MA; blue = depleted",
        fontsize=14.5, fontweight="bold", color="#0f172a", pad=11,
    )
    annotation = (
        f"strict target window: {bamboo_window_pct:.1f}% → {ma_window_pct:.1f}% "
        f"(Δ {window_difference_pct:+.1f} percentage points)"
    )
    axis.text(.02, .02, annotation, transform=axis.transAxes, fontsize=10.2,
              color="#111827", va="bottom", ha="left",
              bbox={"facecolor": "white", "alpha": .88, "edgecolor": "#cbd5e1", "pad": 3.5})
    colorbar = figure.colorbar(image, ax=axis, pad=.02)
    colorbar.set_label("Δ density (percentage points per bin)")
    figure.savefig(output, dpi=240, bbox_inches="tight", facecolor=figure.get_facecolor())
    plt.close(figure)

    bin_rows = []
    for sigma_index in range(len(sigma_edges) - 1):
        for ratio_index in range(len(ratio_edges) - 1):
            bin_rows.append({
                "sigma_low": float(sigma_edges[sigma_index]),
                "sigma_high": float(sigma_edges[sigma_index + 1]),
                "ratio_low": float(ratio_edges[ratio_index]),
                "ratio_high": float(ratio_edges[ratio_index + 1]),
                "released_bamboo_density_pct": 100.0 * float(bamboo_density[sigma_index, ratio_index]),
                "g0_ma_density_pct": 100.0 * float(ma_density[sigma_index, ratio_index]),
                "g0_ma_minus_bamboo_density_pct": 100.0 * float(difference[sigma_index, ratio_index]),
            })
    return ({
        "released_bamboo_target_window_pct": bamboo_window_pct,
        "g0_ma_target_window_pct": ma_window_pct,
        "g0_ma_minus_bamboo_target_window_pct": window_difference_pct,
    }, bin_rows)


def gaussian_kde(values: np.ndarray, grid: np.ndarray) -> np.ndarray:
    """Dependency-free Gaussian KDE for a presentation distribution plot."""
    values = np.asarray(values, dtype=float)
    std = float(values.std(ddof=1)) if len(values) > 1 else 0.0
    bandwidth = max(0.35, 1.06 * std * len(values) ** (-.2))
    z = (grid[:, None] - values[None, :]) / bandwidth
    return np.exp(-.5 * z * z).mean(axis=1) / (bandwidth * math.sqrt(2.0 * math.pi))


def plot_conductivity_density_overlay(
    predictions: list[dict[str, float | str]], arms: tuple[Arm, ...], output: Path,
    target: float,
) -> None:
    """Draw peak-normalized conductivity KDEs as readable shared-axis ridgelines."""
    sigma_low, sigma_high = sigma_grid_limit(predictions, target)
    grid = np.linspace(sigma_low, sigma_high, 700)
    figure, axis = plt.subplots(figsize=(13.8, 8.2), constrained_layout=True)
    figure.set_facecolor("white")
    axis.axvspan(target - 2.0, target + 2.0, color="#fde68a", alpha=.40,
                 label="target window: σ = 20 ± 2")
    axis.axvline(target, color="#d97706", linestyle="--", linewidth=1.7)
    baselines = np.arange(len(arms) - 1, -1, -1, dtype=float)
    for baseline, arm in zip(baselines, arms):
        values = np.asarray([float(row["sigma"]) for row in predictions if row["arm"] == arm.key])
        density = gaussian_kde(values, grid)
        density = density / density.max()
        ridge = baseline + .74 * density
        axis.fill_between(grid, baseline, ridge, color=arm.color, alpha=.22, zorder=2)
        axis.plot(grid, ridge, color=arm.color, linewidth=3.0, zorder=3)
        axis.hlines(baseline, sigma_low, sigma_high, color="#cbd5e1", linewidth=.9, zorder=1)
        median = float(np.median(values))
        axis.plot(median, baseline, marker="o", markersize=5.8, color=arm.color,
                  markeredgecolor="white", markeredgewidth=.8, zorder=4)
    axis.set_xlim(sigma_low, sigma_high)
    axis.set_ylim(-.2, len(arms) - .05)
    axis.set_xlabel("predicted conductivity σ (mS/cm)", fontsize=12)
    axis.set_ylabel("generator group", fontsize=12)
    axis.set_yticks(
        baselines,
        [f"{arm.label.replace(chr(10), ' ')}\ntrusted n={sum(1 for row in predictions if row['arm'] == arm.key):,}"
         for arm in arms],
        fontsize=10.4,
    )
    axis.set_title("Conductivity distribution of trusted generated formulas", fontsize=17,
                   fontweight="bold", color="#0f172a")
    axis.text(.01, .02, "Each ridge is peak-normalized; width and mode are comparable across groups.",
              transform=axis.transAxes, fontsize=10, color="#475569")
    axis.grid(axis="x", color="#cbd5e1", alpha=.55)
    axis.spines[["top", "right"]].set_visible(False)
    legend = axis.legend(loc="upper right", frameon=True, fontsize=10.2)
    legend.get_frame().set_facecolor("white")
    legend.get_frame().set_edgecolor("#cbd5e1")
    figure.savefig(output, dpi=240, bbox_inches="tight", facecolor=figure.get_facecolor())
    plt.close(figure)


def t95_half_width(values: np.ndarray) -> float:
    """Two-sided t 95% half width; n=3 is the intended use here."""
    if len(values) < 2:
        return math.nan
    critical = {2: 12.706, 3: 4.303, 4: 3.182, 5: 2.776}.get(len(values), 1.96)
    return critical * float(values.std(ddof=1)) / math.sqrt(len(values))


def plot_released_bamboo_improvement(
    records: list[dict[str, float | str]], output: Path,
) -> list[dict[str, object]]:
    """Plot seed-paired relative error reductions for G0-MA vs released Bamboo."""
    baseline = {str(row["seed"]): row for row in records if row["arm"] == "Bamboo"}
    treatment = {str(row["seed"]): row for row in records if row["arm"] == "G0-MA"}
    common_seeds = sorted(set(baseline) & set(treatment), key=int)
    if len(common_seeds) < 2:
        raise ValueError("Released Bamboo and G0-MA must share at least two seed labels")
    specs = (ERROR_METRICS[0], ERROR_METRICS[2], JOINT_AND_QUALITY_METRICS[0],
             ERROR_METRICS[1], ERROR_METRICS[3], JOINT_AND_QUALITY_METRICS[1])
    figure, axes = plt.subplots(2, 3, figsize=(16.0, 8.7), constrained_layout=True)
    figure.set_facecolor("#fcfcfd")
    summary: list[dict[str, object]] = []
    dot_y = np.linspace(.18, -.18, len(common_seeds))
    for axis, (metric_name, metric_key, _, _) in zip(axes.flat, specs):
        reductions = np.asarray([
            100.0 * (float(baseline[seed][metric_key]) - float(treatment[seed][metric_key]))
            / float(baseline[seed][metric_key])
            for seed in common_seeds
        ])
        mean, half_width = float(reductions.mean()), t95_half_width(reductions)
        extent = max(abs(mean) + (half_width if math.isfinite(half_width) else 0.0),
                     float(np.abs(reductions).max()), 5.0)
        limit = max(12.0, extent * 1.22)
        axis.axvspan(-limit, 0.0, color="#fff7ed", zorder=0)
        axis.axvspan(0.0, limit, color="#ecfdf5", zorder=0)
        axis.axvline(0.0, color="#475569", linewidth=1.2)
        color = "#10b981" if mean >= 0 else "#f97316"
        axis.errorbar(mean, 0.0, xerr=half_width, fmt="o", markersize=10,
                      color=color, ecolor=color, capsize=6, elinewidth=3, zorder=4)
        axis.scatter(reductions, dot_y, color="#1e293b", edgecolor="white",
                     linewidth=.9, s=58, zorder=5)
        direction = 1 if mean >= 0 else -1
        label_x = mean + direction * ((half_width if math.isfinite(half_width) else 0.0) + limit * .035)
        axis.text(label_x, 0.0, f"{mean:+.1f}%", color=color, fontsize=13,
                  fontweight="bold", ha="left" if direction > 0 else "right", va="center")
        axis.text(.02, .90, "worse", color="#c2410c", transform=axis.transAxes,
                  fontsize=9.5, fontweight="bold")
        axis.text(.98, .90, "better", color="#047857", transform=axis.transAxes,
                  fontsize=9.5, fontweight="bold", ha="right")
        axis.set_xlim(-limit, limit)
        axis.set_yticks([])
        axis.set_title(metric_name, fontsize=13, fontweight="bold")
        axis.set_xlabel("relative error reduction (%)")
        axis.grid(axis="x", color="#cbd5e1", alpha=.45)
        axis.spines[["top", "right", "left"]].set_visible(False)
        summary.append({
            "metric": metric_name,
            "n_seeds": len(reductions),
            "mean_relative_error_reduction_pct": mean,
            "std_pct": float(reductions.std(ddof=1)),
            "ci95_low_pct": mean - half_width,
            "ci95_high_pct": mean + half_width,
            **{f"seed_{seed}_pct": value for seed, value in zip(common_seeds, reductions)},
        })
    figure.suptitle(
        "G0-MA improvement versus released Bamboo: seed-paired relative error reduction\n"
        "point = mean; horizontal line = 95% t CI; small dots = individual seeds; positive is better",
        fontsize=17, fontweight="bold", color="#0f172a",
    )
    figure.savefig(output, dpi=240, bbox_inches="tight", facecolor=figure.get_facecolor())
    plt.close(figure)
    return summary


def markdown_table(headers: list[str], rows: list[list[str]]) -> str:
    return "\n".join([
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
        *["| " + " | ".join(row) + " |" for row in rows],
    ])


def write_report(
    output: Path, summaries: list[dict[str, object]], improvement: list[dict[str, object]],
    density_shift: dict[str, float], diversity_rows: list[dict[str, float | str]],
    args: argparse.Namespace,
) -> None:
    key_metrics = (
        "σ MAE", "r MAE", "Joint normalized MAE", "Joint normalized RMSE",
        "Trusted formula rate", "Strict joint-window coverage",
    )
    table_rows = []
    for row in summaries:
        if row["metric"] not in key_metrics:
            continue
        table_rows.append([
            str(row["label"]), str(row["metric"]), f"{float(row['mean']):.3f}",
            f"{float(row['std']):.3f}", f"[{float(row['min']):.3f}, {float(row['max']):.3f}]",
        ])
    diversity_by_arm = {
        arm: [row for row in diversity_rows if row["arm"] == arm]
        for arm in ("Bamboo", "G0-MA")
    }
    diversity_table = []
    for metric, label, unit in (
            ("target_window_formula_count", "目标窗口配方数 / seed", ""),
            ("mean_active_solvents", "平均活性溶剂数", ""),
            ("effective_dominant_solvents", "有效主导溶剂数", ""),
            ("active_set_unique_rate", "不同活性溶剂集合占比", "%"),
            ("median_pairwise_cosine_similarity", "两两配方余弦相似度中位数", ""),
            ("nearest_neighbor_cosine_ge_0_99_rate", "最近邻余弦相似度 ≥0.99", "%")):
        values = []
        for arm in ("Bamboo", "G0-MA"):
            items = [float(row[metric]) for row in diversity_by_arm[arm]]
            multiplier = 100.0 if unit == "%" else 1.0
            values.append(f"{np.mean(items) * multiplier:.1f}" + unit)
        diversity_table.append([label, *values])
    lines = [
        "# 四组生成架构纵向比较", "",
        "## 组别与统计单位", "",
        "- 发布 Bamboo：发布版 diffusion + decoder 生成配方，经当前双任务 oracle 统一评估。",
        "- G0-MA：固定 Bamboo 词表、当前流程重训，并用真实交互残差作训练期辅助监督；它不是发布 Bamboo checkpoint。",
        "- G1-MA：随机候选词表 + 真实交互残差辅助监督。",
        "- G2-MA：解释头软约束筛选词表 + 相同真实交互残差辅助监督。",
        f"- 共同条件：σ = {args.target_conductivity:g} mS/cm，r = {args.anion_ratio:g}；每组 3 个 seed。",
        "- 柱为 3 个 seed 的均值，误差线为 sample SD，黑点为各 seed 原始值。seed 是统计重复单位。", "",
        "## 结果图", "",
        "![σ 与 r 条件误差](four_group_sigma_r_error_2x2.png)", "",
        "![联合误差、可信率与严格窗口覆盖](four_group_joint_quality_2x2.png)", "",
        "![σ-r 二维密度比较](four_group_sigma_r_density_2x2.png)", "",
        "![G0-MA 相对发布 Bamboo 的差分密度](g0_ma_minus_released_bamboo_density_difference.png)", "",
        "![目标窗口内的混合配方多样性](g0_ma_target_window_composition_diversity.png)", "",
        "![四组 σ 分布叠图](four_group_conductivity_density_overlay.png)", "",
        "![G0-MA 相对发布 Bamboo 的误差下降](released_bamboo_vs_G0_MA_relative_improvement_2x3.png)", "",
        "## 汇总数据", "",
        markdown_table(["组别", "指标", "均值", "seed SD", "范围（3 seed）"], table_rows), "",
        "## G0-MA 相对发布 Bamboo：seed 配对误差下降", "",
        markdown_table(
            ["指标", "均值下降", "seed SD", "95% t CI"],
            [[
                str(row["metric"]),
                f"{float(row['mean_relative_error_reduction_pct']):+.1f}%",
                f"{float(row['std_pct']):.1f}%",
                f"[{float(row['ci95_low_pct']):+.1f}%, {float(row['ci95_high_pct']):+.1f}%]",
            ] for row in improvement],
        ), "",
        "## G0-MA 相对发布 Bamboo 的差分密度", "",
        "- 差分定义为每个 `(σ, r)` 格内的 seed 平均可信配方密度：`Δp = p(G0-MA) − p(G0)`。"
        "红色表示 G0-MA 富集，蓝色表示 G0-MA 减少；整张图的正负差分相加为零。",
        f"- 严格联合目标窗口（σ±2 mS/cm、r±0.10）的密度由 "
        f"{density_shift['released_bamboo_target_window_pct']:.1f}% 提升至 "
        f"{density_shift['g0_ma_target_window_pct']:.1f}%（Δ "
        f"{density_shift['g0_ma_minus_bamboo_target_window_pct']:+.1f} 个百分点）。", "",
        "## G0-MA 目标窗口内的混合配方多样性", "",
        "- 左图以发布 Bamboo = 1 归一化比较目标窗口覆盖与配方组成指标；右图为每个 seed 内两两溶剂混合 BOW 余弦相似度的 CDF，实线为 seed 均值、阴影为 seed 间范围。曲线左移表示配方组成较不相似。",
        "- 两组使用相同的 62 个 Bamboo 溶剂词表，因此该图衡量的是溶剂组合和比例的多样性，而不是新分子骨架的发现。", "",
        markdown_table(["指标", "Released Bamboo", "G0-MA"], diversity_table), "",
        "## 解释边界", "",
        "- 联合误差使用 σ/2 mS/cm 与 r/0.10 的归一化平方和；严格联合窗口为 σ±2 mS/cm 且 r±0.10。",
        "- 可信率必须与误差同时阅读。若某组大量生成配方被 oracle 判为分布外，其余少量可信配方上的误差不能单独代表该生成器整体质量。",
        "- 当前 G1-MA 的可信率显著低于其余三组，因此其误差主要说明该随机词表/生成组合在当前 oracle 可用域内失败；不应与其他组的 Top-100 候选作等预算比较。",
        "- 发布 Bamboo→G0-MA 的下降是端到端比较，不能单独归因于交互辅助：两组还同时不同于发布 checkpoint 与当前重训流程。交互辅助本身的因果证据仍应来自同词表、同数据的 Q/MA/shuffle 消融。",
        "",
        "## 输出", "",
        "- `four_group_seed_metrics.csv`：每组、每 seed 的原始指标。",
        "- `four_group_metric_summary.csv`：均值、seed SD、范围。",
        "- `g0_ma_minus_released_bamboo_density_difference.csv`：差分密度图各个 `(σ, r)` 格的原始密度与差分值。",
        "- `g0_ma_target_window_composition_diversity.csv`：严格目标窗口内、按 seed 计算的混合配方多样性指标。",
        "- `released_bamboo_vs_G0_MA_relative_improvement.csv`：发布 Bamboo→G0-MA 的 seed 配对相对误差下降。",
    ]
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir or args.comparison_root / "four_group_architecture_comparison"
    output_dir.mkdir(parents=True, exist_ok=True)
    arms = make_arms(args.comparison_root)
    records = [record for arm in arms for record in load_arm(
        arm, args.target_conductivity, args.anion_ratio)]
    seed_fields = ["seed", "arm", "label", "group", "n_sampled", "n_trusted",
                   "trusted_formula_rate"] + [spec[1] for spec in ERROR_METRICS + JOINT_AND_QUALITY_METRICS
                                                 if spec[1] != "trusted_formula_rate"]
    write_csv(output_dir / "four_group_seed_metrics.csv", records, seed_fields)
    summaries = summary_rows(records, arms)
    write_csv(output_dir / "four_group_metric_summary.csv", summaries,
              ["arm", "label", "group", "metric", "metric_key", "unit", "n_seeds",
               "mean", "std", "min", "max"])
    plot_dashboard(records, arms, ERROR_METRICS,
                   output_dir / "four_group_sigma_r_error_2x2.png",
                   "Four-group conditional-generation comparison: σ and r error")
    plot_dashboard(records, arms, JOINT_AND_QUALITY_METRICS,
                   output_dir / "four_group_joint_quality_2x2.png",
                   "Four-group conditional-generation comparison: joint quality")
    predictions = [prediction for arm in arms for prediction in load_trusted_predictions(arm)]
    plot_sigma_ratio_density(
        predictions, arms, output_dir / "four_group_sigma_r_density_2x2.png",
        args.target_conductivity, args.anion_ratio,
    )
    density_shift, density_difference_rows = plot_g0_ma_density_difference(
        predictions, output_dir / "g0_ma_minus_released_bamboo_density_difference.png",
        args.target_conductivity, args.anion_ratio,
    )
    write_csv(
        output_dir / "g0_ma_minus_released_bamboo_density_difference.csv",
        density_difference_rows,
        ["sigma_low", "sigma_high", "ratio_low", "ratio_high",
         "released_bamboo_density_pct", "g0_ma_density_pct",
         "g0_ma_minus_bamboo_density_pct"],
    )
    diversity_rows, target_window_similarities = target_window_composition_diversity(
        arms, args.target_conductivity, args.anion_ratio)
    write_csv(
        output_dir / "g0_ma_target_window_composition_diversity.csv",
        diversity_rows,
        ["arm", "seed", "target_window_formula_count",
         "unique_active_solvent_sets", "active_set_unique_rate",
         "unique_compositions_rounded_2pct", "rounded_composition_unique_rate",
         "mean_active_solvents", "effective_dominant_solvents",
         "top_dominant_solvent_share", "median_pairwise_cosine_similarity",
         "mean_nearest_neighbor_cosine_similarity",
         "nearest_neighbor_cosine_ge_0_99_rate",
         "pairwise_cosine_ge_0_90_rate"],
    )
    plot_target_window_composition_diversity(
        diversity_rows, target_window_similarities,
        output_dir / "g0_ma_target_window_composition_diversity.png")
    plot_conductivity_density_overlay(
        predictions, arms, output_dir / "four_group_conductivity_density_overlay.png",
        args.target_conductivity,
    )
    improvement = plot_released_bamboo_improvement(
        records, output_dir / "released_bamboo_vs_G0_MA_relative_improvement_2x3.png")
    improvement_fields = ["metric", "n_seeds", "mean_relative_error_reduction_pct", "std_pct",
                          "ci95_low_pct", "ci95_high_pct", "seed_123_pct", "seed_456_pct", "seed_789_pct"]
    write_csv(output_dir / "released_bamboo_vs_G0_MA_relative_improvement.csv",
              improvement, improvement_fields)
    write_report(output_dir / "four_group_comparison_report_zh.md", summaries, improvement,
                 density_shift, diversity_rows, args)
    print(f"Wrote four-group comparison to {output_dir}")


if __name__ == "__main__":
    main()
