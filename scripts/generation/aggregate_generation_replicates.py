#!/usr/bin/env python3
"""Aggregate seed-matched conditional-generation experiments.

The input directory contains one subdirectory per random seed, for example
``G2_seed_123/generation_experiment_summary.csv``.  Each summary must include
the same controlled experiment groups (normally G2-Q, G2-MA, and
G2-MA-shuffle).  The script treats random seeds--not the generated formulas--
as independent experimental repeats.  It therefore reports mean, sample
standard deviation, and a two-sided Student-t 95% confidence interval across
seeds, together with seed-paired effects against the selected controls.

It deliberately aggregates already evaluated experiments: it never resamples
formulas and never calls an oracle.
"""

import argparse
import csv
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


GROUP_COLORS = {
    "G0": "#0f766e",
    "G0-Q": "#2563eb",
    "G0-MA": "#7c3aed",
    "G0-MA-shuffle": "#059669",
    "G1-Q": "#ea580c",
    "G1-MA": "#7c3aed",
    "G1-MA-shuffle": "#059669",
    "G2-Q": "#2563eb",
    "G2-MA": "#7c3aed",
    "G2-MA-shuffle": "#059669",
}

ALL_METRICS = (
    ("sigma MAE", "kappa_mae_mS_cm", "mS/cm"),
    ("sigma RMSE", "kappa_rmse_mS_cm", "mS/cm"),
    ("r MAE", "anion_ratio_pred_mae_from_condition", "ratio"),
    ("r RMSE", "anion_ratio_pred_rmse_from_condition", "ratio"),
    ("joint normalized MAE", "joint_normalized_mae", "normalized joint error"),
    ("joint normalized RMSE", "joint_normalized_rmse", "normalized joint error"),
)
SELECTED_METRICS = (
    ("Top-k sigma MAE", "selected_kappa_mae_mS_cm", "mS/cm"),
    ("Top-k sigma RMSE", "selected_kappa_rmse_mS_cm", "mS/cm"),
    ("Top-k r MAE", "selected_anion_ratio_pred_mae_from_condition", "ratio"),
    ("Top-k r RMSE", "selected_anion_ratio_pred_rmse_from_condition", "ratio"),
    ("Top-k joint normalized MAE", "selected_joint_normalized_mae", "normalized joint error"),
    ("Top-k joint normalized RMSE", "selected_joint_normalized_rmse", "normalized joint error"),
)
COVERAGE_METRICS = (
    ("all strict joint window", "joint_within_kappa_pm2_ratio_pm0.1"),
    ("selected strict joint window", "selected_joint_within_kappa_pm2_ratio_pm0.1"),
)

# Two-sided 95% Student-t critical values.  The normal approximation is only
# used for n > 31; the intended replicate setting is usually n=3.
T95 = {
    1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571,
    6: 2.447, 7: 2.365, 8: 2.306, 9: 2.262, 10: 2.228,
    11: 2.201, 12: 2.179, 13: 2.160, 14: 2.145, 15: 2.131,
    16: 2.120, 17: 2.110, 18: 2.101, 19: 2.093, 20: 2.086,
    21: 2.080, 22: 2.074, 23: 2.069, 24: 2.064, 25: 2.060,
    26: 2.056, 27: 2.052, 28: 2.048, 29: 2.045, 30: 2.042,
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Aggregate seed-level controlled generation summaries.")
    parser.add_argument("--replicate-root", required=True,
                        help="Directory containing G2_seed_*/generation_experiment_summary.csv.")
    parser.add_argument("--output-dir", default=None,
                        help="Defaults to <replicate-root>/replicate_analysis.")
    parser.add_argument("--groups", nargs="+",
                        default=["G2-Q", "G2-MA", "G2-MA-shuffle"])
    parser.add_argument("--baseline", default="G2-Q",
                        help="Seed-paired primary no-auxiliary control.")
    parser.add_argument("--primary-treatment", default=None,
                        help="Primary treatment for relative-improvement plots and conclusions. "
                             "Defaults to G2-MA when that group is present.")
    parser.add_argument("--counterfactual", default=None,
                        help="Optional counterfactual control paired with --primary-treatment.")
    parser.add_argument("--report-title", default=None,
                        help="Chinese title for the Markdown report.")
    parser.add_argument("--selection-budget", type=int, default=100)
    parser.add_argument("--joint-sigma-scale", type=float, default=2.0)
    parser.add_argument("--joint-anion-ratio-scale", type=float, default=0.10)
    return parser.parse_args()


def read_rows(path):
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path, rows):
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def seed_name(path):
    """Use the trailing numeric seed when possible, while preserving strings."""
    stem = path.parent.name
    return stem.rsplit("_", 1)[-1] if "_" in stem else stem


def comparison_pairs(groups, baseline, primary_treatment, counterfactual):
    """Return requested seed-paired treatment/control comparisons once each."""
    pairs = [(group, baseline) for group in groups if group != baseline]
    if (primary_treatment and counterfactual and primary_treatment in groups
            and counterfactual in groups and primary_treatment != counterfactual):
        pairs.append((primary_treatment, counterfactual))
    elif "G2-MA" in groups and "G2-MA-shuffle" in groups:
        # Preserve the historical G2 default for existing commands.
        pairs.append(("G2-MA", "G2-MA-shuffle"))
    return list(dict.fromkeys(pairs))


def load_experiments(root, groups):
    summaries = sorted(root.glob("*/generation_experiment_summary.csv"),
                       key=lambda path: (int(seed_name(path))
                                         if seed_name(path).isdigit()
                                         else seed_name(path)))
    if not summaries:
        raise FileNotFoundError(
            f"no */generation_experiment_summary.csv found below {root}")

    by_seed = {}
    for path in summaries:
        seed = seed_name(path)
        group_rows = {row["group"]: row for row in read_rows(path)}
        absent = [group for group in groups if group not in group_rows]
        if absent:
            raise ValueError(f"{path}: missing requested groups {absent}")
        by_seed[seed] = {group: group_rows[group] for group in groups}
    return by_seed


def numeric(row, key):
    value = row.get(key, "")
    if value in (None, ""):
        raise ValueError(f"missing numeric value for {key}")
    return float(value)


def t95_half_width(values):
    values = np.asarray(values, dtype=float)
    if len(values) < 2:
        return math.nan
    sd = float(values.std(ddof=1))
    return T95.get(len(values) - 1, 1.96) * sd / math.sqrt(len(values))


def summary_row(group, metric_name, metric_key, values):
    values = np.asarray(values, dtype=float)
    mean = float(values.mean())
    sd = float(values.std(ddof=1)) if len(values) > 1 else math.nan
    half_width = t95_half_width(values)
    return {
        "group": group,
        "metric": metric_name,
        "metric_key": metric_key,
        "n_seeds": len(values),
        "mean": mean,
        "std": sd,
        "sem": sd / math.sqrt(len(values)) if len(values) > 1 else math.nan,
        "ci95_low": mean - half_width if not math.isnan(half_width) else math.nan,
        "ci95_high": mean + half_width if not math.isnan(half_width) else math.nan,
    }


def summarize(by_seed, groups):
    long_rows, summary = [], []
    specs = ALL_METRICS + SELECTED_METRICS + COVERAGE_METRICS
    for seed, group_rows in by_seed.items():
        for group, row in group_rows.items():
            for spec in specs:
                metric_name, metric_key = spec[:2]
                long_rows.append({
                    "seed": seed,
                    "group": group,
                    "metric": metric_name,
                    "metric_key": metric_key,
                    "value": numeric(row, metric_key),
                })
    for group in groups:
        for spec in specs:
            metric_name, metric_key = spec[:2]
            values = [numeric(by_seed[seed][group], metric_key)
                      for seed in by_seed]
            summary.append(summary_row(group, metric_name, metric_key, values))
    return long_rows, summary


def paired_effects(by_seed, comparisons):
    rows = []
    # Coverage is intentionally excluded here: its desired direction is
    # higher, whereas every error metric below is better when lower.
    specs = ALL_METRICS + SELECTED_METRICS
    for treatment, baseline in comparisons:
        for spec in specs:
            metric_name, metric_key = spec[:2]
            baseline_values = np.asarray(
                [numeric(by_seed[seed][baseline], metric_key) for seed in by_seed],
                dtype=float)
            treatment_values = np.asarray(
                [numeric(by_seed[seed][treatment], metric_key) for seed in by_seed],
                dtype=float)
            delta = treatment_values - baseline_values
            mean_delta = float(delta.mean())
            half_width = t95_half_width(delta)
            rows.append({
                "treatment": treatment,
                "baseline": baseline,
                "metric": metric_name,
                "metric_key": metric_key,
                "n_seeds": len(delta),
                "mean_treatment": float(treatment_values.mean()),
                "mean_baseline": float(baseline_values.mean()),
                "mean_delta_treatment_minus_baseline": mean_delta,
                "std_delta": float(delta.std(ddof=1)) if len(delta) > 1 else math.nan,
                "ci95_low_delta": mean_delta - half_width if not math.isnan(half_width) else math.nan,
                "ci95_high_delta": mean_delta + half_width if not math.isnan(half_width) else math.nan,
                "mean_relative_change_pct": 100 * mean_delta / float(baseline_values.mean()),
                "seeds_lower_error": int((delta < 0).sum()),
                "seeds_equal_error": int((delta == 0).sum()),
                "seeds_higher_error": int((delta > 0).sum()),
            })
    return rows


def seed_metric_values(by_seed, group, metric_key):
    return np.asarray([numeric(by_seed[seed][group], metric_key)
                       for seed in by_seed], dtype=float)


def plot_group_metrics(by_seed, groups, specs, output, title):
    figure, axes = plt.subplots(3, 2, figsize=(15, 16), constrained_layout=True)
    x = np.arange(len(groups))
    offsets = np.linspace(-0.10, 0.10, len(by_seed))
    for axis, (metric_name, metric_key, unit) in zip(axes.flat, specs):
        means, cis = [], []
        for group in groups:
            values = seed_metric_values(by_seed, group, metric_key)
            means.append(values.mean())
            cis.append(t95_half_width(values))
        bars = axis.bar(x, means, yerr=cis, capsize=5,
                        color=[GROUP_COLORS.get(group, "#64748b") for group in groups],
                        alpha=.86)
        for index, group in enumerate(groups):
            values = seed_metric_values(by_seed, group, metric_key)
            axis.scatter(np.full(len(values), index) + offsets, values,
                         color="#1f2937", s=28, zorder=3)
            axis.text(index, means[index] + (cis[index] if not math.isnan(cis[index]) else 0),
                      f"{means[index]:.3f}", ha="center", va="bottom", fontsize=10)
        axis.set_title(metric_name)
        axis.set_ylabel(unit)
        axis.set_xticks(x, groups)
        axis.grid(axis="y", alpha=.25)
        axis.margins(y=.18)
    figure.suptitle(title + "\nbar = mean; error bar = seed-level 95% t CI; dot = one seed",
                     fontsize=16)
    figure.savefig(output, dpi=200, bbox_inches="tight")
    plt.close(figure)


def plot_paired_joint_effects(by_seed, comparisons, output, title):
    specs = (
        ("All formulas: joint normalized MAE", "joint_normalized_mae"),
        ("All formulas: joint normalized RMSE", "joint_normalized_rmse"),
        ("Top-k: joint normalized MAE", "selected_joint_normalized_mae"),
        ("Top-k: joint normalized RMSE", "selected_joint_normalized_rmse"),
    )
    figure, axes = plt.subplots(2, 2, figsize=(14, 11), constrained_layout=True)
    seeds = list(by_seed)
    offsets = np.linspace(-.08, .08, len(seeds))
    for axis, (title, metric_key) in zip(axes.flat, specs):
        means, cis = [], []
        labels = []
        for treatment, baseline in comparisons:
            delta = (seed_metric_values(by_seed, treatment, metric_key)
                     - seed_metric_values(by_seed, baseline, metric_key))
            means.append(delta.mean())
            cis.append(t95_half_width(delta))
            labels.append(f"{treatment} - {baseline}")
        x = np.arange(len(comparisons))
        axis.bar(x, means, yerr=cis, capsize=5,
                 color=[GROUP_COLORS.get(group, "#64748b")
                        for group, _ in comparisons],
                 alpha=.86)
        for idx, (treatment, baseline) in enumerate(comparisons):
            delta = (seed_metric_values(by_seed, treatment, metric_key)
                     - seed_metric_values(by_seed, baseline, metric_key))
            axis.scatter(np.full(len(delta), idx) + offsets, delta,
                         color="#1f2937", s=28, zorder=3)
            axis.text(idx, means[idx], f"{means[idx]:+.3f}", ha="center",
                      va="bottom" if means[idx] >= 0 else "top", fontsize=10)
        axis.axhline(0., color="#111827", linewidth=1)
        axis.set_title(title)
        axis.set_ylabel("paired delta (treatment - control)\nlower is better")
        axis.set_xticks(x, labels, rotation=12, ha="right")
        axis.grid(axis="y", alpha=.25)
        axis.margins(y=.22)
    figure.suptitle(title + "\nbar = mean delta; error bar = paired 95% t CI; dot = one seed",
                     fontsize=16)
    figure.savefig(output, dpi=200, bbox_inches="tight")
    plt.close(figure)


def plot_relative_improvement(by_seed, baseline, treatment, specs, output):
    """Plot paired percentage improvements as a compact forest plot."""
    # Arrange MAE / RMSE by rows, and sigma / r / joint by columns.  This is
    # more readable than six one-bar charts and makes the direction of each
    # seed-level effect visible without excessive empty space.
    order = (0, 2, 4, 1, 3, 5)
    ordered_specs = [specs[index] for index in order]
    figure, axes = plt.subplots(2, 3, figsize=(16, 8.6), constrained_layout=True)
    seed_y = np.linspace(.18, -.18, len(by_seed))
    for axis, (metric_name, metric_key, _) in zip(axes.flat, ordered_specs):
        baseline_values = seed_metric_values(by_seed, baseline, metric_key)
        treatment_values = seed_metric_values(by_seed, treatment, metric_key)
        # Positive means that the factual MA treatment has lower error.
        improvement = 100 * (baseline_values - treatment_values) / baseline_values
        mean = float(improvement.mean())
        ci = t95_half_width(improvement)
        color = "#10b981" if mean >= 0 else "#f97316"
        extent = max(abs(mean) + (ci if not math.isnan(ci) else 0),
                     float(np.abs(improvement).max()), 2.0)
        limit = max(8.0, 1.22 * extent)
        axis.axvspan(0, limit, color="#ecfdf5", zorder=0)
        axis.axvspan(-limit, 0, color="#fff7ed", zorder=0)
        axis.axvline(0., color="#334155", linewidth=1.2, zorder=1)
        axis.errorbar(mean, 0, xerr=ci, fmt="o", color=color,
                      ecolor=color, elinewidth=3, capsize=6, markersize=10,
                      zorder=4, label="mean ± 95% t CI")
        axis.scatter(improvement, seed_y, color="#334155", edgecolors="white",
                     linewidth=.8, s=58, zorder=5, label="individual seed")
        label_x = mean + (ci if mean >= 0 else -ci)
        direction = 1 if mean >= 0 else -1
        axis.text(label_x + direction * limit * .045, 0, f"{mean:+.1f}%",
                  color=color, ha="left" if direction > 0 else "right",
                  va="center", fontsize=12, fontweight="bold")
        axis.set_title(metric_name)
        axis.set_xlabel("relative error reduction (%)")
        axis.set_yticks([])
        axis.set_xlim(-limit, limit)
        axis.grid(axis="x", alpha=.20)
        axis.spines[["left", "right", "top"]].set_visible(False)
        axis.text(.02, .90, "worse", color="#c2410c", transform=axis.transAxes,
                  fontsize=9, fontweight="bold")
        axis.text(.98, .90, "better", color="#047857", transform=axis.transAxes,
                  ha="right", fontsize=9, fontweight="bold")
    figure.suptitle(
        f"{treatment} improvement versus {baseline}: seed-paired relative error reduction"
        "\npoint = mean; horizontal line = 95% t CI; small dots = individual seeds; positive is better",
        fontsize=16, fontweight="bold")
    figure.savefig(output, dpi=200, bbox_inches="tight")
    plt.close(figure)


def markdown_table(headers, rows):
    header = "| " + " | ".join(headers) + " |"
    divider = "| " + " | ".join(["---"] * len(headers)) + " |"
    body = ["| " + " | ".join(row) + " |" for row in rows]
    return "\n".join([header, divider, *body])


def fmt(value, digits=3):
    return f"{float(value):.{digits}f}"


def report(output, root, by_seed, groups, baseline, primary_treatment,
           counterfactual, summary, effects, args):
    n_seeds = len(by_seed)
    summary_lookup = {(row["group"], row["metric_key"]): row for row in summary}
    effect_lookup = {(row["treatment"], row["baseline"], row["metric_key"]): row
                     for row in effects}

    core_specs = ALL_METRICS
    overall_rows = []
    for group in groups:
        for metric_name, metric_key, _ in core_specs:
            row = summary_lookup[group, metric_key]
            overall_rows.append([
                group, metric_name, fmt(row["mean"]), fmt(row["std"]),
                f"[{fmt(row['ci95_low'])}, {fmt(row['ci95_high'])}]",
            ])

    comparisons = comparison_pairs(groups, baseline, primary_treatment, counterfactual)

    effect_rows = []
    for treatment, control in comparisons:
        for metric_name, metric_key, _ in core_specs:
            row = effect_lookup[treatment, control, metric_key]
            effect_rows.append([
                f"{treatment} - {control}", metric_name,
                fmt(row["mean_delta_treatment_minus_baseline"]),
                f"[{fmt(row['ci95_low_delta'])}, {fmt(row['ci95_high_delta'])}]",
                f"{row['mean_relative_change_pct']:+.1f}%",
                f"{row['seeds_lower_error']}/{row['n_seeds']}",
            ])

    coverage_rows = []
    for group in groups:
        for metric_name, metric_key in COVERAGE_METRICS:
            row = summary_lookup[group, metric_key]
            coverage_rows.append([
                group, metric_name, f"{100 * row['mean']:.1f}%",
                f"{100 * row['std']:.1f}%",
                f"[{100 * row['ci95_low']:.1f}%, {100 * row['ci95_high']:.1f}%]",
            ])

    primary_comparisons = []
    if primary_treatment and primary_treatment in groups and primary_treatment != baseline:
        primary_comparisons.append((primary_treatment, baseline))
    if (primary_treatment and counterfactual and primary_treatment in groups
            and counterfactual in groups and primary_treatment != counterfactual):
        primary_comparisons.append((primary_treatment, counterfactual))
    primary = [effect_lookup[treatment, control, key]
               for treatment, control in primary_comparisons
               for _, key, _ in ALL_METRICS[-2:]]
    if primary and all(row["seeds_lower_error"] == n_seeds for row in primary):
        conclusion = (
            f"在 {n_seeds} 个 seed 中，{primary_treatment} 相比主要对照的"
            "全候选联合 MAE 与 RMSE 均在每个 seed 中降低；这支持真实交互残差辅助监督"
            "具有可重复的正向趋势。")
    elif primary:
        conclusion = (
            f"{primary_treatment} 的联合误差改善并非在所有 seed 中一致，因此当前结果只能视为趋势，"
            "不能称为稳定提升。")
    else:
        conclusion = (
            "该目录仅包含一个生成组，因而可报告跨 seed 的均值、SD 与 CI，"
            "但没有目录内的配对处理—对照比较，不能据此推断机制贡献。")

    lines = [
        f"# {args.report_title or '条件生成：多随机种子汇总'}", "",
        "## 评价设置", "",
        f"- 独立重复数：{n_seeds}（seed：{', '.join(by_seed)}）；每个 seed 内按相同 seed 配对比较。",
        f"- 主要基线：`{baseline}`；主要处理组：`{primary_treatment or '无（单组汇总）'}`；"
        f"反事实对照：`{counterfactual or '无'}`。",
        f"- 联合误差使用 \\(\\sigma/{args.joint_sigma_scale:g}\\) mS/cm 和 "
        f"\\(r/{args.joint_anion_ratio_scale:g}\\) 归一化；Top-k 的 \\(k={args.selection_budget}\\)。",
        "- 均值、标准差与 95% CI 均以 seed 为统计单位。CI 为双侧 Student-t CI；"
        "不会把同一 seed 内的 1024 条生成配方误当作独立重复。", "",
        "## 全部可信生成配方：跨 seed 汇总", "",
        markdown_table(["组别", "指标", "均值", "seed SD", "95% CI"], overall_rows), "",
        "![全部候选的跨 seed 误差汇总](replicate_all_target_error_3x2.png)", "",
    ]
    if primary_treatment and primary_treatment in groups and primary_treatment != baseline:
        lines.extend([
            f"![{primary_treatment} 相对 {baseline} 的误差下降百分比]"
            "(replicate_primary_vs_baseline_relative_improvement_3x2.png)", "",
        ])
    if comparisons:
        lines.extend([
            "## 配对机制效应：处理组减去对应对照", "",
            "负值表示处理组误差更低；`胜出 seed` 表示该处理组误差低于该行对应对照的 seed 数。", "",
            markdown_table(["配对", "指标", "平均差值", "差值 95% CI", "相对变化", "胜出 seed"], effect_rows), "",
            "![配对联合效应](replicate_paired_joint_effects.png)", "",
        ])
    lines.extend([
        "## 严格联合窗口覆盖", "",
        "严格窗口定义为 \\(\\sigma\\pm2\\) mS/cm 且 \\(r\\pm0.10\\)。覆盖率是补充指标，"
        "不能由平均误差单独推断。", "",
        markdown_table(["组别", "指标", "均值", "seed SD", "95% CI"], coverage_rows), "",
        "## 等预算 Top-k 候选", "",
        "![Top-k 候选的跨 seed 误差汇总](replicate_selected_target_error_3x2.png)", "",
        "## 结论与边界", "",
        conclusion,
        "- 当前仅有 3 个 seed。95% CI 在小样本下会较宽，不宜据此宣称统计显著；建议再补充 seed 或报告预先指定的多目标条件结果。",
        "- 应同时查看 r 的独立误差、严格窗口覆盖和 Top-k 图，避免把主要由 sigma 改善驱动的联合误差下降表述为两个目标均改善。",
        "",
        "## 输出文件", "",
        "- `replicate_seed_metrics.csv`：每个 seed、组别、指标的原始值。",
        "- `replicate_metric_summary.csv`：组别均值、SD、SEM、95% CI。",
    ])
    if primary_treatment and counterfactual:
        lines.insert(-5,
            f"- `{counterfactual}` 保留训练规模和辅助损失，仅打乱同一格内的配方—机制标签对应；"
            f"因此 {primary_treatment} 若持续优于该对照，支持的是**真实配方—机制标签配对**，"
            "而非单纯增加一个辅助损失。")
    if comparisons:
        lines.append("- `replicate_paired_effects.csv`：以 seed 配对计算的处理效应与 CI。")
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    args = parse_args()
    root = Path(args.replicate_root)
    output_dir = Path(args.output_dir) if args.output_dir else root / "replicate_analysis"
    output_dir.mkdir(parents=True, exist_ok=True)
    if args.baseline not in args.groups:
        raise ValueError("--baseline must occur in --groups")
    primary_treatment = args.primary_treatment
    if primary_treatment is None and "G2-MA" in args.groups:
        primary_treatment = "G2-MA"
    if primary_treatment is not None and primary_treatment not in args.groups:
        raise ValueError("--primary-treatment must occur in --groups")
    if args.counterfactual is not None and args.counterfactual not in args.groups:
        raise ValueError("--counterfactual must occur in --groups")

    by_seed = load_experiments(root, args.groups)
    long_rows, summaries = summarize(by_seed, args.groups)
    comparisons = comparison_pairs(args.groups, args.baseline, primary_treatment,
                                   args.counterfactual)
    effects = paired_effects(by_seed, comparisons)
    write_csv(output_dir / "replicate_seed_metrics.csv", long_rows)
    write_csv(output_dir / "replicate_metric_summary.csv", summaries)
    write_csv(output_dir / "replicate_paired_effects.csv", effects)
    plot_group_metrics(
        by_seed, args.groups, ALL_METRICS,
        output_dir / "replicate_all_target_error_3x2.png",
        "All trusted generated formulas: seed-level summary")
    plot_group_metrics(
        by_seed, args.groups, SELECTED_METRICS,
        output_dir / "replicate_selected_target_error_3x2.png",
        f"Equal-budget top-{args.selection_budget} candidates: seed-level summary")
    if comparisons:
        plot_paired_joint_effects(
            by_seed, comparisons,
            output_dir / "replicate_paired_joint_effects.png",
            "Seed-paired conditional-generation effect")
    if primary_treatment and primary_treatment != args.baseline:
        plot_relative_improvement(
            by_seed, args.baseline, primary_treatment, ALL_METRICS,
            output_dir / "replicate_primary_vs_baseline_relative_improvement_3x2.png")
    report(output_dir / "replicate_analysis_report_zh.md", root, by_seed,
           args.groups, args.baseline, primary_treatment, args.counterfactual,
           summaries, effects, args)
    print(f"Aggregated {len(by_seed)} seed(s) into {output_dir}")


if __name__ == "__main__":
    main()
