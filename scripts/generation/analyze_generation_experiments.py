#!/usr/bin/env python3
"""Summarize controlled G0--G3 formula-generation experiments.

The script deliberately separates (1) the proportion of all sampled formulas
that reaches the requested conductivity and (2) the hit rate after an equal
selection budget.  It also writes stage-level wall-clock timing from the CSV
files produced by ``run_generation_experiment.sh``.
"""

import argparse
import csv
import json
import textwrap
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap


GROUP_LABELS = {
    "G0": "G0 released Bamboo generator",
    "G1": "G1 random/diverse pool",
    "G2": "G2 explain-guided pool",
    "G3": "G3 robust re-rank (G2)",
}
GROUPS = tuple(GROUP_LABELS)
MAX_TRUSTED_CONDUCTIVITY = 100.0
DENSITY_CMAP = LinearSegmentedColormap.from_list(
    "density_blue_to_red", ["#2563eb", "#dc2626"])
DENSITY_CMAP.set_bad("white")
GROUP_DESIGNS = {
    "G0": {
        "source": "Bamboo-Mixer 发布的 diffusion + decoder checkpoint；固定 Bamboo 62 溶剂、2 盐词表",
        "goal": "作为不使用本研究候选池筛选与解释约束的基线，衡量原版生成器在同一冻结 oracle 下的输出分布。",
    },
    "G1": {
        "source": "从分子预训练数据的 20000 个候选中随机/多样性抽取 62 个分子；重新训练同规格 diffusion + decoder",
        "goal": "检验仅更换为随机且结构多样的分子库，能否改善目标电导率与阴离子比例的生成结果。",
    },
    "G2": {
        "source": "从相同 20000 个候选中按片段贡献和分子物性软约束筛出 62 个分子；重新训练同规格 diffusion + decoder",
        "goal": "检验解释导出的结构与物性约束是否能将生成分布推向更可信、更接近目标的配方区域。",
    },
    "G3": {
        "source": "复用 G2 生成器与候选池，在温度—浓度网格上进行鲁棒重排序",
        "goal": "检验后处理的鲁棒排序能否提高有限实验预算下候选的可用性。",
    },
}


def read_csv(path):
    with open(path, newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path, rows):
    if not rows:
        return
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def solvent_meta(root, group):
    pool = root / group / "candidate_pool"
    if group == "G3":
        pool = root / "G2" / "candidate_pool"
    return json.loads((pool / "solvents.json").read_text())


def trusted_record(row):
    """Return whether a generated record is safe for quantitative comparison."""
    value = row.get("conductivity_pred_ms_cm")
    return (row.get("prediction_status", "trusted") == "trusted"
            and value is not None
            and np.isfinite(float(value))
            and 0.0 <= float(value) <= MAX_TRUSTED_CONDUCTIVITY)


def formula_label(row, solvents, group, threshold=.02):
    """Use Bamboo names for G0 and SMILES for pool-derived G1/G2 formulas."""
    bow = np.asarray(row["bow"], dtype=float)
    terms = []
    for index in np.where(bow[:len(solvents)] > threshold)[0]:
        identity = (solvents[index]["name"] if group == "G0"
                    else solvents[index]["smiles"])
        terms.append(f"{identity} ({bow[index]:.2f})")
    if not terms:
        index = int(np.argmax(bow[:len(solvents)]))
        identity = (solvents[index]["name"] if group == "G0"
                    else solvents[index]["smiles"])
        terms.append(f"{identity} ({bow[index]:.2f})")
    return " + ".join(terms)


def summarize_group(root, group, target, selection_budget):
    generated = root / group / "generated" / "generated_formulas.json"
    if not generated.exists():
        return None
    records = json.loads(generated.read_text())
    if not records:
        return None
    solvents = solvent_meta(root, group)
    trusted_ids = np.asarray([
        index for index, row in enumerate(records)
        if trusted_record(row)
    ])
    if not len(trusted_ids):
        return None
    values = np.asarray([float(records[index]["conductivity_pred_ms_cm"])
                         for index in trusted_ids])
    ranked_ids = np.asarray([
        index for index in trusted_ids
        if records[index].get("candidate_rank_score") is not None
        and np.isfinite(float(records[index]["candidate_rank_score"]))
    ])
    rank = np.asarray([float(records[index]["candidate_rank_score"])
                       for index in ranked_ids])
    selected_positions = np.argsort(-rank)[:min(selection_budget, len(ranked_ids))]
    selected_ids = ranked_ids[selected_positions]
    selected_values = np.asarray([
        float(records[index]["conductivity_pred_ms_cm"])
        for index in selected_ids
    ])
    trusted_records = [records[index] for index in trusted_ids]
    solvent_counts, dominant = [], []
    formula_keys = []
    for row in trusted_records:
        bow = np.asarray(row["bow"], dtype=float)
        active = np.where(bow[:len(solvents)] > 0.02)[0]
        solvent_counts.append(int(len(active)))
        dominant.append(solvents[int(np.argmax(bow[:len(solvents)]))]["name"])
        formula_keys.append("+".join(
            f"{solvents[i]['name']}:{bow[i]:.2f}" for i in active))
    robust_min = [row.get("robust_min_ms_cm") for row in trusted_records]
    predicted_ratios = np.asarray([
        float(row["anion_ratio_pred"]) for row in trusted_records
        if row.get("anion_ratio_pred") is not None
        and np.isfinite(float(row["anion_ratio_pred"]))
    ])
    condition = float(records[0].get("anion_ratio_condition", .5))
    selected_ratios = np.asarray([
        float(records[index]["anion_ratio_pred"]) for index in selected_ids
        if records[index].get("anion_ratio_pred") is not None
        and np.isfinite(float(records[index]["anion_ratio_pred"]))
    ])
    return {
        "group": group,
        "group_label": GROUP_LABELS[group],
        "records": records,
        "trusted_records": trusted_records,
        "trusted_ids": trusted_ids,
        "solvents": solvents,
        "values": values,
        "selected_ids": selected_ids,
        "summary": {
            "group": group,
            "policy": ("released Bamboo generator checkpoint" if group == "G0" else
                       "random/diverse 62-solvent pool" if group == "G1" else
                       "fragment/property guided 62-solvent pool" if group == "G2" else
                       "G2 generator + robustness/explanation re-ranking"),
            "n_sampled": len(records),
            "n_trusted": len(trusted_ids),
            "n_out_of_domain": len(records) - len(trusted_ids),
            "target_mS_cm": target,
            "all_hit_count": int(np.sum(values >= target)),
            "all_hit_rate": float(np.mean(values >= target)),
            "near_target_rate_pm2": float(np.mean(np.abs(values - target) <= 2.0)),
            "mean_abs_target_gap_mS_cm": float(np.mean(np.abs(values - target))),
            "selected_budget": len(selected_ids),
            "selected_hit_count": int(np.sum(selected_values >= target)),
            "selected_hit_rate": float(np.mean(selected_values >= target)),
            "selected_mean_mS_cm": float(np.mean(selected_values)),
            "selected_min_mS_cm": float(np.min(selected_values)),
            "all_median_mS_cm": float(np.median(values)),
            "all_p90_mS_cm": float(np.percentile(values, 90)),
            "unique_compositions": len(set(formula_keys)),
            "mean_active_solvents": float(np.mean(solvent_counts)),
            "top_dominant_solvent": Counter(dominant).most_common(1)[0][0],
            "anion_ratio_pred_mae_from_condition": (
                float(np.mean(np.abs(predicted_ratios - condition)))
                if len(predicted_ratios) else ""
            ),
            "anion_ratio_pred_within_pm0.1": (
                float(np.mean(np.abs(predicted_ratios - condition) <= .1))
                if len(predicted_ratios) else ""
            ),
            "selected_anion_ratio_pred_mae_from_condition": (
                float(np.mean(np.abs(selected_ratios - condition)))
                if len(selected_ratios) else ""
            ),
            "selected_anion_ratio_pred_within_pm0.1": (
                float(np.mean(np.abs(selected_ratios - condition) <= .1))
                if len(selected_ratios) else ""
            ),
            "g3_robust_min_median_mS_cm": (
                float(np.median([x for x in robust_min if x is not None]))
                if any(x is not None for x in robust_min) else ""),
        },
        "solvent_counts": np.asarray(solvent_counts),
    }


def timing_summary(root, group):
    path = root / group / "timings.csv"
    if not path.exists():
        return None, []
    rows = read_csv(path)
    stages = defaultdict(float)
    for row in rows:
        if row["detail"] == "executed":
            stages[row["stage"]] += float(row["elapsed_seconds"])
    core = ("candidate_pool", "vocabulary_embedding", "synthetic_formula_oracle",
            "write_train_configs", "train_diffusion", "train_decoder", "sample_and_score")
    summary = {"group": group, "total_executed_seconds": round(sum(stages.values()), 3)}
    for stage in core:
        summary[f"{stage}_seconds"] = round(stages.get(stage, 0.0), 3)
    return summary, rows


def save_heatmap(groups, out, target):
    edges = np.asarray([0, 5, 10, 15, 20, 25, 30, 40, np.inf])
    labels = ["0-5", "5-10", "10-15", "15-20", "20-25", "25-30", "30-40", ">=40"]
    matrix = []
    for result in groups:
        matrix.append(np.histogram(result["values"], bins=edges)[0] / len(result["values"]))
    fig, ax = plt.subplots(figsize=(10, max(3.2, 0.75 * len(groups) + 1.6)))
    image = ax.imshow(np.asarray(matrix), cmap="YlGnBu", vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(np.arange(len(labels)), labels=labels)
    ax.set_yticks(np.arange(len(groups)), labels=[x["group"] for x in groups])
    ax.set_xlabel("Predicted conductivity bin (mS/cm); target starts at 20")
    ax.set_ylabel("Generation experiment")
    ax.set_title("Generated formula conductivity coverage")
    for i, row in enumerate(matrix):
        for j, value in enumerate(row):
            ax.text(j, i, f"{100 * value:.1f}%", ha="center", va="center",
                    fontsize=9, color="black" if value < .55 else "white")
    fig.colorbar(image, ax=ax, label="Fraction of sampled formulas")
    fig.tight_layout()
    fig.savefig(out, dpi=220)
    plt.close(fig)


def save_hit_rate(groups, out):
    x = np.arange(len(groups))
    all_rates = [r["summary"]["all_hit_rate"] * 100 for r in groups]
    selected_rates = [r["summary"]["selected_hit_rate"] * 100 for r in groups]
    fig, ax = plt.subplots(figsize=(9, 5))
    width = .36
    bars_a = ax.bar(x - width / 2, all_rates, width, label="all sampled")
    bars_b = ax.bar(x + width / 2, selected_rates, width, label="equal selection budget")
    ax.bar_label(bars_a, labels=[f"{v:.1f}%" for v in all_rates], padding=3, fontsize=8)
    ax.bar_label(bars_b, labels=[f"{v:.1f}%" for v in selected_rates], padding=3, fontsize=8)
    ax.set_xticks(x, [r["group"] for r in groups])
    ax.set_ylim(0, max([10] + all_rates + selected_rates) * 1.22)
    ax.set_ylabel("Target-hit rate (%)")
    ax.set_title("Share of generated formulas reaching 20 mS/cm")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(out, dpi=220)
    plt.close(fig)


def save_formula_range(groups, out, target):
    fig, axes = plt.subplots(2, 2, figsize=(10, 7.6), sharey=True,
                             squeeze=False)
    for ax, result in zip(axes.flat, groups):
        values = result["values"]
        counts = result["solvent_counts"]
        hit = values >= target
        ax.scatter(counts[~hit], values[~hit], s=18, alpha=.55, c="#6b7280", label="below target")
        ax.scatter(counts[hit], values[hit], s=20, alpha=.7, c="#dc2626", label="target hit")
        ax.axhline(target, color="#dc2626", lw=1, ls="--")
        ax.set_title(result["group"], pad=6)
        ax.set_xlabel("Active solvents")
        ax.set_xticks([1, 2, 3])
        ax.grid(alpha=.2)
    axes[0][0].set_ylabel("Predicted conductivity (mS/cm)")
    axes[1][0].set_ylabel("Predicted conductivity (mS/cm)")
    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=2, frameon=False,
               bbox_to_anchor=(.5, .005))
    fig.suptitle("Generated formula complexity and target coverage", y=.985)
    fig.tight_layout(rect=(0, .07, 1, .95))
    fig.savefig(out, dpi=220, bbox_inches="tight")
    plt.close(fig)


def top_near_target(result, target, limit=10):
    """Keep the closest unique solvent mixtures, not repeated diffusion samples."""
    unique = {}
    for record in result["trusted_records"]:
        label = formula_label(record, result["solvents"], result["group"])
        value = float(record["conductivity_pred_ms_cm"])
        candidate = (abs(value - target), value, record, label)
        if label not in unique or candidate[:2] < unique[label][:2]:
            unique[label] = candidate
    selected = sorted(unique.values(), key=lambda item: item[:2])[:limit]
    rows = []
    for rank, (distance, value, record, label) in enumerate(selected, 1):
        rows.append({
            "rank_by_abs_error_to_target": rank,
            "conductivity_pred_ms_cm": round(value, 4),
            "abs_difference_from_target_mS_cm": round(distance, 4),
            "solvent_formula_smiles_and_molar_ratio": label,
            "anion_ratio_condition": record.get("anion_ratio_condition"),
            "salt_molar_ratio": record.get("salt_molar_ratio"),
            "temperature_c": record.get("temperature_c"),
        })
    return rows


def save_top_near_target(result, out, target):
    rows = top_near_target(result, target)
    if not rows:
        return rows
    values = np.asarray([row["conductivity_pred_ms_cm"] for row in rows])
    labels = [textwrap.fill(row["solvent_formula_smiles_and_molar_ratio"], 60)
              for row in rows]
    fig, ax = plt.subplots(figsize=(14, 9.0))
    y = np.arange(len(rows))
    colors = ["#d97706" if value < target else "#2563eb" for value in values]
    ax.barh(y, values, color=colors, alpha=.82, height=.65)
    ax.axvline(target, color="#dc2626", ls="--", lw=1.4,
               label=f"target = {target:g} mS/cm")
    for pos, row in enumerate(rows):
        ax.text(values[pos] + .08, pos, f"{values[pos]:.2f}", va="center",
                fontsize=9)
    ax.set_yticks(y, labels=labels, fontsize=7)
    ax.invert_yaxis()
    ax.set_xlim(0, max(target * 1.16, values.max() * 1.16))
    ax.set_xlabel("Frozen-oracle predicted conductivity (mS/cm)")
    ax.set_title(f"{result['group']}: 10 unique generated formulas closest to {target:g} mS/cm")
    ax.grid(axis="x", alpha=.25)
    ax.legend(frameon=False, loc="lower right")
    fig.subplots_adjust(left=.48, right=.96, top=.92, bottom=.08)
    fig.savefig(out, dpi=240, bbox_inches="tight")
    plt.close(fig)
    return rows


def conductivity_anion_density(result):
    """Fixed-axis density; normalizing by all samples exposes OOD loss."""
    pairs = [(float(row["conductivity_pred_ms_cm"]),
              float(row["anion_ratio_pred"]))
             for row in result["trusted_records"]
             if row.get("anion_ratio_pred") is not None
             and np.isfinite(float(row["anion_ratio_pred"]))]
    if not pairs:
        return None
    values, ratios = (np.asarray(item) for item in zip(*pairs))
    x_edges = np.linspace(0.0, MAX_TRUSTED_CONDUCTIVITY, 31)
    y_edges = np.linspace(0.0, 1.0, 26)
    counts, _, _ = np.histogram2d(values, ratios, bins=(x_edges, y_edges))
    return counts.T / len(result["records"]), x_edges, y_edges


def save_anion_ratio_distribution(result, out, target):
    """Density map of oracle-predicted conductivity and anion ratio."""
    density = conductivity_anion_density(result)
    if density is None:
        return False
    matrix, x_edges, y_edges = density
    matrix = np.ma.masked_less_equal(matrix, 0.0)
    fig, ax = plt.subplots(figsize=(7.4, 5.6))
    image = ax.pcolormesh(x_edges, y_edges, matrix, cmap=DENSITY_CMAP,
                          shading="auto")
    colorbar = fig.colorbar(image, ax=ax, pad=.02)
    colorbar.set_label("Density (fraction of all generated formulas per bin)")
    ax.axvline(target, color="#dc2626", ls="--", lw=1.2,
               label=f"target = {target:g} mS/cm")
    condition = result["trusted_records"][0].get("anion_ratio_condition")
    if condition is not None:
        ax.axhline(float(condition), color="white", ls=":", lw=1.4,
                   label=f"generation condition = {float(condition):g}")
    ax.set_xlim(0, MAX_TRUSTED_CONDUCTIVITY)
    ax.set_ylim(0, 1)
    ax.set_xlabel("Frozen-oracle predicted conductivity (mS/cm)")
    ax.set_ylabel("Frozen-oracle predicted anion ratio")
    ax.set_title(f"{result['group']}: conductivity–anion-ratio density")
    ax.grid(alpha=.22)
    ax.legend(frameon=False, fontsize=8)
    ax.text(.02, .97,
            f"trusted / sampled = {len(result['trusted_records'])} / {len(result['records'])}",
            transform=ax.transAxes, va="top", ha="left", fontsize=8,
            bbox={"facecolor": "white", "alpha": .75, "edgecolor": "none"})
    fig.tight_layout()
    fig.savefig(out, dpi=240, bbox_inches="tight")
    plt.close(fig)
    return True


def save_anion_ratio_density_comparison(results, out, target):
    """Shared-scale density panels for comparing the generated regions."""
    prepared = [(result, conductivity_anion_density(result)) for result in results]
    prepared = [(result, item) for result, item in prepared if item is not None]
    if not prepared:
        return False
    vmax = max(float(item[0].max()) for _, item in prepared)
    fig, axes = plt.subplots(1, len(prepared), figsize=(5.5 * len(prepared), 4.9),
                             sharex=True, sharey=True, squeeze=False)
    image = None
    for ax, (result, (matrix, x_edges, y_edges)) in zip(axes.flat, prepared):
        matrix = np.ma.masked_less_equal(matrix, 0.0)
        image = ax.pcolormesh(x_edges, y_edges, matrix, cmap=DENSITY_CMAP,
                              shading="auto", vmin=0, vmax=vmax)
        ax.axvline(target, color="#dc2626", ls="--", lw=1.1)
        condition = result["trusted_records"][0].get("anion_ratio_condition")
        if condition is not None:
            ax.axhline(float(condition), color="white", ls=":", lw=1.2)
        ax.set_title(result["group"])
        ax.set_xlim(0, MAX_TRUSTED_CONDUCTIVITY)
        ax.set_ylim(0, 1)
        ax.grid(alpha=.18)
        ax.set_xlabel("Predicted conductivity (mS/cm)")
    axes[0][0].set_ylabel("Frozen-oracle predicted anion ratio")
    fig.subplots_adjust(left=.07, right=.87, bottom=.14, top=.82, wspace=.10)
    colorbar_axis = fig.add_axes([.89, .19, .015, .56])
    colorbar = fig.colorbar(image, cax=colorbar_axis)
    colorbar.set_label("Density (fraction of all generated formulas per bin)")
    fig.suptitle("G0–G2 generated conductivity–anion-ratio coverage", y=.96)
    fig.savefig(out, dpi=240, bbox_inches="tight")
    plt.close(fig)
    return True


def save_runtime(timings, out):
    stages = ("candidate_pool", "vocabulary_embedding", "synthetic_formula_oracle",
              "train_diffusion", "train_decoder", "sample_and_score")
    colors = plt.get_cmap("tab20")(np.linspace(0, 1, len(stages)))
    fig, ax = plt.subplots(figsize=(10, 4.8))
    bottom = np.zeros(len(timings))
    for stage, color in zip(stages, colors):
        values = np.asarray([row.get(f"{stage}_seconds", 0.) for row in timings]) / 60
        ax.bar([row["group"] for row in timings], values, bottom=bottom, label=stage.replace("_", " "), color=color)
        bottom += values
    for i, value in enumerate(bottom):
        ax.text(i, value, f"{value:.1f} min", ha="center", va="bottom", fontsize=8)
    ax.set_ylabel("Executed wall-clock time (min)")
    ax.set_title("Generation experiment runtime by stage")
    ax.legend(frameon=False, bbox_to_anchor=(1.02, 1), loc="upper left")
    fig.tight_layout()
    fig.savefig(out, dpi=220, bbox_inches="tight")
    plt.close(fig)


def markdown_table(rows, columns):
    lines = ["| " + " | ".join(columns) + " |",
             "| " + " | ".join(["---"] * len(columns)) + " |"]
    for row in rows:
        lines.append("| " + " | ".join(str(row.get(key, "")) for key in columns) + " |")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment-root", default="run/generation_experiments")
    parser.add_argument("--target-conductivity", type=float, default=20.)
    parser.add_argument("--selection-budget", type=int, default=100)
    parser.add_argument("--groups", nargs="+", choices=GROUPS,
                        default=list(GROUPS),
                        help="Groups to summarize; defaults to every available group.")
    args = parser.parse_args()
    root = Path(args.experiment_root)
    root.mkdir(parents=True, exist_ok=True)
    results = [summarize_group(root, group, args.target_conductivity, args.selection_budget)
               for group in args.groups]
    results = [x for x in results if x is not None]
    if not results:
        raise SystemExit(f"No generated_formulas.json found under {root}")
    summary = [x["summary"] for x in results]
    write_csv(root / "generation_experiment_summary.csv", summary)
    timing_rows = []
    for group in args.groups:
        timing, _ = timing_summary(root, group)
        if timing is not None:
            timing_rows.append(timing)
    if timing_rows:
        write_csv(root / "generation_timing_summary.csv", timing_rows)
    save_heatmap(results, root / "conductivity_coverage_heatmap.png", args.target_conductivity)
    save_hit_rate(results, root / "target_hit_rate.png")
    save_formula_range(results, root / "generated_formula_range.png", args.target_conductivity)
    top_figures = []
    anion_figures = []
    for result in results:
        group = result["group"]
        top_rows = save_top_near_target(
            result, root / f"top10_near_{args.target_conductivity:g}_mS_cm_{group}.png",
            args.target_conductivity)
        if top_rows:
            csv_path = root / f"top10_near_{args.target_conductivity:g}_mS_cm_{group}.csv"
            write_csv(csv_path, top_rows)
            top_figures.append((group, csv_path.name,
                                f"top10_near_{args.target_conductivity:g}_mS_cm_{group}.png"))
        anion_path = root / f"conductivity_anion_density_{group}.png"
        if save_anion_ratio_distribution(result, anion_path,
                                         args.target_conductivity):
            anion_figures.append((group, anion_path.name))
    density_comparison = root / "conductivity_anion_density_comparison.png"
    has_density_comparison = save_anion_ratio_density_comparison(
        results, density_comparison, args.target_conductivity)
    if timing_rows:
        save_runtime(timing_rows, root / "generation_runtime.png")

    design_rows = [{"group": result["group"],
                    "生成器/候选池": GROUP_DESIGNS[result["group"]]["source"],
                    "研究目标": GROUP_DESIGNS[result["group"]]["goal"]}
                   for result in results]
    report = [
        "# G0–G2 联合条件生成实验结果", "",
        f"共同采样条件：目标电导率 **{args.target_conductivity:g} mS/cm**、条件阴离子比例 **0.5**、"
        "温度 25 °C、salt_molar_ratio 0.10；每组生成 1024 个候选并由同一冻结配方 oracle 复评分。",
        "",
        "## 研究目标与分组设计", "",
        markdown_table(design_rows, ["group", "生成器/候选池", "研究目标"]), "",
        "G0 不重训生成器；G1/G2 都使用 62 个溶剂和 2 个盐的候选词表、30,000 条合成配方标签与同规格 diffusion/decoder。"
        "G1 的异常 oracle 标签采用数值回退保留，G2 则采用严格的联合条件可信样本筛选；因此二者是候选池策略对照，但应同时报告域外比例，不能只比较命中率。",
        "",
        "## 指标汇总", "",
        markdown_table(summary, ["group", "n_sampled", "n_trusted", "n_out_of_domain", "all_hit_rate", "near_target_rate_pm2", "mean_abs_target_gap_mS_cm", "selected_budget", "selected_hit_rate", "all_p90_mS_cm", "unique_compositions", "anion_ratio_pred_mae_from_condition", "anion_ratio_pred_within_pm0.1"]),
        "",
        "### 参数与指标含义", "",
        "| 参数/指标 | 含义与使用意义 |",
        "| --- | --- |",
        "| target conductivity | 扩散模型外部条件中的目标电导率；本次为 20 mS/cm。它不是实验实测值，而是生成方向。 |",
        "| anion_ratio_condition | 扩散模型外部条件中的阴离子比例；本次固定 0.5。 |",
        "| conductivity_pred_ms_cm | 冻结 oracle 对解码后配方给出的物理空间电导率预测，用于统一比较。 |",
        "| anion_ratio_pred | 冻结 oracle 对同一配方推断的阴离子比例，用于检查生成条件是否被保持。 |",
        "| temperature_c / salt_molar_ratio | 送入 oracle 复评分的温度与盐浓度条件；本次统一固定为 25 °C 与 0.10，故不是本轮生成器主动探索的变量。 |",
        "| bow | decoder 输出的候选词表组成向量；前段为溶剂比例、后段为盐比例。它用于还原具体配方与统计组成多样性。 |",
        "| candidate_rank_score | 候选排序分数；G0–G2 当前等同或主要来自 oracle 电导率评分，用于选出有限实验预算下的前 100 个，不能视为独立实验标签。 |",
        "| prediction_status | `trusted` 表示可用于定量比较；`out_of_domain` 表示电导率非有限或超出 0–100 mS/cm，保留在原始生成数中但不参与密度与命中统计。 |",
        "| explanation_residual_25c | 当前解释头在 25 °C 给出的电导率残差项；本报告未以其作为 G0–G2 主排序标准，只保留用于后续机制复核。 |",
        "| n_sampled / n_trusted / n_out_of_domain | 总生成数 / 电导率在 0–100 mS/cm 且数值有限的可信数 / 其余域外数。域外数高意味着候选更可能超出 oracle 的可靠适用域。 |",
        "| all_hit_rate | 所有可信样本中 conductivity_pred_ms_cm ≥ 20 的比例，衡量原始生成分布产生达标候选的能力。 |",
        "| near_target_rate_pm2 | 电导率落在 [18, 22] mS/cm 的比例，衡量是否真正围绕目标，而不是单纯越高越好。 |",
        "| mean_abs_target_gap_mS_cm | 与 20 mS/cm 的平均绝对偏差；越小表示整体更接近指定目标。 |",
        "| selected_hit_rate | 按 candidate_rank_score 选前 100 个后的达标比例，衡量有限实验预算下的排序价值。 |",
        "| unique_compositions | 按活性溶剂与比例去重后的配方数，衡量生成多样性而非重复抽样。 |",
        "| anion_ratio_pred_mae_from_condition / within_pm0.1 | 预测阴离子比例距条件 0.5 的平均偏差 / 落在 0.5±0.1 内的比例，用于检查联合条件可控性。 |",
        "",
        "![Conductivity coverage](conductivity_coverage_heatmap.png)", "",
        "![Target hit rate](target_hit_rate.png)", "",
        "![Formula range](generated_formula_range.png)", "",
        "这些补充图分别展示电导率区间覆盖、达标率与有效溶剂数—电导率关系；其中达标率必须结合可信样本数与域外比例一起解释。",
        "",
        "## 电导率—阴离子比例二维密度", "",
        "横轴为冻结 oracle 预测电导率，纵轴为 oracle 预测阴离子比例；红虚线是 20 mS/cm，白点虚线是输入条件 0.5。"
        "色条表示每个二维 bin 占全部 1024 个生成配方的比例，故域外样本不会因重新归一化而被隐藏。",
        "",
    ]
    if has_density_comparison:
        report.extend(["![G0–G2 density comparison](conductivity_anion_density_comparison.png)", ""])
    for group, figure in anion_figures:
        report.extend([f"### {group}", f"![{group} density]({figure})", ""])
    report.extend([
        "## 目标附近的具体配方", "",
        "每组先按组成去重，再取预测电导率最接近 20 mS/cm 的 10 个配方。G0 使用 Bamboo 词表名称；G1/G2 使用 SMILES，括号内为该溶剂在配方中的摩尔比例。"
        "图采用单行换行标签和加宽左边距，避免不同配方文字重叠。",
        "",
    ])
    for group, csv_name, figure in top_figures:
        report.extend([f"### {group}", f"![{group} top-10]({figure})", f"配方明细：[CSV]({csv_name})", ""])
    report.extend([
        "## 运行时间", "",
        markdown_table(timing_rows, ["group", "candidate_pool_seconds", "vocabulary_embedding_seconds", "synthetic_formula_oracle_seconds", "train_diffusion_seconds", "train_decoder_seconds", "sample_and_score_seconds", "total_executed_seconds"]) if timing_rows else "尚无 timing CSV。",
        "",
        "![Runtime](generation_runtime.png)" if timing_rows else "",
        "",
        "## 解释边界", "",
        "G2 的筛选来自片段归因和分子物性软约束，而非直接用测试集电导率标签选分子。当前合成标签与最终复评分均来自同一个冻结 oracle，故报告用于 G0–G2 的内部设计比较；最终实验结论仍需独立 predictor 或实验验证。",
    ])
    (root / "generation_experiment_report_zh.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    print(f"Wrote summary, timing table, figures, and report to {root}")


if __name__ == "__main__":
    main()
