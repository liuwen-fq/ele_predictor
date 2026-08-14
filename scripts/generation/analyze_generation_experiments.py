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


GROUP_LABELS = {
    "G0": "G0 released Bamboo generator",
    "G1": "G1 random/diverse pool",
    "G2": "G2 explain-guided pool",
    "G3": "G3 robust re-rank (G2)",
}
GROUPS = tuple(GROUP_LABELS)
MAX_TRUSTED_CONDUCTIVITY = 100.0


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


def formula_smiles(row, solvents, threshold=.02):
    """Readable solvent-mixture label using SMILES rather than vocabulary names."""
    bow = np.asarray(row["bow"], dtype=float)
    terms = []
    for index in np.where(bow[:len(solvents)] > threshold)[0]:
        terms.append(f"{solvents[index]['smiles']} ({bow[index]:.2f})")
    if not terms:
        index = int(np.argmax(bow[:len(solvents)]))
        terms.append(f"{solvents[index]['smiles']} ({bow[index]:.2f})")
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
    rank = np.asarray([
        float(records[index].get("candidate_rank_score",
                                 records[index]["conductivity_pred_ms_cm"]))
        for index in trusted_ids
    ])
    selected_positions = np.argsort(-rank)[:min(selection_budget, len(trusted_ids))]
    selected_ids = trusted_ids[selected_positions]
    selected_values = values[selected_positions]
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
        label = formula_smiles(record, result["solvents"])
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


def save_anion_ratio_distribution(result, out, target):
    """Plot oracle-predicted anion ratio against conductivity per group."""
    pairs = [(float(row["conductivity_pred_ms_cm"]),
              float(row["anion_ratio_pred"]))
             for row in result["trusted_records"]
             if row.get("anion_ratio_pred") is not None
             and np.isfinite(float(row["anion_ratio_pred"]))]
    if not pairs:
        return False
    values, ratios = (np.asarray(item) for item in zip(*pairs))
    hit = values >= target
    fig, ax = plt.subplots(figsize=(7.2, 5.4))
    ax.scatter(values[~hit], ratios[~hit], s=22, alpha=.48, c="#6b7280",
               label="below target")
    ax.scatter(values[hit], ratios[hit], s=25, alpha=.70, c="#dc2626",
               label="target hit")
    ax.axvline(target, color="#dc2626", ls="--", lw=1.2,
               label=f"target = {target:g} mS/cm")
    condition = result["trusted_records"][0].get("anion_ratio_condition")
    if condition is not None:
        ax.axhline(float(condition), color="#2563eb", ls=":", lw=1.4,
                   label=f"generation condition = {float(condition):g}")
    ax.set_xlim(left=0)
    ax.set_ylim(-.02, 1.02)
    ax.set_xlabel("Frozen-oracle predicted conductivity (mS/cm)")
    ax.set_ylabel("Frozen-oracle predicted anion ratio")
    ax.set_title(f"{result['group']}: generated formula distribution")
    ax.grid(alpha=.22)
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
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
    args = parser.parse_args()
    root = Path(args.experiment_root)
    root.mkdir(parents=True, exist_ok=True)
    results = [summarize_group(root, group, args.target_conductivity, args.selection_budget)
               for group in GROUPS]
    results = [x for x in results if x is not None]
    if not results:
        raise SystemExit(f"No generated_formulas.json found under {root}")
    summary = [x["summary"] for x in results]
    write_csv(root / "generation_experiment_summary.csv", summary)
    timing_rows = []
    for group in GROUPS:
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
        anion_path = root / f"anion_ratio_vs_conductivity_{group}.png"
        if save_anion_ratio_distribution(result, anion_path,
                                         args.target_conductivity):
            anion_figures.append((group, anion_path.name))
    if timing_rows:
        save_runtime(timing_rows, root / "generation_runtime.png")

    report = [
        "# G0–G3 生成实验结果汇总", "",
        f"共同条件：目标电导率 **{args.target_conductivity:g} mS/cm**；条件阴离子比例 **0.5**；"
        "生成时温度 25 °C、salt_molar_ratio 0.10。G3 额外在 25–55 °C、0.05–0.15 的网格上复核。",
        "",
        "## G0–G3 的设计与区别", "",
        "四组在目标电导率、温度、盐浓度、候选数和冻结配方 oracle 上保持一致；差别只来自候选池构建方式、生成器来源或最终排序规则。"
        "当前阴离子比例条件固定为 0.5。",
        "",
        "| 组别 | 生成器/候选池 | 核心目的 | 与其余组的可比点 |",
        "| --- | --- | --- | --- |",
        "| G0 | Bamboo 发布的 diffusion + decoder checkpoint；Bamboo 原始 62 溶剂、2 盐词表 | 作为不使用本研究解释约束的基线 | 不重训 generator；用同一冻结 oracle 复评分 |",
        "| G1 | 从分子预训练数据随机抽取并做结构多样化的 62 分子候选池；同规格 generator 重训 | 评估仅扩大/随机选择分子库的效果 | 与 G2 使用相同词表大小、合成样本数、训练上限和采样数 |",
        "| G2 | 用片段贡献与分子物性软约束筛出的 62 分子候选池；同规格 generator 重训 | 检验解释导出的结构/物性约束能否提升命中质量 | 不直接用测试集电导率标签筛分子 |",
        "| G3 | 复用 G2 generator 与候选池，不新训练；按 25–55 °C、0.05–0.15 浓度网格的最差电导率，并以解释方向稳定性同分决策 | 检验鲁棒重排序能否提高候选可用性 | 原始生成分布与 G2 相同，只有最终候选排序不同 |",
        "",
        "",
        "## 电导率与命中率", "",
        markdown_table(summary, ["group", "policy", "n_sampled", "n_trusted", "n_out_of_domain", "all_hit_rate", "selected_budget", "selected_hit_rate", "selected_mean_mS_cm", "all_p90_mS_cm", "unique_compositions", "top_dominant_solvent"]),
        "",
        "### 字段含义", "",
        "| 字段 | 含义与用途 |",
        "| --- | --- |",
        "| group | 实验组编号。G0–G3 的具体设计见上表。 |",
        "| policy | 该组的候选池、生成器来源和排序策略的简述。 |",
        "| n_sampled | 扩散模型实际生成、并送入冻结 oracle 评分的配方总数。 |",
        "| n_trusted | 预测电导率在 0–100 mS/cm 内的样本数，可用于定量比较。 |",
        "| n_out_of_domain | 预测为非有限数或超过 100 mS/cm 的样本数，视为冻结 oracle 的域外外推并排除。 |",
        "| all_hit_rate | 全部可信生成样本中，预测电导率达到目标 20 mS/cm 的比例，反映原始生成分布的命中能力。 |",
        "| selected_budget | 各组用于后续筛选比较的相同候选数；本实验固定为 100。 |",
        "| selected_hit_rate | 按该组既定排序规则选出的前 `selected_budget` 个候选中，达到目标的比例，反映筛选策略的价值。 |",
        "| selected_mean_mS_cm | 入选候选的平均预测电导率；仅作辅助描述，不代替命中率或实验验证。 |",
        "| all_p90_mS_cm | 所有可信样本电导率的第 90 百分位数，表示高端 10% 样本的分界。 |",
        "| unique_compositions | 以“活性溶剂及其摩尔比例”去重后的配方数，数值越高说明重复生成越少。 |",
        "| top_dominant_solvent | 全部可信配方中，占摩尔比例最大的溶剂最常出现的词表名称；仅用于观察生成偏好，不代表因果结论。 |",
        "",
        "`all_hit_rate` 衡量**生成**是否容易产生达标配方；`selected_hit_rate` 衡量在相同实验预算下，**排序/筛选**能否把达标候选优先排到前面。"
        "对 G3 而言，后者特别反映温度—浓度鲁棒重排序的价值，不能把它当作原始生成分布改变的证据。",
        "",
        "![Conductivity coverage](conductivity_coverage_heatmap.png)", "",
        "![Target hit rate](target_hit_rate.png)", "",
        "![Formula range](generated_formula_range.png)", "",
        "### 图的解读", "",
        "- `conductivity_coverage_heatmap.png`：每行是一组，颜色/数值表示该组可信生成配方落入各电导率区间的比例；20 mS/cm 及右侧区间为达标区。",
        "- `target_hit_rate.png`：左柱为所有可信样本的达标比例（生成能力），右柱为统一取前 100 个候选后的达标比例（筛选能力）。它回答的是“在相同后续实验预算下，哪组更容易把达标配方排到前面”。",
        "- `generated_formula_range.png`：横坐标为配方中有效溶剂数，纵坐标为冻结 oracle 预测电导率；灰点未达标、红点达标、红虚线为 20 mS/cm。它用于查看高电导率是否只来自单溶剂，还是也可由二元/三元溶剂体系得到。图已改为 2×2 面板并将图例移到底部，避免标题和标注重合。",
        "",
        "## 目标附近的具体生成配方", "",
        "下图对每组先按 SMILES+摩尔比例去重，再选取预测电导率与 20 mS/cm 绝对差最小的 10 个配方。"
        "橙色表示略低于目标、蓝色表示略高于目标；虚线是 20 mS/cm。图和 CSV 均用 SMILES 表示溶剂，不依赖词表名称。",
        "",
        *sum(([f"### {group}", f"![{group} top-10]({figure})", f"配方数据：[CSV]({csv_name})", ""]
               for group, csv_name, figure in top_figures), []),
        "## 阴离子比例—电导率分布", "",
        "下图复现 Bamboo 图 3b 的输出逻辑：横轴为冻结 oracle 预测电导率，纵轴为同一 oracle 对生成配方输出的 `anion_ratio_pred`。"
        "蓝色虚线是 diffusion 的输入条件 0.5；散点纵向分散并不表示 diffusion 在生成时自由采样了条件值，而是表示不同生成配方经配方预测器推断出的阴离子比例不同。"
        "因此它可用于检查“条件为 0.5 时，解码出的组成是否仍集中在该比例附近”，但不能替代多条件（0.3/0.5/0.7）生成实验。",
        "",
        *sum(([f"### {group}", f"![{group} anion-ratio distribution]({figure})", ""]
               for group, figure in anion_figures), []),
        "若要比较 anion ratio 条件本身的可控性，仍应在多个外部指定条件下重复 G0–G3 采样；对 G1/G2，这需要用多比例的合成训练配方重训 generator。",
        "",
        "## 运行时间", "",
        markdown_table(timing_rows, ["group", "candidate_pool_seconds", "vocabulary_embedding_seconds", "synthetic_formula_oracle_seconds", "train_diffusion_seconds", "train_decoder_seconds", "sample_and_score_seconds", "total_executed_seconds"]) if timing_rows else "尚无 timing CSV。",
        "",
        "![Runtime](generation_runtime.png)" if timing_rows else "",
        "",
        "## 解释与边界", "",
        "G2 的候选池仅以解释报告导出的片段和分子物性软约束筛选，不直接读取测试集电导率标签；G3 使用固定解释头的残差方向稳定性作为同分时的次级排序依据。"
        "生成器的合成训练标签和最终打分目前均来自同一个冻结配方 oracle，因此这是**内部设计比较**，而不是独立泛化性能估计。正式论文比较应再用未参与 generator 标签构造的独立 predictor 或实验验证候选。",
    ]
    (root / "generation_experiment_report_zh.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    print(f"Wrote summary, timing table, figures, and report to {root}")


if __name__ == "__main__":
    main()
