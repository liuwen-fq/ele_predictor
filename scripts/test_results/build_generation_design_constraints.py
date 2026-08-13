#!/usr/bin/env python3
"""Turn split-wise experimental attribution outputs into generation constraints.

The output is deliberately a *soft design specification*, not a catalogue of
recommended molecules.  Structural motifs nominate chemical directions, while
molecular-property windows reject candidates whose whole-molecule context is
incompatible with the experimentally high-conductivity region.
"""

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from analyze_conductivity_fragment_attributions import (
    physical_property_value, property_unit)
from explain_experimental_fragments import conductivity_ms_cm


SPLITS = ("train", "valid", "test")
DIRECT_PROPERTIES = ("dc", "vis", "Tm")
PROPERTY_NAMES = {
    "dc": "介电常数", "vis": "黏度", "Tm": "熔点", "ST": "表面张力",
    "bp": "沸点", "density": "密度", "nD": "折射率",
    "nD_liquid": "液相折射率", "pka_a": "pKa", "pka_b": "pKb",
    "vapP": "蒸气压",
}


def number(row, key):
    value = row.get(key, 0.0)
    return float(value) if value not in (None, "") else 0.0


def read_csv(path):
    with Path(path).open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path, rows):
    if not rows:
        Path(path).write_text("", encoding="utf-8")
        return
    with Path(path).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def percentile(values, q):
    return float(np.quantile(np.asarray(values, dtype=float), q))


def weighted_properties(record):
    profiles = record.get("molecular_property_profiles", [])
    values = defaultdict(float)
    total = 0.0
    for profile in profiles:
        ratio = float(profile.get("molar_ratio", 0.0))
        total += ratio
        for name, value in profile.get("property_values_normalized", {}).items():
            values[name] += ratio * float(value)
    if total <= 0:
        return {}
    return {name: value / total for name, value in values.items()}


def collect_property_windows(inputs):
    """Compute high-group physical descriptor bands in each split separately."""
    per_split = {}
    for split, path in inputs.items():
        records = json.loads(Path(path).read_text())
        labelled = [r for r in records if r.get("conductivity_mask")]
        targets = [conductivity_ms_cm(float(r["conductivity"]))
                   for r in labelled]
        cutoff = percentile(targets, 0.75)
        high = [r for r in labelled if conductivity_ms_cm(
            float(r["conductivity"])) >= cutoff]
        property_values = defaultdict(list)
        contributions = defaultdict(list)
        for record in high:
            for name, value in weighted_properties(record).items():
                property_values[name].append(physical_property_value(name, value))
            for name, value in record.get("property_explanations", {}).get(
                    "property_sum_by_name_normalized", {}).items():
                contributions[name].append(float(value))
        per_split[split] = {
            "count": len(high), "cutoff": cutoff, "values": property_values,
            "contributions": contributions,
        }
    rows = []
    names = sorted({name for data in per_split.values()
                    for name in data["values"]})
    for name in names:
        split_bands = {}
        for split, data in per_split.items():
            values = data["values"].get(name, [])
            if values:
                split_bands[split] = (percentile(values, .10),
                                      percentile(values, .50),
                                      percentile(values, .90))
        # The historical test result predates property-profile serialization.
        # Two independently held-out fitting splits are still sufficient for a
        # provisional soft window, but never for a hard exclusion rule.
        if len(split_bands) < 2:
            continue
        lower = max(v[0] for v in split_bands.values())
        upper = min(v[2] for v in split_bands.values())
        # Empty intersection means that a narrow hard range would overfit a
        # split.  Use the envelope and explicitly flag it as a soft range.
        overlap = lower <= upper
        if not overlap:
            lower = percentile([v[0] for v in split_bands.values()], .50)
            upper = percentile([v[2] for v in split_bands.values()], .50)
        contribution_means = {}
        for split, data in per_split.items():
            terms = data["contributions"].get(name, [])
            if terms:
                contribution_means[split] = float(np.mean(terms))
        signs = {int(np.sign(v)) for v in contribution_means.values()
                 if abs(v) >= 1e-4}
        if name not in DIRECT_PROPERTIES:
            direction = "非直接项：仅作物性可行性筛选"
        elif len(signs) == 1 and signs and len(contribution_means) >= 2:
            direction = "高组中稳定正向" if next(iter(signs)) > 0 else "高组中稳定负向"
        else:
            direction = "条件依赖：不作为硬方向约束"
        row = {
            "property": name,
            "property_chinese_name": PROPERTY_NAMES.get(name, name),
            "property_unit": property_unit(name),
            "recommended_low": lower,
            "recommended_high": upper,
            "range_evidence": (
                f"{len(split_bands)} 个含逐分子物性画像的 split 的 P10–P90 交集"
                if overlap else
                f"{len(split_bands)} 个含逐分子物性画像的 split 的 P10/P90 中位包络（软范围）"),
            "generation_use": direction,
        }
        for split, (lo, med, hi) in split_bands.items():
            row[f"{split}_p10"] = lo
            row[f"{split}_median"] = med
            row[f"{split}_p90"] = hi
            row[f"{split}_high_property_contribution"] = contribution_means.get(split, "")
        rows.append(row)
    return rows, per_split


def structural_constraints(cross_rows):
    rows = []
    for row in cross_rows:
        if row["direction_consistency"] != "三集合方向一致":
            continue
        enrichment = number(row, "mean_high_minus_low_presence")
        direct = number(row, "mean_direct_effect")
        if enrichment > 0 and direct > 0.01:
            use = "优先保留：作为生成奖励项"
        elif enrichment < 0 and direct < -0.005:
            use = "优先规避：作为生成惩罚项"
        elif enrichment > 0:
            use = "条件候选：需由物性与配方评分复核"
        else:
            use = "条件风险：不单独否决，需由物性与配方评分复核"
        rows.append({
            "motif_key": row["motif_key"],
            "fragment_chinese_name": row["fragment_chinese_name"],
            "fragment_group_formula": row["fragment_group_formula"],
            "carrier_solvents_in_dataset": row["solvent_molecules"],
            "mean_high_minus_low_presence": enrichment,
            "mean_direct_effect": direct,
            "generation_use": use,
        })
    return sorted(rows, key=lambda r: (-abs(r["mean_high_minus_low_presence"]),
                                       -abs(r["mean_direct_effect"])))


def plot_motifs(rows, output):
    rows = rows[:14]
    labels = [r["fragment_group_formula"].replace("（环）", " (cyclic)")
              .replace("；", ";") for r in rows]
    x = [r["mean_high_minus_low_presence"] for r in rows]
    y = [r["mean_direct_effect"] for r in rows]
    fig, ax = plt.subplots(figsize=(10, 7))
    colors = ["#c44e52" if a > 0 and b > 0 else "#4c72b0"
              if a < 0 and b < 0 else "#999999" for a, b in zip(x, y)]
    ax.scatter(x, y, s=100, c=colors, edgecolors="white", linewidths=.7)
    for xx, yy, label in zip(x, y, labels):
        ax.annotate(label, (xx, yy), xytext=(5, 3), textcoords="offset points",
                    fontsize=8)
    ax.axhline(0, color="#444", linewidth=.8)
    ax.axvline(0, color="#444", linewidth=.8)
    ax.set_xlabel("Mean high-group presence − low-group presence")
    ax.set_ylabel("Mean direct contribution (normalized log conductivity)")
    ax.set_title("Generation structural constraints from three-split evidence")
    fig.tight_layout()
    fig.savefig(output, format="svg", bbox_inches="tight")
    plt.close(fig)


def plot_properties(rows, output):
    rows = [r for r in rows if r["property"] in DIRECT_PROPERTIES]
    fig, ax = plt.subplots(figsize=(10, 4.8))
    y = np.arange(len(rows))
    for index, row in enumerate(rows):
        lo, hi = float(row["recommended_low"]), float(row["recommended_high"])
        medians = [float(row[f"{s}_median"]) for s in SPLITS
                   if row.get(f"{s}_median", "") not in ("", None)]
        med = np.mean(medians)
        ax.hlines(index, lo, hi, color="#4c72b0", linewidth=8, alpha=.7)
        ax.plot(med, index, "o", color="#c44e52", markersize=7)
        ax.text(hi, index + .16, row["property_unit"], fontsize=8,
                ha="right", va="bottom")
    ax.set_yticks(y, [r["property"] for r in rows])
    ax.set_xlabel("Physical-property value (each row has its own unit/scale)")
    ax.set_title("High-conductivity molecular-property soft windows")
    ax.grid(axis="x", alpha=.25)
    fig.tight_layout()
    fig.savefig(output, format="svg", bbox_inches="tight")
    plt.close(fig)


def markdown(structures, properties, split_data):
    positive = [r for r in structures if r["generation_use"].startswith("优先保留")]
    risk = [r for r in structures if r["generation_use"].startswith("优先规避")]
    lines = [
        "# 面向生成模型的高电导率溶剂设计约束",
        "",
        "## 目的与证据边界",
        "",
        "本报告把现有解释头的结果转换为生成前筛选的**软约束**，而不是直接推荐训练/测试集中出现过的具体溶剂。结构基元来自 train/valid/test 各自实验电导率高/低四分位分析；物性窗口来自含逐分子物性画像的高电导率组的组成加权分子物性 P10–P90。候选分子仍必须在给定盐、浓度、温度、比例下通过配方主模型复核。",
        "",
        "## 使用顺序",
        "",
        "1. 生成器提出候选溶剂分子；2. 分子预训练模型预测其 11 项物性；3. 用下方物性窗口进行软筛选；4. 用结构基元奖励/惩罚重排；5. 组成配方后以 Bamboo 主模型、条件化片段/物性贡献和不确定性做最终筛选。",
        "",
        "## 结构约束：生成奖励与惩罚",
        "",
        "![三集合结构约束](generation_structural_constraints.svg)",
        "",
        "红色右上区域表示在三个数据集均更常出现在高电导率组且平均直接贡献为正，适合作为奖励项；蓝色左下区域表示稳定低组富集且直接贡献为负，适合作为惩罚项。灰色不是无效，而是说明结构效应依赖完整分子、盐或条件，不能凭片段硬筛。",
        "",
        "| 化学基元 | 官能团式 | 数据集承载溶剂（仅作证据，不是推荐清单） | 高−低富集差 | 直接贡献 | 生成用途 |",
        "| --- | --- | --- | ---: | ---: | --- |",
    ]
    for row in structures:
        lines.append(f"| {row['fragment_chinese_name']} | `{row['fragment_group_formula']}` | {row['carrier_solvents_in_dataset']} | {row['mean_high_minus_low_presence']:.3f} | {row['mean_direct_effect']:.4f} | {row['generation_use']} |")
    lines += ["", "## 分子物性软范围", "",
              "![高电导率物性软窗口](generation_property_windows.svg)", "",
              "蓝条是含逐分子物性画像的高电导率组得到的推荐软范围；红点是这些 split 中位数的平均。历史 test 结果没有为带实验标签样本保存逐分子物性画像，故其结构归因仍计入三集合一致性，但不伪装成物性范围证据。各行单位和数值尺度不同，图中不能横向比较条的长度。`dc`、`vis`、`Tm` 还具有解释头直接贡献：只有当可用高组贡献方向一致时才建议作为方向性奖励/惩罚；其余物性仅用于让生成分子保持在训练分布附近。", "",
              "| 性质 | 推荐软范围 | 单位 | 贡献/使用规则 | 证据范围 |", "| --- | ---: | --- | --- | --- |"]
    for row in properties:
        lines.append(f"| {row['property_chinese_name']} (`{row['property']}`) | {float(row['recommended_low']):.4g} – {float(row['recommended_high']):.4g} | {row['property_unit']} | {row['generation_use']} | {row['range_evidence']} |")
    lines += ["", "## 如何避免‘相同片段但分子效果不同’", "",
              "例如环状酯不能作为单一硬规则：BUTY 类与 EC/PC 类共享环状酯相关结构，却有不同的环类型、取代、熔点、黏度和介电常数。生成时应先把环状酯作为候选基元，再用上表的分子物性软范围和最终配方级复核区分具体分子。若某候选仅满足片段奖励、但物性远离窗口，或在目标盐/浓度条件下物性贡献方向不稳定，应降级或剔除。", "",
              "## 推荐评分而非硬阈值", "",
              "建议对每个生成分子计算 `S = 结构奖励 − 结构风险 + 物性窗口相似度 + 直接物性方向一致性 + 配方主模型电导率 − 分布外惩罚`。窗口外的距离可连续扣分；不要因为一个性质略超范围就直接排除，否则会误杀需要共溶剂补偿的 EC 类候选。",
              "",
              "## 数据支撑", "",
              *[f"- {split} 高电导率组：{data['count']} 个配方，组内阈值 ≥ {data['cutoff']:.3f} mS/cm；逐分子物性画像：{'有' if data['values'] else '无'}。" for split, data in split_data.items()],
              "",
              "该约束表是生成模型的先验与筛选器，不是化学因果定律。生成后的分子必须检查合成可行性、安全性、电化学稳定窗口以及与目标盐的相容性。",
    ]
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-input", required=True)
    parser.add_argument("--valid-input", required=True)
    parser.add_argument(
        "--test-input",
        help=("optional test prediction JSON when it contains labelled "
              "per-molecule property profiles; structural constraints always "
              "come from --cross-split-csv"))
    parser.add_argument("--cross-split-csv", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    inputs = {"train": args.train_input, "valid": args.valid_input}
    if args.test_input:
        inputs["test"] = args.test_input
    properties, split_data = collect_property_windows(inputs)
    structures = structural_constraints(read_csv(args.cross_split_csv))
    write_csv(output / "generation_structural_constraints.csv", structures)
    write_csv(output / "generation_property_constraints.csv", properties)
    plot_motifs(structures, output / "generation_structural_constraints.svg")
    plot_properties(properties, output / "generation_property_windows.svg")
    (output / "generation_design_constraints_report_zh.md").write_text(
        markdown(structures, properties, split_data), encoding="utf-8")
    print(f"Wrote generation design constraints to: {output}")


if __name__ == "__main__":
    main()
