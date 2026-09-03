#!/usr/bin/env python3
"""Create paired conductivity/anion explanation figures and a Chinese report.

The input prediction export can be multi-GB, so conductivity summaries are
computed in a streaming pass.  Anion summaries are read from the already
generated compact analysis files.
"""

from __future__ import annotations

import argparse
import csv
import heapq
import json
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from analyze_anion_ratio_explanations import explanation_records, iter_json_array


PROPERTY_LABELS = {"dc": "介电常数 (dc)", "vis": "黏度 (vis)", "Tm": "熔点 (Tm)"}


def update_stat(stats, key, value):
    stats[key]["n"] += 1
    stats[key]["sum"] += float(value)


def write_csv(path: Path, rows: list[dict], fields: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def read_csv(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def collect_conductivity(input_path: Path, max_interactions: int) -> tuple[list[dict], list[dict], list[dict], dict]:
    fragments = defaultdict(lambda: {"n": 0, "sum": 0.0})
    properties = defaultdict(lambda: {"n": 0, "sum": 0.0})
    interactions, counter = [], 0
    records, selected, total_pairs = 0, 0, 0
    for row in iter_json_array(input_path):
        records += 1
        if not row.get("conductivity_mask", False):
            continue
        selected += 1
        for molecule in row.get("fragment_attention", {}).get("solvents", []):
            for fragment in molecule.get("fragments", []):
                if "conductivity_main_effect_normalized" in fragment:
                    update_stat(fragments, fragment.get("smiles", ""),
                                fragment["conductivity_main_effect_normalized"])
        for explanation in explanation_records(
                row.get("property_explanations"), "property_explanations"):
            for molecule in explanation.get("solvent_property_contributions", []):
                for name, value in molecule.get("property_effects_normalized", {}).items():
                    update_stat(properties, name, value)
        for explanation in explanation_records(
                row.get("fragment_explanations"), "fragment_explanations"):
            for pair in explanation.get("top_fragment_pairs", []):
                effect = float(pair["effect_logit"])
                total_pairs += 1
                counter += 1
                candidate = (abs(effect), counter, {
                    "effect_logit": effect,
                    "effect_delta_ln": pair.get("effect_delta_ln"),
                    "left": pair["left"], "right": pair["right"],
                })
                if len(interactions) < max_interactions:
                    heapq.heappush(interactions, candidate)
                elif candidate[0] > interactions[0][0]:
                    heapq.heapreplace(interactions, candidate)
    fragment_rows = [{"fragment_smiles": key, "n": value["n"],
                      "mean_normalized_effect": value["sum"] / value["n"]}
                     for key, value in fragments.items()]
    property_rows = [{"property": key, "n": value["n"],
                      "mean_normalized_effect": value["sum"] / value["n"]}
                     for key, value in properties.items()]
    fragment_rows.sort(key=lambda row: -abs(row["mean_normalized_effect"]))
    property_rows.sort(key=lambda row: row["property"])
    interaction_rows = [record for _, _, record in sorted(
        interactions, key=lambda item: item[0], reverse=True)]
    return fragment_rows, property_rows, interaction_rows, {
        "prediction_records": records,
        "conductivity_labelled_records": selected,
        "fragment_pair_candidates": total_pairs,
        "retained_strongest_fragment_pairs": len(interaction_rows),
    }


def interaction_label(row: dict) -> str:
    return f"{row['left'].get('molecule_name', '?')} × {row['right'].get('molecule_name', '?')}"


def plot_bars(left_rows: list[dict], right_rows: list[dict], left_title: str,
              right_title: str, value_key: str, labeler, output: Path,
              x_label: str, top_n: int = 12) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(15, 7), constrained_layout=True)
    for ax, rows, title in zip(axes, (left_rows, right_rows), (left_title, right_title)):
        selected = sorted(rows, key=lambda row: abs(float(row[value_key])), reverse=True)[:top_n]
        selected = list(reversed(selected))
        values = [float(row[value_key]) for row in selected]
        labels = [labeler(row) for row in selected]
        colors = ["#c62828" if value > 0 else "#1565c0" for value in values]
        ax.barh(range(len(selected)), values, color=colors, alpha=0.85)
        ax.axvline(0, color="black", linewidth=0.8)
        ax.set_yticks(range(len(selected)), labels, fontsize=8)
        ax.set_title(title)
        ax.set_xlabel(x_label)
        ax.grid(axis="x", alpha=0.25)
    fig.savefig(output, dpi=220, bbox_inches="tight")
    plt.close(fig)


def format_table(rows: list[dict], value_key: str, labeler, count_key: str | None = None,
                 limit: int = 10) -> str:
    lines = ["| 项目 | 平均贡献 | 出现数 |", "|---|---:|---:|"]
    for row in sorted(rows, key=lambda item: abs(float(item[value_key])), reverse=True)[:limit]:
        count = row.get(count_key, "—") if count_key else "—"
        lines.append(f"| `{labeler(row)}` | {float(row[value_key]):+.4f} | {count} |")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="ft_test_predictions.json")
    parser.add_argument("--analysis-dir", required=True)
    parser.add_argument("--max-interactions", type=int, default=10000)
    args = parser.parse_args()
    if args.max_interactions <= 0:
        raise ValueError("--max-interactions must be positive")

    analysis_dir = Path(args.analysis_dir)
    analysis_dir.mkdir(parents=True, exist_ok=True)
    conductivity_fragments, conductivity_properties, conductivity_pairs, conductivity_summary = (
        collect_conductivity(Path(args.input), args.max_interactions))
    write_csv(analysis_dir / "conductivity_fragment_contributions.csv",
              conductivity_fragments,
              ["fragment_smiles", "n", "mean_normalized_effect"])
    write_csv(analysis_dir / "conductivity_property_contributions.csv",
              conductivity_properties,
              ["property", "n", "mean_normalized_effect"])
    (analysis_dir / "conductivity_interactions.json").write_text(
        json.dumps(conductivity_pairs, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (analysis_dir / "conductivity_analysis_summary.json").write_text(
        json.dumps(conductivity_summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    anion_fragments = read_csv(analysis_dir / "anion_fragment_contributions.csv")
    for row in anion_fragments:
        row["mean_effect"] = float(row["mean_logit_effect"])
        row["mean_normalized_effect"] = row["mean_effect"]
    anion_properties = read_csv(analysis_dir / "anion_property_contributions.csv")
    for row in anion_properties:
        row["mean_effect"] = float(row["mean_logit_effect"])
    anion_pairs = json.loads((analysis_dir / "anion_interactions.json").read_text(encoding="utf-8"))

    plot_bars(conductivity_fragments, anion_fragments,
              "电导率：片段主贡献", "Anion ratio：片段主贡献",
              "mean_normalized_effect", lambda row: row["fragment_smiles"],
              analysis_dir / "fragment_main_contributions_comparison.png",
              "平均贡献（归一化；正/负为模型方向）")
    # Map anion's column name onto the same plotting interface.
    for row in anion_properties:
        row["mean_normalized_effect"] = row["mean_effect"]
    plot_bars(conductivity_properties, anion_properties,
              "电导率：溶剂物性贡献", "Anion ratio：溶剂物性贡献",
              "mean_normalized_effect", lambda row: PROPERTY_LABELS.get(row["property"], row["property"]),
              analysis_dir / "property_contributions_comparison.png",
              "平均贡献（电导率：归一化 log-σ；anion：logit）", top_n=3)
    plot_bars(conductivity_pairs, anion_pairs,
              "电导率：最强片段交互", "Anion ratio：最强片段交互",
              "effect_logit", interaction_label,
              analysis_dir / "fragment_interactions_comparison.png",
              "单配方交互贡献（归一化尺度）")

    anion_metric = json.loads((analysis_dir / "analysis_provenance.json").read_text(encoding="utf-8"))
    conductivity_metric = json.loads((analysis_dir.parent / "results" /
                                      "conductivity_explain_metrics.json").read_text(encoding="utf-8"))
    report = f"""# 电导率与 Anion Ratio 解释头分析报告

## 模型一致性与适用范围

- Anion-ratio 常规头 / 解释头 MAE：{anion_metric['ordinary']['mae']:.4f} / {anion_metric['explain']['mae']:.4f}；解释头劣化 {anion_metric['mae_degradation']:.4f}，通过预设 0.01 门槛。
- Conductivity 解释头：MAE {conductivity_metric['conductivity_mae_mS_cm']:.3f} mS/cm，R² {conductivity_metric['conductivity_r2']:.4f}（仅实验测试配方）。
- 电导率贡献仅汇总 {conductivity_summary['conductivity_labelled_records']} 条实验标注配方；anion-ratio 贡献使用 MD 标注配方。两者的样本群体不同，适合比较方向与模式，不应把绝对量直接横向等同。

## 图件

1. [片段主贡献对比](fragment_main_contributions_comparison.png)
2. [溶剂物性贡献对比](property_contributions_comparison.png)
3. [片段交互对比](fragment_interactions_comparison.png)

红色表示模型的正向贡献，蓝色表示负向贡献。电导率主贡献与交互使用解释头的归一化 log-conductivity 尺度；anion-ratio 使用 logit 尺度。

## 电导率：片段主贡献（前十）

{format_table(conductivity_fragments, 'mean_normalized_effect', lambda row: row['fragment_smiles'], 'n')}

## Anion ratio：片段主贡献（前十）

{format_table(anion_fragments, 'mean_effect', lambda row: row['fragment_smiles'], 'n')}

## 溶剂物性贡献

电导率与 anion ratio 的物性项均只包含 `dc`、`vis`、`Tm`。它们是条件化解释头的模型关联项，而不是单变量因果效应；方向随盐种、浓度、温度和溶剂配比而改变，图中为测试集合的平均值。

{format_table(conductivity_properties, 'mean_normalized_effect', lambda row: PROPERTY_LABELS.get(row['property'], row['property']), 'n', 3)}

{format_table(anion_properties, 'mean_effect', lambda row: PROPERTY_LABELS.get(row['property'], row['property']), 'n', 3)}

## 片段交互

交互图展示每个目标中绝对贡献最大的单配方片段对，而非全局因果规则。为保证输出规模受控，分别从 {conductivity_summary['fragment_pair_candidates']:,} 个电导率候选 pair 和 anion 分析文件保留的 pair 中，最多保存 {args.max_interactions:,} 条最强记录。

结论应优先基于“跨样本稳定的片段主贡献 + 物性贡献”形成；单个最强交互适合作为后续配方或分子组合的定向验证线索。
"""
    (analysis_dir / "joint_explanation_analysis_report_zh.md").write_text(
        report, encoding="utf-8")
    print(f"Wrote figures and report to: {analysis_dir}")


if __name__ == "__main__":
    main()
