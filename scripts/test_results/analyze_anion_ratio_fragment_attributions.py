#!/usr/bin/env python3
"""Detailed, solvent-traceable anion-ratio fragment attribution analysis.

This is the anion-ratio counterpart of
``analyze_conductivity_fragment_attributions.py``.  It deliberately reports
attachment-aware, hydrogen-free fragment SMILES, rather than the explicit-H
BRICS fragments stored by the model.  It also retains every carrier solvent
and its dc/vis/Tm contribution for each reported chemical motif.

The prediction export can be multiple GB, therefore this module reads it as a
JSON stream and retains only the fields required by the detailed analysis.
"""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
from typing import Any

from analyze_anion_ratio_explanations import iter_json_array
import analyze_conductivity_fragment_attributions as base
from explain_experimental_fragments import percentile


def _required_record(record: dict[str, Any]) -> dict[str, Any]:
    """Make an analysis-only view and map anion fields to shared helpers."""

    anion_fragment = record.get("anion_ratio_fragment_explanations")
    anion_property = record.get("anion_ratio_property_explanations")
    if not isinstance(anion_fragment, dict) or not isinstance(anion_property, dict):
        raise ValueError("anion explanation objects are absent from prediction JSON")
    attention = copy.deepcopy(record.get("fragment_attention", {}))
    for molecule in attention.get("solvents", []):
        for fragment in molecule.get("fragments", []):
            if "anion_ratio_main_effect_logit" in fragment:
                # The shared detailed functions use this field name.  Values
                # remain in the anion head's native logit space.
                fragment["conductivity_main_effect_normalized"] = float(
                    fragment["anion_ratio_main_effect_logit"])
    fragment_explanation = copy.deepcopy(anion_fragment)
    for pair in fragment_explanation.get("top_fragment_pairs", []):
        if "effect_logit" in pair:
            pair["effect_normalized"] = float(pair["effect_logit"])
    return {
        "solvents": record.get("solvents", []),
        "salts": record.get("salts", []),
        "salt_molar_ratio": record.get("salt_molar_ratio", 0.0),
        "temperature": record.get("temperature", float("nan")),
        "anion_ratio": float(record["anion_ratio"]),
        "anion_ratio_mask": bool(record.get("anion_ratio_mask", False)),
        "anion_ratio_pred": record.get("anion_ratio_pred"),
        "anion_ratio_explain_pred": record.get("anion_ratio_explain_pred"),
        "fragment_attention": attention,
        "fragment_explanations": fragment_explanation,
        "property_explanations": copy.deepcopy(anion_property),
        "molecular_property_profiles": record.get("molecular_property_profiles", []),
    }


def load_samples(path: str, quantile: float) -> tuple[list[dict[str, Any]], float, float]:
    """Load only labelled records and group them by observed anion ratio."""

    samples: list[dict[str, Any]] = []
    for index, record in enumerate(iter_json_array(path)):
        if index and index % 1000 == 0:
            print(f"Read {index} prediction records", flush=True)
        if not record.get("anion_ratio_mask", False):
            continue
        compact = _required_record(record)
        samples.append({
            "sample_index": index,
            "record": compact,
            "target_anion_ratio": compact["anion_ratio"],
            # Compatibility field consumed by the shared row collectors; it
            # carries the raw anion-ratio label, not a conductivity value.
            "target_ms_cm": compact["anion_ratio"],
            # Keep the field names expected by the shared carrier-property
            # helpers.  They are not interpreted as conductivity here.
            "concentration": float(compact["salt_molar_ratio"]),
            "temperature_c": float(compact["temperature"]),
            "bamboo_prediction_ms_cm": 0.0,
        })
    if not samples:
        raise ValueError("no anion-ratio-labelled records were found")
    values = [sample["target_anion_ratio"] for sample in samples]
    low_cutoff, high_cutoff = percentile(values, quantile), percentile(values, 1 - quantile)
    for sample in samples:
        value = sample["target_anion_ratio"]
        # The shared aggregation functions retain these internal labels; all
        # exported columns/report text rename them to high/low anion ratio.
        sample["conductivity_group"] = (
            "低电导率" if value <= low_cutoff else
            "高电导率" if value >= high_cutoff else "中间电导率")
    return samples, low_cutoff, high_cutoff


def _rename_columns(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    replacements = {
        "conductivity": "anion_ratio",
        "main_effect_normalized": "main_effect_logit",
        "interaction_effect_normalized": "interaction_effect_logit",
        "effect_normalized": "effect_logit",
    }
    output = []
    for row in rows:
        output.append({
            next((key.replace(old, new) for old, new in replacements.items() if old in key), key): value
            for key, value in row.items()
        })
    return output


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    base.write_csv(path, _rename_columns(rows))


def _replace_svg_titles(path: Path) -> None:
    """Make terminology correct and SVG rendering font-independent.

    Labels intentionally remain ASCII/SMILES/solvent identifiers: several
    batch nodes lack any CJK font, which is the cause of the square glyphs.
    Chinese interpretation belongs in the Markdown report, where the IDE/web
    renderer has a proper fallback font.
    """

    text = path.read_text()
    text = text.replace("high-conductivity", "high-anion-ratio")
    text = text.replace("low-conductivity", "low-anion-ratio")
    text = text.replace("High-conductivity", "High-anion-ratio")
    text = text.replace("Low-conductivity", "Low-anion-ratio")
    text = text.replace("log-conductivity", "anion-ratio logit")
    text = text.replace("conductivity", "anion ratio")
    text = text.replace('font-family="Arial, sans-serif"',
                        'font-family="Arial, Helvetica, sans-serif"')
    path.write_text(text)


def make_figures(output_dir: Path, motifs: list[dict[str, Any]], pairs: list[dict[str, Any]],
                 profiles: list[dict[str, Any]], min_formulas: int) -> None:
    high = base.candidates(motifs, "high", min_formulas, "high_mean_main_effect_normalized")
    low = base.candidates(motifs, "low", min_formulas, "low_mean_main_effect_normalized")
    high_pairs = base.candidates(pairs, "high", min_formulas, "high_mean_interaction_normalized")
    low_pairs = base.candidates(pairs, "low", min_formulas, "low_mean_interaction_normalized")
    base.write_svg(output_dir / "high_low_fragment_direct_contributions.svg", 1900, 590,
        base.svg_bar_panel(high, "high_mean_main_effect_normalized", base.plot_label_fragment,
                           "High-anion-ratio common motifs", 15, 15, 920) +
        base.svg_bar_panel(low, "low_mean_main_effect_normalized", base.plot_label_fragment,
                           "Low-anion-ratio common motifs", 965, 15, 920,
                           zero_fraction=0.3, label_width=450))
    base.write_svg(output_dir / "high_low_fragment_pair_interactions.svg", 1900, 590,
        base.svg_bar_panel(high_pairs, "high_mean_interaction_normalized", base.plot_label_pair,
                           "High-anion-ratio fragment-pair interactions", 15, 15, 920) +
        base.svg_bar_panel(low_pairs, "low_mean_interaction_normalized", base.plot_label_pair,
                           "Low-anion-ratio fragment-pair interactions", 965, 15, 920))
    base.make_motif_property_figures(output_dir, high, low, profiles)
    # Enrichment versus signed direct contribution: same calculation as the
    # established conductivity report, without an irrelevant concentration plot.
    eligible = [row for row in motifs if row["unique_formula_count"] >= min_formulas]
    width, height, margin = 1000, 700, 75
    x_values = [r["high_presence_rate"] - r["low_presence_rate"] for r in eligible]
    y_values = [r["mean_main_effect_normalized"] for r in eligible]
    x_span = max(max((abs(v) for v in x_values), default=.1), .1)
    y_span = max(max((abs(v) for v in y_values), default=.1), .1)
    cx, cy = width / 2, height / 2
    xs, ys = (width / 2 - margin) / x_span, (height / 2 - margin) / y_span
    elements = [base.svg_text(cx, 28, "Motif enrichment versus direct anion-ratio attribution", 17, "middle", "bold"),
                f'<line x1="{margin}" y1="{cy}" x2="{width-margin}" y2="{cy}" stroke="#222"/>',
                f'<line x1="{cx}" y1="{margin}" x2="{cx}" y2="{height-margin}" stroke="#222"/>',
                base.svg_text(cx, height - 20, "High minus low anion-ratio presence rate", 13, "middle"),
                base.svg_text(18, 40, "Mean direct attribution (logit)", 13)]
    coordinates = {}
    for row in eligible:
        x, y = row["high_presence_rate"] - row["low_presence_rate"], row["mean_main_effect_normalized"]
        px, py = cx + x * xs, cy - y * ys
        coordinates[row["fragment_smiles"]] = (px, py)
        radius = min(14, 3 + .45 * row["unique_formula_count"] ** .5)
        elements.append(f'<circle cx="{px:.1f}" cy="{py:.1f}" r="{radius:.1f}" fill="{"#c74343" if y > 0 else "#356ea2"}" fill-opacity=".68" stroke="white"/>')
    selected = sorted(eligible, key=lambda r: -r["unique_formula_count"])[:12]
    elements.extend(base.non_overlapping_scatter_labels(selected, coordinates))
    base.write_svg(output_dir / "fragment_enrichment_vs_direct_contribution.svg", width, height, elements)
    for svg in output_dir.glob("*.svg"):
        _replace_svg_titles(svg)


def report(samples: list[dict[str, Any]], low_cutoff: float, high_cutoff: float,
           motifs: list[dict[str, Any]], pairs: list[dict[str, Any]], properties: list[dict[str, Any]],
           profiles: list[dict[str, Any]], min_formulas: int) -> str:
    high_total = sum(s["conductivity_group"] == "高电导率" for s in samples)
    low_total = sum(s["conductivity_group"] == "低电导率" for s in samples)
    high = base.candidates(motifs, "high", min_formulas, "high_mean_main_effect_normalized")
    low = base.candidates(motifs, "low", min_formulas, "low_mean_main_effect_normalized")
    high_pairs = base.candidates(pairs, "high", min_formulas, "high_mean_interaction_normalized")
    low_pairs = base.candidates(pairs, "low", min_formulas, "low_mean_interaction_normalized")
    lines = [
        "# Anion ratio 条件感知片段归因报告", "",
        "## 数据范围与表示方式", "",
        f"- 有 anion ratio 标签的配方数：{len(samples)}", f"- 低 anion ratio 组：{low_total} 个，标签 ≤ {low_cutoff:.6g}",
        f"- 高 anion ratio 组：{high_total} 个，标签 ≥ {high_cutoff:.6g}", f"- 候选项最小独立配方支持度：{min_formulas}", "",
        "- 片段 SMILES 已由原始显式加氢 BRICS 表示恢复为**去氢、带 `*` 连接位点**的表示；`*` 是母体溶剂中的断键位点，避免把醚键端误读成羟基。",
        "- 每个化学基元均列出承载它的全部溶剂名称和 SMILES；精确模型片段及其合并后的化学基元都保留在 CSV 中，便于审计。",
        "- 直接贡献、物性项、片段对交互均为解释头的 **anion-ratio logit** 项：正/负仅表示在给定盐、温度和其余组成条件下推高/推低解释值，不是因果结论，也不是百分比。", "",
        "## 高 / 低 anion ratio 常见化学基元", "",
        *base.fragment_table(high, 20, "high_mean_main_effect_normalized", "高组内平均直接贡献（logit）"), "",
        *base.fragment_table(low, 20, "low_mean_main_effect_normalized", "低组内平均直接贡献（logit）"), "",
        "## 片段对应溶剂的 dc / vis / Tm 贡献", "",
        "下表将排名靠前的片段基元与其**承载溶剂**的物性画像和解释项绑定。`dc`、`vis`、`Tm` 的物性原值来自冻结分子模型的输出反归一化；三个有符号项来自 anion-ratio 解释头。一个溶剂可承载多个基元，故跨行不能相加。", "",
        *base.motif_property_table(high, low, profiles), "",
        "## 高 / 低 anion ratio 组中的跨溶剂片段交互", "",
        *base.pair_table(high_pairs, 20, "high_mean_interaction_normalized", "高组内平均交互（logit）"), "",
        *base.pair_table(low_pairs, 20, "low_mean_interaction_normalized", "低组内平均交互（logit）"), "",
        "## 图", "",
        "图内刻意只使用英文、官能团式、SMILES 和溶剂代号，避免计算节点缺少中文字体时出现方框；中文解释以本 Markdown 报告和 CSV 为准。", "",
        "![化学基元的直接贡献](high_low_fragment_direct_contributions.svg)", "",
        "![跨溶剂片段对交互](high_low_fragment_pair_interactions.svg)", "",
        "![富集与直接贡献](fragment_enrichment_vs_direct_contribution.svg)", "",
        "![基元承载溶剂物性](motif_carrier_solvent_property_profiles.svg)", "",
        "![基元承载溶剂物性贡献](motif_carrier_property_terms.svg)", "",
        "## 结果使用边界", "",
        "高/低组富集与模型直接贡献必须结合看：高组富集且正贡献、或低组富集且负贡献的基元，才是较一致的候选信号。低支持度、方向不稳定或只在单一承载溶剂中出现的结果，应只作为后续配方验证线索。",
    ]
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="Detailed anion-ratio fragment attribution analysis")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--quantile", type=float, default=.25)
    parser.add_argument("--min-formulas", type=int, default=5)
    args = parser.parse_args()
    if not 0 < args.quantile < .5 or args.min_formulas < 1:
        raise ValueError("quantile must be in (0, .5) and min-formulas must be positive")
    samples, low_cutoff, high_cutoff = load_samples(args.input, args.quantile)
    high_total = sum(s["conductivity_group"] == "高电导率" for s in samples)
    low_total = sum(s["conductivity_group"] == "低电导率" for s in samples)
    exact = base.collect_fragment_occurrences(samples)
    fragments = base.summarize_fragments(exact, high_total, low_total)
    motifs = base.summarize_motifs(base.collect_motif_occurrences(exact), high_total, low_total)
    pairs = base.summarize_pairs(base.collect_pair_occurrences(samples), high_total, low_total)
    properties = base.summarize_molecular_properties(samples)
    profiles = base.summarize_motif_property_profiles(base.collect_motif_property_occurrences(samples))
    output = Path(args.output_dir); output.mkdir(parents=True, exist_ok=True)
    _write_csv(output / "fragment_attribution_summary.csv", fragments)
    _write_csv(output / "chemical_motif_attribution_summary.csv", motifs)
    _write_csv(output / "fragment_pair_interaction_summary.csv", pairs)
    _write_csv(output / "molecular_property_mechanism_summary.csv", properties)
    _write_csv(output / "motif_carrier_solvent_property_profiles.csv", profiles)
    metadata: dict[str, Any] = {"input": args.input, "labelled_formula_count": len(samples), "quantile": args.quantile,
        "low_anion_ratio_cutoff": low_cutoff, "high_anion_ratio_cutoff": high_cutoff,
        "min_formulas": args.min_formulas, "fragment_smiles_representation": "hydrogen-free, attachment-aware", "plot_language": "English/SMILES to avoid missing CJK font"}
    (output / "analysis_metadata.json").write_text(json.dumps(metadata, indent=2, ensure_ascii=False) + "\n")
    make_figures(output, motifs, pairs, profiles, args.min_formulas)
    (output / "anion_ratio_fragment_attribution_report_zh.md").write_text(
        report(samples, low_cutoff, high_cutoff, motifs, pairs, properties, profiles, args.min_formulas))
    print(f"Detailed anion-ratio analysis written to {output}")


if __name__ == "__main__":
    main()
