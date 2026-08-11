"""Comprehensively summarize signed conductivity fragment attributions.

This script consumes the JSON produced by ``predict.py`` with the independent
``fragment_explanations`` head.  It separates (1) experimental high/low
conductivity enrichment from (2) signed, conditional model contributions.
The output is intended for screening hypotheses, not causal claims.
"""

from __future__ import annotations

import argparse
import csv
import html
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean
from typing import Any, Iterable

from explain_experimental_fragments import (canonical_fragment_smiles,
                                            attachment_smiles_by_fragment,
                                            classify_functional_group,
                                            conductivity_ms_cm,
                                            functional_group_shorthand,
                                            percentile)


MODIFIER_SHORTHANDS = {
    "含三氟甲基（-CF3）": "-CF3",
    "含二氟亚甲基（-CF2-）": "-CF2-",
    "芳香环": "-Ar-",
    "非芳香环状骨架": "-(CH2)n-",
}


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("")
        return
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def finite_mean(values: Iterable[float]) -> float | None:
    values = [float(value) for value in values if math.isfinite(float(value))]
    return mean(values) if values else None


def rate(numerator: int, denominator: int) -> float:
    return numerator / denominator if denominator else 0.0


def odds_ratio(high_present: int, high_total: int, low_present: int,
               low_total: int) -> float:
    """Haldane-corrected high-vs-low presence odds ratio."""

    return ((high_present + 0.5) * (low_total - low_present + 0.5) /
            ((high_total - high_present + 0.5) * (low_present + 0.5)))


def fragment_descriptor(smiles: str) -> dict[str, str]:
    """Return an attachment-aware structural-fragment descriptor.

    ``smiles`` may contain ``*`` dummy atoms at bonds cut during
    fragmentation.  Keeping them is essential: a capped fragment such as
    ``CO`` would otherwise look like methanol and be falsely classified as an
    alcohol, even if the O atom is an ether linkage in its parent solvent.
    """

    canonical = canonical_fragment_smiles(smiles)
    info = classify_functional_group(canonical)
    primary = info["functional_group"]
    labels = [functional_group_shorthand(primary)]
    labels.extend(MODIFIER_SHORTHANDS[modifier]
                  for modifier in info["functional_group_modifiers"]
                  if modifier in MODIFIER_SHORTHANDS)
    shorthand = "；".join(dict.fromkeys(label for label in labels
                                        if label and label != "--")) or "--"
    return {
        "fragment_smiles": canonical,
        "fragment_chinese_name": info["functional_group_label"],
        # This intentionally reports a functional-group expression such as
        # -OH or -CF3, rather than a capped-fragment molecular formula.
        "fragment_group_formula": shorthand,
    }


def mark_high_low(records: list[dict[str, Any]], quantile: float) -> tuple[
        list[dict[str, Any]], float, float]:
    samples = []
    for index, record in enumerate(records):
        if not record.get("conductivity_mask", False):
            continue
        if "fragment_explanations" not in record:
            raise ValueError(
                "fragment_explanations is missing; run predict.py with the "
                "fragment explanation head first")
        samples.append({
            "sample_index": index,
            "record": record,
            "target_ms_cm": conductivity_ms_cm(float(record["conductivity"])),
        })
    if not samples:
        raise ValueError("no experimental conductivity records were found")
    values = [sample["target_ms_cm"] for sample in samples]
    low_cutoff = percentile(values, quantile)
    high_cutoff = percentile(values, 1 - quantile)
    for sample in samples:
        value = sample["target_ms_cm"]
        sample["conductivity_group"] = (
            "低电导率" if value <= low_cutoff else
            "高电导率" if value >= high_cutoff else "中间电导率")
    return samples, low_cutoff, high_cutoff


def solvent_lookup(record: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {
        str(solvent.get("name", "未知溶剂")): solvent
        for solvent in record.get("solvents", [])
    }


def collect_fragment_occurrences(samples: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Aggregate attachment-aware fragments once per formula.

    The model still receives the capped fragment representation used during
    training.  The attachment-aware representation is only for correct
    chemical naming and reporting.
    """

    rows = []
    for sample in samples:
        record = sample["record"]
        per_formula: dict[str, dict[str, Any]] = {}
        lookup = solvent_lookup(record)
        for molecule in record.get("fragment_attention", {}).get("solvents", []):
            attached_smiles = attachment_smiles_by_fragment(molecule)
            solvent_name = str(molecule.get("molecule_name", "未知溶剂"))
            solvent = lookup.get(solvent_name, {})
            ratio = float(solvent.get("molar_ratio", 0.0))
            solvent_smiles = solvent.get("smiles", molecule.get("mapped_smiles", ""))
            for fragment in molecule.get("fragments", []):
                if "conductivity_main_effect_normalized" not in fragment:
                    continue
                fragment_id = int(fragment.get("fragment_id", -1))
                descriptor = fragment_descriptor(attached_smiles.get(
                    fragment_id, fragment.get("smiles", "")))
                key = descriptor["fragment_smiles"]
                row = per_formula.setdefault(key, {
                    **descriptor,
                    "sample_index": sample["sample_index"],
                    "conductivity_group": sample["conductivity_group"],
                    "target_conductivity_ms_cm": sample["target_ms_cm"],
                    "main_effect_normalized": 0.0,
                    "solvent_names": set(),
                    "solvent_smiles": set(),
                    "molar_ratios": [],
                    "fragment_instance_count": 0,
                })
                row["main_effect_normalized"] += float(
                    fragment["conductivity_main_effect_normalized"])
                row["solvent_names"].add(solvent_name)
                row["solvent_smiles"].add(str(solvent_smiles))
                row["molar_ratios"].append(ratio)
                row["fragment_instance_count"] += 1
        for row in per_formula.values():
            row["solvent_names"] = "；".join(sorted(row["solvent_names"]))
            row["solvent_smiles"] = "；".join(sorted(row["solvent_smiles"]))
            row["mean_molar_ratio_when_present"] = finite_mean(
                row.pop("molar_ratios"))
            rows.append(row)
    return rows


def summarize_fragments(rows: list[dict[str, Any]], high_total: int,
                        low_total: int) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[row["fragment_smiles"]].append(row)
    summary = []
    for smiles, values in grouped.items():
        high = [row for row in values if row["conductivity_group"] == "高电导率"]
        low = [row for row in values if row["conductivity_group"] == "低电导率"]
        effects = [row["main_effect_normalized"] for row in values]
        high_effects = [row["main_effect_normalized"] for row in high]
        low_effects = [row["main_effect_normalized"] for row in low]
        effect = finite_mean(effects) or 0.0
        high_count, low_count = len(high), len(low)
        high_rate, low_rate = rate(high_count, high_total), rate(low_count,
                                                                 low_total)
        summary.append({
            "fragment_smiles": smiles,
            "fragment_chinese_name": values[0]["fragment_chinese_name"],
            "fragment_group_formula": values[0]["fragment_group_formula"],
            "unique_formula_count": len(values),
            "high_formula_count": high_count,
            "low_formula_count": low_count,
            "high_presence_rate": high_rate,
            "low_presence_rate": low_rate,
            "high_vs_low_presence_odds_ratio": odds_ratio(
                high_count, high_total, low_count, low_total),
            "mean_main_effect_normalized": effect,
            "positive_main_effect_fraction": rate(
                sum(value > 0 for value in effects), len(effects)),
            "high_mean_main_effect_normalized": finite_mean(high_effects),
            "low_mean_main_effect_normalized": finite_mean(low_effects),
            "high_minus_low_main_effect_normalized": (
                (finite_mean(high_effects) or 0.0) -
                (finite_mean(low_effects) or 0.0)),
            "solvent_molecules": "；".join(sorted({
                name for row in values for name in row["solvent_names"].split("；")
                if name
            })),
            "solvent_smiles": "；".join(sorted({
                item for row in values for item in row["solvent_smiles"].split("；")
                if item
            })),
            "mean_molar_ratio_when_present": finite_mean(
                row["mean_molar_ratio_when_present"] for row in values),
            "interpretation": (
                "条件下正向直接贡献候选" if effect > 0 else
                "条件下负向直接贡献候选" if effect < 0 else "直接贡献接近零"),
        })
    return sorted(summary, key=lambda row: (
        -row["unique_formula_count"],
        -abs(row["mean_main_effect_normalized"])))


def collect_pair_occurrences(samples: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for sample in samples:
        record = sample["record"]
        per_formula: dict[tuple[str, str, str, str], dict[str, Any]] = {}
        attachment_lookup = {
            (molecule_index, int(fragment_id)): smiles
            for molecule_index, molecule in enumerate(
                record.get("fragment_attention", {}).get("solvents", []))
            for fragment_id, smiles in attachment_smiles_by_fragment(molecule).items()
        }
        for pair in record["fragment_explanations"].get("top_fragment_pairs", []):
            left, right = pair["left"], pair["right"]
            left_desc = fragment_descriptor(attachment_lookup.get(
                (int(left.get("molecule_index", -1)),
                 int(left.get("fragment_id", -1))),
                left.get("fragment_smiles", "")))
            right_desc = fragment_descriptor(attachment_lookup.get(
                (int(right.get("molecule_index", -1)),
                 int(right.get("fragment_id", -1))),
                right.get("fragment_smiles", "")))
            # Make a pair structure order-invariant while retaining its two
            # solvent identities for reporting.
            members = sorted([
                (left_desc["fragment_smiles"], left_desc, str(left.get("molecule_name", "未知溶剂"))),
                (right_desc["fragment_smiles"], right_desc, str(right.get("molecule_name", "未知溶剂"))),
            ], key=lambda item: (item[0], item[2]))
            key = (members[0][0], members[1][0], members[0][2], members[1][2])
            row = per_formula.setdefault(key, {
                "sample_index": sample["sample_index"],
                "conductivity_group": sample["conductivity_group"],
                "target_conductivity_ms_cm": sample["target_ms_cm"],
                "fragment_a_smiles": members[0][0],
                "fragment_a_chinese_name": members[0][1]["fragment_chinese_name"],
                "fragment_a_group_formula": members[0][1]["fragment_group_formula"],
                "solvent_a": members[0][2],
                "fragment_b_smiles": members[1][0],
                "fragment_b_chinese_name": members[1][1]["fragment_chinese_name"],
                "fragment_b_group_formula": members[1][1]["fragment_group_formula"],
                "solvent_b": members[1][2],
                "interaction_effect_normalized": 0.0,
                "pair_instance_count": 0,
            })
            row["interaction_effect_normalized"] += float(
                pair["effect_normalized"])
            row["pair_instance_count"] += 1
        rows.extend(per_formula.values())
    return rows


def summarize_pairs(rows: list[dict[str, Any]], high_total: int,
                    low_total: int) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(row["fragment_a_smiles"], row["fragment_b_smiles"],
                 row["solvent_a"], row["solvent_b"])].append(row)
    summary = []
    for _, values in grouped.items():
        high = [row for row in values if row["conductivity_group"] == "高电导率"]
        low = [row for row in values if row["conductivity_group"] == "低电导率"]
        effects = [row["interaction_effect_normalized"] for row in values]
        high_count, low_count = len(high), len(low)
        effect = finite_mean(effects) or 0.0
        summary.append({
            key: values[0][key] for key in (
                "fragment_a_smiles", "fragment_a_chinese_name",
                "fragment_a_group_formula", "solvent_a", "fragment_b_smiles",
                "fragment_b_chinese_name", "fragment_b_group_formula", "solvent_b")
        } | {
            "unique_formula_count": len(values),
            "high_formula_count": high_count,
            "low_formula_count": low_count,
            "high_presence_rate": rate(high_count, high_total),
            "low_presence_rate": rate(low_count, low_total),
            "high_vs_low_presence_odds_ratio": odds_ratio(
                high_count, high_total, low_count, low_total),
            "mean_interaction_effect_normalized": effect,
            "positive_interaction_fraction": rate(
                sum(value > 0 for value in effects), len(effects)),
            "high_mean_interaction_normalized": finite_mean(
                row["interaction_effect_normalized"] for row in high),
            "low_mean_interaction_normalized": finite_mean(
                row["interaction_effect_normalized"] for row in low),
            "interaction_interpretation": (
                "模型正协同候选" if effect > 0 else
                "模型负协同/拮抗候选" if effect < 0 else "交互接近零"),
        })
    return sorted(summary, key=lambda row: (
        -row["unique_formula_count"],
        -abs(row["mean_interaction_effect_normalized"])))


def candidates(rows: list[dict[str, Any]], group: str, min_formulas: int,
               effect_key: str) -> list[dict[str, Any]]:
    count_key = "high_formula_count" if group == "high" else "low_formula_count"
    # Ties are not group-specific evidence.  Keeping them in both lists made
    # the former report claim that the same fragment was common in both high
    # and low conductivity groups.
    association = (lambda row: row["high_presence_rate"] > row["low_presence_rate"]
                   if group == "high" else
                   row["low_presence_rate"] > row["high_presence_rate"])
    selected = [row for row in rows if row[count_key] >= min_formulas
                and association(row)]
    return sorted(selected, key=lambda row: (
        -abs(float(row[effect_key] or 0.0)), -row[count_key]))


def short_text(value: Any, limit: int = 52) -> str:
    value = str(value).replace("|", "\\|").replace("\n", " ")
    return value if len(value) <= limit else value[:limit - 1] + "…"


def number(value: Any, digits: int = 3) -> str:
    if value is None or value == "":
        return "—"
    return f"{float(value):.{digits}f}"


def markdown_table(headers: list[str], rows: list[list[str]]) -> list[str]:
    if not rows:
        return ["无满足最小支持度的候选项。"]
    return [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
        *["| " + " | ".join(row) + " |" for row in rows],
    ]


def fragment_table(rows: list[dict[str, Any]], limit: int,
                   effect_key: str, effect_title: str) -> list[str]:
    table_rows = [[
        short_text(row["fragment_chinese_name"], 28),
        short_text(row["fragment_group_formula"], 25),
        short_text(row["fragment_smiles"], 32),
        short_text(row["solvent_molecules"], 35),
        f'{row["high_formula_count"]}/{row["low_formula_count"]}',
        f'{number(row["high_presence_rate"])} / {number(row["low_presence_rate"])}',
        number(row[effect_key], 4),
    ] for row in rows[:limit]]
    return markdown_table([
        "中文名称", "官能团式", "带连接位点的片段 SMILES", "所属溶剂",
        "高/低支持度", "高/低出现率", effect_title,
    ], table_rows)


def pair_table(rows: list[dict[str, Any]], limit: int,
               effect_key: str, effect_title: str) -> list[str]:
    table_rows = [[
        short_text(row["fragment_a_chinese_name"], 22) + "<br>" +
        short_text(row["fragment_a_group_formula"], 18) + "<br>" +
        short_text(row["fragment_a_smiles"], 28),
        short_text(row["solvent_a"], 18),
        short_text(row["fragment_b_chinese_name"], 22) + "<br>" +
        short_text(row["fragment_b_group_formula"], 18) + "<br>" +
        short_text(row["fragment_b_smiles"], 28),
        short_text(row["solvent_b"], 18),
        f'{row["high_formula_count"]}/{row["low_formula_count"]}',
        number(row[effect_key], 4),
        ("模型正协同候选" if float(row[effect_key] or 0.0) > 0 else
         "模型负协同/拮抗候选" if float(row[effect_key] or 0.0) < 0 else
         "交互接近零"),
    ] for row in rows[:limit]]
    return markdown_table([
        "片段 A（中文名 / 官能团式 / 带连接位点 SMILES）", "溶剂 A",
        "片段 B（中文名 / 官能团式 / 带连接位点 SMILES）", "溶剂 B",
        "高/低支持度", effect_title, "模型判断",
    ], table_rows)


def plot_label_fragment(row: dict[str, Any]) -> list[str]:
    solvents = [item for item in str(row["solvent_molecules"]).split("；") if item]
    preview = "+".join(solvents[:2])
    if len(solvents) > 2:
        preview += "+…"
    return [row["fragment_group_formula"],
            f"solv: {preview} ({len(solvents)})"]


def plot_label_pair(row: dict[str, Any]) -> list[str]:
    """Use two lines so both pair members remain readable without IDs."""
    return [f'{row["solvent_a"]}: {row["fragment_a_group_formula"]}',
            f'× {row["solvent_b"]}: {row["fragment_b_group_formula"]}']


def svg_text(x: float, y: float, value: str, size: int = 12,
             anchor: str = "start", weight: str = "normal") -> str:
    return (f'<text x="{x:.1f}" y="{y:.1f}" font-size="{size}" '
            f'font-family="Arial, sans-serif" text-anchor="{anchor}" '
            f'font-weight="{weight}">{html.escape(value)}</text>')


def write_svg(path: Path, width: int, height: int, elements: list[str]) -> None:
    path.write_text(
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}">\n<rect width="100%" height="100%" '
        'fill="white"/>\n' + "\n".join(elements) + "\n</svg>\n")


def svg_bar_panel(rows: list[dict[str, Any]], effect_key: str, label_fn,
                  title: str, x0: float, y0: float, width: float,
                  limit: int = 12, zero_fraction: float = 0.5,
                  label_width: float = 390) -> list[str]:
    rows = sorted(rows, key=lambda row: abs(row[effect_key]), reverse=True)[:limit]
    lines = [svg_text(x0 + width / 2, y0 + 20, title, 15, "middle", "bold")]
    if not rows:
        return lines + [svg_text(x0 + width / 2, y0 + 70,
                                 "No supported candidates", 12, "middle")]
    plot_left, plot_width, row_height = x0 + label_width, width - label_width - 25, 42
    zero_x = plot_left + plot_width * zero_fraction
    max_abs = max(max(abs(row[effect_key]) for row in rows), 1e-8)
    scale = (min(zero_x - plot_left, plot_left + plot_width - zero_x) - 8) / max_abs
    lines.append(f'<line x1="{zero_x:.1f}" y1="{y0 + 34:.1f}" '
                 f'x2="{zero_x:.1f}" y2="{y0 + 35 + len(rows) * row_height:.1f}" '
                 'stroke="#222" stroke-width="1"/>')
    for index, row in enumerate(rows):
        y, value = y0 + 55 + index * row_height, row[effect_key]
        bar_width = abs(value) * scale
        bar_x = zero_x if value >= 0 else zero_x - bar_width
        color = "#c74343" if value > 0 else "#356ea2"
        labels = label_fn(row)
        for line_index, label in enumerate(labels):
            lines.append(svg_text(plot_left - 9, y + 8 + line_index * 13, label,
                                  11 if line_index == 0 else 9,
                                  "end"))
        # With two-line labels, place the bar below both text lines.  The
        # previous offset put the bar on the first line and visually covered
        # the ``solv:`` line.
        bar_y = y + (30 if len(labels) > 1 else 0)
        lines.append(f'<rect x="{bar_x:.1f}" y="{bar_y:.1f}" width="{bar_width:.1f}" '
                     f'height="15" fill="{color}"/>')
        lines.append(svg_text(bar_x + bar_width + 4 if value >= 0 else bar_x - 4,
                              bar_y + 11, f"{value:.3f}", 9,
                              "start" if value >= 0 else "end"))
    return lines


def scatter_annotation_label(row: dict[str, Any]) -> str:
    """Short, chemistry-first label for selected enrichment-scatter points."""

    solvents = [item for item in str(row["solvent_molecules"]).split("；")
                if item]
    solvent = solvents[0] if solvents else "unknown"
    formula = (str(row["fragment_group_formula"]).replace("；", "+")
               .replace("（环）", "(ring)"))
    return f"{formula} ({solvent})"


def non_overlapping_scatter_labels(selected: list[dict[str, Any]],
                                   coordinates: dict[str, tuple[float, float]]) -> list[str]:
    """Lay out labels with small leader lines; preserve rare strong signals."""

    elements: list[str] = []
    occupied: list[tuple[float, float, float, float]] = []
    # Try near-point locations first, then progressively farther locations.
    offsets = ((8, -9), (8, 16), (-8, -9), (-8, 16),
               (8, -25), (-8, -25), (8, 32), (-8, 32))
    for row in selected:
        key = row["fragment_smiles"]
        x, y = coordinates[key]
        text = scatter_annotation_label(row)
        text_width = max(32.0, 4.6 * len(text))
        placement = None
        for dx, dy in offsets:
            anchor = "start" if dx > 0 else "end"
            tx, ty = x + dx, y + dy
            left = tx if anchor == "start" else tx - text_width
            right = tx + text_width if anchor == "start" else tx
            top, bottom = ty - 9, ty + 3
            if not any(left < old_right and right > old_left and
                       top < old_bottom and bottom > old_top
                       for old_left, old_right, old_top, old_bottom in occupied):
                placement = (tx, ty, anchor, left, right, top, bottom)
                break
        if placement is None:
            # Keep the label visible even in a dense region.
            dx, dy = offsets[-1]
            anchor = "start" if dx > 0 else "end"
            tx, ty = x + dx, y + dy
            left = tx if anchor == "start" else tx - text_width
            right = tx + text_width if anchor == "start" else tx
            placement = (tx, ty, anchor, left, right, ty - 9, ty + 3)
        tx, ty, anchor, left, right, top, bottom = placement
        occupied.append((left, right, top, bottom))
        elements.append(f'<line x1="{x:.1f}" y1="{y:.1f}" x2="{tx:.1f}" '
                        f'y2="{ty - 3:.1f}" stroke="#555" stroke-width="0.6"/>')
        elements.append(svg_text(tx, ty, text, 8, anchor, "bold"))
    return elements


def make_plots(output_dir: Path, fragment_rows: list[dict[str, Any]],
               pair_rows: list[dict[str, Any]], min_formulas: int) -> None:
    """Write dependency-free SVG figures with concise formula/solvent labels."""
    high_fragments = candidates(fragment_rows, "high", min_formulas,
                                "high_mean_main_effect_normalized")
    low_fragments = candidates(fragment_rows, "low", min_formulas,
                               "low_mean_main_effect_normalized")
    high_pairs = candidates(pair_rows, "high", min_formulas,
                            "high_mean_interaction_normalized")
    low_pairs = candidates(pair_rows, "low", min_formulas,
                           "low_mean_interaction_normalized")
    height = 590
    write_svg(output_dir / "high_low_fragment_direct_contributions.svg", 1900, height,
              svg_bar_panel(high_fragments, "high_mean_main_effect_normalized",
                            plot_label_fragment, "High-conductivity common fragments",
                            15, 15, 920) +
              svg_bar_panel(low_fragments, "low_mean_main_effect_normalized",
                            plot_label_fragment, "Low-conductivity common fragments",
                            # Reserve a wider label area and move the low-panel
                            # plotting area/zero axis right to prevent
                            # horizontal overlap with long labels.
                            965, 15, 920, zero_fraction=0.3,
                            label_width=450))
    write_svg(output_dir / "high_low_fragment_pair_interactions.svg", 1900, height,
              svg_bar_panel(high_pairs, "high_mean_interaction_normalized",
                            plot_label_pair, "High-group fragment-pair interactions",
                            15, 15, 920) +
              svg_bar_panel(low_pairs, "low_mean_interaction_normalized",
                            plot_label_pair, "Low-group fragment-pair interactions",
                            965, 15, 920))

    eligible = [row for row in fragment_rows
                if row["unique_formula_count"] >= min_formulas]
    width, height, margin = 1000, 700, 75
    x_values = [row["high_presence_rate"] - row["low_presence_rate"] for row in eligible]
    y_values = [row["mean_main_effect_normalized"] for row in eligible]
    x_span = max(max((abs(value) for value in x_values), default=0.1), 0.1)
    y_span = max(max((abs(value) for value in y_values), default=0.1), 0.1)
    center_x, center_y = width / 2, height / 2
    x_scale, y_scale = (width / 2 - margin) / x_span, (height / 2 - margin) / y_span
    elements = [svg_text(width / 2, 28, "Fragment enrichment versus direct contribution",
                         17, "middle", "bold"),
                f'<line x1="{margin}" y1="{center_y}" x2="{width - margin}" y2="{center_y}" stroke="#222"/>',
                f'<line x1="{center_x}" y1="{margin}" x2="{center_x}" y2="{height - margin}" stroke="#222"/>',
                svg_text(width / 2, height - 20, "High minus low presence rate", 13, "middle"),
                svg_text(18, 40, "Mean direct attribution", 13)]
    coordinates = {}
    for row in eligible:
        x_value = row["high_presence_rate"] - row["low_presence_rate"]
        y_value = row["mean_main_effect_normalized"]
        x, y = center_x + x_value * x_scale, center_y - y_value * y_scale
        coordinates[row["fragment_smiles"]] = (x, y)
        radius = min(14, 3 + 0.45 * row["unique_formula_count"] ** 0.5)
        color = "#c74343" if y_value > 0 else "#356ea2"
        elements.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="{radius:.1f}" '
                        f'fill="{color}" fill-opacity="0.68" stroke="white"/>')
    # Frequency alone misses chemically informative but less common fragments
    # such as AN's -C#N.  Retain frequent labels and add any point that is
    # both clearly enriched and has a strong signed direct attribution.
    frequent = sorted(eligible,
                      key=lambda item: item["unique_formula_count"],
                      reverse=True)[:12]
    strong_directional = [
        row for row in eligible
        if abs(row["high_presence_rate"] - row["low_presence_rate"]) >= 0.02
        and abs(row["mean_main_effect_normalized"]) >= 0.10
    ]
    selected_by_smiles = {row["fragment_smiles"]: row for row in frequent}
    selected_by_smiles.update({row["fragment_smiles"]: row
                               for row in strong_directional})
    selected = sorted(selected_by_smiles.values(),
                      key=lambda item: (-item["unique_formula_count"],
                                        item["fragment_smiles"]))
    elements.extend(non_overlapping_scatter_labels(selected, coordinates))
    write_svg(output_dir / "fragment_enrichment_vs_direct_contribution.svg",
              width, height, elements)


def make_report(samples: list[dict[str, Any]], low_cutoff: float,
                high_cutoff: float, fragment_rows: list[dict[str, Any]],
                pair_rows: list[dict[str, Any]], min_formulas: int) -> str:
    high_total = sum(s["conductivity_group"] == "高电导率" for s in samples)
    low_total = sum(s["conductivity_group"] == "低电导率" for s in samples)
    high_fragments = candidates(fragment_rows, "high", min_formulas,
                                "high_mean_main_effect_normalized")
    low_fragments = candidates(fragment_rows, "low", min_formulas,
                               "low_mean_main_effect_normalized")
    high_pairs = candidates(pair_rows, "high", min_formulas,
                            "high_mean_interaction_normalized")
    low_pairs = candidates(pair_rows, "low", min_formulas,
                           "low_mean_interaction_normalized")
    lines = [
        "# 电导率条件感知片段归因报告",
        "",
        "## 数据范围与分组",
        "",
        f"- 有实验标签配方数：{len(samples)}",
        f"- 低电导率组：{low_total} 个，实验电导率 ≤ {low_cutoff:.6g} mS/cm",
        f"- 高电导率组：{high_total} 个，实验电导率 ≥ {high_cutoff:.6g} mS/cm",
        f"- 候选项最小配方支持度：{min_formulas}",
        "",
        "## 名词、含义与使用意义",
        "",
        "- **高/低电导率常见片段**：按实验电导率上下分位数组划分后，在对应组中出现较多的片段。它反映数据集富集，而非单独的因果作用。",
        "- **出现率**：含某片段（或片段对）的配方数除以该高/低组总配方数。高/低出现率及其 odds ratio 用于判断片段偏向出现在哪一组。",
        "- **直接贡献（main effect）**：解释头在固定盐组成、盐浓度和温度条件下，分给一个溶剂片段的有符号归一化 log 电导率项。正值表示提高解释值，负值表示降低解释值；高/低表分别显示该组内平均值。",
        "- **协同/拮抗（interaction effect）**：来自不同溶剂分子的两个片段同时出现时的额外有符号项。正值是模型正协同候选，负值是模型负协同/拮抗候选；片段对表按对应高/低组内平均交互绝对值排序，不再优先按出现次数排序。",
        "- **条件基准**：盐组成、盐浓度、温度所给出的参考电导率。片段项描述的是相对于该基准的组成差异，不是盐效应本身。",
        "- **官能团式**：如 `-OH`、`-CF3`、`-C#N` 的连接式，仅用于结构阅读；它不是把被切分片段自动补氢后的完整分子式。",
        "- **带连接位点的片段 SMILES**：`*` 表示该片段在母体溶剂中被切开的连接位点。例如 `*OC` 是醚氧连接的甲氧基端，而不是甲醇；它用于避免把断键后的氢封端误判为羟基。",
        "- **结构片段与官能团类别**：同一个官能团类别（如碳酸酯或醚键）可以对应多个不同的结构片段 SMILES，表示它们在分子中的连接位置、相邻原子或片段边界不同；它们不是“同一种片段却写了不同 SMILES”。",
        "- **复合结构描述**：如 `-O-；-CF2-` 表示同一个片段的主官能团是醚键，并且该片段还含二氟亚甲基；分号不是把两个候选片段或两个分子拼在一起。出现多行时，以带连接位点 SMILES 区分不同的局部结构。",
        "- **支持度**：某候选出现于多少个独立配方。支持度太低的项目只保留在完整 CSV 中，不应作为稳定设计结论。",
        "",
        "## 高电导率常见片段",
        "",
        "以下要求高组出现率**严格高于**低组，且达到最小支持度。正直接贡献更适合作为提高电导率的结构候选；负直接贡献说明它虽高组富集，但其作用可能依赖其他组分或条件。出现率相等的片段不会同时列入高、低表。",
        "",
        *fragment_table(high_fragments, 20, "high_mean_main_effect_normalized",
                        "高组内平均直接贡献"),
        "",
        "## 低电导率常见片段",
        "",
        "以下要求低组出现率**严格高于**高组，且达到最小支持度。负直接贡献且低组富集的片段是更优先的待避开候选。出现率相等的片段不会同时列入高、低表。",
        "",
        *fragment_table(low_fragments, 20, "low_mean_main_effect_normalized",
                        "低组内平均直接贡献"),
        "",
        "## 高电导率组中的片段对协同与拮抗",
        "",
        "正交互为模型正协同候选，负交互为模型负协同/拮抗候选。仅统计来自不同溶剂分子的片段对，且按高组内平均交互绝对值排序，因此腈基、磷酸酯等较低频片段若交互足够强也可出现。",
        "",
        *pair_table(high_pairs, 20, "high_mean_interaction_normalized",
                    "高组内平均交互"),
        "",
        "## 低电导率组中的片段对协同与拮抗",
        "",
        "正交互表示该片段对在低电导率配方中仍给出提高解释值的方向；负交互是更值得关注的低电导率拮抗候选。表格按低组内平均交互绝对值排序。",
        "",
        *pair_table(low_pairs, 20, "low_mean_interaction_normalized",
                    "低组内平均交互"),
        "",
        "## 可视化",
        "",
        "三张图均来自本报告上方同一批、通过最小支持度筛选的候选项。直接贡献图的第二行 `solv:` 显示前两种代表溶剂和该片段覆盖的溶剂总数；片段对图将两个成员拆为两行，并在每行前标出对应溶剂。完整结构和完整溶剂清单仍以表格与 CSV 为准。红色为正贡献/正交互，蓝色为负贡献/负交互。",
        "",
        "**图 1：常见结构片段的直接贡献。** 左、右面板分别对应高、低电导率组。横轴为解释头在对应组内的平均归一化 log 电导率直接贡献，条越长说明在本数据与条件范围内模型归因越强。它回答“该片段出现时，模型倾向将解释值往上还是往下推”，不能单独证明因果。",
        "",
        "![高低电导率组中常见片段的直接贡献](high_low_fragment_direct_contributions.svg)",
        "",
        "**图 2：跨溶剂片段对的交互。** 左、右面板分别在高、低电导率组筛选，并按该组内平均交互绝对值排列。横轴为两个来自不同溶剂分子的片段同时出现时，超出各自直接贡献之和的平均交互项：正值为模型协同候选，负值为模型拮抗候选。交互很小或支持度很低时，只能作为后续实验组合筛选线索。",
        "",
        "![高低电导率组中片段对交互](high_low_fragment_pair_interactions.svg)",
        "",
        "**图 3：富集与直接贡献的交叉判断。** 横轴为高组出现率减低组出现率：右侧表示高组富集，左侧表示低组富集；纵轴为全样本平均直接贡献：上方为正、下方为负。图中优先标注高支持度片段，此外也强制标注同时具有明显高/低组富集和强直接贡献的低频片段（如 `-C#N (AN)`）；标签使用官能团式和代表溶剂，并以引线避开相邻点。最值得优先验证的高电导率候选通常位于右上象限，低电导率待避开候选通常位于左下象限。气泡大小随独立配方支持度增加而增大。",
        "",
        "![片段富集与直接贡献关系](fragment_enrichment_vs_direct_contribution.svg)",
        "",
        "## 使用边界",
        "",
        "- 本报告的直接贡献和交互贡献严格加和到解释头的片段残差；它们不修改 Bamboo 最终预测。",
        "- 片段及片段对的正负号是模型在给定条件下学到的关联，不等同于化学实验中的单因素因果效应。",
        "- 设计候选应优先选择：高支持度、在不同配方中方向一致、同时具有高组富集和正贡献（或低组富集和负贡献）的项。",
        "- 建议在盐种、浓度、温度和摩尔比例尽量可比时，使用真实溶剂替换或补充实验进行验证。",
    ]
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Analyze high/low conductivity fragment attributions")
    parser.add_argument("--input", required=True,
                        help="prediction JSON generated by predict.py")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--quantile", type=float, default=0.25,
                        help="low/high experimental conductivity quantile")
    parser.add_argument("--min-formulas", type=int, default=5,
                        help="minimum independent formulas in screened tables")
    args = parser.parse_args()
    if not 0 < args.quantile < 0.5:
        raise ValueError("--quantile must be in (0, 0.5)")
    if args.min_formulas < 1:
        raise ValueError("--min-formulas must be positive")

    records = json.loads(Path(args.input).read_text())
    samples, low_cutoff, high_cutoff = mark_high_low(records, args.quantile)
    high_total = sum(s["conductivity_group"] == "高电导率" for s in samples)
    low_total = sum(s["conductivity_group"] == "低电导率" for s in samples)
    fragment_rows = summarize_fragments(
        collect_fragment_occurrences(samples), high_total, low_total)
    pair_rows = summarize_pairs(
        collect_pair_occurrences(samples), high_total, low_total)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(output_dir / "fragment_attribution_summary.csv", fragment_rows)
    write_csv(output_dir / "high_conductivity_common_fragments.csv",
              candidates(fragment_rows, "high", args.min_formulas,
                         "high_mean_main_effect_normalized"))
    write_csv(output_dir / "low_conductivity_common_fragments.csv",
              candidates(fragment_rows, "low", args.min_formulas,
                         "low_mean_main_effect_normalized"))
    write_csv(output_dir / "fragment_pair_interaction_summary.csv", pair_rows)
    write_csv(output_dir / "high_conductivity_fragment_pairs.csv",
              candidates(pair_rows, "high", args.min_formulas,
                         "high_mean_interaction_normalized"))
    write_csv(output_dir / "low_conductivity_fragment_pairs.csv",
              candidates(pair_rows, "low", args.min_formulas,
                         "low_mean_interaction_normalized"))
    metadata = {
        "input": str(args.input), "experimental_formula_count": len(samples),
        "quantile": args.quantile, "low_cutoff_ms_cm": low_cutoff,
        "high_cutoff_ms_cm": high_cutoff, "min_formulas": args.min_formulas,
        "high_formula_count": high_total, "low_formula_count": low_total,
    }
    (output_dir / "analysis_metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False) + "\n")
    make_plots(output_dir, fragment_rows, pair_rows, args.min_formulas)
    (output_dir / "fragment_attribution_report_zh.md").write_text(
        make_report(samples, low_cutoff, high_cutoff, fragment_rows, pair_rows,
                    args.min_formulas))
    print(f"Wrote comprehensive fragment attribution analysis to: {output_dir}")


if __name__ == "__main__":
    main()
