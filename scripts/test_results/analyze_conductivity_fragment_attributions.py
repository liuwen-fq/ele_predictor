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
from statistics import mean, median
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


PROPERTY_NORMALIZATION_PATH = (Path(__file__).resolve().parents[2] /
                               "config/molecular_property_normalization.json")


def load_property_normalization() -> dict[str, Any]:
    """Load the min/max transform used by the fixed molecular checkpoint."""

    if not PROPERTY_NORMALIZATION_PATH.is_file():
        raise FileNotFoundError(
            "missing molecular-property normalization metadata: "
            f"{PROPERTY_NORMALIZATION_PATH}")
    return json.loads(PROPERTY_NORMALIZATION_PATH.read_text())


PROPERTY_NORMALIZATION = load_property_normalization()


def physical_property_value(name: str, normalized_value: float) -> float:
    """Invert molecular-pretraining normalization into a physical value.

    This is an inversion of the property label transform, not a rigorous
    thermodynamic mixture rule.  Mixture tables use molar-ratio-weighted
    solvent descriptors solely to make the learned trends readable.
    """

    bounds = PROPERTY_NORMALIZATION["linear_min_max"].get(name)
    if bounds is None:
        return float(normalized_value)
    value = float(bounds[0]) + float(normalized_value) * (
        float(bounds[1]) - float(bounds[0]))
    if name in PROPERTY_NORMALIZATION["exp_after_linear"]:
        value = math.exp(value)
    if name in PROPERTY_NORMALIZATION["kelvin_to_celsius"]:
        value -= 273.15
    return value


def property_unit(name: str) -> str:
    return PROPERTY_NORMALIZATION["units"].get(name, "source-label scale")


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
    # This key deliberately discards BRICS cut-boundary details while keeping
    # chemistry meaningful to experimental interpretation.  For example,
    # *OCC(*)(F)F, *CC(F)(F)O* and *OC(F)(F)C(*)(F)F are all fluorinated
    # ether motifs, not three independent chemistry conclusions.
    motif_key = "|".join([primary, *sorted(info["functional_group_modifiers"])])
    return {
        "fragment_smiles": canonical,
        "fragment_chinese_name": info["functional_group_label"],
        # This intentionally reports a functional-group expression such as
        # -OH or -CF3, rather than a capped-fragment molecular formula.
        "fragment_group_formula": shorthand,
        "motif_key": motif_key,
        "motif_chinese_name": info["functional_group_label"],
        "motif_group_formula": shorthand,
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
            # The configured model maps this source field to concentration.
            # Keep its source name in the final report instead of inventing a
            # mol/L unit that the preprocessed data do not declare.
            "concentration": float(record.get("salt_molar_ratio", 0.0)),
            "temperature_c": float(record.get("temperature", float("nan"))),
            "bamboo_prediction_ms_cm": float(record.get("conductivity_pred", 0.0)),
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


def collect_motif_occurrences(fragment_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Merge cut-boundary variants into one chemical motif per formula.

    The neural model is still attributed at its exact, non-overlapping BRICS
    fragments.  This reporting layer only sums exact-fragment contributions
    that share an experimentally readable motif in the *same formula*.
    Consequently a fluorinated ether split at three different bonds has one
    support count and one total contribution, rather than three repeated rows.
    """

    merged: dict[tuple[int, str], dict[str, Any]] = {}
    for row in fragment_rows:
        key = (int(row["sample_index"]), row["motif_key"])
        item = merged.setdefault(key, {
            "motif_key": row["motif_key"],
            "fragment_chinese_name": row["motif_chinese_name"],
            "fragment_group_formula": row["motif_group_formula"],
            "sample_index": row["sample_index"],
            "conductivity_group": row["conductivity_group"],
            "target_conductivity_ms_cm": row["target_conductivity_ms_cm"],
            "main_effect_normalized": 0.0,
            "solvent_names": set(),
            "solvent_smiles": set(),
            "exact_fragment_smiles": set(),
            "molar_ratios": [],
            "fragment_instance_count": 0,
        })
        item["main_effect_normalized"] += float(row["main_effect_normalized"])
        item["solvent_names"].update(filter(None, row["solvent_names"].split("；")))
        item["solvent_smiles"].update(filter(None, row["solvent_smiles"].split("；")))
        item["exact_fragment_smiles"].add(row["fragment_smiles"])
        if row["mean_molar_ratio_when_present"] is not None:
            item["molar_ratios"].append(row["mean_molar_ratio_when_present"])
        item["fragment_instance_count"] += int(row["fragment_instance_count"])
    output = []
    for item in merged.values():
        item["solvent_names"] = "；".join(sorted(item["solvent_names"]))
        item["solvent_smiles"] = "；".join(sorted(item["solvent_smiles"]))
        item["exact_fragment_smiles"] = "；".join(sorted(item["exact_fragment_smiles"]))
        item["mean_molar_ratio_when_present"] = finite_mean(item.pop("molar_ratios"))
        output.append(item)
    return output


def summarize_motifs(rows: list[dict[str, Any]], high_total: int,
                     low_total: int) -> list[dict[str, Any]]:
    """Summarize chemical motifs while retaining exact fragment audit trails."""

    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[row["motif_key"]].append(row)
    summary = []
    for motif_key, values in grouped.items():
        high = [row for row in values if row["conductivity_group"] == "高电导率"]
        low = [row for row in values if row["conductivity_group"] == "低电导率"]
        effects = [row["main_effect_normalized"] for row in values]
        high_effects = [row["main_effect_normalized"] for row in high]
        low_effects = [row["main_effect_normalized"] for row in low]
        effect = finite_mean(effects) or 0.0
        high_count, low_count = len(high), len(low)
        summary.append({
            "motif_key": motif_key,
            # Keep this field name for shared plot/table helpers.  It now
            # carries one representative exact fragment; all variants are in
            # ``exact_fragment_smiles`` for audit.
            "fragment_smiles": sorted({smiles for row in values
                                        for smiles in row["exact_fragment_smiles"].split("；")})[0],
            "exact_fragment_smiles": "；".join(sorted({smiles for row in values
                                                         for smiles in row["exact_fragment_smiles"].split("；")})),
            "fragment_chinese_name": values[0]["fragment_chinese_name"],
            "fragment_group_formula": values[0]["fragment_group_formula"],
            "unique_formula_count": len(values),
            "high_formula_count": high_count,
            "low_formula_count": low_count,
            "high_presence_rate": rate(high_count, high_total),
            "low_presence_rate": rate(low_count, low_total),
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
                smiles for row in values for smiles in row["solvent_smiles"].split("；")
                if smiles
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


PROPERTY_ROLES = {
    "dc": ("介电常数", "盐解离/自由离子线索", True),
    "vis": ("黏度", "离子迁移与扩散阻力线索", True),
    "Tm": ("熔点", "液态范围、结晶与低温输运线索", True),
    "density": ("密度", "堆积和单位体积组成线索", False),
    "nD": ("折射率", "极化率/分子极性间接线索", False),
    "nD_liquid": ("液相折射率", "液相极化率间接线索", False),
    "bp": ("沸点", "挥发性与高温稳定性约束", False),
    "ST": ("表面张力", "界面润湿与工艺辅助线索", False),
    "vapP": ("蒸气压", "挥发性与安全性约束", False),
    "pka_a": ("pKa_a", "酸性/反应稳定性辅助线索", False),
    "pka_b": ("pKa_b", "碱性/反应稳定性辅助线索", False),
}


def _weighted_property_profile(sample: dict[str, Any]) -> dict[str, float]:
    """Return molar-ratio-weighted solvent descriptors in physical space."""

    record = sample["record"]
    profiles = record.get("molecular_property_profiles")
    if profiles is None:
        # Backward-compatible fallback for result JSON written before the
        # all-property export was added.  It has only dc/vis/Tm.
        profiles = record.get("property_explanations", {}).get(
            "solvent_property_contributions", [])
    totals: dict[str, float] = defaultdict(float)
    for solvent in profiles:
        ratio = float(solvent.get("molar_ratio", 0.0))
        values = solvent.get("property_values_normalized", {})
        for name, value in values.items():
            totals[name] += ratio * physical_property_value(name, float(value))
    return dict(totals)


def summarize_molecular_properties(samples: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Compare composition-weighted molecular properties between high/low groups.

    Values are model-normalized and are deliberately reported separately from
    trained property contributions.  Only dc, vis and Tm are direct terms in
    the current explanation head.
    """

    grouped: dict[str, list[tuple[str, float]]] = defaultdict(list)
    direct_effects: dict[str, list[tuple[str, float]]] = defaultdict(list)
    for sample in samples:
        for name, value in _weighted_property_profile(sample).items():
            grouped[name].append((sample["conductivity_group"], value))
        explanation = sample["record"].get("property_explanations", {})
        for name, value in explanation.get(
                "property_sum_by_name_normalized", {}).items():
            direct_effects[name].append((sample["conductivity_group"],
                                         float(value)))
    rows = []
    for name, values in sorted(grouped.items()):
        high = [value for group, value in values if group == "高电导率"]
        low = [value for group, value in values if group == "低电导率"]
        effects = [value for _, value in direct_effects.get(name, [])]
        high_effects = [value for group, value in direct_effects.get(name, [])
                        if group == "高电导率"]
        low_effects = [value for group, value in direct_effects.get(name, [])
                       if group == "低电导率"]
        label, role, is_direct = PROPERTY_ROLES.get(
            name, (name, "冻结分子多任务模型的辅助性质", False))
        rows.append({
            "property": name,
            "property_chinese_name": label,
            "mechanistic_role": role,
            "is_direct_explanation_term": is_direct,
            "formula_count": len(values),
            "property_unit": property_unit(name),
            "high_mean_value_physical": finite_mean(high),
            "low_mean_value_physical": finite_mean(low),
            "high_minus_low_value_physical": (
                (finite_mean(high) or 0.0) - (finite_mean(low) or 0.0)),
            "mean_direct_effect_normalized": finite_mean(effects),
            "high_mean_direct_effect_normalized": finite_mean(high_effects),
            "low_mean_direct_effect_normalized": finite_mean(low_effects),
        })
    return sorted(rows, key=lambda row: (
        not row["is_direct_explanation_term"],
        -abs(row["high_minus_low_value_physical"])))


def _solvent_property_lookup(record: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """Map solvent name to physical profile and direct property effects."""

    profiles = record.get("molecular_property_profiles")
    if profiles is None:
        profiles = record.get("property_explanations", {}).get(
            "solvent_property_contributions", [])
    result = {}
    for profile in profiles:
        name = str(profile.get("molecule_name", "未知溶剂"))
        values = {
            key: physical_property_value(key, float(value))
            for key, value in profile.get("property_values_normalized", {}).items()
        }
        result[name] = {
            "molar_ratio": float(profile.get("molar_ratio", 0.0)),
            "values": values,
            "effects": {
                key: float(value) for key, value in profile.get(
                    "property_effects_normalized", {}).items()
            },
        }
    return result


def collect_motif_property_occurrences(samples: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Link each motif to physical descriptors of its carrier solvent(s)."""

    rows = []
    for sample in samples:
        record = sample["record"]
        solvent_profiles = _solvent_property_lookup(record)
        motifs_by_solvent: dict[str, set[str]] = defaultdict(set)
        motif_labels: dict[str, tuple[str, str]] = {}
        for molecule in record.get("fragment_attention", {}).get("solvents", []):
            solvent_name = str(molecule.get("molecule_name", "未知溶剂"))
            attached_smiles = attachment_smiles_by_fragment(molecule)
            for fragment in molecule.get("fragments", []):
                if "conductivity_main_effect_normalized" not in fragment:
                    continue
                descriptor = fragment_descriptor(attached_smiles.get(
                    int(fragment.get("fragment_id", -1)), fragment.get("smiles", "")))
                motif_key = descriptor["motif_key"]
                motifs_by_solvent[solvent_name].add(motif_key)
                motif_labels[motif_key] = (
                    descriptor["motif_chinese_name"],
                    descriptor["motif_group_formula"],
                )
        per_formula: dict[str, dict[str, Any]] = {}
        for solvent_name, motif_keys in motifs_by_solvent.items():
            profile = solvent_profiles.get(solvent_name)
            if profile is None:
                continue
            ratio = profile["molar_ratio"]
            for motif_key in motif_keys:
                item = per_formula.setdefault(motif_key, {
                    "motif_key": motif_key,
                    "motif_chinese_name": motif_labels[motif_key][0],
                    "motif_group_formula": motif_labels[motif_key][1],
                    "conductivity_group": sample["conductivity_group"],
                    "sample_index": sample["sample_index"],
                    "carrier_solvents": set(),
                    "ratio_sum": 0.0,
                    "properties": defaultdict(float),
                    "effects": defaultdict(float),
                })
                item["carrier_solvents"].add(solvent_name)
                item["ratio_sum"] += ratio
                for name, value in profile["values"].items():
                    item["properties"][name] += ratio * value
                # These are already molar-ratio-weighted contributions from
                # the explanation head; do not multiply by ratio again.
                for name, value in profile["effects"].items():
                    item["effects"][name] += value
        for item in per_formula.values():
            weight = max(item.pop("ratio_sum"), 1e-12)
            item["properties"] = {
                name: value / weight for name, value in item["properties"].items()
            }
            item["effects"] = dict(item["effects"])
            item["carrier_solvents"] = "；".join(sorted(item["carrier_solvents"]))
            rows.append(item)
    return rows


def summarize_motif_property_profiles(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Summarize carrier-solvent properties for every chemical motif."""

    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[row["motif_key"]].append(row)
    output = []
    for motif_key, values in grouped.items():
        high = [row for row in values if row["conductivity_group"] == "高电导率"]
        low = [row for row in values if row["conductivity_group"] == "低电导率"]
        row = {
            "motif_key": motif_key,
            "motif_chinese_name": values[0]["motif_chinese_name"],
            "motif_group_formula": values[0]["motif_group_formula"],
            "carrier_solvents": "；".join(sorted({
                solvent for item in values
                for solvent in item["carrier_solvents"].split("；") if solvent
            })),
            "formula_count": len(values),
        }
        for name in ("dc", "vis", "Tm"):
            for group_name, group_rows in (("high", high), ("low", low)):
                row[f"{group_name}_{name}_physical"] = finite_mean(
                    item["properties"].get(name, float("nan"))
                    for item in group_rows)
                row[f"{group_name}_{name}_effect_normalized"] = finite_mean(
                    item["effects"].get(name, float("nan"))
                    for item in group_rows)
        for group_name, group_rows in (("high", high), ("low", low)):
            row[f"{group_name}_property_effect_total_normalized"] = finite_mean(
                sum(item["effects"].get(name, 0.0)
                    for name in ("dc", "vis", "Tm"))
                for item in group_rows)
        output.append(row)
    return output


def _ratio_bucket(value: float, width: float = 0.05) -> float:
    return round(round(value / width) * width, 3)


def _component_signature(components: list[dict[str, Any]]) -> str:
    return "+".join(
        f'{item.get("name", "未知")}:{_ratio_bucket(float(item.get("molar_ratio", 0.0))):.2f}'
        for item in sorted(components, key=lambda item: str(item.get("name", ""))))


def _condition_context(sample: dict[str, Any]) -> tuple[str, str, str, float]:
    record = sample["record"]
    salts = _component_signature(record.get("salts", []))
    solvents = _component_signature(record.get("solvents", []))
    temperature_bin = round(sample["temperature_c"] / 5.0) * 5.0
    return salts, solvents, f"{temperature_bin:.0f} °C 邻域", temperature_bin


def _signature_ratios(signature: str) -> list[tuple[str, float]]:
    """Parse a binned component signature such as ``DEC:0.25+PC:0.75``."""

    values = []
    for token in signature.split("+"):
        name, separator, ratio = token.rpartition(":")
        if separator:
            values.append((name, float(ratio)))
    return values


def _whole_mixture_mole_fraction_endpoints(salt_context: str,
                                           solvent_context: str,
                                           concentration_low: float,
                                           concentration_high: float) -> str:
    """Format approximate total-mixture mole fractions at both c endpoints.

    Solvent and salt ratios are normalized within their own groups.  The
    concentration field is the total salt mole fraction, so it supplies the
    missing bridge between the two groups.
    """

    salts = _signature_ratios(salt_context)
    solvents = _signature_ratios(solvent_context)

    def endpoint(concentration: float) -> str:
        values = [(name, concentration * ratio) for name, ratio in salts]
        values += [(name, (1.0 - concentration) * ratio)
                   for name, ratio in solvents]
        return ", ".join(f"{name}:{value:.4f}" for name, value in values)

    return f"c={concentration_low:.4f}: {endpoint(concentration_low)}; " \
           f"c={concentration_high:.4f}: {endpoint(concentration_high)}"


def _direction(values: list[float], tolerance: float = 0.002) -> str:
    nonzero = [value for value in values if abs(value) >= tolerance]
    if not nonzero:
        return "接近零且稳定"
    positive = sum(value > 0 for value in nonzero)
    fraction = max(positive, len(nonzero) - positive) / len(nonzero)
    if fraction < 0.75:
        return "方向不稳定"
    return "稳定正向" if positive >= len(nonzero) - positive else "稳定负向"


def concentration_windows(samples: list[dict[str, Any]], min_points: int = 3
                          ) -> tuple[list[dict[str, Any]], list[dict[str, Any]],
                                     list[dict[str, Any]]]:
    """Find global-high and context-local observed concentration windows.

    This is intentionally an observational recommendation, not an unmeasured
    counterfactual scan.  Contexts use solvent/salt ratio bins of 0.05 and a
    5 °C temperature neighborhood; recommendations require at least three
    distinct observed concentration points and stable dc, vis and fragment
    directions within the selected window.
    """

    context_rows: dict[tuple[str, str, float], list[dict[str, Any]]] = defaultdict(list)
    all_predictions = [sample["bamboo_prediction_ms_cm"] for sample in samples]
    global_high = percentile(all_predictions, 0.75)
    for sample in samples:
        record = sample["record"]
        explanation = record.get("fragment_explanations", {})
        property_effects = explanation.get("property_sum_by_name_normalized", {})
        salts, solvents, temp_label, temp_bin = _condition_context(sample)
        context_rows[(salts, solvents, temp_bin)].append({
            "concentration": sample["concentration"],
            "temperature_c": sample["temperature_c"],
            "bamboo_prediction_ms_cm": sample["bamboo_prediction_ms_cm"],
            "condition_reference": float(explanation.get(
                "condition_reference_normalized", 0.0)),
            "dc_effect": float(property_effects.get("dc", 0.0)),
            "vis_effect": float(property_effects.get("vis", 0.0)),
            "fragment_effect": float(explanation.get(
                "total_residual_normalized", 0.0)),
            "salt_context": salts,
            "solvent_context": solvents,
            "temperature_context": temp_label,
        })
    all_contexts, global_recommendations, local_candidates = [], [], []
    for (salts, solvents, temp_bin), values in context_rows.items():
        by_concentration: dict[float, list[dict[str, Any]]] = defaultdict(list)
        for value in values:
            by_concentration[round(value["concentration"], 6)].append(value)
        bins = []
        for concentration, entries in sorted(by_concentration.items()):
            bins.append({
                "concentration": concentration,
                "sample_count": len(entries),
                **{key: finite_mean(entry[key] for entry in entries) or 0.0
                   for key in ("bamboo_prediction_ms_cm", "condition_reference",
                               "dc_effect", "vis_effect", "fragment_effect")},
            })
        if len(bins) < min_points:
            continue
        # Use the upper 40% of each context, rather than only its top quartile:
        # a three-point experimental window cannot exist in a six-point scan
        # if the top quartile contains at most two points.
        local_high = percentile([item["bamboo_prediction_ms_cm"] for item in bins], 0.60)
        high_indices = [index for index, item in enumerate(bins)
                        if item["bamboo_prediction_ms_cm"] >= local_high]
        all_contexts.append({
            "salt_context": salts,
            "solvent_context": solvents,
            "temperature_context": f"{temp_bin:.0f} °C 邻域",
            "distinct_concentration_count": len(bins),
            "formula_count": len(values),
            "concentration_min": bins[0]["concentration"],
            "concentration_max": bins[-1]["concentration"],
            "global_high_prediction_threshold_ms_cm": global_high,
            "best_bamboo_prediction_ms_cm": max(
                item["bamboo_prediction_ms_cm"] for item in bins),
        })
        start = 0
        while start < len(high_indices):
            end = start
            while (end + 1 < len(high_indices)
                   and high_indices[end + 1] == high_indices[end] + 1):
                end += 1
            window = bins[high_indices[start]:high_indices[end] + 1]
            start = end + 1
            if len(window) < min_points:
                continue
            directions = {
                key: _direction([item[key] for item in window])
                for key in ("dc_effect", "vis_effect", "fragment_effect")
            }
            if any(value == "方向不稳定" for value in directions.values()):
                continue
            item = {
                "salt_context": salts,
                "solvent_context": solvents,
                "temperature_context": f"{temp_bin:.0f} °C 邻域",
                "concentration_low": window[0]["concentration"],
                "concentration_high": window[-1]["concentration"],
                "distinct_concentration_count": len(window),
                "formula_count": sum(item["sample_count"] for item in window),
                "mean_bamboo_prediction_ms_cm": finite_mean(
                    item["bamboo_prediction_ms_cm"] for item in window),
                "min_bamboo_prediction_ms_cm": min(
                    item["bamboo_prediction_ms_cm"] for item in window),
                "mean_condition_reference_normalized": finite_mean(
                    item["condition_reference"] for item in window),
                "mean_dc_effect_normalized": finite_mean(
                    item["dc_effect"] for item in window),
                "mean_vis_effect_normalized": finite_mean(
                    item["vis_effect"] for item in window),
                "mean_fragment_effect_normalized": finite_mean(
                    item["fragment_effect"] for item in window),
                "dc_direction_stability": directions["dc_effect"],
                "vis_direction_stability": directions["vis_effect"],
                "fragment_direction_stability": directions["fragment_effect"],
                "evidence_type": "同一近邻条件上下文中的已观测浓度窗口",
                "local_high_prediction_threshold_ms_cm": local_high,
                "global_high_prediction_threshold_ms_cm": global_high,
            }
            item["whole_mixture_mole_fraction_endpoints"] = (
                _whole_mixture_mole_fraction_endpoints(
                    salts, solvents, item["concentration_low"],
                    item["concentration_high"]))
            # A strict global recommendation requires every observed point in
            # the window to clear the test-set-wide 75th percentile.  The
            # remaining windows are still valuable for concentration scans,
            # but are explicitly labelled context-local rather than globally
            # high-performing formulas.
            if item["min_bamboo_prediction_ms_cm"] >= global_high:
                item["recommendation_tier"] = "全局高预测优先窗口"
                global_recommendations.append(item)
            else:
                item["recommendation_tier"] = "体系内局部最优扫描候选"
                local_candidates.append(item)
    return (sorted(all_contexts, key=lambda row: -row["best_bamboo_prediction_ms_cm"]),
            sorted(global_recommendations,
                   key=lambda row: (-float(row["mean_bamboo_prediction_ms_cm"] or 0.0),
                                    -row["formula_count"])),
            sorted(local_candidates,
                   key=lambda row: (-float(row["mean_bamboo_prediction_ms_cm"] or 0.0),
                                    -row["formula_count"])))


def candidates(rows: list[dict[str, Any]], group: str, min_formulas: int,
               effect_key: str, min_enrichment_delta: float = 0.02) -> list[dict[str, Any]]:
    count_key = "high_formula_count" if group == "high" else "low_formula_count"
    # Ties are not group-specific evidence.  Keeping them in both lists made
    # the former report claim that the same fragment was common in both high
    # and low conductivity groups.
    association = (lambda row: row["high_presence_rate"] - row["low_presence_rate"]
                   >= min_enrichment_delta
                   if group == "high" else
                   row["low_presence_rate"] - row["high_presence_rate"]
                   >= min_enrichment_delta)
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
        "化学基元", "官能团式", "代表性带连接位点 SMILES", "所属溶剂",
        "高/低支持度", "高/低出现率", effect_title,
    ], table_rows)


def concentration_window_table(rows: list[dict[str, Any]],
                               limit: int = 15) -> list[str]:
    table_rows = [[
        short_text(row["salt_context"], 28),
        short_text(row["temperature_context"], 14),
        short_text(row["solvent_context"], 44),
        f'{number(row["concentration_low"], 4)} – {number(row["concentration_high"], 4)}',
        short_text(row["whole_mixture_mole_fraction_endpoints"], 82),
        str(row["distinct_concentration_count"]),
        number(row["mean_bamboo_prediction_ms_cm"], 3),
        "/".join([
            row["dc_direction_stability"], row["vis_direction_stability"],
            row["fragment_direction_stability"],
        ]),
    ] for row in rows[:limit]]
    return markdown_table([
        "盐组成（比例 0.05 分箱）", "温度", "溶剂组成（比例 0.05 分箱）",
        "建议扫描范围（高预测观测点 min–max；salt_molar_ratio）",
        "全混合物摩尔分数端点（近似）", "浓度点数", "平均 Bamboo 预测 (mS/cm)",
        "dc / 黏度 / 片段方向稳定性",
    ], table_rows)


def property_table(rows: list[dict[str, Any]], limit: int = 15) -> list[str]:
    table_rows = [[
        row["property_chinese_name"] + f' (`{row["property"]}`)',
        short_text(row["mechanistic_role"], 28),
        "直接贡献项" if row["is_direct_explanation_term"] else "辅助诊断",
        number(row["high_mean_value_physical"], 3),
        number(row["low_mean_value_physical"], 3),
        number(row["high_minus_low_value_physical"], 3),
        short_text(row["property_unit"], 16),
        number(row["high_mean_direct_effect_normalized"], 4)
        if row["is_direct_explanation_term"] else "—",
    ] for row in rows[:limit]]
    return markdown_table([
        "性质", "在电导率解释中的角色", "使用方式", "高组均值", "低组均值",
        "高−低", "单位", "高组直接贡献（归一化 log 电导率项）",
    ], table_rows)


def motif_property_table(high_motifs: list[dict[str, Any]],
                         low_motifs: list[dict[str, Any]],
                         profiles: list[dict[str, Any]],
                         limit_each: int = 12) -> list[str]:
    """Show carrier-solvent property profiles beside screened motifs."""

    profile_by_key = {row["motif_key"]: row for row in profiles}
    table_rows = []
    for group, motifs in (("高组", high_motifs[:limit_each]),
                          ("低组", low_motifs[:limit_each])):
        prefix = "high" if group == "高组" else "low"
        for motif in motifs:
            profile = profile_by_key.get(motif["motif_key"])
            if profile is None:
                continue
            table_rows.append([
                group,
                short_text(profile["motif_chinese_name"], 24),
                short_text(profile["carrier_solvents"], 32),
                number(profile[f"{prefix}_dc_physical"], 2),
                number(profile[f"{prefix}_vis_physical"], 2),
                number(profile[f"{prefix}_Tm_physical"], 1),
                "/".join(number(profile[f"{prefix}_{name}_effect_normalized"], 3)
                         for name in ("dc", "vis", "Tm")),
                number(profile[
                    f"{prefix}_property_effect_total_normalized"], 3),
            ])
    return markdown_table([
        "电导率组", "化学基元", "承载该基元的溶剂", "介电常数 εr",
        "黏度 (mPa·s)", "熔点 (°C)", "dc / vis / Tm 物性项",
        "物性项合计",
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


def motif_profile_plot_rows(high_motifs: list[dict[str, Any]],
                            low_motifs: list[dict[str, Any]],
                            profiles: list[dict[str, Any]],
                            limit_each: int = 12) -> list[dict[str, Any]]:
    """Return plotted motif rows with their group-specific carrier profile."""

    profile_by_key = {row["motif_key"]: row for row in profiles}
    output = []
    for group, motifs in (("High", high_motifs[:limit_each]),
                          ("Low", low_motifs[:limit_each])):
        prefix = group.lower()
        for motif in motifs:
            profile = profile_by_key.get(motif["motif_key"])
            if profile is None:
                continue
            carriers = str(profile["carrier_solvents"]).split("；")
            carrier_text = "+".join(carriers[:2])
            if len(carriers) > 2:
                carrier_text += "+…"
            output.append({
                "group": group,
                "label": f'{motif["fragment_group_formula"]} | {carrier_text}',
                **{name: profile.get(f"{prefix}_{name}_physical")
                   for name in ("dc", "vis", "Tm")},
                **{f"{name}_effect": profile.get(
                    f"{prefix}_{name}_effect_normalized")
                   for name in ("dc", "vis", "Tm")},
            })
    return output


def _finite_plot_values(rows: list[dict[str, Any]], key: str) -> list[float]:
    return [float(row[key]) for row in rows
            if row.get(key) is not None and math.isfinite(float(row[key]))]


def property_profile_dot_panel(rows: list[dict[str, Any]], key: str,
                               title: str, x0: float, y0: float,
                               width: float, row_height: float) -> list[str]:
    """Draw one physical-property dot panel with an independent x scale."""

    values = _finite_plot_values(rows, key)
    if not values:
        return [svg_text(x0 + width / 2, y0 + 25, title, 13, "middle", "bold")]
    value_min, value_max = min(values), max(values)
    padding = max((value_max - value_min) * 0.10, 0.1)
    value_min -= padding
    value_max += padding
    scale = width / max(value_max - value_min, 1e-12)
    lines = [svg_text(x0 + width / 2, y0 + 20, title, 13, "middle", "bold"),
             f'<line x1="{x0:.1f}" y1="{y0 + 34:.1f}" '
             f'x2="{x0 + width:.1f}" y2="{y0 + 34:.1f}" stroke="#333"/>',
             svg_text(x0, y0 + 48, f"{value_min:.2f}", 9),
             svg_text(x0 + width, y0 + 48, f"{value_max:.2f}", 9, "end")]
    for index, row in enumerate(rows):
        value = row.get(key)
        if value is None or not math.isfinite(float(value)):
            continue
        y = y0 + 72 + index * row_height
        x = x0 + (float(value) - value_min) * scale
        color = "#c74343" if row["group"] == "High" else "#356ea2"
        lines.append(f'<line x1="{x0:.1f}" y1="{y:.1f}" x2="{x0 + width:.1f}" '
                     f'y2="{y:.1f}" stroke="#eeeeee"/>')
        lines.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="5" fill="{color}"/>')
        lines.append(svg_text(x + 8, y + 3, f"{float(value):.2f}", 9))
    return lines


def property_effect_bar_panel(rows: list[dict[str, Any]], key: str,
                              title: str, x0: float, y0: float,
                              width: float, row_height: float,
                              max_abs: float) -> list[str]:
    """Draw signed, bounded-scale direct property-attribution bars."""

    zero_x = x0 + width / 2
    scale = (width / 2 - 12) / max(max_abs, 1e-8)
    lines = [svg_text(x0 + width / 2, y0 + 20, title, 13, "middle", "bold"),
             f'<line x1="{zero_x:.1f}" y1="{y0 + 34:.1f}" '
             f'x2="{zero_x:.1f}" y2="{y0 + 60 + len(rows) * row_height:.1f}" '
             'stroke="#222"/>',
             svg_text(x0, y0 + 48, f"−{max_abs:.2f}", 9),
             svg_text(zero_x, y0 + 48, "0", 9, "middle"),
             svg_text(x0 + width, y0 + 48, f"+{max_abs:.2f}", 9, "end")]
    for index, row in enumerate(rows):
        value = float(row.get(f"{key}_effect") or 0.0)
        y = y0 + 72 + index * row_height
        bar_width = abs(value) * scale
        x = zero_x if value >= 0 else zero_x - bar_width
        color = "#c74343" if value >= 0 else "#356ea2"
        lines.append(f'<rect x="{x:.1f}" y="{y - 6:.1f}" width="{bar_width:.1f}" '
                     f'height="12" fill="{color}"/>')
        lines.append(svg_text(x + bar_width + 4 if value >= 0 else x - 4,
                              y + 3, f"{value:.3f}", 8,
                              "start" if value >= 0 else "end"))
    return lines


def make_motif_property_figures(output_dir: Path,
                                high_motifs: list[dict[str, Any]],
                                low_motifs: list[dict[str, Any]],
                                profiles: list[dict[str, Any]]) -> None:
    """Write physical-property and signed-property-term motif figures."""

    rows = motif_profile_plot_rows(high_motifs, low_motifs, profiles)
    row_height, label_width = 42, 440
    height, width, y0 = max(250, 105 + len(rows) * row_height), 1900, 34
    labels = [svg_text(width / 2, 22,
                       "Carrier-solvent physical properties by chemical motif",
                       16, "middle", "bold"),
              svg_text(16, 42, "red: high-conductivity group; blue: low-conductivity group", 10)]
    for index, row in enumerate(rows):
        y = y0 + 72 + index * row_height
        color = "#c74343" if row["group"] == "High" else "#356ea2"
        labels.append(svg_text(label_width - 10, y + 3,
                               f'{row["group"]}: {row["label"]}', 10,
                               "end", "bold"))
    high_count = sum(row["group"] == "High" for row in rows)
    if 0 < high_count < len(rows):
        separator_y = y0 + 72 + (high_count - 0.5) * row_height
        labels.append(f'<line x1="15" y1="{separator_y:.1f}" x2="{width - 15}" '
                      f'y2="{separator_y:.1f}" stroke="#999" stroke-dasharray="4,3"/>')
    panels = [
        property_profile_dot_panel(rows, "dc", "Dielectric constant εr",
                                   470, y0, 410, row_height),
        property_profile_dot_panel(rows, "vis", "Viscosity (mPa·s)",
                                   970, y0, 410, row_height),
        property_profile_dot_panel(rows, "Tm", "Melting point (°C)",
                                   1470, y0, 410, row_height),
    ]
    write_svg(output_dir / "motif_carrier_solvent_property_profiles.svg",
              width, height, labels + panels[0] + panels[1] + panels[2])

    effect_values = [abs(value) for key in ("dc_effect", "vis_effect", "Tm_effect")
                     for value in _finite_plot_values(rows, key)]
    max_abs = max(effect_values, default=0.01)
    effect_title = [svg_text(width / 2, 22,
                             "Signed property terms of carrier solvents by motif",
                             16, "middle", "bold"),
                    svg_text(16, 42,
                             "Each bar is a normalized log-conductivity term, not a percentage.",
                             10)]
    effect_labels = []
    for index, row in enumerate(rows):
        y = y0 + 72 + index * row_height
        effect_labels.append(svg_text(label_width - 10, y + 3,
                                      f'{row["group"]}: {row["label"]}', 10,
                                      "end", "bold"))
    if 0 < high_count < len(rows):
        separator_y = y0 + 72 + (high_count - 0.5) * row_height
        effect_labels.append(f'<line x1="15" y1="{separator_y:.1f}" x2="{width - 15}" '
                             f'y2="{separator_y:.1f}" stroke="#999" stroke-dasharray="4,3"/>')
    effect_panels = [
        property_effect_bar_panel(rows, "dc", "dc term", 470, y0, 410,
                                  row_height, max_abs),
        property_effect_bar_panel(rows, "vis", "vis term", 970, y0, 410,
                                  row_height, max_abs),
        property_effect_bar_panel(rows, "Tm", "Tm term", 1470, y0, 410,
                                  row_height, max_abs),
    ]
    write_svg(output_dir / "motif_carrier_property_terms.svg", width, height,
              effect_title + effect_labels + effect_panels[0] +
              effect_panels[1] + effect_panels[2])


def make_concentration_window_figure(output_dir: Path,
                                     samples: list[dict[str, Any]],
                                     windows: list[dict[str, Any]]) -> None:
    """Show observed concentration--prediction response in each context.

    The previous interval-only graphic could look like a conductivity
    distribution.  This replaces it with small multiples: x is concentration,
    y is the Bamboo prediction, grey is every observed point in that context,
    and red/shading identifies the selected high-prediction window.
    """

    windows_by_context: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for window in windows:
        key = (window["salt_context"], window["solvent_context"],
               window["temperature_context"])
        windows_by_context[key].append(window)
    observations: dict[tuple[str, str, str], list[dict[str, float]]] = defaultdict(list)
    for sample in samples:
        salt, solvent, temperature, _ = _condition_context(sample)
        key = (salt, solvent, temperature)
        if key in windows_by_context:
            observations[key].append({
                "concentration": float(sample["concentration"]),
                "prediction": float(sample["bamboo_prediction_ms_cm"]),
            })

    panels = [(key, windows_by_context[key], observations[key])
              for key in windows_by_context]
    panels.sort(key=lambda item: -max(
        float(window["mean_bamboo_prediction_ms_cm"]) for window in item[1]))
    columns, panel_width, panel_height = 4, 470, 280
    rows = max(1, math.ceil(len(panels) / columns))
    width, height = columns * panel_width + 30, rows * panel_height + 80
    elements = [
        svg_text(width / 2, 23,
                 "Observed concentration--conductivity response in global-high contexts",
                 16, "middle", "bold"),
        svg_text(16, 43,
                 "grey: all observed samples in the same binned condition; red: selected global-high window; line joins concentration-bin means",
                 10),
        svg_text(width - 16, 43, "Each panel has its own y scale (mS/cm).", 10,
                 "end"),
    ]
    for panel_index, (key, context_windows, context_observations) in enumerate(panels):
        column, row = panel_index % columns, panel_index // columns
        panel_x, panel_y = 15 + column * panel_width, 60 + row * panel_height
        chart_left, chart_right = panel_x + 58, panel_x + panel_width - 18
        chart_top, chart_bottom = panel_y + 68, panel_y + panel_height - 44
        salt, solvent, temperature = key
        title = f"{salt} | {temperature}"
        elements.append(f'<rect x="{panel_x}" y="{panel_y}" width="{panel_width - 8}" '
                        f'height="{panel_height - 10}" fill="#ffffff" stroke="#bbbbbb"/>')
        elements.append(svg_text(panel_x + (panel_width - 8) / 2, panel_y + 17,
                                 short_text(title, 44), 11, "middle", "bold"))
        elements.append(svg_text(panel_x + (panel_width - 8) / 2, panel_y + 33,
                                 short_text(solvent, 48), 10, "middle"))
        if not context_observations:
            elements.append(svg_text(panel_x + panel_width / 2, panel_y + 125,
                                     "No observations", 11, "middle"))
            continue
        concentration_values = [item["concentration"] for item in context_observations]
        prediction_values = [item["prediction"] for item in context_observations]
        x_min, x_max = min(concentration_values), max(concentration_values)
        y_min, y_max = min(prediction_values), max(prediction_values)
        x_pad = max((x_max - x_min) * 0.10, 0.005)
        y_pad = max((y_max - y_min) * 0.12, 0.25)
        x_min, x_max = max(0.0, x_min - x_pad), x_max + x_pad
        y_min, y_max = max(0.0, y_min - y_pad), y_max + y_pad
        x_scale = (chart_right - chart_left) / max(x_max - x_min, 1e-12)
        y_scale = (chart_bottom - chart_top) / max(y_max - y_min, 1e-12)
        def point_x(value: float) -> float:
            return chart_left + (value - x_min) * x_scale
        def point_y(value: float) -> float:
            return chart_bottom - (value - y_min) * y_scale
        for window in context_windows:
            shade_left = point_x(float(window["concentration_low"]))
            shade_right = point_x(float(window["concentration_high"]))
            elements.append(f'<rect x="{shade_left:.1f}" y="{chart_top:.1f}" '
                            f'width="{max(1.0, shade_right - shade_left):.1f}" '
                            f'height="{chart_bottom - chart_top:.1f}" '
                            'fill="#f6caca" fill-opacity="0.45"/>')
        for guide_fraction in (0.0, 0.5, 1.0):
            y = chart_bottom - guide_fraction * (chart_bottom - chart_top)
            value = y_min + guide_fraction * (y_max - y_min)
            elements.append(f'<line x1="{chart_left:.1f}" y1="{y:.1f}" '
                            f'x2="{chart_right:.1f}" y2="{y:.1f}" stroke="#e7e7e7"/>')
            elements.append(svg_text(chart_left - 5, y + 3, f"{value:.2f}", 8,
                                     "end"))
        by_concentration: dict[float, list[float]] = defaultdict(list)
        for observation in context_observations:
            by_concentration[round(observation["concentration"], 6)].append(
                observation["prediction"])
        mean_points = [(concentration, finite_mean(values) or 0.0)
                       for concentration, values in sorted(by_concentration.items())]
        polyline = " ".join(
            f"{point_x(concentration):.1f},{point_y(prediction):.1f}"
            for concentration, prediction in mean_points)
        elements.append(f'<polyline points="{polyline}" fill="none" '
                        'stroke="#8c8c8c" stroke-width="1.4"/>')
        def in_selected_window(concentration: float) -> bool:
            return any(float(window["concentration_low"]) - 1e-6 <= concentration <=
                       float(window["concentration_high"]) + 1e-6
                       for window in context_windows)
        for concentration, prediction in mean_points:
            x, y = point_x(concentration), point_y(prediction)
            selected = in_selected_window(concentration)
            color = "#c74343" if selected else "#777777"
            radius = 5 if selected else 3.5
            elements.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="{radius}" '
                            f'fill="{color}" stroke="white" stroke-width="1"/>')
        elements.append(f'<line x1="{chart_left:.1f}" y1="{chart_bottom:.1f}" '
                        f'x2="{chart_right:.1f}" y2="{chart_bottom:.1f}" stroke="#222"/>')
        elements.append(f'<line x1="{chart_left:.1f}" y1="{chart_top:.1f}" '
                        f'x2="{chart_left:.1f}" y2="{chart_bottom:.1f}" stroke="#222"/>')
        elements.append(svg_text(chart_left, chart_bottom + 16, f"{x_min:.3f}", 8))
        elements.append(svg_text(chart_right, chart_bottom + 16, f"{x_max:.3f}", 8,
                                 "end"))
        elements.append(svg_text((chart_left + chart_right) / 2, chart_bottom + 30,
                                 "salt_molar_ratio", 9, "middle"))
        mean_prediction = finite_mean(
            float(window["mean_bamboo_prediction_ms_cm"])
            for window in context_windows)
        elements.append(svg_text(chart_right, panel_y + 52,
                                 f"window mean={mean_prediction:.2f} mS/cm", 9,
                                 "end", "bold"))
    write_svg(output_dir / "global_high_concentration_windows.svg", width,
              height, elements)


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
               pair_rows: list[dict[str, Any]],
               motif_property_rows: list[dict[str, Any]],
               global_windows: list[dict[str, Any]],
               samples: list[dict[str, Any]],
               min_formulas: int) -> None:
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
    make_motif_property_figures(output_dir, high_fragments, low_fragments,
                                motif_property_rows)
    make_concentration_window_figure(output_dir, samples, global_windows)


def make_report(samples: list[dict[str, Any]], low_cutoff: float,
                high_cutoff: float, motif_rows: list[dict[str, Any]],
                pair_rows: list[dict[str, Any]], property_rows: list[dict[str, Any]],
                motif_property_rows: list[dict[str, Any]],
                contexts: list[dict[str, Any]],
                global_windows: list[dict[str, Any]],
                local_windows: list[dict[str, Any]],
                min_formulas: int, has_all_property_profiles: bool) -> str:
    high_total = sum(s["conductivity_group"] == "高电导率" for s in samples)
    low_total = sum(s["conductivity_group"] == "低电导率" for s in samples)
    high_fragments = candidates(motif_rows, "high", min_formulas,
                                "high_mean_main_effect_normalized")
    low_fragments = candidates(motif_rows, "low", min_formulas,
                               "low_mean_main_effect_normalized")
    low_negative_fragments = [
        row for row in low_fragments
        if float(row["low_mean_main_effect_normalized"] or 0.0) < 0.0
    ]
    low_context_dependent_fragments = [
        row for row in low_fragments
        if float(row["low_mean_main_effect_normalized"] or 0.0) >= 0.0
    ]
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
        f"- 可形成浓度上下文的近邻体系数：{len(contexts)}",
        f"- 全测试集高预测且归因方向稳定的浓度窗口数：{len(global_windows)}",
        f"- 体系内局部最优、归因方向稳定的浓度扫描候选数：{len(local_windows)}",
        "",
        "## 名词、含义与使用意义",
        "",
        "- **化学基元（motif）**：将模型的精确 BRICS 切割片段按主官能团和关键修饰基（如 `-O-；-CF2-`）归并。归并仅发生在报告中：同一配方中同一基元只计一次支持度，并将其多个精确片段贡献求和；模型输入与训练不变。",
        "- **高/低电导率常见基元**：按实验电导率上下分位数组划分后，在对应组中出现较多的化学基元。它反映数据集富集，而非单独的因果作用。",
        "- **出现率**：含某片段（或片段对）的配方数除以该高/低组总配方数。高/低出现率及其 odds ratio 用于判断片段偏向出现在哪一组。",
        "- **直接贡献（main effect）**：解释头在固定盐组成、盐浓度和温度条件下，分给一个溶剂片段的有符号归一化 log 电导率项。正值表示提高解释值，负值表示降低解释值；高/低表分别显示该组内平均值。",
        "- **协同/拮抗（interaction effect）**：来自不同溶剂分子的两个片段同时出现时的额外有符号项。正值是模型正协同候选，负值是模型负协同/拮抗候选；片段对表按对应高/低组内平均交互绝对值排序，不再优先按出现次数排序。",
        "- **条件基准**：盐组成、数据集盐浓度标量（源字段 `salt_molar_ratio`）和温度所给出的参考电导率。片段项描述的是相对于该基准的组成差异，不是盐效应本身。",
        "- **盐/溶剂组成比例**：`LiTFSI:1.00` 表示 LiTFSI 占该样本**盐混合物内部**的 100%；`AN:1.00` 表示 AN 占该样本**溶剂混合物内部**的 100%。两者不构成盐:溶剂 = 1:1。",
        "- **数据集盐摩尔分数（`salt_molar_ratio = c`）**：预处理文档定义 `c = n_salt / (n_salt + n_solvent)`。故盐 i 的全混合物摩尔分数为 `x_salt,i = c × r_salt,i`，溶剂 j 的全混合物摩尔分数为 `x_solvent,j = (1−c) × r_solvent,j`，其中 `r` 是表中的盐内/溶剂内比例。`c` 是无量纲摩尔分数，仍不是 mol/L。",
        "- **浓度窗口**：在盐组成、温度和溶剂比例均相近的已观测样本中，若干较高 Bamboo 主预测标量点覆盖的扫描范围。它不是对未观测浓度的因果外推；比例以 0.05 分箱、温度以 5 °C 邻域定义“相近”。",
        "- **官能团式**：如 `-OH`、`-CF3`、`-C#N` 的连接式，仅用于结构阅读；它不是把被切分片段自动补氢后的完整分子式。",
        "- **带连接位点的片段 SMILES**：`*` 表示该片段在母体溶剂中被切开的连接位点。例如 `*OC` 是醚氧连接的甲氧基端，而不是甲醇；它用于避免把断键后的氢封端误判为羟基。",
        "- **精确模型片段**：带 `*` 的 SMILES 是实际输入模型的非重叠切割片段。它们保留在 `fragment_attribution_summary.csv` 中作审计；主表不再将仅切割边界不同的雷同片段重复列出。",
        "- **复合结构描述**：如 `-O-；-CF2-` 表示同一个化学基元中同时有醚键和二氟亚甲基；分号不是把两个候选片段或两个分子拼在一起。",
        "- **支持度**：某候选出现于多少个独立配方。支持度太低的项目只保留在完整 CSV 中，不应作为稳定设计结论。",
        "",
        "## 条件化浓度窗口：实验优先尝试范围",
        "",
        "本节不表示“只有表内两种配方值得做”。它只列出测试集恰好含有至少 3 个相邻标量点、且组成/温度相近的配方序列。多数多溶剂体系虽存在于测试集，但没有足够密的同组成浓度序列，因此不能从观察数据中宣称一个稳定窗口。",
        "表中范围是这些较高预测观测点的最小—最大覆盖范围，不表示所有未观测的内部标量点都已验证；实验中仍需补测中间点和范围外边界点。",
        "",
        "### 全测试集高预测优先窗口",
        "",
        "以下窗口要求每个点均达到全测试集 Bamboo 预测上四分位，同时 `dc`、`vis` 和片段总残差方向稳定。因此它们是最强的实验优先证据，而非仅有的候选体系。",
        "“全混合物摩尔分数端点”由上式根据 0.05 分箱后的组成近似换算。例如 `LiPF6:1.00 / DEC:0.25+PC:0.75 / c=0.0612–0.1021` 对应端点约为：低端 `LiPF6:0.0612, DEC:0.2347, PC:0.7041`；高端 `LiPF6:0.1021, DEC:0.2245, PC:0.6734`。因此扫描时保持 DEC:PC=1:3，仅提高总盐摩尔分数。",
        "",
        *concentration_window_table(global_windows),
        "",
        "### 体系内局部最优扫描候选",
        "",
        "以下窗口在各自盐种/温度/溶剂比例上下文中位于预测较高的 40% 区域，且归因方向稳定，但未必达到全测试集绝对高值。它们特别适合用于多组分体系的下一轮浓度扫描，而不应与上表混称为全局最优。",
        "",
        *concentration_window_table(local_windows, limit=20),
        "",
        "## 分子性质机制：为何该配方可能获得较高电导率",
        "",
        "表中是先按固定分子预训练的 min/max 反归一化、再按溶剂摩尔比例汇总的物理性质描述符。`dc`、`vis`、`Tm` 同时是解释头的直接贡献项；其余性质只提供与极化、相态、挥发性和稳定性相关的支持线索，不能被误读为独立电导率贡献。混合黏度与混合介电常数并非严格线性混合，本表仅用于比较模型所见的组成趋势。",
        "",
        *property_table(property_rows),
        "",
        ("完整 11 项分子性质已导出。" if has_all_property_profiles else
         "当前 JSON 仅含 `dc/vis/Tm`；重新运行 `predict` 后将自动补齐冻结分子模型的 11 项性质诊断。"),
        "",
        "## 高电导率常见化学基元",
        "",
        "以下要求高组出现率至少比低组高 **2 个百分点**，且达到最小支持度。每一行是配方内去重、精确片段贡献求和后的化学基元。正直接贡献更适合作为提高电导率的结构候选；负直接贡献说明它虽高组富集，但其作用可能依赖其他组分或条件。",
        "",
        *fragment_table(high_fragments, 20, "high_mean_main_effect_normalized",
                        "高组内平均直接贡献"),
        "",
        "## 低电导率常见化学基元",
        "",
        "以下要求低组出现率至少比高组高 **2 个百分点**，且达到最小支持度。只有**负直接贡献且低组富集**的基元，才可优先列作待避开候选；精确 SMILES 变体见 CSV 审计附件。",
        "",
        *fragment_table(low_negative_fragments, 20, "low_mean_main_effect_normalized",
                        "低组内平均直接贡献"),
        "",
        "### 低组富集、但直接贡献为正或接近零的基元",
        "",
        "这些基元常出现在低电导率配方中，但在控制盐种、数据集盐浓度标量、温度和其他结构条件后，解释头仍给出正向或接近零的局部直接贡献。因此**不能**据此断言“该基元会降低电导率”，也不应直接加入待避开清单；它更可能与高黏度、低迁移率、盐浓度、其他溶剂或片段交互等共同条件相关。",
        "",
        *fragment_table(low_context_dependent_fragments, 20,
                        "low_mean_main_effect_normalized",
                        "低组内平均直接贡献"),
        "",
        "## 化学基元与承载溶剂的物性画像",
        "",
        "下表把高/低组常见基元与**含有该基元的溶剂**的物性画像联系起来。介电常数、黏度和熔点为反归一化后的载体溶剂描述符；`dc / vis / Tm` 是解释头的三个**有符号归一化 log 电导率项**，最后一列是三者之和。正值将解释预测上推，负值下推；因此不是“数值越小越差”，而是**负值越大（绝对值越大）才表示更强的降低方向**。这些项不是百分比，不要求合计为 1：单一配方内三项求和只得到受 `tanh` 限制的物性子项（上限 0.25），而完整解释还包含条件基准、片段主效应和片段对交互；跨 motif 行也不能相加，因为同一载体溶剂可承载多个基元。",
        "",
        *motif_property_table(high_fragments, low_fragments,
                              motif_property_rows),
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
        "前五张图均来自本报告上方同一批、通过最小支持度筛选的候选项；第六张图来自全部全局高预测窗口。直接贡献图已按化学基元归并，第二行 `solv:` 显示前两种代表溶剂和该基元覆盖的溶剂总数；片段对图仍保留精确模型片段，以便审计交互来源。完整结构和完整溶剂清单仍以 CSV 为准。贡献/交互/物性项图中红色为正方向、蓝色为负方向；物性原值图中红蓝仅表示高/低电导率组。",
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
        "**图 4：化学基元的承载溶剂物性。** 每一行对应上方表中的一个基元—承载溶剂组合；红色为高组、蓝色为低组。三列分别在各自实际单位中显示介电常数、黏度和熔点，因此用于辨别结构候选伴随的盐解离、迁移率和低温液态范围线索；不同列的横轴独立，不能直接比较数值大小。",
        "",
        "![化学基元承载溶剂的物性画像](motif_carrier_solvent_property_profiles.svg)",
        "",
        "**图 5：承载溶剂的 dc / vis / Tm 有符号物性项。** 每根条对应表内一个物性直接贡献，红色为正、蓝色为负；它们是归一化 log 电导率项而非占比。应沿同一行比较三项的方向与相对大小，而不应跨行求和。",
        "",
        "![化学基元承载溶剂的物性直接贡献](motif_carrier_property_terms.svg)",
        "",
        "**图 6：全局高预测体系的浓度—电导率响应。** 每个小面板固定一个盐组成、温度邻域和溶剂比例上下文。横轴为 `salt_molar_ratio`，纵轴为该上下文中各观测浓度点的 Bamboo 主预测（mS/cm）；灰点及灰线表示全部观测浓度点及其按浓度汇总的均值，红点与浅红阴影表示最终筛出的全局高预测窗口。连线只连接观测点，不表示对中间浓度的插值。为看清各体系内部的变化趋势，每个面板采用独立纵轴，面板右上角的 `window mean` 用于跨体系比较绝对预测水平。",
        "",
        "![全局高预测浓度窗口](global_high_concentration_windows.svg)",
        "",
        "## 使用边界",
        "",
        "- 本报告的直接贡献和交互贡献严格加和到解释头的片段残差；它们不修改 Bamboo 最终预测。",
        "- 片段及片段对的正负号是模型在给定条件下学到的关联，不等同于化学实验中的单因素因果效应。",
        "- 设计候选应优先选择：高支持度、在不同配方中方向一致、同时具有高组富集和正贡献（或低组富集和负贡献）的项。",
        "- 浓度窗口的第一轮实验应优先在窗口内部取低/中/高三个点，并在边界外各补一个点，以验证是否存在真实平台而非数据偶然聚集；如原始数据可以提供 `salt_molar_ratio` 到 mol/L 的换算，请在实验计划中先完成该换算。",
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
    exact_fragment_occurrences = collect_fragment_occurrences(samples)
    fragment_rows = summarize_fragments(exact_fragment_occurrences,
                                       high_total, low_total)
    motif_rows = summarize_motifs(
        collect_motif_occurrences(exact_fragment_occurrences), high_total,
        low_total)
    pair_rows = summarize_pairs(
        collect_pair_occurrences(samples), high_total, low_total)
    property_rows = summarize_molecular_properties(samples)
    motif_property_rows = summarize_motif_property_profiles(
        collect_motif_property_occurrences(samples))
    contexts, global_windows, local_windows = concentration_windows(samples)
    has_all_property_profiles = any(
        "molecular_property_profiles" in sample["record"] for sample in samples)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    high_motif_candidates = candidates(
        motif_rows, "high", args.min_formulas,
        "high_mean_main_effect_normalized")
    low_motif_candidates = candidates(
        motif_rows, "low", args.min_formulas,
        "low_mean_main_effect_normalized")
    low_negative_motif_candidates = [
        row for row in low_motif_candidates
        if float(row["low_mean_main_effect_normalized"] or 0.0) < 0.0
    ]
    low_context_dependent_motif_candidates = [
        row for row in low_motif_candidates
        if float(row["low_mean_main_effect_normalized"] or 0.0) >= 0.0
    ]
    write_csv(output_dir / "fragment_attribution_summary.csv", fragment_rows)
    write_csv(output_dir / "chemical_motif_attribution_summary.csv", motif_rows)
    write_csv(output_dir / "high_conductivity_common_fragments.csv",
              high_motif_candidates)
    write_csv(output_dir / "low_conductivity_common_fragments.csv",
              low_motif_candidates)
    write_csv(output_dir / "low_conductivity_negative_contribution_motifs.csv",
              low_negative_motif_candidates)
    write_csv(output_dir / "low_conductivity_context_dependent_motifs.csv",
              low_context_dependent_motif_candidates)
    write_csv(output_dir / "fragment_pair_interaction_summary.csv", pair_rows)
    write_csv(output_dir / "high_conductivity_fragment_pairs.csv",
              candidates(pair_rows, "high", args.min_formulas,
                         "high_mean_interaction_normalized"))
    write_csv(output_dir / "low_conductivity_fragment_pairs.csv",
              candidates(pair_rows, "low", args.min_formulas,
                         "low_mean_interaction_normalized"))
    write_csv(output_dir / "molecular_property_mechanism_summary.csv",
              property_rows)
    write_csv(output_dir / "motif_carrier_solvent_property_profiles.csv",
              motif_property_rows)
    write_csv(output_dir / "concentration_context_summary.csv", contexts)
    write_csv(output_dir / "recommended_concentration_windows.csv",
              global_windows)
    write_csv(output_dir / "local_concentration_scan_candidates.csv",
              local_windows)
    metadata = {
        "input": str(args.input), "experimental_formula_count": len(samples),
        "quantile": args.quantile, "low_cutoff_ms_cm": low_cutoff,
        "high_cutoff_ms_cm": high_cutoff, "min_formulas": args.min_formulas,
        "high_formula_count": high_total, "low_formula_count": low_total,
        "motif_count": len(motif_rows),
        "concentration_context_count": len(contexts),
        "recommended_concentration_window_count": len(global_windows),
        "local_concentration_scan_candidate_count": len(local_windows),
        "has_all_molecular_property_profiles": has_all_property_profiles,
    }
    (output_dir / "analysis_metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False) + "\n")
    make_plots(output_dir, motif_rows, pair_rows, motif_property_rows,
               global_windows, samples, args.min_formulas)
    (output_dir / "fragment_attribution_report_zh.md").write_text(
        make_report(samples, low_cutoff, high_cutoff, motif_rows, pair_rows,
                    property_rows, motif_property_rows, contexts,
                    global_windows, local_windows, args.min_formulas,
                    has_all_property_profiles))
    print(f"Wrote comprehensive fragment attribution analysis to: {output_dir}")


if __name__ == "__main__":
    main()
