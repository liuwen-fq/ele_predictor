"""Build attention-based fragment explanation artifacts for experimental tests.

This script analyzes the ``fragment_attention`` field produced by
``scripts/test_results/predict.py``. The resulting scores describe relative
fragment pooling attention within each neutral solvent molecule; they are not
causal conductivity attributions.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter, defaultdict
from itertools import combinations
from pathlib import Path
from statistics import mean, median
from typing import Any, Iterable

from rdkit import Chem
from rdkit.Chem import rdMolDescriptors

LOG_SHIFT = 4.6002
LOG_SCALE = 8.3344


PRIMARY_GROUP_SMARTS = (
    ("环状碳酸酯", "[O;R][C;R](=[O])[O;R]"),
    ("内酯（环状酯）", "[C;R](=[O])[O;R]"),
    ("碳酸酯", "[OX2][CX3](=[OX1])[OX2]"),
    ("磺酰胺", "[S](=[O])(=[O])[N]"),
    ("硫酸酯", "[OX2][S](=[OX1])(=[OX1])[OX2]"),
    ("亚硫酸酯", "[OX2][S](=[OX1])[OX2]"),
    ("磺酰基", "[S](=[O])(=[O])"),
    ("磷酸酯/磷酰基", "[P](=[O])"),
    ("酰胺", "[CX3](=[OX1])[NX3]"),
    ("羧酸/羧酸盐", "[CX3](=[OX1])[O;H1,-1]"),
    ("酯基", "[CX3](=[OX1])[OX2][!#1]"),
    ("腈基", "[CX2]#[NX1]"),
    ("醚键", "[!#1][OX2][!#1]"),
    ("羟基", "[OX2H]"),
)
PRIMARY_GROUP_PATTERNS = tuple(
    (label, Chem.MolFromSmarts(smarts))
    for label, smarts in PRIMARY_GROUP_SMARTS)

FUNCTIONAL_GROUP_SHORTHANDS = {
    "环状碳酸酯": "-O-C(=O)-O-（环）",
    "内酯（环状酯）": "-C(=O)-O-（环）",
    "碳酸酯": "-O-C(=O)-O-",
    "磺酰胺": "-SO2-NH-",
    "硫酸酯": "-O-S(=O)2-O-",
    "亚硫酸酯": "-O-S(=O)-O-",
    "磺酰基": "-SO2-",
    "磷酸酯/磷酰基": "-O-P(=O)(-O-)-O-",
    "酰胺": "-C(=O)-NH-",
    "羧酸/羧酸盐": "-C(=O)OH",
    "酯基": "-C(=O)-O-",
    "腈基": "-C#N",
    "醚键": "-O-",
    "羟基": "-OH",
    "三氟甲基（-CF3）": "-CF3",
    "二氟亚甲基（-CF2-）": "-CF2-",
    "芳香环": "-Ar-",
    "非芳香环状骨架": "-(CH2)n-",
    "烃基/未匹配特征基团": "-CH2-/-CH3-",
    "未识别片段": "--",
}


def percentile(values: list[float], quantile: float) -> float:
    """Return an interpolated quantile without requiring NumPy."""

    if not values:
        raise ValueError("cannot calculate a percentile of an empty list")
    sorted_values = sorted(values)
    position = (len(sorted_values) - 1) * quantile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return sorted_values[lower]
    fraction = position - lower
    return sorted_values[lower] * (1 - fraction) + sorted_values[upper] * fraction


def conductivity_ms_cm(normalized_conductivity: float) -> float:
    return math.exp(normalized_conductivity * LOG_SCALE - LOG_SHIFT)


def canonical_fragment_smiles(fragment_smiles: str) -> str:
    """Remove explicit hydrogens and atom maps from a fragment SMILES."""
    molecule = Chem.MolFromSmiles(fragment_smiles)
    if molecule is None:
        return fragment_smiles
    molecule = Chem.RemoveHs(molecule)
    for atom in molecule.GetAtoms():
        atom.SetAtomMapNum(0)
        if atom.GetAtomicNum() == 0:
            atom.SetIsotope(0)
    return Chem.MolToSmiles(molecule, canonical=True, isomericSmiles=True)


def fragment_formula(fragment_smiles: str) -> str:
    """Return the molecular formula of the capped fragment SMILES.

    The fragmentation records use explicit hydrogens to cap cut bonds.  The
    returned formula therefore describes the stored fragment itself and does
    not count ``*`` connection placeholders used in attachment SMILES.
    """

    molecule = Chem.MolFromSmiles(fragment_smiles)
    if molecule is None:
        return ""
    molecule = Chem.RemoveHs(molecule)
    return rdMolDescriptors.CalcMolFormula(molecule).replace("*", "")


def functional_group_shorthand(functional_group: str) -> str:
    """Return a human-readable connection-style shorthand for a group."""

    return FUNCTIONAL_GROUP_SHORTHANDS.get(functional_group, "--")


def classify_functional_group(fragment_smiles: str) -> dict[str, Any]:
    """Assign one specific primary group plus non-overlapping modifiers."""
    molecule = Chem.MolFromSmiles(fragment_smiles)
    if molecule is None:
        return {
            "functional_group": "未识别片段",
            "functional_group_modifiers": [],
            "functional_group_label": "未识别片段",
        }
    primary = next((label for label, pattern in PRIMARY_GROUP_PATTERNS
                    if pattern is not None
                    and molecule.HasSubstructMatch(pattern)), None)
    fluorine_counts = [
        sum(neighbor.GetAtomicNum() == 9 for neighbor in atom.GetNeighbors())
        for atom in molecule.GetAtoms() if atom.GetAtomicNum() == 6
    ]
    modifiers = []
    if 3 in fluorine_counts:
        modifiers.append("含三氟甲基（-CF3）")
    if 2 in fluorine_counts:
        modifiers.append("含二氟亚甲基（-CF2-）")
    if any(atom.GetIsAromatic() for atom in molecule.GetAtoms()):
        modifiers.append("芳香环")
    elif molecule.GetRingInfo().NumRings() > 0:
        modifiers.append("非芳香环状骨架")
    if primary is None:
        if "含三氟甲基（-CF3）" in modifiers:
            primary = "三氟甲基（-CF3）"
            modifiers.remove("含三氟甲基（-CF3）")
        elif "含二氟亚甲基（-CF2-）" in modifiers:
            primary = "二氟亚甲基（-CF2-）"
            modifiers.remove("含二氟亚甲基（-CF2-）")
        elif "芳香环" in modifiers:
            primary = "芳香环"
            modifiers.remove("芳香环")
        elif "非芳香环状骨架" in modifiers:
            primary = "非芳香环状骨架"
            modifiers.remove("非芳香环状骨架")
        else:
            primary = "烃基/未匹配特征基团"
    label = primary
    if modifiers:
        label += "；" + "；".join(modifiers)
    return {
        "functional_group": primary,
        "functional_group_modifiers": modifiers,
        "functional_group_label": label,
    }


def attachment_smiles_by_fragment(solvent: dict[str, Any]) -> dict[int, str]:
    """Rebuild canonical fragments with ``*`` at every cut-bond endpoint."""
    fallback = {
        int(fragment["fragment_id"]): canonical_fragment_smiles(
            fragment.get("smiles", ""))
        for fragment in solvent.get("fragments", [])
    }
    connections = solvent.get("connections", [])
    if not connections:
        return fallback
    molecule = Chem.MolFromSmiles(solvent.get("mapped_smiles", ""))
    if molecule is None:
        return fallback
    map_to_atom = {
        atom.GetAtomMapNum(): atom.GetIdx() for atom in molecule.GetAtoms()
        if atom.GetAtomMapNum() > 0
    }
    cut_bonds = []
    for connection in connections:
        original_indices = connection.get("atom_indices", [])
        if len(original_indices) != 2:
            continue
        begin = map_to_atom.get(int(original_indices[0]) + 1)
        end = map_to_atom.get(int(original_indices[1]) + 1)
        if begin is None or end is None:
            continue
        bond = molecule.GetBondBetweenAtoms(begin, end)
        if bond is not None:
            cut_bonds.append(bond.GetIdx())
    if not cut_bonds:
        return fallback
    fragmented = Chem.FragmentOnBonds(
        molecule, sorted(set(cut_bonds)), addDummies=True)
    # Parsing mapped SMILES removes explicit H atoms. Match fragments by the
    # retained heavy-atom map numbers and ignore hydrogen indices in metadata.
    fragment_atom_sets = {
        int(fragment["fragment_id"]): {
            int(atom_index) for atom_index in fragment.get("atom_indices", [])
            if int(atom_index) + 1 in map_to_atom
        }
        for fragment in solvent.get("fragments", [])
    }
    output = dict(fallback)
    for fragment_molecule in Chem.GetMolFrags(fragmented,
                                              asMols=True,
                                              sanitizeFrags=True):
        original_atoms = {
            atom.GetAtomMapNum() - 1 for atom in fragment_molecule.GetAtoms()
            if atom.GetAtomicNum() != 0 and atom.GetAtomMapNum() > 0
        }
        fragment_id = next((fragment_id for fragment_id, atom_indices
                            in fragment_atom_sets.items()
                            if original_atoms == atom_indices), None)
        if fragment_id is None:
            continue
        for atom in fragment_molecule.GetAtoms():
            atom.SetAtomMapNum(0)
            if atom.GetAtomicNum() == 0:
                atom.SetIsotope(0)
        fragment_molecule = Chem.RemoveHs(fragment_molecule)
        output[fragment_id] = Chem.MolToSmiles(
            fragment_molecule, canonical=True, isomericSmiles=True)
    return output


def finite_mean(values: Iterable[float]) -> float | None:
    values = [value for value in values if math.isfinite(value)]
    return mean(values) if values else None


def write_json(path: Path, content: Any) -> None:
    with path.open("w") as file:
        json.dump(content, file, indent=2)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("")
        return
    columns = list(rows[0].keys())
    with path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def experimental_records(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    selected = []
    for sample_index, record in enumerate(records):
        if not record.get("conductivity_mask", False):
            continue
        if "fragment_attention" not in record:
            raise ValueError(
                "fragment_attention is missing. Run scripts/test_results/predict.py "
                "with the fragment-aware model first.")
        target = conductivity_ms_cm(float(record["conductivity"]))
        prediction = float(record["conductivity_pred"])
        selected.append({
            "sample_index": sample_index,
            "target_conductivity_ms_cm": target,
            "prediction_conductivity_ms_cm": prediction,
            "residual_ms_cm": prediction - target,
            "abs_error_ms_cm": abs(prediction - target),
            "log_residual": math.log(max(prediction, 1e-12)) - math.log(target),
            "temperature_c": float(record["temperature"]),
            "salt_molar_ratio": float(record["salt_molar_ratio"]),
            "salts": record.get("salts", []),
            "solvents": record.get("solvents", []),
            "fragment_attention": record["fragment_attention"],
        })
    if not selected:
        raise ValueError("no experimental conductivity samples were found")
    return selected


def annotate_solvent_fragments(sample: dict[str, Any]) -> list[dict[str, Any]]:
    """Create explainable solvent fragment rows for one formula sample."""

    molar_ratios = {
        solvent.get("name"): float(solvent.get("molar_ratio", 0.0))
        for solvent in sample["solvents"]
    }
    output = []
    for solvent in sample["fragment_attention"].get("solvents", []):
        if not solvent.get("fragment_encoder_applied", False):
            continue
        fragments = solvent.get("fragments", [])
        fragment_count = len(fragments)
        if fragment_count == 0:
            continue
        attachment_smiles = attachment_smiles_by_fragment(solvent)
        solvent_name = solvent.get("molecule_name", "unknown")
        molar_ratio = molar_ratios.get(solvent_name, 0.0)
        for fragment in fragments:
            attention = float(fragment.get("pooling_attention", 0.0))
            raw_smiles = fragment.get("smiles", "")
            connected_smiles = attachment_smiles.get(
                int(fragment.get("fragment_id", -1)),
                canonical_fragment_smiles(raw_smiles))
            group = classify_functional_group(connected_smiles)
            is_whole_molecule = fragment_count == 1
            output.append({
                "sample_index": sample["sample_index"],
                "target_conductivity_ms_cm": sample["target_conductivity_ms_cm"],
                "prediction_conductivity_ms_cm": sample[
                    "prediction_conductivity_ms_cm"],
                "temperature_c": sample["temperature_c"],
                "salt_molar_ratio": sample["salt_molar_ratio"],
                "solvent_name": solvent_name,
                "solvent_smiles": solvent.get("mapped_smiles", ""),
                "solvent_molar_ratio": molar_ratio,
                "fragment_id": fragment.get("fragment_id"),
                "fragment_smiles": raw_smiles,
                "raw_fragment_smiles": raw_smiles,
                "canonical_fragment_smiles": canonical_fragment_smiles(
                    raw_smiles),
                "attachment_smiles": connected_smiles,
                "fragment_formula": fragment_formula(raw_smiles),
                **group,
                "functional_group_shorthand": functional_group_shorthand(
                    group["functional_group"]),
                "atom_indices": fragment.get("atom_indices", []),
                "attachment_atom_indices": fragment.get(
                    "attachment_atom_indices", []),
                "pooling_attention": attention,
                "attention_enrichment": attention * fragment_count,
                "formula_weighted_attention": attention * molar_ratio,
                "fragment_count_in_molecule": fragment_count,
                "is_whole_molecule_fragment": is_whole_molecule,
                "attention_comparable": not is_whole_molecule,
            })
    return output


def classify_samples(samples: list[dict[str, Any]], quantile: float) -> tuple[float, float]:
    targets = [sample["target_conductivity_ms_cm"] for sample in samples]
    return percentile(targets, quantile), percentile(targets, 1 - quantile)


def case_explanations(samples: list[dict[str, Any]], low_cutoff: float,
                      high_cutoff: float) -> list[dict[str, Any]]:
    output = []
    for sample in samples:
        fragments = annotate_solvent_fragments(sample)
        fragments.sort(key=lambda row: row["formula_weighted_attention"],
                       reverse=True)
        conductivity_group = "middle"
        if sample["target_conductivity_ms_cm"] <= low_cutoff:
            conductivity_group = "low"
        elif sample["target_conductivity_ms_cm"] >= high_cutoff:
            conductivity_group = "high"
        output.append({
            **sample,
            "conductivity_group": conductivity_group,
            "neutral_solvent_fragments": fragments,
            "ranked_multi_fragment_attention": [
                fragment for fragment in fragments
                if fragment["attention_comparable"]
            ],
            "whole_molecule_fragments": [
                fragment for fragment in fragments
                if fragment["is_whole_molecule_fragment"]
            ],
            "excluded_charged_salt_fragments": [
                salt for salt in sample["fragment_attention"].get("salts", [])
                if not salt.get("fragment_encoder_applied", False)
            ],
        })
        output[-1].pop("fragment_attention")
    return output


def fragment_summary(fragment_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in fragment_rows:
        grouped[row["attachment_smiles"]].append(row)
    summary = []
    for attachment_smiles, rows in grouped.items():
        attention = [row["pooling_attention"] for row in rows]
        enrichment = [row["attention_enrichment"] for row in rows]
        weighted = [row["formula_weighted_attention"] for row in rows]
        raw_smiles = Counter(row["raw_fragment_smiles"] for row in rows)
        formulas = Counter(row["fragment_formula"] for row in rows)
        summary.append({
            "attachment_smiles": attachment_smiles,
            "representative_raw_fragment_smiles": raw_smiles.most_common(1)[0][0],
            "fragment_formula": formulas.most_common(1)[0][0],
            "functional_group": rows[0]["functional_group"],
            "functional_group_label": rows[0]["functional_group_label"],
            "functional_group_shorthand": rows[0]["functional_group_shorthand"],
            "occurrences": len(rows),
            "unique_formulas": len({row["sample_index"] for row in rows}),
            "unique_solvents": len({row["solvent_name"] for row in rows}),
            "mean_pooling_attention": mean(attention),
            "median_pooling_attention": median(attention),
            "mean_attention_enrichment": mean(enrichment),
            "mean_formula_weighted_attention": mean(weighted),
            "mean_target_conductivity_ms_cm": mean(
                row["target_conductivity_ms_cm"] for row in rows),
        })
    return sorted(summary,
                  key=lambda row: row["mean_formula_weighted_attention"],
                  reverse=True)


def formula_functional_group_rows(
        fragment_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Merge repeated occurrences of one functional group within a formula."""
    grouped: dict[tuple[int, str], list[dict[str, Any]]] = defaultdict(list)
    for row in fragment_rows:
        grouped[(row["sample_index"], row["functional_group"])].append(row)
    output = []
    for (sample_index, functional_group), rows in grouped.items():
        structures = Counter((row["attachment_smiles"], row["fragment_formula"])
                             for row in rows)
        representative_structure, representative_formula = structures.most_common(1)[0][0]
        output.append({
            "sample_index": sample_index,
            "functional_group": functional_group,
            "functional_group_shorthand": rows[0]["functional_group_shorthand"],
            "representative_attachment_smiles": representative_structure,
            "representative_fragment_formula": representative_formula,
            "fragment_occurrences_in_formula": len(rows),
            "formula_weighted_attention": sum(
                row["formula_weighted_attention"] for row in rows),
            "mean_attention_enrichment": mean(
                row["attention_enrichment"] for row in rows),
            "target_conductivity_ms_cm": rows[0]["target_conductivity_ms_cm"],
        })
    return output


def functional_group_summary(
        formula_group_rows: list[dict[str, Any]],
        experimental_sample_count: int) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in formula_group_rows:
        grouped[row["functional_group"]].append(row)
    output = []
    for functional_group, rows in grouped.items():
        structures = Counter((row["representative_attachment_smiles"],
                              row["representative_fragment_formula"])
                             for row in rows)
        representatives = structures.most_common(3)
        weighted = [row["formula_weighted_attention"] for row in rows]
        output.append({
            "functional_group": functional_group,
            "functional_group_shorthand": rows[0]["functional_group_shorthand"],
            "representative_attachment_smiles": "；".join(
                structure for (structure, _), _ in representatives),
            "representative_fragment_formula": "；".join(
                formula for (_, formula), _ in representatives),
            "unique_formulas": len(rows),
            "formula_prevalence": len(rows) / experimental_sample_count,
            "fragment_occurrences": sum(
                row["fragment_occurrences_in_formula"] for row in rows),
            "mean_formula_weighted_attention_when_present": mean(weighted),
            "dataset_mean_formula_weighted_attention": sum(weighted) /
            experimental_sample_count,
            "mean_attention_enrichment": mean(
                row["mean_attention_enrichment"] for row in rows),
        })
    return sorted(output,
                  key=lambda row: row[
                      "dataset_mean_formula_weighted_attention"],
                  reverse=True)


def functional_group_association(
        formula_group_rows: list[dict[str, Any]], samples: list[dict[str, Any]],
        low_cutoff: float, high_cutoff: float) -> list[dict[str, Any]]:
    score_by_formula_group = {
        (row["sample_index"], row["functional_group"]):
        row["formula_weighted_attention"] for row in formula_group_rows
    }
    structures: dict[str, Counter] = defaultdict(Counter)
    for row in formula_group_rows:
        structures[row["functional_group"]][
            (row["representative_attachment_smiles"],
             row["representative_fragment_formula"])] += 1
    low_ids = [sample["sample_index"] for sample in samples
               if sample["target_conductivity_ms_cm"] <= low_cutoff]
    high_ids = [sample["sample_index"] for sample in samples
                if sample["target_conductivity_ms_cm"] >= high_cutoff]
    output = []
    for functional_group in sorted(structures):
        low_values = [score_by_formula_group.get((sample_id, functional_group), 0.0)
                      for sample_id in low_ids]
        high_values = [score_by_formula_group.get((sample_id, functional_group), 0.0)
                       for sample_id in high_ids]
        low_mean = mean(low_values)
        high_mean = mean(high_values)
        output.append({
            "functional_group": functional_group,
            "functional_group_shorthand": functional_group_shorthand(
                functional_group),
            "representative_attachment_smiles": "；".join(
                structure for (structure, _), _
                in structures[functional_group].most_common(3)),
            "representative_fragment_formula": "；".join(
                formula for (_, formula), _
                in structures[functional_group].most_common(3)),
            "low_formula_count": sum(value > 0 for value in low_values),
            "high_formula_count": sum(value > 0 for value in high_values),
            "low_group_size": len(low_ids),
            "high_group_size": len(high_ids),
            "low_mean_formula_weighted_attention": low_mean,
            "high_mean_formula_weighted_attention": high_mean,
            "high_minus_low_weighted_attention": high_mean - low_mean,
        })
    return sorted(output,
                  key=lambda row: abs(row[
                      "high_minus_low_weighted_attention"]),
                  reverse=True)


def _odds_ratio(present_a: int, absent_a: int, present_b: int,
                absent_b: int) -> float:
    """Return a continuity-corrected odds ratio for sparse group counts."""

    return ((present_a + 0.5) * (absent_b + 0.5) /
            ((absent_a + 0.5) * (present_b + 0.5)))


def functional_group_composition_association(
        formula_group_rows: list[dict[str, Any]], samples: list[dict[str, Any]],
        low_cutoff: float, high_cutoff: float) -> list[dict[str, Any]]:
    """Compare solvent functional-group presence in high and low formulas.

    Unlike attention, this treats a functional group as present once per
    formula. It includes whole-molecule solvents so EC, DMC, and other
    uncut molecules remain part of composition-level screening.
    """

    group_sample_ids: dict[str, set[int]] = defaultdict(set)
    representatives: dict[str, Counter] = defaultdict(Counter)
    for row in formula_group_rows:
        group = row["functional_group"]
        group_sample_ids[group].add(row["sample_index"])
        representatives[group][(row["representative_attachment_smiles"],
                                row["representative_fragment_formula"])] += 1
    low_ids = {
        sample["sample_index"] for sample in samples
        if sample["target_conductivity_ms_cm"] <= low_cutoff
    }
    high_ids = {
        sample["sample_index"] for sample in samples
        if sample["target_conductivity_ms_cm"] >= high_cutoff
    }
    output = []
    for group, sample_ids in group_sample_ids.items():
        high_count = len(sample_ids & high_ids)
        low_count = len(sample_ids & low_ids)
        high_prevalence = high_count / len(high_ids)
        low_prevalence = low_count / len(low_ids)
        odds_ratio = _odds_ratio(high_count, len(high_ids) - high_count,
                                 low_count, len(low_ids) - low_count)
        representative, formula = representatives[group].most_common(1)[0][0]
        if odds_ratio > 1:
            direction = "高电导率关联候选"
        elif odds_ratio < 1:
            direction = "低电导率关联候选"
        else:
            direction = "无组间差异"
        output.append({
            "functional_group": group,
            "functional_group_shorthand": functional_group_shorthand(group),
            "representative_attachment_smiles": representative,
            "representative_fragment_formula": formula,
            "high_formula_count": high_count,
            "low_formula_count": low_count,
            "high_formula_prevalence": high_prevalence,
            "low_formula_prevalence": low_prevalence,
            "high_minus_low_prevalence": high_prevalence - low_prevalence,
            "high_vs_low_odds_ratio": odds_ratio,
            "log_high_vs_low_odds_ratio": math.log(odds_ratio),
            "association_candidate": direction,
        })
    return sorted(output,
                  key=lambda row: abs(row["log_high_vs_low_odds_ratio"]),
                  reverse=True)


def functional_group_pair_coordination(
        formula_group_rows: list[dict[str, Any]], samples: list[dict[str, Any]],
        low_cutoff: float, high_cutoff: float) -> list[dict[str, Any]]:
    """Screen functional-group pairs for high/low-specific co-occurrence.

    ``relative_cooccurrence_ratio`` compares a pair's observed co-occurrence
    to the product of its component prevalence in each group. A value above
    one means the pair co-occurs more often than expected from its marginals.
    Comparing the high and low values highlights formulation coordination
    candidates; it does not establish a physical or causal interaction.
    """

    groups_by_sample: dict[int, set[str]] = defaultdict(set)
    for row in formula_group_rows:
        groups_by_sample[row["sample_index"]].add(row["functional_group"])
    low_ids = {
        sample["sample_index"] for sample in samples
        if sample["target_conductivity_ms_cm"] <= low_cutoff
    }
    high_ids = {
        sample["sample_index"] for sample in samples
        if sample["target_conductivity_ms_cm"] >= high_cutoff
    }
    group_ids: dict[str, set[int]] = defaultdict(set)
    pair_ids: dict[tuple[str, str], set[int]] = defaultdict(set)
    for sample_index, groups in groups_by_sample.items():
        for group in groups:
            group_ids[group].add(sample_index)
        for pair in combinations(sorted(groups), 2):
            pair_ids[pair].add(sample_index)

    output = []
    for (group_a, group_b), sample_ids in pair_ids.items():
        high_pair_count = len(sample_ids & high_ids)
        low_pair_count = len(sample_ids & low_ids)
        high_a_count = len(group_ids[group_a] & high_ids)
        high_b_count = len(group_ids[group_b] & high_ids)
        low_a_count = len(group_ids[group_a] & low_ids)
        low_b_count = len(group_ids[group_b] & low_ids)
        high_relative = ((high_pair_count + 0.5) * len(high_ids) /
                         ((high_a_count + 0.5) * (high_b_count + 0.5)))
        low_relative = ((low_pair_count + 0.5) * len(low_ids) /
                        ((low_a_count + 0.5) * (low_b_count + 0.5)))
        pair_odds_ratio = _odds_ratio(high_pair_count,
                                      len(high_ids) - high_pair_count,
                                      low_pair_count,
                                      len(low_ids) - low_pair_count)
        coordination_ratio = high_relative / low_relative
        if pair_odds_ratio > 1 and coordination_ratio > 1:
            candidate = "高电导率协同共现候选"
        elif pair_odds_ratio < 1 and coordination_ratio < 1:
            candidate = "低电导率共现候选"
        else:
            candidate = "混合关联"
        output.append({
            "functional_group_a": group_a,
            "functional_group_a_shorthand": functional_group_shorthand(group_a),
            "functional_group_b": group_b,
            "functional_group_b_shorthand": functional_group_shorthand(group_b),
            "pair_formula_count": len(sample_ids),
            "high_pair_formula_count": high_pair_count,
            "low_pair_formula_count": low_pair_count,
            "high_pair_prevalence": high_pair_count / len(high_ids),
            "low_pair_prevalence": low_pair_count / len(low_ids),
            "high_vs_low_pair_odds_ratio": pair_odds_ratio,
            "high_relative_cooccurrence": high_relative,
            "low_relative_cooccurrence": low_relative,
            "high_vs_low_relative_cooccurrence_ratio": coordination_ratio,
            "coordination_candidate": candidate,
        })
    return sorted(output,
                  key=lambda row: abs(math.log(
                      row["high_vs_low_relative_cooccurrence_ratio"])),
                  reverse=True)


def whole_molecule_summary(fragment_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in fragment_rows:
        grouped[row["canonical_fragment_smiles"]].append(row)
    output = []
    for molecule_smiles, rows in grouped.items():
        solvent_names = Counter(row["solvent_name"] for row in rows)
        output.append({
            "whole_molecule_smiles": molecule_smiles,
            "molecule_formula": fragment_formula(molecule_smiles),
            "solvent_names": "；".join(name for name, _
                                       in solvent_names.most_common()),
            "functional_group_label": rows[0]["functional_group_label"],
            "functional_group_shorthand": rows[0]["functional_group_shorthand"],
            "occurrences": len(rows),
            "unique_formulas": len({row["sample_index"] for row in rows}),
            "mean_solvent_molar_ratio": mean(
                row["solvent_molar_ratio"] for row in rows),
        })
    return sorted(output, key=lambda row: row["unique_formulas"], reverse=True)


def representative_cases(cases: list[dict[str, Any]], low_cutoff: float,
                         high_cutoff: float, count: int) -> dict[str, list[dict[str, Any]]]:
    by_error = sorted(cases, key=lambda case: case["abs_error_ms_cm"])
    low = [case for case in by_error if case["target_conductivity_ms_cm"] <= low_cutoff]
    high = [case for case in by_error if case["target_conductivity_ms_cm"] >= high_cutoff]
    return {
        "accurate_low_conductivity": low[:count],
        "accurate_high_conductivity": high[:count],
        "largest_under_predictions": sorted(
            cases, key=lambda case: case["residual_ms_cm"])[:count],
        "largest_over_predictions": sorted(
            cases, key=lambda case: case["residual_ms_cm"], reverse=True)[:count],
    }


def write_report(path: Path, samples: list[dict[str, Any]],
                 group_summary: list[dict[str, Any]],
                 association: list[dict[str, Any]],
                 composition_association: list[dict[str, Any]],
                 pair_coordination: list[dict[str, Any]],
                 whole_summary: list[dict[str, Any]], low_cutoff: float,
                 high_cutoff: float, comparable_fragment_count: int,
                 whole_fragment_count: int, min_group_formulas: int,
                 min_pair_formulas: int) -> None:
    lines = [
        "# 实验测试集片段注意力分析报告",
        "",
        f"实验电导率样本数：{len(samples)}",
        f"低电导率组阈值（mS/cm）：{low_cutoff:.6g}",
        f"高电导率组阈值（mS/cm）：{high_cutoff:.6g}",
        "",
        "## 解释边界",
        "",
        "`pooling_attention` 是中性溶剂分子内部片段池化的相对权重。",
        "它不表示片段对电导率的正负或因果贡献。由于 `neutral_only=true`，带电盐/阴离子片段的残差不会进入编码器输出，因此本报告将其排除在统计解释之外。",
        "",
        "## 去冗余与统计口径",
        "",
        "原始片段会去除显式氢，并使用 `*` 标记与其他片段的连接位置。功能团采用最具体类别优先的互斥规则；同一配方中的同类功能团先合并，再参与全局统计。",
        f"可进行分子内 attention 比较的片段记录数：{comparable_fragment_count}",
        f"完整分子单片段记录数：{whole_fragment_count}（不参与 attention 排名）",
        f"主报告最小支持度：至少出现在 {min_group_formulas} 个实验配方中。",
        "",
        "## 指标定义",
        "",
        "- `pooling_attention`：同一个中性溶剂内各片段的池化权重，单个溶剂内所有片段之和为 1，只能在该溶剂内部比较。",
        "- 配方覆盖率：包含该功能团的实验配方数除以全部实验配方数。注意力主表仅统计可分为多个片段的溶剂；完整分子单片段另列，不参与该表。",
        "- 配方加权 attention：对一个片段计算 `pooling_attention x 该溶剂摩尔比例`。同一配方内同类功能团先求和，因此同时体现该结构在分子内部的权重和该溶剂在配方中的占比。",
        "- 数据集平均配方加权 attention：上述配方级数值在全部实验配方上的平均值；不含该功能团的配方按 0 计入。它适合衡量该类片段在整个数据集中的总体信号。",
        "- 平均相对富集度：`pooling_attention x 该分子片段数` 的平均值。大于 1 表示高于该分子内的均匀分配基线，等于 1 表示接近均匀，小于 1 表示低于均匀基线。",
        "- 基团简写保留连接方向，例如 `-CF3`、`-O-`、`-C(=O)-O-`；片段化学式由预处理保存的氢封端片段计算，`*` 连接占位符不计入化学式。",
        "",
        "## 配方加权注意力最高的功能团",
        "",
        "| 主功能团 | 基团简写 | 代表性连接点 SMILES | 片段化学式 | 唯一配方数 | 配方覆盖率 | 数据集平均配方加权 attention | 平均相对富集度 |",
        "| --- | --- | --- | --- | ---: | ---: | ---: | ---: |",
    ]
    supported_groups = [row for row in group_summary
                        if row["unique_formulas"] >= min_group_formulas]
    for row in supported_groups[:20]:
        lines.append(
            f"| {row['functional_group']} | "
            f"`{row['functional_group_shorthand']}` | "
            f"`{row['representative_attachment_smiles']}` | "
            f"`{row['representative_fragment_formula']}` | "
            f"{row['unique_formulas']} | {row['formula_prevalence']:.2%} | "
            f"{row['dataset_mean_formula_weighted_attention']:.6g} | "
            f"{row['mean_attention_enrichment']:.6g} |")
    lines.extend([
        "",
        "## 高低电导率组的注意力关联",
        "",
        "该结果是观察性关联，尚未控制温度、盐浓度、盐种类和共溶剂共现等混杂因素。",
        "",
        "| 主功能团 | 基团简写 | 代表性连接点 SMILES | 片段化学式 | 低组配方数 | 高组配方数 | 高组 - 低组配方加权 attention |",
        "| --- | --- | --- | --- | ---: | ---: | ---: |",
    ])
    supported_names = {row["functional_group"] for row in supported_groups}
    supported_association = [row for row in association
                             if row["functional_group"] in supported_names]
    for row in supported_association[:20]:
        difference = row["high_minus_low_weighted_attention"]
        lines.append(f"| {row['functional_group']} | "
                     f"`{row['functional_group_shorthand']}` | "
                     f"`{row['representative_attachment_smiles']}` | "
                     f"`{row['representative_fragment_formula']}` | "
                     f"{row['low_formula_count']} | {row['high_formula_count']} | "
                     f"{difference:.6g} |")
    lines.extend([
        "",
        "## 高电导率配方的组分候选",
        "",
        "此处按功能团是否出现在一个配方中统计，包含完整分子单片段，因此不使用片段 attention。attention 只是在同一溶剂内部的相对权重，不能跨溶剂比较，且完整分子单片段恒为 1；配方筛选更适合先使用结构是否出现及其共现关系。高/低组出现优势比大于 1 表示高组更常见。它用于筛选候选结构，仍未控制温度、盐浓度、盐种类及其他共溶剂等混杂因素，不能表述为该片段单独导致电导率升高。",
        "",
        "| 主功能团 | 基团简写 | 代表性连接点 SMILES | 片段化学式 | 高组配方数 | 低组配方数 | 高/低组出现优势比 |",
        "| --- | --- | --- | --- | ---: | ---: | ---: |",
    ])
    supported_composition = [
        row for row in composition_association
        if row["high_formula_count"] + row["low_formula_count"] >=
        min_group_formulas
    ]
    high_composition = [row for row in supported_composition
                        if row["high_vs_low_odds_ratio"] > 1]
    low_composition = [row for row in supported_composition
                       if row["high_vs_low_odds_ratio"] < 1]
    for row in high_composition[:20]:
        lines.append(
            f"| {row['functional_group']} | "
            f"`{row['functional_group_shorthand']}` | "
            f"`{row['representative_attachment_smiles']}` | "
            f"`{row['representative_fragment_formula']}` | "
            f"{row['high_formula_count']} | {row['low_formula_count']} | "
            f"{row['high_vs_low_odds_ratio']:.6g} |")
    lines.extend([
        "",
        "## 低电导率配方的组分候选",
        "",
        "高/低组出现优势比小于 1 表示该结构在低电导率组更常见。它只能作为待规避或重点验证候选，不能表述为该片段单独导致电导率降低。",
        "",
        "| 主功能团 | 基团简写 | 代表性连接点 SMILES | 片段化学式 | 高组配方数 | 低组配方数 | 高/低组出现优势比 |",
        "| --- | --- | --- | --- | ---: | ---: | ---: |",
    ])
    for row in low_composition[:20]:
        lines.append(
            f"| {row['functional_group']} | "
            f"`{row['functional_group_shorthand']}` | "
            f"`{row['representative_attachment_smiles']}` | "
            f"`{row['representative_fragment_formula']}` | "
            f"{row['high_formula_count']} | {row['low_formula_count']} | "
            f"{row['high_vs_low_odds_ratio']:.6g} |")
    lines.extend([
        "",
        "## 高电导率配方的共现协调候选",
        "",
        "功能团对的“协同共现”是统计筛选：先比较该对在高/低组出现的优势比，再比较其相对共现度（实际共现相对于两个单独结构出现率乘积的比值）。两个指标同时偏向高组才列入本表。它描述配方中的共现模式，不证明两个片段存在物理协同作用；后续应在相同温度、盐种和盐浓度下进行替换/消融验证。",
        "",
        "| 功能团 A | 基团简写 A | 功能团 B | 基团简写 B | 配方总数 | 高组配方数 | 低组配方数 | 高/低组配对优势比 | 高/低组相对共现比 |",
        "| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: |",
    ])
    supported_pairs = [
        row for row in pair_coordination
        if row["pair_formula_count"] >= min_pair_formulas
    ]
    high_pairs = [row for row in supported_pairs
                  if row["coordination_candidate"] == "高电导率协同共现候选"]
    low_pairs = [row for row in supported_pairs
                 if row["coordination_candidate"] == "低电导率共现候选"]
    for row in high_pairs[:20]:
        lines.append(
            f"| {row['functional_group_a']} | "
            f"`{row['functional_group_a_shorthand']}` | "
            f"{row['functional_group_b']} | "
            f"`{row['functional_group_b_shorthand']}` | "
            f"{row['pair_formula_count']} | {row['high_pair_formula_count']} | "
            f"{row['low_pair_formula_count']} | "
            f"{row['high_vs_low_pair_odds_ratio']:.6g} | "
            f"{row['high_vs_low_relative_cooccurrence_ratio']:.6g} |")
    lines.extend([
        "",
        "## 低电导率配方的共现候选",
        "",
        "两个指标同时偏向低组的功能团对列于此处。若表格为空，说明在当前最低支持度下没有稳定的低电导率共现候选。",
        "",
        "| 功能团 A | 基团简写 A | 功能团 B | 基团简写 B | 配方总数 | 高组配方数 | 低组配方数 | 高/低组配对优势比 | 高/低组相对共现比 |",
        "| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: |",
    ])
    if low_pairs:
        for row in low_pairs[:20]:
            lines.append(
                f"| {row['functional_group_a']} | "
                f"`{row['functional_group_a_shorthand']}` | "
                f"{row['functional_group_b']} | "
                f"`{row['functional_group_b_shorthand']}` | "
                f"{row['pair_formula_count']} | "
                f"{row['high_pair_formula_count']} | "
                f"{row['low_pair_formula_count']} | "
                f"{row['high_vs_low_pair_odds_ratio']:.6g} | "
                f"{row['high_vs_low_relative_cooccurrence_ratio']:.6g} |")
    else:
        lines.append("| 暂无 | -- | 暂无 | -- | -- | -- | -- | -- | -- |")
    lines.extend([
        "",
        "## 完整分子单片段",
        "",
        "以下分子没有可比较的内部片段，其 `pooling_attention` 必然为 1，因此仅作为完整分子记录，不解释为高注意力功能团。",
        "",
        "| 溶剂 | 去氢规范 SMILES | 化学式 | 基团简写 | 化学类别 | 唯一配方数 |",
        "| --- | --- | --- | --- | --- | ---: |",
    ])
    for row in whole_summary[:20]:
        lines.append(f"| {row['solvent_names']} | "
                     f"`{row['whole_molecule_smiles']}` | "
                     f"`{row['molecule_formula']}` | "
                     f"`{row['functional_group_shorthand']}` | "
                     f"{row['functional_group_label']} | "
                     f"{row['unique_formulas']} |")
    path.write_text("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Analyze fragment attention for experimental conductivity tests")
    parser.add_argument("--input", required=True, help="Prediction JSON from predict.py")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--quantile",
                        type=float,
                        default=0.25,
                        help="Low/high experimental conductivity quantile")
    parser.add_argument("--representative-count", type=int, default=10)
    parser.add_argument("--min-group-formulas", type=int, default=10)
    parser.add_argument("--min-pair-formulas", type=int, default=10)
    args = parser.parse_args()
    if not 0 < args.quantile < 0.5:
        raise ValueError("--quantile must be in (0, 0.5)")

    with Path(args.input).open("r") as file:
        records = json.load(file)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    samples = experimental_records(records)
    low_cutoff, high_cutoff = classify_samples(samples, args.quantile)
    cases = case_explanations(samples, low_cutoff, high_cutoff)
    fragment_rows = [fragment for case in cases
                     for fragment in case["neutral_solvent_fragments"]]
    comparable_rows = [row for row in fragment_rows
                       if row["attention_comparable"]]
    whole_rows = [row for row in fragment_rows
                  if row["is_whole_molecule_fragment"]]
    summary = fragment_summary(comparable_rows)
    formula_group_rows = formula_functional_group_rows(comparable_rows)
    composition_group_rows = formula_functional_group_rows(fragment_rows)
    group_summary = functional_group_summary(formula_group_rows, len(samples))
    association = functional_group_association(
        formula_group_rows, samples, low_cutoff, high_cutoff)
    composition_association = functional_group_composition_association(
        composition_group_rows, samples, low_cutoff, high_cutoff)
    pair_coordination = functional_group_pair_coordination(
        composition_group_rows, samples, low_cutoff, high_cutoff)
    whole_summary = whole_molecule_summary(whole_rows)
    representatives = representative_cases(cases, low_cutoff, high_cutoff,
                                           args.representative_count)

    write_json(output_dir / "case_explanations.json", cases)
    write_json(output_dir / "representative_cases.json", representatives)
    write_csv(output_dir / "fragment_attention_summary.csv", summary)
    write_csv(output_dir / "functional_group_summary.csv", group_summary)
    write_csv(output_dir / "functional_group_high_low_association.csv",
              association)
    write_csv(output_dir / "functional_group_high_low_composition.csv",
              composition_association)
    write_csv(output_dir / "functional_group_pair_coordination.csv",
              pair_coordination)
    write_csv(output_dir / "whole_molecule_fragments.csv", whole_summary)
    write_json(output_dir / "analysis_summary.json", {
        "input": str(Path(args.input)),
        "experimental_sample_count": len(samples),
        "neutral_solvent_fragment_occurrences": len(fragment_rows),
        "attention_comparable_fragment_occurrences": len(comparable_rows),
        "whole_molecule_fragment_occurrences": len(whole_rows),
        "low_conductivity_cutoff_ms_cm": low_cutoff,
        "high_conductivity_cutoff_ms_cm": high_cutoff,
        "quantile": args.quantile,
        "min_group_formulas": args.min_group_formulas,
        "min_pair_formulas": args.min_pair_formulas,
        "interpretation": "attention-based, non-causal; neutral solvents only",
    })
    write_report(output_dir / "interpretation_report.md", samples,
                 group_summary, association, composition_association,
                 pair_coordination, whole_summary, low_cutoff,
                 high_cutoff, len(comparable_rows), len(whole_rows),
                 args.min_group_formulas, args.min_pair_formulas)
    print(f"Wrote experimental fragment analysis to: {output_dir}")


if __name__ == "__main__":
    main()
