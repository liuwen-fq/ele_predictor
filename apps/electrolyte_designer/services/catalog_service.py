"""GDB17 catalogue filtering and reuse of the project's soft-scoring logic."""

from __future__ import annotations

import csv
import io
import sys
from functools import lru_cache
from pathlib import Path
from typing import Any, Iterable

from apps.electrolyte_designer.services.data_access import DemoPaths, load_json, require_file


DIRECT_PROPERTIES = ("dc", "vis", "Tm")
PROPERTY_LABELS = {"dc": "介电常数 (dc)", "vis": "黏度 (vis)", "Tm": "熔点 (Tm)"}


def _candidate_pool_helpers(root: Path):
    """Import the current selection implementation rather than duplicating it."""
    root_text = str(root)
    if root_text not in sys.path:
        sys.path.insert(0, root_text)
    from scripts.generation.build_candidate_pool import (  # pylint: disable=import-outside-toplevel
        fragment_profile,
        guided_coverage_top,
        property_score,
    )
    return fragment_profile, guided_coverage_top, property_score


def property_constraints(paths: DemoPaths) -> list[dict[str, Any]]:
    with require_file(paths.property_constraints, "property-constraint table").open(
        newline="", encoding="utf-8"
    ) as handle:
        rows = list(csv.DictReader(handle))
    return [
        row for row in rows if row.get("property") in DIRECT_PROPERTIES
    ]


@lru_cache(maxsize=2)
def catalogue_rows(catalogue_path: str) -> tuple[dict[str, Any], ...]:
    rows = load_json(catalogue_path)
    if not isinstance(rows, list):
        raise ValueError("GDB17 catalogue must be a JSON list")
    return tuple(rows)


@lru_cache(maxsize=2)
def catalogue_summary(root: str, catalogue_path: str) -> tuple[dict[str, Any], ...]:
    """Precompute presentation fields once; the GPU property pass is already done."""
    fragment_profile, _, _ = _candidate_pool_helpers(Path(root))
    summary = []
    for row in catalogue_rows(catalogue_path):
        profile = fragment_profile(row["smiles"])
        properties = row.get("predicted_properties", {})
        summary.append({
            **row,
            "motifs": tuple(sorted(profile["positive_terms"])),
            "matched_structure_terms": profile["details"],
            "functional_family": profile["functional_family"],
            "murcko_scaffold": profile["murcko_scaffold"],
            "fragment_risk_score": float(profile["risk_score"]),
            "anion_tendency_score": float(profile["anion_tendency_score"]),
            "dc": float(properties.get("dc", float("nan"))),
            "vis": float(properties.get("vis", float("nan"))),
            "Tm": float(properties.get("Tm", float("nan"))),
        })
    return tuple(summary)


def filter_catalogue(
    rows: Iterable[dict[str, Any]],
    motifs: Iterable[str],
    dc_range: tuple[float, float],
    vis_range: tuple[float, float],
    tm_range: tuple[float, float],
) -> list[dict[str, Any]]:
    requested = set(motifs)
    filtered = []
    for row in rows:
        if requested and not requested.intersection(row["motifs"]):
            continue
        if not dc_range[0] <= row["dc"] <= dc_range[1]:
            continue
        if not vis_range[0] <= row["vis"] <= vis_range[1]:
            continue
        if not tm_range[0] <= row["Tm"] <= tm_range[1]:
            continue
        filtered.append(dict(row))
    return filtered


def soft_property_score(values: dict[str, float], constraints: list[dict[str, Any]], root: Path) -> float:
    _, _, score = _candidate_pool_helpers(root)
    return float(score(values, constraints)[0])


def presentation_rows(
    rows: Iterable[dict[str, Any]], constraints: list[dict[str, Any]], root: Path
) -> list[dict[str, Any]]:
    """Create sortable rows while preserving the scoring terms visible to users."""
    output = []
    for row in rows:
        property_part = soft_property_score(row["predicted_properties"], constraints, root)
        motif_part = len(row["motifs"]) * 0.4 + row["fragment_risk_score"]
        output.append({
            **row,
            "property_window_score": property_part,
            "motif_score": motif_part,
            "total_score": property_part + motif_part,
        })
    return sorted(output, key=lambda item: item["total_score"], reverse=True)


def automatic_selection(
    rows: Iterable[dict[str, Any]],
    constraints: list[dict[str, Any]],
    root: Path,
    pool_size: int = 62,
    seed: int = 20260909,
    property_weight: float = 1.0,
    fragment_weight: float = 1.0,
    family_weight: float = 0.7,
    scaffold_weight: float = 0.25,
    redundancy_weight: float = 1.0,
    anion_weight: float = 0.35,
) -> list[dict[str, Any]]:
    """Run the existing coverage_v1 selector over an already screened subset."""
    rows = list(rows)
    if len(rows) < pool_size:
        raise ValueError(f"Need at least {pool_size} candidates; only {len(rows)} remain after filters.")
    fragment_profile, guided_coverage_top, property_score = _candidate_pool_helpers(root)
    prepared = []
    for row in rows:
        profile = fragment_profile(row["smiles"])
        property_value, _ = property_score(row["predicted_properties"], constraints)
        prepared.append({
            **row,
            "property_window_score": float(property_value),
            "property_score": float(property_value),
            "_profile": profile,
        })
    return guided_coverage_top(
        prepared, pool_size, seed, property_weight, fragment_weight,
        family_weight, scaffold_weight, redundancy_weight, anion_weight,
    )


def selection_csv(rows: Iterable[dict[str, Any]]) -> bytes:
    fields = [
        "selected_step", "name", "smiles", "selection_marginal_score",
        "property_window_score", "matched_structure_terms", "functional_family",
        "murcko_scaffold", "max_tanimoto_to_earlier", "fragment_risk_score",
        "anion_tendency_score",
    ]
    output = io.StringIO()
    writer = csv.DictWriter(output, fieldnames=fields, extrasaction="ignore")
    writer.writeheader()
    writer.writerows(rows)
    return output.getvalue().encode("utf-8-sig")


def molecule_png(smiles: str) -> bytes | None:
    """Render a tiny RDKit structure image for a candidate card."""
    try:
        from rdkit import Chem
        from rdkit.Chem import Draw
        molecule = Chem.MolFromSmiles(smiles)
        if molecule is None:
            return None
        image = Draw.MolToImage(molecule, size=(280, 160))
        buffer = io.BytesIO()
        image.save(buffer, format="PNG")
        return buffer.getvalue()
    except Exception:  # Rendering should never make catalogue browsing fail.
        return None


def fragment_topology_png(smiles: str) -> bytes | None:
    """Draw a molecule with ``element:fragment-id`` labels and red cut bonds."""
    try:
        from rdkit import Chem
        from rdkit.Chem import Draw
        from formula_design.fragment import build_fragment_graph

        molecule = Chem.MolFromSmiles(smiles)
        if molecule is None:
            return None
        fragment_graph = build_fragment_graph(molecule)
        drawer = Draw.MolDraw2DCairo(440, 260)
        options = drawer.drawOptions()
        for atom_index, fragment_id in enumerate(fragment_graph.atom_to_fragment):
            symbol = molecule.GetAtomWithIdx(atom_index).GetSymbol()
            options.atomLabels[atom_index] = f"{symbol}:{fragment_id + 1}"
        cut_bonds = list(fragment_graph.cut_bond_indices)
        drawer.DrawMolecule(
            molecule,
            [], cut_bonds, None,
            {bond: (0.9, 0.15, 0.15) for bond in cut_bonds},
        )
        drawer.FinishDrawing()
        return drawer.GetDrawingText()
    except Exception:
        return None
