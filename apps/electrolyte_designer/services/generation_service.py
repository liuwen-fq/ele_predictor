"""Read and rank frozen generator outputs without starting another training run."""

from __future__ import annotations

import json
from typing import Any


def _components(bow: list[float], solvents: list[dict[str, Any]], salts: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    threshold = 0.02
    solvent_rows = [
        {"name": item["name"], "smiles": item["smiles"], "ratio": float(bow[index])}
        for index, item in enumerate(solvents) if float(bow[index]) > threshold
    ]
    offset = len(solvents)
    salt_rows = [
        {"name": item["name"], "smiles": item["smiles"], "ratio": float(bow[offset + index])}
        for index, item in enumerate(salts) if float(bow[offset + index]) > threshold
    ]
    return solvent_rows, salt_rows


def rank_frozen_samples(
    records: list[dict[str, Any]],
    solvents: list[dict[str, Any]],
    salts: list[dict[str, Any]],
    target_conductivity: float,
    target_anion_ratio: float,
    conductivity_tolerance: float,
    anion_tolerance: float,
    allowed_solvents: set[str] | None = None,
    required_solvents: set[str] | None = None,
    diversify_compositions: bool = True,
) -> list[dict[str, Any]]:
    """Re-rank existing frozen G0-MA samples by the joint target shown in the demo."""
    ranked = []
    for record in records:
        solvent_rows, salt_rows = _components(record["bow"], solvents, salts)
        if allowed_solvents and any(row["name"] not in allowed_solvents for row in solvent_rows):
            continue
        names = {row["name"] for row in solvent_rows}
        if required_solvents and not required_solvents.issubset(names):
            continue
        conductivity = float(record.get("conductivity_pred_ms_cm", float("nan")))
        ratio = float(record.get("anion_ratio_pred", float("nan")))
        sigma_error = abs(conductivity - target_conductivity)
        ratio_error = abs(ratio - target_anion_ratio)
        joint_loss = sigma_error / max(conductivity_tolerance, 1e-8) + ratio_error / max(anion_tolerance, 1e-8)
        ranked.append({
            "solvents": solvent_rows,
            "salts": salt_rows,
            "conductivity_ms_cm": conductivity,
            "anion_ratio": ratio,
            "conductivity_error": sigma_error,
            "anion_error": ratio_error,
            "joint_loss": joint_loss,
            "prediction_status": record.get("prediction_status", "unknown"),
            "explanation_residual": float(record.get("explanation_residual_25c", 0.0)),
            "temperature_c": float(record.get("temperature_c", 25.0)),
            "salt_molar_ratio": float(record.get("salt_molar_ratio", 0.1)),
        })
    ranked.sort(key=lambda row: row["joint_loss"])
    if not diversify_compositions:
        return ranked
    # Exact target matching alone can repeatedly return one solvent-rich mode
    # (for example Glyme).  Surface the best example of each composition first,
    # then append remaining records without pretending new compositions exist.
    unique, remainder, signatures = [], [], set()
    for row in ranked:
        signature = tuple(sorted(component["name"] for component in row["solvents"]))
        if signature in signatures:
            remainder.append(row)
        else:
            signatures.add(signature)
            unique.append(row)
    return unique + remainder


def card_json(cards: list[dict[str, Any]]) -> bytes:
    return json.dumps(cards, ensure_ascii=False, indent=2).encode("utf-8")
