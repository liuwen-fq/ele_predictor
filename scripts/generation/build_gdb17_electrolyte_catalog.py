#!/usr/bin/env python3
"""Build one reusable GDB17 catalogue for electrolyte-solvent generation.

The catalogue is deliberately target independent: it applies only structural
and encoder-compatibility checks, then records frozen mono-model property
predictions once.  G1 random selection and G2 soft-guided selection can thus
start from exactly the same eligible GDB17 universe without repeating GPU
property inference.
"""

import argparse
import csv
import hashlib
import json
import math
import sys
from collections import Counter
from pathlib import Path

from rdkit import Chem
from rdkit.Chem import Descriptors, Lipinski, rdMolDescriptors

# These sibling scripts are command-line entry points rather than a package.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_candidate_pool import (  # noqa: E402
    DIRECT_PROPERTIES,
    candidate_ineligibility_reason,
    load_mono,
    predict_properties,
)


# Chlorinated/brominated GDB molecules are common but are not retained in this
# default liquid-electrolyte solvent universe.  Fluorinated molecules remain
# eligible because fluorinated ethers/carbonates are relevant electrolyte
# families.  This is a structural eligibility rule, not an oracle-derived
# ranking criterion.
ALLOWED_ELEMENTS = {"C", "N", "O", "F", "P", "S", "Si"}
REACTIVE_SMARTS = {
    "free_carboxylic_acid": "[CX3](=O)[OX2H1]",
    "acid_chloride": "[CX3](=O)[Cl]",
    "acid_anhydride": "[CX3](=O)O[CX3](=O)",
    "sulfonyl_chloride": "[SX4](=O)(=O)Cl",
    "isocyanate": "[NX2]=[CX2]=[OX1]",
    "aldehyde": "[CX3H1](=O)[#6]",
    "guanidine_or_amidine": "[NX3][CX3](=[NX2])[NX3,NX2]",
}
REACTIVE_PATTERNS = {
    name: Chem.MolFromSmarts(smarts) for name, smarts in REACTIVE_SMARTS.items()
}


def aromatic_ring_count(mol):
    return sum(
        all(mol.GetAtomWithIdx(index).GetIsAromatic() for index in ring)
        for ring in mol.GetRingInfo().AtomRings()
    )


def electrolyte_solvent_reason(smiles, declared_charge):
    """Return ``(reason, mol, descriptor_dict)`` for one GDB17 record."""
    if not smiles or "." in smiles:
        return "missing_or_multicomponent_smiles", None, None
    try:
        if int(float(declared_charge)) != 0:
            return "nonzero_declared_charge", None, None
    except (TypeError, ValueError):
        return "invalid_declared_charge", None, None
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return "rdkit_parse_failed", None, None
    if Chem.GetFormalCharge(mol) != 0:
        return "non_neutral_rdkit", mol, None
    if not all(atom.GetSymbol() in ALLOWED_ELEMENTS for atom in mol.GetAtoms()):
        return "unsupported_element_for_electrolyte_solvent", mol, None

    values = {
        "heavy_atoms": mol.GetNumHeavyAtoms(),
        "mol_wt": Descriptors.MolWt(mol),
        "ring_count": rdMolDescriptors.CalcNumRings(mol),
        "aromatic_ring_count": aromatic_ring_count(mol),
        "rotatable_bonds": Lipinski.NumRotatableBonds(mol),
        "h_bond_donors": Lipinski.NumHDonors(mol),
    }
    limits = (
        ("heavy_atom_count_outside_3_20", 3 <= values["heavy_atoms"] <= 20),
        ("molecular_weight_outside_40_300", 40 <= values["mol_wt"] <= 300),
        ("ring_count_above_1", values["ring_count"] <= 1),
        ("aromatic_ring_not_allowed", values["aromatic_ring_count"] == 0),
        ("rotatable_bonds_above_8", values["rotatable_bonds"] <= 8),
        ("h_bond_donors_above_1", values["h_bond_donors"] <= 1),
    )
    for reason, passed in limits:
        if not passed:
            return reason, mol, values
    for name, pattern in REACTIVE_PATTERNS.items():
        if mol.HasSubstructMatch(pattern):
            return name, mol, values
    return None, mol, values


def stable_catalog_subset(rows, size, seed):
    """Choose a reproducible, order-independent subset of structural hits."""
    def key(row):
        text = f"{seed}|{row['smiles']}".encode("utf-8")
        return hashlib.sha256(text).hexdigest()

    return sorted(rows, key=key)[:size]


def valid_property_values(values):
    if not isinstance(values, dict):
        return False
    try:
        return all(name in values and math.isfinite(float(values[name]))
                   for name in DIRECT_PROPERTIES)
    except (TypeError, ValueError):
        return False


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-csv", required=True)
    parser.add_argument("--output", required=True,
                        help="JSON catalogue consumable by --precomputed-candidate-source.")
    parser.add_argument("--audit-output", required=True)
    parser.add_argument("--model-config", required=True)
    parser.add_argument("--mono-checkpoint", required=True)
    parser.add_argument("--catalog-size", type=int, default=20000)
    parser.add_argument("--selection-seed", type=int, default=20260908)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--min-candidates", type=int, default=62)
    args = parser.parse_args()

    if args.catalog_size < args.min_candidates:
        raise ValueError("--catalog-size must be at least --min-candidates")

    seen, structural_rows, rejected = set(), [], Counter()
    input_count = duplicate_count = 0
    with open(args.input_csv, newline="", encoding="utf-8") as handle:
        for input_count, record in enumerate(csv.DictReader(handle), start=1):
            reason, mol, values = electrolyte_solvent_reason(
                record.get("smiles", ""), record.get("formal_charge", ""))
            if reason is not None:
                rejected[reason] += 1
                continue
            smiles = Chem.MolToSmiles(mol, canonical=True)
            if smiles in seen:
                duplicate_count += 1
                continue
            seen.add(smiles)
            structural_rows.append({
                "name": f"gdb17_{record['molecule_id']}",
                "smiles": smiles,
                "gdb17_molecule_id": str(record["molecule_id"]),
                "source": "GDB17_strictly_uncharged",
                "structural_descriptors": values,
            })
            if input_count % 10000 == 0:
                print(f"structural screen: {input_count}", flush=True)

    selected = stable_catalog_subset(structural_rows, args.catalog_size,
                                     args.selection_seed)
    encoder_rows, encoder_rejected = [], Counter()
    for row in selected:
        reason = candidate_ineligibility_reason(row)
        if reason is not None:
            encoder_rejected[reason] += 1
            continue
        encoder_rows.append(row)
    if len(encoder_rows) < args.min_candidates:
        raise RuntimeError(
            f"Only {len(encoder_rows)} candidates remain after encoder checks; "
            f"need at least {args.min_candidates}.")

    print(f"Frozen property prediction for {len(encoder_rows)} candidates", flush=True)
    model = load_mono(args.model_config, args.mono_checkpoint)
    predictions = predict_properties(encoder_rows, model, args.batch_size)
    catalogue, property_rejected = [], Counter()
    for row in encoder_rows:
        values = predictions.get(row["name"])
        if not valid_property_values(values):
            property_rejected["missing_or_nonfinite_property_prediction"] += 1
            continue
        catalogue.append({
            **row,
            "predicted_properties": {
                name: float(values[name]) for name in DIRECT_PROPERTIES
            },
        })
    if len(catalogue) < args.min_candidates:
        raise RuntimeError(
            f"Only {len(catalogue)} property-eligible candidates remain; "
            f"need at least {args.min_candidates}.")

    output, audit = Path(args.output), Path(args.audit_output)
    output.parent.mkdir(parents=True, exist_ok=True)
    audit.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(catalogue, indent=2) + "\n")
    audit.write_text(json.dumps({
        "catalogue_name": "gdb17_electrolyte_solvent_v1",
        "purpose": (
            "Shared target-independent GDB17 candidate universe for G1 random "
            "and G2 soft-guided vocabulary construction."
        ),
        "input_csv": str(Path(args.input_csv).resolve()),
        "model_config": str(Path(args.model_config).resolve()),
        "mono_checkpoint": str(Path(args.mono_checkpoint).resolve()),
        "selection": {
            "catalog_size_requested": args.catalog_size,
            "selection_seed": args.selection_seed,
            "selection_rule": "lowest SHA256(seed|canonical_smiles)",
        },
        "structural_eligibility": {
            "allowed_elements": sorted(ALLOWED_ELEMENTS),
            "heavy_atoms": [3, 20], "mol_wt": [40, 300],
            "ring_count_max": 1, "aromatic_ring_count_exact": 0,
            "rotatable_bonds_max": 8, "h_bond_donors_max": 1,
            "reactive_smarts": REACTIVE_SMARTS,
        },
        "counts": {
            "input_records": input_count,
            "structural_unique_eligible": len(structural_rows),
            "canonical_duplicates": duplicate_count,
            "deterministic_structural_subset": len(selected),
            "encoder_eligible": len(encoder_rows),
            "property_eligible_written": len(catalogue),
        },
        "structural_rejected_by_reason": dict(sorted(rejected.items())),
        "encoder_rejected_by_reason": dict(sorted(encoder_rejected.items())),
        "property_rejected_by_reason": dict(sorted(property_rejected.items())),
        "output": str(output.resolve()),
    }, indent=2) + "\n")
    print(f"Wrote {len(catalogue)} frozen-property GDB17 candidates to {output}")
    print(f"Wrote audit to {audit}")


if __name__ == "__main__":
    main()
