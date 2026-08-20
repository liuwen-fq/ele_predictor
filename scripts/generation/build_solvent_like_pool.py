#!/usr/bin/env python3
"""Build a reusable, structure-screened solvent-like candidate pool.

This is intentionally an oracle-independent *eligibility* screen.  It removes
charged species and obviously drug-/reagent-like structures, but does not use
conductivity, explanation scores, or a target-condition prediction.  The
published Bamboo solvent vocabulary is always retained and explicitly tagged
when it needs an override, so it remains an auditable reference set rather
than an accidental hidden filter.
"""

import argparse
import json
from collections import Counter
from pathlib import Path

from rdkit import Chem
from rdkit.Chem import Descriptors, rdMolDescriptors

ALLOWED_ELEMENTS = {"C", "N", "O", "F", "P", "S", "Cl", "Br", "Si"}

# Reactive groups and strongly basic motifs that are poor default choices for
# a generic liquid-solvent candidate universe.  This is not a safety list and
# does not replace later experimental compatibility screening.
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


def descriptors(mol):
    return {
        "heavy_atoms": mol.GetNumHeavyAtoms(),
        "mol_wt": Descriptors.MolWt(mol),
        "ring_count": rdMolDescriptors.CalcNumRings(mol),
        "aromatic_ring_count": aromatic_ring_count(mol),
    }


def solvent_like_reason(smiles):
    """Return a single deterministic rejection reason or ``None``."""
    if not smiles or "." in smiles:
        return "missing_or_multicomponent_smiles", None, None
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return "rdkit_parse_failed", None, None
    if Chem.GetFormalCharge(mol) != 0:
        return "non_neutral", None, None
    if not all(atom.GetSymbol() in ALLOWED_ELEMENTS for atom in mol.GetAtoms()):
        return "unsupported_element", None, None
    values = descriptors(mol)
    limits = (
        ("heavy_atom_count_outside_3_30", 3 <= values["heavy_atoms"] <= 30),
        ("molecular_weight_outside_40_400", 40 <= values["mol_wt"] <= 400),
        ("ring_count_above_2", values["ring_count"] <= 2),
        ("aromatic_ring_count_above_1", values["aromatic_ring_count"] <= 1),
    )
    for reason, passed in limits:
        if not passed:
            return reason, mol, values
    for name, pattern in REACTIVE_PATTERNS.items():
        if mol.HasSubstructMatch(pattern):
            return name, mol, values
    return None, mol, values


def canonical_smiles(mol):
    return Chem.MolToSmiles(mol, canonical=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mono-data", required=True)
    parser.add_argument("--bamboo-solvents", required=True)
    parser.add_argument("--output", required=True,
                        help="Output JSON list consumable by --candidate-source.")
    parser.add_argument("--audit-output", required=True)
    args = parser.parse_args()

    inputs = (
        ("mono_data", json.loads(Path(args.mono_data).read_text()), False),
        ("bamboo_vocab", json.loads(Path(args.bamboo_solvents).read_text()), True),
    )
    accepted, rejected, overrides = {}, [], []
    for source, records, allow_override in inputs:
        print(f"Screening {len(records)} records from {source}", flush=True)
        for index, record in enumerate(records, start=1):
            raw_smiles = record.get("smiles", "")
            reason, mol, values = solvent_like_reason(raw_smiles)
            if reason is not None and not allow_override:
                rejected.append({"source": source, "name": record.get("name", ""),
                                 "smiles": raw_smiles, "reason": reason,
                                 "descriptors": values})
                continue
            if mol is None:
                # Bamboo entries must be chemically parseable to be usable by
                # the generator; this is a data error rather than an override.
                rejected.append({"source": source, "name": record.get("name", ""),
                                 "smiles": raw_smiles, "reason": reason,
                                 "descriptors": values})
                continue
            canonical = canonical_smiles(mol)
            entry = accepted.setdefault(canonical, {
                "name": record.get("name", ""),
                "smiles": canonical,
                "sources": [],
                "solvent_like_status": "eligible",
                "descriptors": values,
            })
            if source not in entry["sources"]:
                entry["sources"].append(source)
            if reason is not None:
                entry["solvent_like_status"] = "bamboo_override"
                entry["bamboo_override_reason"] = reason
                overrides.append({"name": entry["name"], "smiles": canonical,
                                  "reason": reason, "descriptors": values})
            if index % 10000 == 0:
                print(f"  {source}: {index}/{len(records)}", flush=True)

    pool = sorted(accepted.values(), key=lambda row: (row["smiles"], row["name"]))
    output, audit = Path(args.output), Path(args.audit_output)
    output.parent.mkdir(parents=True, exist_ok=True)
    audit.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(pool, indent=2) + "\n")
    audit.write_text(json.dumps({
        "screen_name": "solvent_like_v1",
        "definition": "oracle-independent structural eligibility; Bamboo vocabulary retained by override",
        "summary": {
            "eligible_unique_molecules": len(pool),
            "bamboo_overrides": len(overrides),
            "rejected_mono_records": len(rejected),
            "rejected_by_reason": dict(Counter(row["reason"] for row in rejected)),
        },
        "thresholds": {
            "heavy_atoms": [3, 30], "mol_wt": [40, 400], "ring_count_max": 2,
            "aromatic_ring_count_max": 1,
        },
        "bamboo_overrides": overrides,
        "rejected_mono_records": rejected,
    }, indent=2) + "\n")
    print(f"Wrote {len(pool)} solvent-like candidates to {output}")
    print(f"Bamboo overrides: {len(overrides)}; rejected mono records: {len(rejected)}")
    print(f"Wrote audit to {audit}")


if __name__ == "__main__":
    main()
