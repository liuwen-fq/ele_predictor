#!/usr/bin/env python3
"""Create one reusable, label-independent solvent candidate universe.

The pool combines molecular-pretraining records and Bamboo's released solvent
vocabulary.  It deliberately applies only shared technical eligibility rules;
G2's structure/property soft scores remain downstream selection variables.
"""

import argparse
import json
from collections import Counter
from pathlib import Path

from rdkit import Chem

from formula_design.mol import Molecule


ALLOWED_ELEMENTS = {"C", "N", "O", "F", "P", "S", "Cl", "Br", "Si"}


def eligibility_reason(smiles):
    if not smiles or "." in smiles:
        return "missing_or_multicomponent_smiles"
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return "rdkit_parse_failed"
    if Chem.GetFormalCharge(mol) != 0:
        return "non_neutral"
    if not 3 <= mol.GetNumHeavyAtoms() <= 40:
        return "heavy_atom_count_outside_3_40"
    if not all(atom.GetSymbol() in ALLOWED_ELEMENTS for atom in mol.GetAtoms()):
        return "unsupported_element"
    try:
        Molecule.from_smiles(smiles).get_mapped_smiles()
    except Exception:
        return "molecule_encoder_mapping_failed"
    return None


def canonical_smiles(smiles):
    return Chem.MolToSmiles(Chem.MolFromSmiles(smiles), canonical=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mono-data", required=True)
    parser.add_argument("--bamboo-solvents", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--audit-output", required=True)
    args = parser.parse_args()

    sources = (
        ("mono_data", json.loads(Path(args.mono_data).read_text())),
        ("bamboo_vocab", json.loads(Path(args.bamboo_solvents).read_text())),
    )
    accepted, rejected = {}, []
    for source, records in sources:
        for record in records:
            smiles = record.get("smiles", "")
            reason = eligibility_reason(smiles)
            if reason is not None:
                rejected.append({"source": source, "name": record.get("name", ""),
                                 "smiles": smiles, "status": "rejected",
                                 "reason": reason})
                continue
            canonical = canonical_smiles(smiles)
            if canonical not in accepted:
                accepted[canonical] = {
                    "name": record.get("name", ""), "smiles": canonical,
                    "sources": [source],
                }
            elif source not in accepted[canonical]["sources"]:
                accepted[canonical]["sources"].append(source)

    pool = sorted(accepted.values(), key=lambda row: (row["smiles"], row["name"]))
    output = Path(args.output)
    audit = Path(args.audit_output)
    output.parent.mkdir(parents=True, exist_ok=True)
    audit.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(pool, indent=2) + "\n")
    audit_rows = rejected + [
        {"source": ";".join(row["sources"]), "name": row["name"],
         "smiles": row["smiles"], "status": "eligible", "reason": ""}
        for row in pool
    ]
    audit.write_text(json.dumps({
        "summary": {
            "eligible_unique_molecules": len(pool),
            "rejected_records": len(rejected),
            "rejected_by_reason": dict(Counter(row["reason"] for row in rejected)),
            "sources": [source for source, _ in sources],
        },
        "records": audit_rows,
    }, indent=2) + "\n")
    print(f"Wrote {len(pool)} unique eligible molecules to {output}")
    print(f"Wrote audit to {audit}")


if __name__ == "__main__":
    main()
