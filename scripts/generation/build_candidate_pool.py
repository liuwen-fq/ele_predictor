#!/usr/bin/env python3
"""Build fixed, random, or explanation-guided solvent vocabularies.

All pools use the same number of solvents as Bamboo.  G1/G2 therefore differ
in selection policy, not vocabulary size.  G2 uses frozen molecular-property
predictions and the generated soft-constraint table; no test conductivity
label is read here.
"""

import argparse
import csv
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch
import yaml
from rdkit import Chem

from formula_design.data import MonoData, collate_data
from formula_design.mol import Molecule
from formula_design.predictor import Mono

# ``scripts`` is not a regular Python package in this repository.  Add the
# sibling analysis folder explicitly so this entry point also works when
# invoked directly by a shell pipeline.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "test_results"))
from analyze_conductivity_fragment_attributions import physical_property_value

DIRECT_PROPERTIES = ("dc", "vis", "Tm")


def read_csv(path):
    with open(path, newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path, rows):
    if not rows:
        raise RuntimeError("candidate selection produced no molecules")
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def valid_candidate(entry):
    smiles = entry.get("smiles", "")
    if not smiles or "." in smiles:
        return False
    mol = Chem.MolFromSmiles(smiles)
    if mol is None or Chem.GetFormalCharge(mol) != 0:
        return False
    if not 3 <= mol.GetNumHeavyAtoms() <= 40:
        return False
    allowed = {"C", "N", "O", "F", "P", "S", "Cl", "Br", "Si"}
    return all(atom.GetSymbol() in allowed for atom in mol.GetAtoms())


def structural_score(smiles):
    mol = Chem.MolFromSmiles(smiles)
    patterns = {
        "nitrile": ("[C]#[N]", 1.0),
        "ester": ("[CX3](=O)[OX2]", 0.7),
        "ether": ("[OD2]([#6])[#6]", 0.3),
        "sulfonamide": ("[SX4](=O)(=O)[NX3]", -1.0),
        # The attribution report found that carbonates are context dependent:
        # e.g. a cyclic ester can be useful whereas EC-like high-melting
        # structures can be poor under another salt/temperature condition.
        # Do not turn this family into a blanket exclusion at the *molecule
        # pool* stage.  Formula-level role balance and property prediction are
        # the appropriate later constraints.
        "carbonate_contextual": ("[OX2][CX3](=O)[OX2]", 0.0),
    }
    details, score = [], 0.0
    for name, (smarts, weight) in patterns.items():
        if mol.HasSubstructMatch(Chem.MolFromSmarts(smarts)):
            details.append(name)
            score += weight
    if (mol.HasSubstructMatch(Chem.MolFromSmarts("[OD2]([#6])[#6]"))
            and sum(atom.GetSymbol() == "F" for atom in mol.GetAtoms()) >= 2):
        details.append("fluorinated_ether_risk")
        score -= 0.8
    if any(bond.IsInRing() for bond in mol.GetBonds()) and "ester" in details:
        details.append("lactone_bonus")
        score += 0.8
    return score, ";".join(details) or "none"


def load_mono(model_config_path, checkpoint):
    config = yaml.safe_load(Path(model_config_path).read_text())
    model = Mono(**config["model"]["pretrain"]["model_config"])
    model.load_ckpt(checkpoint)
    return model.eval().to("cuda")


def predict_properties(entries, model, batch_size):
    output = {}
    # Mono.predict expects one scalar temperature per molecule, shape [B].
    # Passing [B, 1] would make its internal ``unsqueeze(-1)`` produce a
    # three-dimensional tensor and fail at ``repeat(1, 8)``.
    temperature = torch.tensor((25.0 + 273.15) / 373.15, device="cuda")
    for start in range(0, len(entries), batch_size):
        data, kept = [], []
        for entry in entries[start:start + batch_size]:
            try:
                mapped = Molecule.from_smiles(entry["smiles"]).get_mapped_smiles()
                data.append(MonoData(entry["name"], mapped))
                kept.append(entry)
            except Exception:
                continue
        if not data:
            continue
        with torch.no_grad():
            graph = collate_data(data).to("cuda")
            pred = model.predict(graph, temperature.repeat(len(data)))
        for index, entry in enumerate(kept):
            output[entry["name"]] = {
                name: physical_property_value(name, values[index].item())
                for name, values in pred.items()
            }
        print(f"property screen: {min(start + batch_size, len(entries))}/{len(entries)}")
    return output


def property_score(values, constraints):
    score, details = 0.0, []
    for row in constraints:
        name = row["property"]
        if name not in DIRECT_PROPERTIES or name not in values:
            continue
        lower, upper = float(row["recommended_low"]), float(row["recommended_high"])
        value, width = values[name], max(upper - lower, 1e-8)
        distance = 0.0 if lower <= value <= upper else min(
            abs(value - lower), abs(value - upper)) / width
        score += 1.0 - min(distance, 2.0)
        details.append(f"{name}={value:.4g}")
    return score, ";".join(details)


def diverse_top(rows, pool_size, seed):
    from rdkit.Chem import AllChem, DataStructs
    rng = random.Random(seed)
    ranked = sorted(rows, key=lambda row: (-float(row["total_score"]), rng.random()))
    selected, fingerprints = [], []
    for position, row in enumerate(ranked):
        fingerprint = AllChem.GetMorganFingerprintAsBitVect(
            Chem.MolFromSmiles(row["smiles"]), radius=2, nBits=2048)
        similarity = max((DataStructs.TanimotoSimilarity(fingerprint, old)
                          for old in fingerprints), default=0.0)
        remaining = len(ranked) - position - 1
        if similarity <= 0.75 or len(selected) + remaining < pool_size:
            row["max_tanimoto_to_earlier"] = similarity
            selected.append(row)
            fingerprints.append(fingerprint)
        if len(selected) == pool_size:
            break
    if len(selected) != pool_size:
        raise RuntimeError(f"selected {len(selected)} rather than {pool_size} solvents")
    return selected


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("baseline", "random", "guided"), required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--baseline-solvents", required=True)
    parser.add_argument("--salts", required=True)
    parser.add_argument("--mono-data")
    parser.add_argument("--property-constraints")
    parser.add_argument("--model-config")
    parser.add_argument("--mono-checkpoint")
    parser.add_argument("--pool-size", type=int, default=62)
    parser.add_argument("--screen-size", type=int, default=20000)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--batch-size", type=int, default=256)
    args = parser.parse_args()

    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    salts = json.loads(Path(args.salts).read_text())
    if args.mode == "baseline":
        solvents = json.loads(Path(args.baseline_solvents).read_text())
        if len(solvents) != args.pool_size:
            raise ValueError("baseline vocabulary size differs from --pool-size")
        rows = [{"name": item["name"], "smiles": item["smiles"],
                 "selection": "fixed_bamboo_vocabulary", "total_score": ""}
                for item in solvents]
    else:
        if not args.mono_data:
            raise ValueError("random/guided modes require --mono-data")
        if args.mode == "guided" and not all((args.property_constraints,
                                               args.model_config,
                                               args.mono_checkpoint)):
            raise ValueError("guided mode requires property constraints, model config, and checkpoint")
        raw = json.loads(Path(args.mono_data).read_text())
        unique = {}
        for item in raw:
            if valid_candidate(item):
                unique.setdefault(item["smiles"], {"name": item["name"], "smiles": item["smiles"]})
        candidates = list(unique.values())
        rng = random.Random(args.seed)
        rng.shuffle(candidates)
        candidates = candidates[:args.screen_size]
        if args.mode == "random":
            rows = diverse_top([{**item, "selection": "size_matched_random_diverse", "total_score": 0.0}
                                for item in candidates], args.pool_size, args.seed)
        else:
            constraints = read_csv(args.property_constraints)
            predictions = predict_properties(candidates, load_mono(
                args.model_config, args.mono_checkpoint), args.batch_size)
            scored = []
            for item in candidates:
                values = predictions.get(item["name"])
                if values is None:
                    continue
                fragment, fragments = structural_score(item["smiles"])
                prop, property_values = property_score(values, constraints)
                scored.append({**item, "selection": "fragment_property_soft_constraints",
                               "fragment_score": fragment, "property_score": prop,
                               "total_score": fragment + prop,
                               "matched_structure_terms": fragments,
                               "predicted_properties": property_values})
            rows = diverse_top(scored, args.pool_size, args.seed)
    solvents = [{"name": row["name"], "smiles": row["smiles"], "temperature": 25}
                for row in rows]
    (output / "solvents.json").write_text(json.dumps(solvents, indent=2) + "\n")
    (output / "salts.json").write_text(json.dumps(salts, indent=2) + "\n")
    write_csv(output / "candidate_pool_scores.csv", rows)
    (output / "candidate_pool_metadata.json").write_text(json.dumps({
        "mode": args.mode, "pool_size": args.pool_size,
        "screen_size": args.screen_size if args.mode != "baseline" else None,
        "seed": args.seed,
    }, indent=2) + "\n")


if __name__ == "__main__":
    main()
