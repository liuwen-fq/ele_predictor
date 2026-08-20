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
import math
import random
import sys
from pathlib import Path

import numpy as np
import torch
import yaml
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem
from rdkit.Chem.Scaffolds import MurckoScaffold

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


def candidate_ineligibility_reason(entry):
    """Return a label-independent common-eligibility failure, if any."""
    smiles = entry.get("smiles", "")
    if not smiles or "." in smiles:
        return "missing_or_multicomponent_smiles"
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return "rdkit_parse_failed"
    if Chem.GetFormalCharge(mol) != 0:
        return "non_neutral"
    if not 3 <= mol.GetNumHeavyAtoms() <= 40:
        return "heavy_atom_count_outside_3_40"
    allowed = {"C", "N", "O", "F", "P", "S", "Cl", "Br", "Si"}
    if not all(atom.GetSymbol() in allowed for atom in mol.GetAtoms()):
        return "unsupported_element"
    try:
        Molecule.from_smiles(smiles).get_mapped_smiles()
    except Exception:
        return "molecule_encoder_mapping_failed"
    return None


def fragment_profile(smiles):
    """Return explanation-derived fragment terms without a top-k bias.

    Positive terms are later scored as a *marginal* gain relative to the
    molecules already selected.  Risks stay per-molecule penalties: selecting
    one risky molecule must not make later risky molecules less risky.
    """
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
    details, positive_terms, risk_score = [], {}, 0.0
    for name, (smarts, weight) in patterns.items():
        if mol.HasSubstructMatch(Chem.MolFromSmarts(smarts)):
            details.append(name)
            if weight > 0:
                positive_terms[name] = weight
            elif weight < 0:
                risk_score += weight
    if (mol.HasSubstructMatch(Chem.MolFromSmarts("[OD2]([#6])[#6]"))
            and sum(atom.GetSymbol() == "F" for atom in mol.GetAtoms()) >= 2):
        details.append("fluorinated_ether_risk")
        risk_score -= 0.8
    if any(bond.IsInRing() for bond in mol.GetBonds()) and "ester" in details:
        details.append("lactone_bonus")
        positive_terms["lactone_bonus"] = 0.8
    # Functional-family coverage is intentionally coarser than individual
    # terms.  It captures repeated nitrile-esters/lactones that can pass a
    # molecule-level Tanimoto threshold but still form one chemical family.
    family_tags = [name for name in ("nitrile", "ester", "ether", "lactone_bonus")
                   if name in positive_terms]
    if not family_tags:
        family_tags = ["other"]
    scaffold = MurckoScaffold.MurckoScaffoldSmiles(mol=mol) or "acyclic"
    return {
        "positive_terms": positive_terms,
        "risk_score": risk_score,
        "details": ";".join(details) or "none",
        "functional_family": "+".join(family_tags),
        "murcko_scaffold": scaffold,
        "fingerprint": AllChem.GetMorganFingerprintAsBitVect(
            mol, radius=2, nBits=2048),
    }


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
    """A flat, soft property window; values inside get no extra extremum bonus."""
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


def concave_marginal(count, amount=1):
    """Automatic diminishing-return gain for repeated family/fragment use."""
    return math.log1p(count + amount) - math.log1p(count)


def guided_coverage_top(rows, pool_size, seed, property_weight,
                        fragment_weight, family_coverage_weight,
                        scaffold_coverage_weight, redundancy_weight):
    """Greedily select explanation-guided molecules by current marginal gain.

    No fragment has a hard occurrence limit.  Its gain follows ``log(1+n)``:
    the second and later representatives remain eligible, but must justify
    themselves against unused families and lower-redundancy alternatives.
    """
    from collections import Counter

    rng = random.Random(seed)
    remaining = [dict(row, _tie_break=rng.random()) for row in rows]
    selected, fingerprints = [], []
    term_counts, family_counts, scaffold_counts = Counter(), Counter(), Counter()
    while len(selected) < pool_size:
        best, best_terms = None, None
        for row in remaining:
            profile = row["_profile"]
            max_similarity = max((DataStructs.TanimotoSimilarity(
                profile["fingerprint"], old) for old in fingerprints), default=0.0)
            fragment_gain = sum(
                weight * concave_marginal(term_counts[term])
                for term, weight in profile["positive_terms"].items())
            family_gain = family_coverage_weight * concave_marginal(
                family_counts[profile["functional_family"]])
            scaffold_gain = 0.0
            if profile["murcko_scaffold"] != "acyclic":
                scaffold_gain = scaffold_coverage_weight * concave_marginal(
                    scaffold_counts[profile["murcko_scaffold"]])
            redundancy_penalty = redundancy_weight * max_similarity
            marginal = (
                property_weight * row["property_window_score"]
                + fragment_weight * fragment_gain
                + profile["risk_score"]
                + family_gain + scaffold_gain - redundancy_penalty)
            key = (marginal, -row["_tie_break"])
            if best is None or key > best[0]:
                best = (key, row, max_similarity, fragment_gain, family_gain,
                        scaffold_gain, redundancy_penalty, marginal)
                best_terms = profile["positive_terms"]
        _, row, max_similarity, fragment_gain, family_gain, scaffold_gain, penalty, marginal = best
        profile = row.pop("_profile")
        row.pop("_tie_break")
        row.update({
            "selection": "guided_coverage_v1",
            "fragment_score": sum(profile["positive_terms"].values()) + profile["risk_score"],
            "fragment_risk_score": profile["risk_score"],
            "selected_fragment_marginal": fragment_gain,
            "family_coverage_gain": family_gain,
            "scaffold_coverage_gain": scaffold_gain,
            "redundancy_penalty": penalty,
            "max_tanimoto_to_earlier": max_similarity,
            "selection_marginal_score": marginal,
            "total_score": marginal,
            "matched_structure_terms": profile["details"],
            "functional_family": profile["functional_family"],
            "murcko_scaffold": profile["murcko_scaffold"],
            "selected_step": len(selected) + 1,
        })
        selected.append(row)
        fingerprints.append(profile["fingerprint"])
        family_counts[profile["functional_family"]] += 1
        if profile["murcko_scaffold"] != "acyclic":
            scaffold_counts[profile["murcko_scaffold"]] += 1
        for term in best_terms:
            term_counts[term] += 1
        remaining.remove(row)
    return selected


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
    parser.add_argument("--mono-data", help="Legacy raw pretraining source.")
    parser.add_argument("--candidate-source",
                        help="Reusable common eligible-pool JSON.")
    parser.add_argument("--property-constraints")
    parser.add_argument("--guided-selection-strategy",
                        choices=("legacy_topk", "coverage_v1"),
                        default="coverage_v1")
    parser.add_argument("--guided-property-weight", type=float, default=1.0)
    parser.add_argument("--guided-fragment-weight", type=float, default=1.0)
    parser.add_argument("--guided-family-coverage-weight", type=float, default=0.7)
    parser.add_argument("--guided-scaffold-coverage-weight", type=float, default=0.25)
    parser.add_argument("--guided-redundancy-weight", type=float, default=1.0)
    parser.add_argument("--model-config")
    parser.add_argument("--mono-checkpoint")
    parser.add_argument("--pool-size", type=int, default=62)
    parser.add_argument("--screen-size", type=int,
                        help="Optional seeded cap of the common pool; omit to use all.")
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
        if bool(args.mono_data) == bool(args.candidate_source):
            raise ValueError("random/guided modes require exactly one of --mono-data or --candidate-source")
        if not all((args.model_config, args.mono_checkpoint)):
            raise ValueError(
                "random/guided modes require model config and checkpoint for "
                "the shared encoder/property eligibility preflight")
        if args.mode == "guided" and not args.property_constraints:
            raise ValueError("guided mode requires property constraints")
        raw = json.loads(Path(args.candidate_source or args.mono_data).read_text())
        unique = {}
        for item in raw:
            smiles = item.get("smiles", "")
            if smiles and smiles not in unique:
                unique[smiles] = {"name": item.get("name", ""), "smiles": smiles}
        basic_eligible, basic_rejected = [], []
        for item in unique.values():
            reason = candidate_ineligibility_reason(item)
            if reason is None:
                basic_eligible.append(item)
            else:
                basic_rejected.append({**item, "eligibility_status": "rejected",
                                       "eligibility_reason": reason})
        candidates = list(basic_eligible)
        if args.screen_size is not None:
            rng = random.Random(args.seed)
            rng.shuffle(candidates)
            candidates = candidates[:args.screen_size]
        predictions = predict_properties(candidates, load_mono(
            args.model_config, args.mono_checkpoint), args.batch_size)
        eligible, preflight_rejected = [], []
        for item in candidates:
            values = predictions.get(item["name"])
            if values is None:
                preflight_rejected.append({**item, "eligibility_status": "rejected",
                                           "eligibility_reason": "property_prediction_failed"})
            elif not all(np.isfinite(value) for value in values.values()):
                preflight_rejected.append({**item, "eligibility_status": "rejected",
                                           "eligibility_reason": "nonfinite_property_prediction"})
            else:
                eligible.append({**item, "eligibility_status": "eligible",
                                 "eligibility_reason": ""})
        if len(eligible) < args.pool_size:
            raise RuntimeError(
                f"only {len(eligible)} commonly eligible molecules for pool size "
                f"{args.pool_size}; increase --screen-size")
        audit_rows = basic_rejected + preflight_rejected + eligible
        write_csv(output / "candidate_eligibility_audit.csv", audit_rows)
        write_csv(output / "eligible_pool.csv", eligible)
        if args.mode == "random":
            rows = diverse_top([{**item, "selection": "size_matched_random_diverse", "total_score": 0.0}
                                for item in eligible], args.pool_size, args.seed)
        else:
            constraints = read_csv(args.property_constraints)
            scored = []
            for item in eligible:
                values = predictions.get(item["name"])
                profile = fragment_profile(item["smiles"])
                prop, property_values = property_score(values, constraints)
                scored.append({**item, "selection": "fragment_property_soft_constraints",
                               "property_window_score": prop,
                               "property_score": prop,
                               "total_score": sum(profile["positive_terms"].values())
                                              + profile["risk_score"] + prop,
                               "matched_structure_terms": profile["details"],
                               "functional_family": profile["functional_family"],
                               "murcko_scaffold": profile["murcko_scaffold"],
                               "predicted_properties": property_values,
                               "_profile": profile})
            if args.guided_selection_strategy == "legacy_topk":
                for row in scored:
                    row["fragment_score"] = (
                        sum(row["_profile"]["positive_terms"].values())
                        + row["_profile"]["risk_score"])
                    row.pop("_profile")
                rows = diverse_top(scored, args.pool_size, args.seed)
            else:
                rows = guided_coverage_top(
                    scored, args.pool_size, args.seed,
                    args.guided_property_weight, args.guided_fragment_weight,
                    args.guided_family_coverage_weight,
                    args.guided_scaffold_coverage_weight,
                    args.guided_redundancy_weight)
    solvents = [{"name": row["name"], "smiles": row["smiles"], "temperature": 25}
                for row in rows]
    (output / "solvents.json").write_text(json.dumps(solvents, indent=2) + "\n")
    (output / "salts.json").write_text(json.dumps(salts, indent=2) + "\n")
    write_csv(output / "candidate_pool_scores.csv", rows)
    (output / "candidate_pool_metadata.json").write_text(json.dumps({
        "mode": args.mode, "pool_size": args.pool_size,
        "screen_size": args.screen_size if args.mode != "baseline" else None,
        "candidate_source": args.candidate_source if args.mode != "baseline" else None,
        "seed": args.seed,
        "guided_selection": ({
            "strategy": args.guided_selection_strategy,
            "property_weight": args.guided_property_weight,
            "fragment_weight": args.guided_fragment_weight,
            "family_coverage_weight": args.guided_family_coverage_weight,
            "scaffold_coverage_weight": args.guided_scaffold_coverage_weight,
            "redundancy_weight": args.guided_redundancy_weight,
        } if args.mode == "guided" else None),
        "common_eligibility": {
            "basic_unique_candidates": len(unique) if args.mode != "baseline" else None,
            "screened_candidates": len(candidates) if args.mode != "baseline" else None,
            "eligible_candidates": len(eligible) if args.mode != "baseline" else None,
            "rejected_before_screen": len(basic_rejected) if args.mode != "baseline" else None,
            "rejected_after_preflight": len(preflight_rejected) if args.mode != "baseline" else None,
            "audit_file": "candidate_eligibility_audit.csv" if args.mode != "baseline" else None,
            "eligible_pool_file": "eligible_pool.csv" if args.mode != "baseline" else None,
        },
    }, indent=2) + "\n")


if __name__ == "__main__":
    main()
