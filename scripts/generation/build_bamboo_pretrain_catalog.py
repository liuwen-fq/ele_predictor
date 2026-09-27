#!/usr/bin/env python3
"""Shared candidate catalogue strictly from Bamboo molecular pretraining records."""
import argparse
import hashlib
import json
from pathlib import Path
from rdkit import Chem
from scripts.generation.build_common_eligible_pool import eligibility_reason
from formula_design.conditional_screening import file_hash


def mono_candidates(records):
    accepted, rejected = {}, {}
    for record in records:
        smiles = record.get('smiles', '')
        reason = eligibility_reason(smiles)
        if reason:
            rejected[reason] = rejected.get(reason, 0) + 1
            continue
        canonical = Chem.MolToSmiles(Chem.MolFromSmiles(smiles), canonical=True)
        if canonical not in accepted:
            accepted[canonical] = dict(name='bamboo_mono_' + hashlib.sha256(canonical.encode()).hexdigest()[:20],
                                       smiles=canonical, source_name=record.get('name', ''), source='bamboo_mono_data')
    return [accepted[s] for s in sorted(accepted)], rejected


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--mono-data', required=True)
    p.add_argument('--model-config', required=True)
    p.add_argument('--mono-checkpoint', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--batch-size', type=int, default=256)
    args = p.parse_args()
    from scripts.generation.build_candidate_pool import load_mono, predict_properties
    rows, rejected = mono_candidates(json.loads(Path(args.mono_data).read_text()))
    predictions = predict_properties(rows, load_mono(args.model_config, args.mono_checkpoint), args.batch_size)
    for row in rows:
        row['predicted_properties'] = predictions.get(row['name'])
    out = Path(args.output); out.parent.mkdir(parents=True, exist_ok=True)
    metadata = dict(source='bamboo_molecular_pretraining_only', eligible_count=len(rows),
                    rejected_by_reason=rejected, temperature_c=25.,
                    input_sha256={str(Path(f).resolve()):file_hash(f) for f in (args.mono_data,args.model_config,args.mono_checkpoint)})
    out.with_suffix('.metadata.json').write_text(json.dumps(metadata, indent=2)+'\n')
    temp = out.with_suffix('.tmp');temp.write_text(json.dumps(rows, indent=2)+'\n');temp.replace(out)
    print(f'Wrote {len(rows)} pretraining candidates to {out}')


if __name__ == '__main__':
    main()
