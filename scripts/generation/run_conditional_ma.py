#!/usr/bin/env python3
"""Isolated G1-MA / conditional G2-MA experiments, one seed/arm at a time."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
from formula_design.conditional_screening import load_task, file_hash

ROOT = Path(__file__).resolve().parents[2]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('group', choices=('G1-MA', 'G2-Q', 'G2-MA', 'G2-MA-shuffle'))
    parser.add_argument('seed', type=int, choices=(123, 456, 789))
    parser.add_argument('task')
    parser.add_argument('--plan', action='store_true')
    args = parser.parse_args()
    os.chdir(ROOT)
    task = load_task(args.task)
    if task['target']['temperature_c'] != 25.:
        raise ValueError('Current molecular property predictor is defined at 25 C; provide a temperature-aware catalog adapter before other temperatures')
    env = os.environ.copy()
    env.update(CONDITIONAL_TASK=str(Path(args.task).resolve()),
               TARGET_CONDUCTIVITY=str(task['target']['conductivity_ms_cm']),
               ANION_RATIO=str(task['target']['anion_ratio']),
               TEMPERATURE_C=str(task['target']['temperature_c']),
               SALT_MOLAR_RATIO=str(task['target']['salt_molar_ratio']))
    # Legacy MA evaluator uses these fixed scales. Fail rather than silently
    # fit for one target loss while comparing against a different one.
    if task['target']['conductivity_tolerance'] != 2. or task['target']['anion_ratio_tolerance'] != .1:
        raise ValueError('This MA pipeline requires sigma/r tolerances 2.0/0.1; standalone pool fitting supports other tolerances')
    defaults = {
        'MONO_DATA': str(ROOT.parent / 'bamboo_mixer/run/mono_data.json'),
        'ORACLE_CONFIG': 'config/test_results/anion_ratio_property_fragment_predict.yaml',
        'ORACLE_CKPT': 'run/anion_ratio_property_fragment_explanation/ckpt_formula_ft',
        'MONO_CKPT': 'run/checkpoints/fixed_fragment_mono.pt',
        'BAMBOO_SOLVENTS': 'emb_dict/solvents.json', 'BAMBOO_SALTS': 'emb_dict/salts.json'}
    for key, value in defaults.items():
        env[key] = str(Path(env.get(key, value)).resolve())
    salt_names = {s['name'] for s in json.loads(Path(env['BAMBOO_SALTS']).read_text())}
    if salt_names != set(task['target']['salts']):
        raise ValueError('Task salts differ from the generation salt vocabulary')
    pipeline = ROOT / 'submit/generation/run_bamboo_pretrain_mechanism_auxiliary_replicate.sh'
    keys = set(re.findall(r'\$\{([A-Z][A-Z0-9_]*)[:-]', pipeline.read_text())) - {'EXPERIMENT_ROOT', 'SEED'}
    paths = [args.task, *(s['path'] for s in task['sources'])]
    paths += [env[k] for k in defaults if k != 'ORACLE_CKPT']
    paths += [str(Path(env['ORACLE_CKPT']) / n) for n in ('optimal.pt', 'pretrain.pt')]
    for directory in ('formula_design', 'scripts/generation', 'submit/generation'):
        paths += [str(p) for p in (ROOT / directory).rglob('*') if p.suffix in ('.py', '.sh')]
    spec = {'candidate_origin': 'bamboo_molecular_pretraining_only', 'task': task, 'settings': {k: env.get(k) for k in sorted(keys)},
            'sha256': {str(Path(p).resolve()): file_hash(p) for p in paths}}
    token = hashlib.sha256(json.dumps(spec, sort_keys=True).encode()).hexdigest()[:16]
    base = Path(env.get('CONDITIONAL_EXPERIMENT_ROOT', 'run/generation_conditional_bamboo_ma'))
    out = base.resolve() / token
    env['EXPERIMENT_ROOT'] = str(out)
    print(f'Experiment: {out}\nJob: {args.group}, seed {args.seed}', flush=True)
    if args.plan:
        return
    out.mkdir(parents=True, exist_ok=True)
    # Same immutable signature across arms/seeds; publish atomically.
    temp = out / f'.task-{os.getpid()}.tmp'
    temp.write_text(json.dumps(spec, indent=2) + '\n')
    temp.replace(out / 'conditional_experiment.json')
    subprocess.run(['bash', str(pipeline), args.group, str(args.seed)], env=env, check=True)


if __name__ == '__main__':
    main()
