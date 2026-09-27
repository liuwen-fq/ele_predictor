#!/usr/bin/env python3
"""Isolated, fingerprinted legacy G0 triplet runner (standard library only)."""
import argparse
import csv
from contextlib import ExitStack
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[2]
GROUPS = ('G0-Q', 'G0-MA', 'G0-MA-shuffle')


def resolve(value):
    p = Path(value).expanduser()
    return (p if p.is_absolute() else ROOT / p).resolve()


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def validate(c):
    if c['schema_version'] != 1 or c['mode'] != 'legacy_reproduction':
        raise ValueError('Only schema 1 legacy_reproduction is supported')
    if not c['seeds'] or len(set(c['seeds'])) != len(c['seeds']):
        raise ValueError('Seeds must be nonempty and unique')
    if any(type(s) is not int or s < 0 for s in c['seeds']):
        raise ValueError('Seeds must be nonnegative integers')
    required_env = {
        'CONDA_ENV', 'TRAIN_MAX_EPOCH', 'GEN_BATCHES', 'GEN_BATCH_SIZE',
        'TARGET_CONDUCTIVITY', 'ANION_RATIO', 'TEMPERATURE_C', 'SALT_MOLAR_RATIO',
        'ANION_RATIO_TOLERANCE', 'MECHANISM_AUX_WEIGHT', 'MECHANISM_AUX_HIDDEN_DIM',
        'ORACLE_CONFIG', 'ORACLE_CKPT'}
    if set(c['environment']) != required_env:
        raise ValueError('environment must specify exactly the documented legacy settings')
    for key in ('TRAIN_MAX_EPOCH', 'GEN_BATCHES', 'GEN_BATCH_SIZE', 'MECHANISM_AUX_HIDDEN_DIM'):
        if int(c['environment'][key]) <= 0:
            raise ValueError(f'{key} must be positive')
    if not c['comparison']['metrics']:
        raise ValueError('At least one comparison metric is required')
    output = resolve(c['output_root'])
    for old in (resolve(c['source_q']), resolve(c['reference_root'])):
        if output == old or output in old.parents or old in output.parents:
            raise ValueError('Output must be isolated from source and reference directories')
    for key in ('absolute_tolerance', 'relative_tolerance'):
        v = c['comparison'][key]
        if not math.isfinite(v) or v < 0:
            raise ValueError('Comparison tolerances must be finite and nonnegative')


def inputs(c):
    q = resolve(c['source_q'])
    paths = [q / 'generator_data' / n for n in
             ('processed_data.pkl', 'synthetic_data.json', 'dataset_config.yaml', 'synthetic_metadata.json')]
    paths += [q / 'candidate_pool' / n for n in ('solvents.json', 'salts.json', 'solv_embs.pt', 'salt_embs.pt')]
    paths += [resolve(c['environment']['ORACLE_CONFIG'])]
    paths += [resolve(c['environment']['ORACLE_CKPT']) / n for n in ('pretrain.pt', 'optimal.pt')]
    return paths


def fingerprint(c):
    paths = inputs(c)
    for directory in ('formula_design', 'scripts/generation', 'scripts/train', 'submit/generation'):
        paths += sorted(p for p in (ROOT / directory).rglob('*') if p.suffix in ('.py', '.sh'))
    return {'configuration': c, 'sha256': {str(p): digest(p) for p in paths}}


def guard(path, manifest):
    target = path / 'reproduction_manifest.json'
    if target.exists():
        if json.loads(target.read_text()) != manifest:
            raise ValueError(f'Inputs/configuration/code changed: choose a fresh output root ({path})')
    else:
        if path.exists() and any(path.iterdir()):
            raise ValueError(f'Refusing untracked existing output: {path}')
        path.mkdir(parents=True, exist_ok=True)
        temporary = path / '.manifest.tmp'
        temporary.write_text(json.dumps(manifest, indent=2) + '\n')
        temporary.replace(target)


def compare(c):
    report = {'meaning': 'Numerical regression check, not statistical equivalence', 'checks': [], 'passed': True}
    for seed in c['seeds']:
        tables = []
        for directory in (c['reference_root'], c['output_root']):
            p = resolve(directory) / f'G0_seed_{seed}' / 'generation_experiment_summary.csv'
            if not p.is_file():
                raise ValueError(f'Missing comparison summary: {p}')
            with p.open() as f:
                rows = list(csv.DictReader(f))
            table = {r['group']: r for r in rows}
            if len(table) != len(rows):
                raise ValueError(f'Duplicate groups: {p}')
            tables.append(table)
        for group in GROUPS:
            for metric in c['comparison']['metrics']:
                a, b = (float(t[group][metric]) for t in tables)
                ok = math.isfinite(a) and math.isfinite(b) and math.isclose(
                    a, b, rel_tol=c['comparison']['relative_tolerance'],
                    abs_tol=c['comparison']['absolute_tolerance'])
                report['checks'].append(dict(seed=seed, group=group, metric=metric,
                                             reference=a, current=b, passed=ok))
                report['passed'] &= ok
    out = resolve(c['output_root'])
    out.mkdir(parents=True, exist_ok=True)
    (out / 'reproduction_comparison.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))
    return 0 if report['passed'] else 1


def selected_jobs(c, seed=None, group=None):
    if seed is not None and seed not in c['seeds']:
        raise ValueError(f'Seed {seed} is not listed in the task configuration')
    if group is not None and group not in GROUPS:
        raise ValueError(f'Unknown group: {group}')
    return [(s, g) for s in ([seed] if seed is not None else c['seeds'])
            for g in ([group] if group else GROUPS)]


def require_completed_q(seed_root):
    q = seed_root / 'G0-Q'
    required = ('train_decoder/FINISHED', 'train_decoder/optimal.pt',
                'train_diffusion/FINISHED', 'train_diffusion/optimal.pt',
                'generated/generated_formulas.json',
                'generator_data/processed_data.pkl', 'configs/decoder.yaml')
    missing = [str(q / name) for name in required if not (q / name).is_file()]
    if missing:
        raise ValueError('Complete the same-seed G0-Q first:\n' + '\n'.join(missing))
    return q


def run_job(c, manifest, seed, group):
    import fcntl
    out = resolve(c['output_root'])
    locks = out / '.locks'
    locks.mkdir(parents=True, exist_ok=True)
    seed_root = out / f'G0_seed_{seed}'
    # Lock a group exclusively. MA readers share a Q lock so Q cannot be
    # rewritten while its decoder/data are in use. Different seeds are free.
    with ExitStack() as stack:
        q_lock = stack.enter_context((locks / f'{seed}-G0-Q.lock').open('a'))
        fcntl.flock(q_lock, (fcntl.LOCK_EX if group == 'G0-Q' else fcntl.LOCK_SH)
                    | fcntl.LOCK_NB)
        if group != 'G0-Q':
            arm_lock = stack.enter_context((locks / f'{seed}-{group}.lock').open('a'))
            fcntl.flock(arm_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with (locks / f'{seed}-manifest.lock').open('a') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            guard(seed_root, manifest)
        q = resolve(c['source_q']) if group == 'G0-Q' else require_completed_q(seed_root)
        env = os.environ.copy()
        env.update({k: str(v) for k, v in c['environment'].items()})
        env.update(SEED=str(seed), EXPERIMENT_ROOT=str(seed_root),
                   SOURCE_GROUP_DIR=str(q), SOURCE_DATA_DIR=str(q / 'generator_data'),
                   VOCAB_DIR=str(q / 'candidate_pool'),
                   SOURCE_DECODER_DIR='' if group == 'G0-Q' else str(q / 'train_decoder'))
        with (seed_root / f'{group}.log').open('a') as log:
            print(f'Running {group}, seed {seed}; log {log.name}', flush=True)
            subprocess.run(['bash', str(ROOT / 'submit/generation/run_mechanism_auxiliary.sh'), group],
                           cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default='config/reproduction/g0_legacy.json')
    parser.add_argument('--seed', type=int, help='Run only this configured seed')
    parser.add_argument('--group', choices=GROUPS, help='Run only this experiment arm')
    parser.add_argument('action', choices=('plan', 'preflight', 'run', 'compare'))
    args = parser.parse_args()
    c = json.loads(resolve(args.config).read_text())
    validate(c)
    if args.action == 'compare':
        if args.seed is not None or args.group is not None:
            raise ValueError('compare checks the full configured triplet; omit --seed/--group')
        return compare(c)
    jobs = selected_jobs(c, args.seed, args.group)
    for seed, group in jobs:
        print(f'Seed {seed}: {group}; output {resolve(c["output_root"]) / f"G0_seed_{seed}" / group}', flush=True)
    if args.action == 'plan':
        print('Legacy source data reused unchanged; complete same-seed Q before either MA arm.')
        return 0
    missing = [str(p) for p in inputs(c) if not p.is_file()]
    if missing:
        raise ValueError('Missing required assets:\n' + '\n'.join(missing))
    manifest = fingerprint(c)
    if args.action == 'preflight':
        if args.group and args.group != 'G0-Q':
            for seed, _ in jobs:
                require_completed_q(resolve(c['output_root']) / f'G0_seed_{seed}')
        print('Required input files found and hashed. GPU/runtime validation occurs at launch.')
        return 0
    for seed, group in jobs:
        run_job(c, manifest, seed, group)
    print('Selected jobs completed. Compare after all configured groups and seeds finish.')
    return 0


if __name__ == '__main__':
    try:
        raise SystemExit(main())
    except (ValueError, KeyError, OSError, subprocess.CalledProcessError) as exc:
        raise SystemExit(str(exc))
