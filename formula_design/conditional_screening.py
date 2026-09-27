"""Target-local, explanation-derived molecular screening (no fixed motif scores).

The score is a surrogate explanation sensitivity, NOT a molecular causal effect.
Only explicit fitting exports are accepted. Candidate property units must match
those of physical_property_value in the Bamboo export adapter.
"""
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np
from rdkit import Chem, DataStructs
from rdkit.Chem import rdFingerprintGenerator

PROPERTIES = ('dc', 'vis', 'Tm')
DEFAULTS = dict(temperature_bandwidth=10., concentration_bandwidth=.03,
                salt_ratio_bandwidth=.25, max_context_distance=3.,
                min_effective_samples=5., shrinkage=10., uncertainty_z=1.,
                exploration_fraction=.15, diversity_weight=.5, property_weight=1.,
                fragment_weight=1.)


def file_hash(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def canonical(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f'Invalid SMILES: {smiles}')
    mol = Chem.RemoveHs(mol)
    for atom in mol.GetAtoms():
        atom.SetAtomMapNum(0)
    return Chem.MolToSmiles(mol)


def load_task(path):
    path = Path(path).resolve()
    task = json.loads(path.read_text())
    if task.get('schema_version') != 1 or task.get('adapter') != 'bamboo_prediction_export_v1':
        raise ValueError('Unsupported screening schema/adapter')
    if not task.get('model_id') or not task.get('sources'):
        raise ValueError('model_id and fitting sources are required')
    options = task.get('options', {})
    if set(options) - set(DEFAULTS):
        raise ValueError('Unknown screening option')
    task['options'] = {**DEFAULTS, **options}
    for key, value in task['options'].items():
        if not math.isfinite(value) or value < 0:
            raise ValueError(f'Invalid option: {key}')
    for key in ('temperature_bandwidth', 'concentration_bandwidth', 'salt_ratio_bandwidth',
                'max_context_distance', 'min_effective_samples', 'shrinkage'):
        if task['options'][key] <= 0:
            raise ValueError(f'{key} must be positive')
    if task['options']['exploration_fraction'] > 1:
        raise ValueError('exploration_fraction must be <= 1')
    t = task['target']
    for key in ('conductivity_ms_cm', 'anion_ratio', 'temperature_c', 'salt_molar_ratio',
                'conductivity_tolerance', 'anion_ratio_tolerance'):
        if not math.isfinite(t[key]):
            raise ValueError(f'Nonfinite target {key}')
    if t['conductivity_ms_cm'] <= 0 or min(t['conductivity_tolerance'], t['anion_ratio_tolerance']) <= 0:
        raise ValueError('Conductivity and tolerances must be positive')
    if not 0 <= t['anion_ratio'] <= 1 or not 0 < t['salt_molar_ratio'] < 1:
        raise ValueError('Ratio target outside its domain')
    salts = t['salts']
    if not salts or any(not math.isfinite(v) or v <= 0 for v in salts.values()) or not math.isclose(sum(salts.values()), 1.):
        raise ValueError('target.salts must map salt names to positive normalized ratios')
    for source in task['sources']:
        # A declaration cannot detect dishonest relabeling; obvious test files
        # and explicit test/validation roles are rejected rather than guessed.
        if source.get('role') != 'rule_fit' or source.get('split') != 'train':
            raise ValueError('Only role=rule_fit, split=train sources may fit rules')
        p = Path(source['path']).expanduser()
        p = (p if p.is_absolute() else path.parent / p).resolve()
        if 'test' in p.stem.lower() or 'valid' in p.stem.lower():
            raise ValueError('Held-out prediction exports cannot fit rules')
        source['path'] = str(p)
    return task


def effective_count(weights):
    w = np.asarray(weights, dtype=float)
    return float(w.sum() ** 2 / (w @ w)) if len(w) and w @ w > 0 else 0.


def context_weight(record, task):
    t, o = task['target'], task['options']
    salts = {s['name']: float(s['molar_ratio']) for s in record['salts']}
    if set(salts) != set(t['salts']):
        return 0.
    distance = ((float(record['temperature']) - t['temperature_c']) / o['temperature_bandwidth']) ** 2
    distance += ((float(record['salt_molar_ratio']) - t['salt_molar_ratio']) / o['concentration_bandwidth']) ** 2
    distance += sum(((salts[k] - v) / o['salt_ratio_bandwidth']) ** 2 for k, v in t['salts'].items())
    if not math.isfinite(distance) or distance > o['max_context_distance'] ** 2:
        return 0.
    return math.exp(-distance / 2)


def target_distance(sigma, ratio, target):
    return ((sigma - target['conductivity_ms_cm']) / target['conductivity_tolerance']) ** 2 + ((ratio - target['anion_ratio']) / target['anion_ratio_tolerance']) ** 2


def quantile(values, weights, q):
    order = np.argsort(values)
    v, w = np.asarray(values)[order], np.asarray(weights)[order]
    return float(np.interp(q, (np.cumsum(w) - .5 * w) / w.sum(), v))


def fit_rules(task, physical_property_value):
    o, t = task['options'], task['target']
    terms, props = defaultdict(list), defaultdict(list)
    seen, local_weights, provenance = set(), [], []
    total = 0
    for source in task['sources']:
        provenance.append(dict(path=source['path'], sha256=file_hash(source['path']), split='train'))
        records = json.loads(Path(source['path']).read_text())
        if not isinstance(records, list):
            raise ValueError('Bamboo prediction export must be a list')
        for r in records:
            total += 1
            # Repeated formula exports must not inflate statistical support.
            key = json.dumps([sorted((canonical(s['smiles']), s['molar_ratio']) for s in r['solvents']),
                              sorted((s['name'], s['molar_ratio']) for s in r['salts']),
                              r['temperature'], r['salt_molar_ratio']], sort_keys=True)
            if key in seen:
                continue
            seen.add(key)
            w = context_weight(r, task)
            if not w:
                continue
            sigma, ratio = float(r['conductivity_explain_pred']), float(r['anion_ratio_explain_pred'])
            if not math.isfinite(sigma) or sigma <= 0 or not 0 < ratio < 1:
                raise ValueError('Invalid explanation predictions in fitting export')
            base = target_distance(sigma, ratio, t)
            w *= math.exp(-min(base, 1500.) / 2)
            if w < 1e-8:
                continue
            local_weights.append(w)
            contributions = defaultdict(lambda: [0., 0.])
            for molecule in r['fragment_attention']['solvents']:
                for fragment in molecule['fragments']:
                    name = canonical(fragment['smiles'])
                    delta = contributions[name]
                    delta[0] += float(fragment['conductivity_main_effect_delta_ln'])
                    delta[1] += float(fragment['anion_ratio_main_effect_logit'])
            for name, (ds, dr) in contributions.items():
                if not math.isfinite(ds) or not math.isfinite(dr):
                    raise ValueError('Nonfinite fragment contribution')
                without_sigma = sigma * math.exp(np.clip(-ds, -50, 50))
                without_ratio = 1 / (1 + math.exp(-float(np.clip(math.log(ratio / (1-ratio)) - dr, -50, 50))))
                utility = float(np.clip(target_distance(without_sigma, without_ratio, t) - base, -10, 10))
                terms[name].append((utility, w))
            profiles = r['molecular_property_profiles']
            ratio_sum = sum(float(p['molar_ratio']) for p in profiles)
            if ratio_sum <= 0:
                raise ValueError('Missing solvent ratios in property profiles')
            for p in profiles:
                for name in PROPERTIES:
                    value = physical_property_value(name, float(p['property_values_normalized'][name]))
                    if not math.isfinite(value):
                        raise ValueError('Nonfinite molecular property')
                    weight = w * float(p['molar_ratio']) / ratio_sum
                    if weight > 0:
                        props[name].append((value, weight))
    neff = effective_count(local_weights)
    fallback = neff < o['min_effective_samples']
    learned = {}
    for name, samples in sorted(terms.items()):
        a = np.asarray(samples); values, weights = a[:, 0], a[:, 1]
        n = effective_count(weights); mean = float(np.average(values, weights=weights))
        variance = float(np.average((values - mean)**2, weights=weights))
        uncertainty = o['uncertainty_z'] * math.sqrt(variance / max(n, 1.))
        score = math.copysign(max(0., abs(mean) - uncertainty), mean) * n / (n + o['shrinkage'])
        if fallback or n < o['min_effective_samples']:
            score = 0.
        learned[name] = dict(score=score, mean=mean, standard_error=math.sqrt(variance / max(n, 1.)), effective_samples=n)
    windows = {}
    if not fallback:
        for name, samples in props.items():
            values, weights = zip(*samples)
            windows[name] = [quantile(values, weights, .1), quantile(values, weights, .9)]
    return dict(schema_version=1, target=t, model_id=task['model_id'], options=o,
                provenance=provenance, input_rows=total, unique_formulas=len(seen),
                local_formulas=len(local_weights), effective_samples=neff,
                fallback='insufficient_local_evidence' if fallback else None,
                fragments=learned, property_windows=windows,
                interpretation='target-local explanation sensitivity; not a causal effect')


def select_candidates(entries, predictions, rules, count, seed):
    if count <= 0 or count > len(entries):
        raise ValueError('Invalid pool size')
    rng = np.random.default_rng(seed)
    o = rules['options']
    patterns = [(name, Chem.MolFromSmiles(name), rule['score']) for name, rule in rules['fragments'].items() if rule['score']]
    generator = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
    rows, fps = [], []
    for item in entries:
        mol = Chem.MolFromSmiles(item['smiles'])
        matches = {name: score for name, pattern, score in patterns if mol.HasSubstructMatch(pattern)}
        fragment = float(np.mean(list(matches.values()))) if matches else 0.
        penalties = []
        for name, (lo, hi) in rules['property_windows'].items():
            value = float(predictions[item['name']][name])
            # Molecular distributions, not a mixture average applied as a gate.
            scale = max(hi-lo, abs(lo)*.05, 1e-6)
            penalties.append(-min(max(lo-value, value-hi, 0.)/scale, 2.))
        prop = float(np.mean(penalties)) if penalties else 0.
        rows.append({**item, 'selection': 'conditional_soft_v1', 'fragment_score': fragment,
                     'property_score': prop, 'matched_structure_terms': json.dumps(matches, sort_keys=True),
                     'fallback': rules['fallback'] or '', 'total_score': 0.})
        fps.append(generator.GetFingerprint(mol))
    # Bounded channel scales preserve evidence shrinkage. Re-standardizing
    # over candidates would amplify weak evidence back to unit variance.
    scores = np.asarray([o['fragment_weight'] * row['fragment_score'] / 10.
                         + o['property_weight'] * row['property_score'] / 2.
                         for row in rows])
    ties = rng.random(len(rows)); similarity = np.zeros(len(rows)); remaining = set(range(len(rows)))
    selected = []
    explore_count = math.ceil(count*o['exploration_fraction'])
    for step in range(count):
        explore = bool(rules['fallback']) or step >= count-explore_count
        best = max(remaining, key=lambda i: ((0. if explore else scores[i])-o['diversity_weight']*similarity[i], ties[i]))
        row = rows[best]
        row.update(total_score=float(scores[best]), selected_step=step+1, exploration=explore,
                   max_tanimoto_to_earlier=float(similarity[best]))
        selected.append(row); remaining.remove(best)
        similarity = np.maximum(similarity, DataStructs.BulkTanimotoSimilarity(fps[best], fps))
    return selected
