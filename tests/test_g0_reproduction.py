import contextlib
import csv
import importlib.util
import io
import json
from pathlib import Path
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('reproduce_g0', ROOT / 'scripts/generation/reproduce_g0.py')
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)


class ReproductionTests(unittest.TestCase):
    def test_guard_rejects_changed_inputs_and_untracked_results(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / 'run'
            m.guard(p, {'hash': 'a'})
            m.guard(p, {'hash': 'a'})
            with self.assertRaises(ValueError):
                m.guard(p, {'hash': 'b'})
            q = Path(d) / 'legacy'
            q.mkdir()
            (q / 'checkpoint').write_text('existing')
            with self.assertRaises(ValueError):
                m.guard(q, {})

    def test_comparison_requires_all_arms_and_detects_regression(self):
        with tempfile.TemporaryDirectory() as d:
            c = {'reference_root': d + '/reference', 'output_root': d + '/output', 'seeds': [123],
                 'comparison': {'metrics': ['metric'], 'absolute_tolerance': 1e-6, 'relative_tolerance': 0}}
            def write(directory, value=1, groups=m.GROUPS):
                p = Path(directory) / 'G0_seed_123'
                p.mkdir(parents=True, exist_ok=True)
                with (p / 'generation_experiment_summary.csv').open('w') as f:
                    w = csv.DictWriter(f, fieldnames=['group', 'metric'])
                    w.writeheader()
                    w.writerows({'group': g, 'metric': value} for g in groups)
            write(c['reference_root'])
            write(c['output_root'])
            with contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(m.compare(c), 0)
                write(c['output_root'], 2)
                self.assertEqual(m.compare(c), 1)
                write(c['output_root'], groups=['G0-Q'])
                with self.assertRaises(KeyError):
                    m.compare(c)

    def test_job_selection_and_same_seed_dependency(self):
        c = {'seeds': [123, 456, 789]}
        self.assertEqual(m.selected_jobs(c, 456, 'G0-MA'), [(456, 'G0-MA')])
        self.assertEqual(len(m.selected_jobs(c)), 9)
        with self.assertRaises(ValueError):
            m.selected_jobs(c, 999, 'G0-Q')
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            with self.assertRaises(ValueError):
                m.require_completed_q(root)
            for name in ('train_decoder/FINISHED', 'train_decoder/optimal.pt',
                         'train_diffusion/FINISHED', 'train_diffusion/optimal.pt',
                         'generated/generated_formulas.json',
                         'generator_data/processed_data.pkl', 'configs/decoder.yaml'):
                p = root / 'G0-Q' / name
                p.parent.mkdir(parents=True, exist_ok=True)
                p.touch()
            self.assertEqual(m.require_completed_q(root), root / 'G0-Q')

    def test_source_output_overlap_rejected(self):
        c = json.loads((ROOT / 'config/reproduction/g0_legacy.json').read_text())
        c['output_root'] = c['reference_root']
        with self.assertRaises(ValueError):
            m.validate(c)


if __name__ == '__main__':
    unittest.main()
