"""A relocated or symlinked source must not retain its stale save_dir."""
import importlib.util
from pathlib import Path
import tempfile
import unittest
import yaml

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('generator_configs', ROOT / 'scripts/generation/write_generator_train_configs.py')
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)


class RelocationTests(unittest.TestCase):
    def test_relocated_source_is_read_without_mutation(self):
        with tempfile.TemporaryDirectory() as d:
            source = Path(d) / 'historical'
            source.mkdir()
            original = 'data_cls: FormulaData\nsave_dir: missing/G0-comp\nkwargs: {}\n'
            (source / 'dataset_config.yaml').write_text(original)
            (source / 'processed_data.pkl').write_bytes(b'unchanged')
            run = Path(d) / 'new_run'
            run.mkdir()
            (run / 'generator_data').symlink_to(source, target_is_directory=True)
            result = m.relocated_dataset_config(run)
            config = yaml.safe_load(result.read_text())
            self.assertEqual(config['save_dir'], str(source.resolve()))
            self.assertEqual(config['data_cls'], 'FormulaData')
            self.assertEqual((source / 'dataset_config.yaml').read_text(), original)
            self.assertEqual((source / 'processed_data.pkl').read_bytes(), b'unchanged')

    def test_missing_processed_data_fails(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            (root / 'generator_data').mkdir()
            (root / 'generator_data/dataset_config.yaml').write_text('data_cls: FormulaData\n')
            with self.assertRaises(FileNotFoundError):
                m.relocated_dataset_config(root)


if __name__ == '__main__':
    unittest.main()
