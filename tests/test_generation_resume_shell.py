"""Exercise the real shell dispatch with mock training, without importing CUDA."""
from pathlib import Path
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]


class ResumeDispatchTests(unittest.TestCase):
    def dispatch(self, files):
        source = (ROOT / 'submit/generation/run_mechanism_auxiliary.sh').read_text()
        function = source.split('train_or_resume() {', 1)[1].split('\n}\n', 1)[0]
        with tempfile.TemporaryDirectory() as d:
            work = Path(d) / 'work'
            work.mkdir()
            for name in files:
                p = work / name
                p.parent.mkdir(parents=True, exist_ok=True)
                p.touch()
            script = ('set -euo pipefail\nrecord_skip() { echo "SKIP $*"; }\n'
                      'run_stage() { echo "RUN $*"; }\ntrain_or_resume() {'
                      + function + '\n}\ntrain_or_resume train_diffusion "$1" config.yaml\n')
            return subprocess.run(['bash', '-c', script, 'test', str(work)], capture_output=True, text=True)

    def test_full_checkpoint_takes_precedence_over_best_model(self):
        r = self.dispatch(['optimal.pt', 'ckpt/ckpt_epoch_400.pt'])
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertIn('--restart', r.stdout)

    def test_finished_training_is_skipped(self):
        r = self.dispatch(['FINISHED', 'optimal.pt', 'ckpt/ckpt_epoch_350.pt'])
        self.assertEqual(r.returncode, 0)
        self.assertIn('SKIP', r.stdout)
        self.assertNotIn('RUN', r.stdout)

    def test_best_weights_alone_do_not_silently_restart_training(self):
        self.assertNotEqual(self.dispatch(['optimal.pt']).returncode, 0)


if __name__ == '__main__':
    unittest.main()
