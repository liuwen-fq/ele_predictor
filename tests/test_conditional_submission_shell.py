"""Exercise real preparation/resume shell functions with no GPU training."""
from pathlib import Path
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
SOURCE = (ROOT/'submit/generation/run_bamboo_pretrain_mechanism_auxiliary_replicate.sh').read_text()


def function(name):
    return name + '() {' + SOURCE.split(name+'() {', 1)[1].split('\n}\n', 1)[0] + '\n}\n'


class SubmissionTests(unittest.TestCase):
    def test_complete_shared_data_is_not_regenerated_on_repeated_submission(self):
        for arm in ('g1', 'g2'):
            with self.subTest(arm=arm), tempfile.TemporaryDirectory() as d:
                root = Path(d)
                for directory in ('candidate_pool', 'generator_data_source', 'generator_data_q', 'generator_data_ma_true'):
                    p = root/directory
                    p.mkdir()
                    for name in ('solvents.json', 'solv_embs.pt', 'processed_data.pkl', 'synthetic_data.json', 'dataset_config.yaml', 'synthetic_metadata.json', 'mechanism_auxiliary.json'):
                        (p/name).write_text('fixture')
                script = '''set -euo pipefail
G1_SHARED="$1"; G2_SHARED="$1"
CONDITIONAL_TASK=task.json; BAMBOO_SOLVENTS=solvents; BAMBOO_SALTS=salts
BAMBOO_CATALOG=catalog; POOL_SIZE=62; G2_VOCAB_SELECTION_SEED=123
COMMON_GATE_ARGS=()
python3() {
  # G2 always verifies the cached vocabulary; no other build is permitted.
  [[ "$*" == *build_candidate_pool.py* ]] || { echo "unexpected rebuild: $*" >&2; return 99; }
}
'''
                script += function('source_data_ready') + function('auxiliary_data_ready') + function('prepare_'+arm+'_shared')
                script += '\nprepare_'+arm+'_shared\nprepare_'+arm+'_shared\n'
                r = subprocess.run(['bash','-c',script,'test',d],capture_output=True,text=True)
                self.assertEqual(r.returncode,0,r.stdout+r.stderr)
                self.assertEqual(r.stdout.count('shared data ready'),2)

    def test_training_resume_states(self):
        for files, expected in [([], 'RUN'), (['ckpt/ckpt_epoch_100.pt','optimal.pt'], '--restart'),
                                (['FINISHED','optimal.pt'],'SKIP'), (['optimal.pt'],'ERROR')]:
            with self.subTest(files=files), tempfile.TemporaryDirectory() as d:
                work=Path(d)/'work'
                for name in files:
                    p=work/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_text('fixture')
                script='set -euo pipefail\nrun_stage() { echo "RUN $*"; }\nrecord_skip() { echo "SKIP $*"; }\n'
                script+=function('train_or_resume')+'\ntrain_or_resume train_decoder timing "$1" config\n'
                r=subprocess.run(['bash','-c',script,'test',str(work)],capture_output=True,text=True)
                if expected=='ERROR':
                    self.assertNotEqual(r.returncode,0)
                    self.assertIn('without FINISHED',r.stderr)
                else:
                    self.assertEqual(r.returncode,0,r.stderr)
                    self.assertIn(expected,r.stdout)

    def test_separate_entrypoints_forward_args_and_failure_status(self):
        # Copy wrappers into a temporary repository with a harmless mock dispatcher.
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);submit=root/'submit/generation';submit.mkdir(parents=True)
            task=root/'config/screening/conditional_g2.example.json';task.parent.mkdir(parents=True);task.write_text('{}')
            dispatcher=submit/'run_conditional_G1G2_MA.sh'
            dispatcher.write_text('printf "ARGS:<%s>\\n" "$@"\necho STDERR_CAPTURED >&2\nexit 17\n')
            for arm in ('G1','G2'):
                name=f'run_conditional_{arm}_MA.sh'
                entry=submit/name;entry.write_text((ROOT/'submit/generation'/name).read_text())
                r=subprocess.run(['bash',str(entry),'456','--plan'],capture_output=True,text=True)
                self.assertEqual(r.returncode,17,r.stdout+r.stderr)
                self.assertIn(f'ARGS:<{arm}-MA>',r.stdout)
                self.assertIn('ARGS:<456>',r.stdout)
                self.assertIn('ARGS:<--plan>',r.stdout)
                self.assertIn('STDERR_CAPTURED',r.stdout)
                self.assertIn('FAILED:',r.stdout)
                log=next((root/'run/submission_logs').glob(f'{arm}-MA_*.log'))
                self.assertIn('STDERR_CAPTURED',log.read_text())
                bad=subprocess.run(['bash',str(entry),'999'],capture_output=True,text=True)
                self.assertEqual(bad.returncode,2)


if __name__=='__main__':
    unittest.main()
