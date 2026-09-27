from pathlib import Path
import subprocess
import tempfile
import unittest
from scripts.generation.build_bamboo_pretrain_catalog import mono_candidates

ROOT = Path(__file__).resolve().parents[1]


class BambooSourceTests(unittest.TestCase):
    def test_source_is_strictly_pretraining_and_deduplicates(self):
        rows, rejected = mono_candidates([
            {'name':'same','smiles':'CCO'}, {'name':'alias','smiles':'OCC'},
            {'name':'same','smiles':'CCN'}, {'name':'ion','smiles':'[Na+]'}])
        self.assertEqual({r['smiles'] for r in rows}, {'CCO','CCN'})
        self.assertEqual(len({r['name'] for r in rows}),2)
        self.assertTrue(all(r['source']=='bamboo_mono_data' for r in rows))
        self.assertEqual(sum(rejected.values()),1)

    def test_property_gates_are_opt_in_and_shared(self):
        import os
        source = (ROOT/'submit/generation/run_bamboo_pretrain_mechanism_auxiliary_replicate.sh').read_text()
        block = source[source.index('COMMON_MINIMUM_DC='):source.index('# Within that shared universe')]
        script = 'set -euo pipefail\n' + block + '\nprintf "<%s>" "${COMMON_GATE_ARGS[@]}"'
        env = {k:v for k,v in os.environ.items() if k not in ('COMMON_MINIMUM_DC', 'COMMON_MAXIMUM_VIS')}
        for overrides, expected in [({}, '<>'),
                ({'COMMON_MINIMUM_DC':'8'}, '<--minimum-dc><8>'),
                ({'COMMON_MAXIMUM_VIS':'4'}, '<--maximum-vis><4>'),
                ({'COMMON_MINIMUM_DC':'8','COMMON_MAXIMUM_VIS':'4'}, '<--minimum-dc><8><--maximum-vis><4>')]:
            result = subprocess.run(['bash','-c',script], env={**env, **overrides}, capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(result.stdout, expected)
        for arm, end in [('prepare_g1_shared()', 'prepare_g2_shared()'),
                         ('prepare_g2_shared()', '# Both groups use')]:
            body = source[source.index(arm):source.index(end)]
            self.assertIn('"${COMMON_GATE_ARGS[@]}"', body)

    def test_existing_auxiliary_data_is_reusable(self):
        source = (ROOT/'submit/generation/run_bamboo_pretrain_mechanism_auxiliary_replicate.sh').read_text()
        functions = source[source.index('source_data_ready()'):source.index('for path in')]
        with tempfile.TemporaryDirectory() as d:
            for name in ('processed_data.pkl','synthetic_data.json','dataset_config.yaml','synthetic_metadata.json','mechanism_auxiliary.json'):
                (Path(d)/name).write_text('fixture')
            script = functions + '\nauxiliary_data_ready "$1"'
            result = subprocess.run(['bash','-c',script,'test',d], capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            (Path(d)/'mechanism_auxiliary.json').write_text('')
            result = subprocess.run(['bash','-c',script,'test',d], capture_output=True, text=True)
            self.assertNotEqual(result.returncode, 0)

    def test_both_ma_groups_train_decoder_without_q_dependency(self):
        source=(ROOT/'submit/generation/run_bamboo_pretrain_mechanism_auxiliary_replicate.sh').read_text()
        start=source.index('if [[ "$GROUP" == "G1-MA" ]]; then\n  mkdir')
        block=source[start:source.index('\nmkdir -p "$RUN"',start)]
        with tempfile.TemporaryDirectory() as d:
            script='''set -euo pipefail
GROUP="$1"; SEED=123
G1_ROOT="$2/G1"; G2_ROOT="$2/G2"; G1_SHARED="$2/shared1"; G2_SHARED="$2/shared2"
prepare_g1_shared() { :; }
prepare_g2_shared() { :; }
need_file() { [[ -f "$1" ]] || exit 9; }
'''+block+'\nprintf "%s %s %s" "$RETRAIN_DECODER" "$AUXILIARY" "$DATA"\n'
            for group in ('G1-MA','G2-MA'):
                r=subprocess.run(['bash','-c',script,'test',group,d],capture_output=True,text=True)
                self.assertEqual(r.returncode,0,r.stderr)
                self.assertTrue(r.stdout.startswith('1 1 '),r.stdout)
            r=subprocess.run(['bash','-c',script,'test','G2-MA-shuffle',d],capture_output=True,text=True)
            self.assertEqual(r.returncode,9)


if __name__=='__main__':unittest.main()
