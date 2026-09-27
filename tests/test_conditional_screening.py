import copy
import json
from pathlib import Path
import tempfile
import unittest
from formula_design.conditional_screening import load_task, fit_rules, select_candidates


class ConditionalTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.records = []
        for i in range(8):
            self.records.append(dict(solvents=[dict(smiles='CCO', molar_ratio=.3+i*.01)],
                salts=[dict(name='LiFSI', molar_ratio=1.)], temperature=25., salt_molar_ratio=.1,
                conductivity_explain_pred=10., anion_ratio_explain_pred=.5,
                fragment_attention={'solvents':[{'fragments':[dict(smiles='CO',
                    conductivity_main_effect_delta_ln=.05, anion_ratio_main_effect_logit=0.)]}]},
                molecular_property_profiles=[dict(molar_ratio=1., property_values_normalized={'dc':10.,'vis':1.,'Tm':250.})]))
        self.source = self.root/'train_predictions.json'
        self.source.write_text(json.dumps(self.records))
        self.config = dict(schema_version=1, adapter='bamboo_prediction_export_v1', model_id='fixture',
            sources=[dict(path=self.source.name, role='rule_fit', split='train')],
            target=dict(conductivity_ms_cm=12., anion_ratio=.5, conductivity_tolerance=2.,
                        anion_ratio_tolerance=.1, temperature_c=25., salt_molar_ratio=.1, salts={'LiFSI':1.}), options={})

    def fit(self):
        p=self.root/'task.json';p.write_text(json.dumps(self.config))
        return fit_rules(load_task(p), lambda name, value:value)

    def test_target_changes_fragment_preference(self):
        high=self.fit()
        self.config['target']['conductivity_ms_cm']=8.
        low=self.fit()
        self.assertGreater(high['fragments']['CO']['score'],0)
        self.assertLess(low['fragments']['CO']['score'],0)

    def test_sparse_or_wrong_salt_falls_back(self):
        self.config['target']['salts']={'LiPF6':1.}
        r=self.fit()
        self.assertEqual(r['fallback'],'insufficient_local_evidence')
        self.assertEqual(r['property_windows'],{})

    def test_duplicate_exports_do_not_inflate_support(self):
        self.source.write_text(json.dumps([self.records[0]]*30))
        r=self.fit()
        self.assertEqual(r['unique_formulas'],1)
        self.assertIsNotNone(r['fallback'])

    def test_test_split_rejected(self):
        self.config['sources'][0]['split']='test'
        with self.assertRaises(ValueError):self.fit()

    def test_missing_explanation_is_not_silently_accepted(self):
        del self.records[0]['fragment_attention']
        self.source.write_text(json.dumps(self.records))
        with self.assertRaises(KeyError):self.fit()

    def test_selection_is_deterministic_unique_and_keeps_exploration(self):
        r=self.fit()
        entries=[dict(name=str(i),smiles=s) for i,s in enumerate(['CCO','CCCO','CCCC','CCN','CCOC'])]
        predictions={e['name']:dict(dc=10.,vis=1.,Tm=250.) for e in entries}
        a=select_candidates(entries,predictions,r,3,123)
        b=select_candidates(entries,predictions,r,3,123)
        self.assertEqual(a,b)
        self.assertEqual(len({x['name'] for x in a}),3)
        self.assertTrue(a[-1]['exploration'])
        self.assertEqual(r['property_windows']['dc'],[10.,10.])

    def test_candidate_cli_cache_and_stale_target_rejection(self):
        import os
        import subprocess
        import sys
        root = Path(__file__).resolve().parents[1]
        task = self.root/'task.json'; task.write_text(json.dumps(self.config))
        catalog=self.root/'catalog.json'
        catalog.write_text(json.dumps([dict(name=str(i),smiles=smiles,
            predicted_properties=dict(dc=1.,vis=10.,Tm=20.))
            for i,smiles in enumerate(['CCO','CCCO','CCN','CCOC','CCCC'])]))
        salts=self.root/'salts.json';salts.write_text(json.dumps([dict(name='LiFSI',smiles='F')]))
        output=self.root/'pool'
        command=[sys.executable, str(root/'scripts/generation/build_candidate_pool.py'),
            '--mode','conditional','--conditional-task',str(task),'--output-dir',str(output),
            '--baseline-solvents',str(catalog),'--salts',str(salts),
            '--precomputed-candidate-source',str(catalog),'--pool-size','3']
        env={**os.environ,'PYTHONPATH':str(root),'MPLCONFIGDIR':str(self.root/'mpl')}
        a=subprocess.run(command,cwd=root,env=env,capture_output=True,text=True)
        self.assertEqual(a.returncode,0,a.stdout+a.stderr)
        self.assertEqual(len(json.loads((output/'solvents.json').read_text())),3)
        b=subprocess.run(command,cwd=root,env=env,capture_output=True,text=True)
        self.assertEqual(b.returncode,0,b.stderr)
        self.assertIn('Reusing verified',b.stdout)
        # Even outside historical dc/vis gates, both arms retain the full pool.
        metadata = json.loads((output/'candidate_pool_metadata.json').read_text())
        self.assertEqual(metadata['common_eligibility']['eligible_candidates'], 5)
        self.assertIsNone(metadata['common_eligibility']['minimum_dc'])
        self.assertIsNone(metadata['common_eligibility']['maximum_vis'])
        random_command = list(command)
        random_command[random_command.index('--mode')+1] = 'random'
        random_output = self.root/'random_pool'
        random_command[random_command.index('--output-dir')+1] = str(random_output)
        random_command.remove('--conditional-task')
        random_command.remove(str(task))
        result = subprocess.run(random_command,cwd=root,env=env,capture_output=True,text=True)
        self.assertEqual(result.returncode,0,result.stdout+result.stderr)
        metadata = json.loads((random_output/'candidate_pool_metadata.json').read_text())
        self.assertEqual(metadata['common_eligibility']['eligible_candidates'],5)
        self.config['target']['conductivity_ms_cm']=8.
        task.write_text(json.dumps(self.config))
        c=subprocess.run(command,cwd=root,env=env,capture_output=True,text=True)
        self.assertNotEqual(c.returncode,0)
        self.assertIn('inputs changed',c.stderr)


if __name__=='__main__':unittest.main()
