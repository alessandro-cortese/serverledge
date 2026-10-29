"""Deterministic structural tests; do not train an alternative model."""
from __future__ import annotations
import importlib.util
from pathlib import Path
import unittest

import numpy as np
import pandas as pd

MODULE = Path(__file__).with_name('paper_original_prediction_pipeline.py')
spec = importlib.util.spec_from_file_location('paper_original_prediction_pipeline', MODULE)
paper = importlib.util.module_from_spec(spec)
spec.loader.exec_module(paper)


class FixedPaperPipelineTests(unittest.TestCase):
    def test_exact_author_model_parameters(self):
        from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
        cls = RandomForestClassifier(**paper.MODEL_CLASSIFIER)
        reg = RandomForestRegressor(**paper.MODEL_REGRESSOR)
        self.assertEqual(cls.get_params()['random_state'], 42)
        self.assertEqual(reg.get_params()['n_estimators'], 1000)
        self.assertEqual(reg.get_params()['random_state'], 42)
        self.assertEqual(len(paper.PREDICTORS_14), 14)
        self.assertEqual(paper.REGRESSORS_3, ['cpuUserDelta','cpuKernelDelta','cpuIdleDelta'])

    def test_pairing_is_always_per_function_and_within_architecture(self):
        values=[]
        for fn,shift in [('A',0),('B',10000)]:
            for arch,scale in [('amd64',1),('arm64',2)]:
                for j in range(100):
                    values.append(dict(function_name=fn,machine_tag=arch,request_id=f'{fn}-{arch}-{j}',experiment_id='test-01',
                                       runtime=shift+scale*(j+1),cpuUserDelta=1.,
                                       cpuKernelDelta=0.2,cpuIdleDelta=0.3))
        frame=pd.DataFrame(values).sample(frac=1,random_state=42)
        pairs=paper.make_pairs(frame,['A','B'])
        self.assertEqual(len(pairs),200)
        for fn in ('A','B'):
            sub=pairs[pairs.function_name==fn]
            self.assertEqual(len(sub),100)
            self.assertTrue(np.allclose(sub.runtime_arm_ms,sub.runtime_x86_ms*2 - (10000 if fn=='B' else 0)))
        # In particular, a shuffled input order may not affect percentile alignment.
        again=paper.make_pairs(frame.sample(frac=1,random_state=23),['A','B'])
        pd.testing.assert_frame_equal(pairs,again)

    def test_ties_are_deterministic(self):
        prediction=np.array(['ARM-similar','ARM-faster','ARM-similar','ARM-faster'])
        self.assertEqual(paper.majority_class(prediction),'ARM-faster')

    def test_resource_bound_threshold_is_strictly_more_than_10(self):
        self.assertEqual(paper.bound_label(10),'cpu-user')
        self.assertEqual(paper.bound_label(10.01),'cpu-kernel')

    def test_no_training_target_in_paired_input(self):
        rows=[]
        for f in ('train_A','train_B','heldout'):
            for arch in ('amd64','arm64'):
                for i in range(100):
                    rows.append({'function_name':f,'machine_tag':arch,'request_id':f'{f}_{arch}_{i}', 'experiment_id':'test-01',
                                 'runtime':float(i+1),'cpuUserDelta':1.,'cpuKernelDelta':1.,'cpuIdleDelta':0.})
        paired=paper.make_pairs(pd.DataFrame(rows),['train_A','train_B'])
        self.assertEqual(len(paired),200)
        self.assertNotIn('heldout',set(paired.function_name))


if __name__=='__main__':
    unittest.main(verbosity=2)
