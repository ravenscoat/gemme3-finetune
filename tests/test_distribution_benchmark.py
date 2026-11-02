import sys
import unittest
from pathlib import Path
import numpy as np
import pandas as pd
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from distribution_benchmark import mmd2,median_gamma,words,text_views,features_frame,categorical_distributions


class DistributionBenchmarkTests(unittest.TestCase):
    def test_mmd_identical_permuted_and_shifted(self):
        x=np.array([[0.,0.],[1.,0.],[0.,1.],[1.,1.]])
        gamma=median_gamma(x)
        self.assertAlmostEqual(mmd2(x,x[::-1],gamma),0.)
        self.assertGreater(mmd2(x,x+10,gamma),.1)
        self.assertAlmostEqual(mmd2(x,x+1,gamma),mmd2(x+1,x,gamma))

    def test_keyword_normalization_and_placeholder_removal(self):
        self.assertEqual(words('The PAYMENT, payment! {{policy_number}} deductible.'),['payment','payment','deductible'])

    def test_joint_view_and_features_do_not_include_record_identifier(self):
        row={'domain':'auto-insurance','question':'What is my deductible?','answer':'Check your policy.',
             'source_id':'DO_NOT_EMBED_THIS'}
        self.assertNotIn('DO_NOT_EMBED_THIS',text_views('insuranceqa',row)['joint'])
        self.assertIn('What is my deductible?',text_views('insuranceqa',row)['joint'])
        self.assertNotIn('source_id',features_frame('insuranceqa',[row]).columns)

    def test_categorical_numeric_codes_and_missing_values(self):
        reference=pd.DataFrame({'code':[9.,None]});generated=pd.DataFrame({'code':[9,9]})
        metadata={'tables':{'records':{'columns':{'code':{'sdtype':'categorical'}}}}}
        metric=categorical_distributions(reference,generated,metadata)['code']
        self.assertEqual(metric['reference_counts'],{'9':1,'<MISSING>':1})
        self.assertEqual(metric['generated_counts'],{'9':2})
        self.assertAlmostEqual(metric['total_variation'],.5)


if __name__=='__main__':unittest.main()
