import json
from pathlib import Path
import sys
import unittest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from dataset_catalog import data,core,read_json
from prefix_prompt_pilot import prefix_for,SIZES,OUT


class PrefixPromptTests(unittest.TestCase):
    def test_prefix_matches_actual_training_record_serialization(self):
        fields={'reliability':['category','subcategory','difficulty'],'bitext':['intent','category','tags'],'insuranceqa':['domain']}
        for key in SIZES:
            for source in data(key)['train'][:20]:
                record=core(key,source);prefix=prefix_for(key,{f:record[f] for f in fields[key]})
                serialized=json.dumps(record,ensure_ascii=False)
                self.assertTrue(serialized.startswith(prefix),(key,prefix,serialized[:100]))
                self.assertEqual(json.loads(prefix+serialized[len(prefix):]),record)

    def test_user_prompt_is_original_and_has_no_free_text_template(self):
        for key,n in SIZES.items():
            p=read_json(OUT/key/'protocol.json')
            self.assertEqual(p['prompts'],[data(key)['prompt']]*n)
            for prefix,c in zip(p['assistant_prefixes'],p['conditions']):
                self.assertNotIn('<GENERATE:',prefix)
                self.assertEqual(prefix,prefix_for(key,c['required']))
                self.assertEqual(c['ranges'],{})


if __name__=='__main__':unittest.main()
