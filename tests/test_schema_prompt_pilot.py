import json
from pathlib import Path
import sys
import unittest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from dataset_catalog import read_json,validate_record,value_at
from schema_prompt_pilot import OUT,SIZES,nested_template,improved_prompt,PARENT


class SchemaPromptTests(unittest.TestCase):
    def test_templates_preserve_schema_and_required_values(self):
        for key in SIZES:
            p=read_json(OUT/key/'protocol.json')
            for c in p['conditions']:
                template=nested_template(key,c)
                self.assertEqual(validate_record(key,template),[])
                for path,value in c['required'].items():self.assertEqual(value_at(template,path),value)
                if key=='reliability':
                    self.assertEqual(len(template['input']['conversation_history']),c['history_turns'])
                    self.assertEqual(len(template['expected_output']['required_actions']),c['required_action_count'])
                    self.assertNotIn('input.customer_message',template)

    def test_representation_change_preserves_original_constraints(self):
        separator='\nRequired field values (dotted paths refer to nested JSON fields): '
        for key in SIZES:
            parent=read_json(PARENT/key/'protocol.json');pilot=read_json(OUT/key/'protocol.json')
            for old,new,c in zip(parent['prompts'],pilot['prompts'],pilot['conditions']):
                prefix,body=old.split(separator,1);_,end=json.JSONDecoder().raw_decode(body)
                self.assertTrue(new.startswith(prefix));self.assertTrue(new.endswith(body[end+1:]))
                self.assertEqual(new,improved_prompt(key,old,c))
                self.assertNotIn('dotted paths',new)


if __name__=='__main__':unittest.main()
