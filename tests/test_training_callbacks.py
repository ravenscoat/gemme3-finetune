"""Exercise the actual nested callback without importing the GPU stack."""
import ast
from pathlib import Path
from types import SimpleNamespace
import unittest


class TrainingCallbackTest(unittest.TestCase):
    def test_training_arguments_do_not_shadow_source_checkpoint(self):
        path = Path(__file__).resolve().parents[1] / 'scripts/train_bitext.py'
        tree = ast.parse(path.read_text(encoding='utf-8'))
        callback = next(node for node in ast.walk(tree) if isinstance(node, ast.ClassDef) and node.name == 'EvaluateFinalStep')
        namespace = {'TrainerCallback': object, 'source_checkpoint': 'saved-pilot',
                     'before': {'eval_loss': .5}, 'resume_step': 200}
        exec(compile(ast.Module(body=[callback], type_ignores=[]), str(path), 'exec'), namespace)
        state = SimpleNamespace()
        args = SimpleNamespace(logging_steps=25, eval_steps=500, save_steps=500)
        control = object()
        result = namespace['EvaluateFinalStep']().on_train_begin(args, state, control)
        self.assertIs(result, control)
        self.assertEqual(state.best_model_checkpoint, 'saved-pilot')
        self.assertEqual(state.best_metric, .5)
        self.assertEqual(state.eval_steps, 500)


if __name__ == '__main__':
    unittest.main()
