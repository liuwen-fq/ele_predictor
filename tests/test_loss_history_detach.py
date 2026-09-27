import ast
from pathlib import Path
import unittest
import torch


class LossHistoryTests(unittest.TestCase):
    def test_actual_history_expression_drops_graph_and_backing_storage(self):
        source = Path('formula_design/train/trainer.py').read_text()
        tree = ast.parse(source)
        method = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == 'train_epoch')
        call = next(n for n in ast.walk(method) if isinstance(n, ast.Call)
                    and isinstance(n.func, ast.Attribute) and n.func.attr == 'append'
                    and 'loss_term_history' in ast.unparse(n.func))
        expression = ast.Expression(call.args[0].elts[2])
        x = torch.ones(10000, requires_grad=True)
        full = x * 2
        loss_terms = {'loss': full[0]}
        history = eval(compile(expression, '<history>', 'eval'), {'torch': torch, 'loss_terms': loss_terms})
        self.assertEqual(history['loss'].item(), 2)
        self.assertIsNone(history['loss'].grad_fn)
        self.assertFalse(history['loss'].requires_grad)
        self.assertEqual(history['loss'].untyped_storage().nbytes(), history['loss'].element_size())
        loss_terms['loss'].backward()
        self.assertEqual(x.grad[0].item(), 2)

    def test_checkpoint_restore_compacts_history_and_restores_optimizer(self):
        import tempfile
        from types import SimpleNamespace
        tree = ast.parse(Path('formula_design/train/trainer.py').read_text())
        method = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == 'load_ckpt')
        namespace = {'torch': torch}
        exec(compile(ast.Module(body=[method], type_ignores=[]), '<restore>', 'exec'), namespace)
        model = torch.nn.Linear(1, 1)
        opt = torch.optim.Adam(model.parameters(), lr=0.01)
        model(torch.ones(1, 1)).sum().backward(); opt.step()
        state = {'epoch': 400, 'loss_term_history': [[[1, 0, {'loss': torch.ones(10000)[0]}]]]}
        checkpoint = {'model_state_dict': model.state_dict(), 'optimal_state_dict': model.state_dict(),
                      'optimizer_state_dict': opt.state_dict(), 'scheduler_state_dict': None,
                      'trainer_state_dict': state}
        target = torch.nn.Linear(1, 1)
        obj = SimpleNamespace(model=target, world_size=1, scheduler=None,
                              optimizer=torch.optim.Adam(target.parameters(), lr=0.1),
                              trainer_state_variables=list(state))
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / 'checkpoint.pt'; torch.save(checkpoint, path)
            namespace['load_ckpt'](obj, str(path), model_only=False)
        self.assertEqual(obj.epoch, 400)
        self.assertEqual(obj.optimizer.param_groups[0]['lr'], 0.01)
        self.assertTrue(torch.equal(target.weight, model.weight))
        scalar = obj.loss_term_history[0][0][2]['loss']
        self.assertEqual(scalar.untyped_storage().nbytes(), scalar.element_size())


if __name__ == '__main__':
    unittest.main()
