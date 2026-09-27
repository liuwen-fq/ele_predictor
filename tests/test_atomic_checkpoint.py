"""Test actual trainer checkpoint I/O in a subprocess: SIGBUS cannot kill tests."""
import subprocess
import sys
import unittest


class AtomicCheckpointTests(unittest.TestCase):
    def test_mmap_same_path_rewrite_and_failed_save_preserve_original(self):
        code = r'''
import ast, os, tempfile
from pathlib import Path
from unittest.mock import patch
import torch
source = ast.parse(Path('formula_design/train/trainer.py').read_text())
fn = next(n for n in source.body if isinstance(n, ast.FunctionDef) and n.name == 'atomic_torch_save')
ns = dict(torch=torch, os=os, tempfile=tempfile)
exec(compile(ast.Module(body=[fn], type_ignores=[]), '<atomic>', 'exec'), ns)
save = ns['atomic_torch_save']
with tempfile.TemporaryDirectory() as d:
    p = Path(d) / 'checkpoint.pt'
    expected = torch.arange(100000, dtype=torch.float32)
    torch.save({'tensor': expected}, p)
    mapped = torch.load(p, mmap=True, weights_only=False, map_location='cpu')
    save(mapped, p)
    assert torch.equal(mapped['tensor'], expected)
    assert torch.equal(torch.load(p, weights_only=False)['tensor'], expected)
    # Verify restart selection skips a damaged latest archive.
    import zipfile
    from glob import glob
    from types import SimpleNamespace
    get_latest = next(n for n in ast.walk(source) if isinstance(n, ast.FunctionDef) and n.name == 'get_latest_ckpt')
    ns.update(zipfile=zipfile, glob=glob)
    exec(compile(ast.Module(body=[get_latest], type_ignores=[]), '<latest>', 'exec'), ns)
    good = Path(d) / 'ckpt_epoch_350.pt'
    torch.save({'tensor': expected}, good)
    (Path(d) / 'ckpt_epoch_400.pt').write_bytes(b'truncated')
    assert ns['get_latest_ckpt'](SimpleNamespace(ckpt_folder=d)) == str(good)
    before = p.read_bytes()
    def fail(value, stream):
        stream.write(b'incomplete')
        raise OSError('simulated full filesystem')
    with patch.object(torch, 'save', side_effect=fail):
        try:
            save(mapped, p)
        except OSError:
            pass
        else:
            raise AssertionError('save should fail')
    assert p.read_bytes() == before
    assert not list(Path(d).glob('.checkpoint-*'))
'''
        result = subprocess.run([sys.executable, '-c', code], capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)


if __name__ == '__main__':
    unittest.main()
