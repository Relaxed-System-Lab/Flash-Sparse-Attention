"""CPU-only regression tests for the module's selected-attention dispatch.

Execute the dispatch statements from forward with mocked kernels, avoiding
imports of CUDA-only dependencies. These tests do not validate GPU numerics
or performance. Run with: python -m unittest discover -s test -p test_fsa_dispatch.py
"""

import ast
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import Mock


def load_dispatch():
    source = Path(__file__).resolve().parents[1] / "fsa/module/fsa.py"
    module = ast.parse(source.read_text())
    cls = next(node for node in module.body
               if isinstance(node, ast.ClassDef) and node.name == "FlashSparseAttention")
    forward = next(node for node in cls.body
                   if isinstance(node, ast.FunctionDef) and node.name == "forward")
    # Include the actual shape unpacking, group-size calculation and branch.
    start = next(i for i, node in enumerate(forward.body)
                 if isinstance(node, ast.Assign)
                 and any(isinstance(target, ast.Tuple)
                         and any(isinstance(item, ast.Name) and item.id == "num_q_heads"
                                 for item in target.elts)
                         for target in node.targets))
    end = next(i for i in range(start, len(forward.body))
               if isinstance(forward.body[i], ast.If))
    return compile(ast.Module(body=forward.body[start:end + 1], type_ignores=[]),
                   str(source), "exec")


class DispatchTest(unittest.TestCase):
    def test_selected_attention_dispatch(self):
        dispatch = load_dispatch()
        for group_size, expected in ((1, "fsa"), (4, "fsa"), (8, "nsa"), (16, "nsa")):
            for kv_heads in (1, 4):
                with self.subTest(group_size=group_size, kv_heads=kv_heads):
                    q = SimpleNamespace(shape=(64, kv_heads * group_size, 128))
                    k = SimpleNamespace(shape=(64, kv_heads, 128))
                    v, topk_idx, cu_seqlens = object(), object(), object()
                    fsa, nsa = Mock(), Mock()
                    scope = dict(q=q, k=k, v=v, topk_idx=topk_idx,
                                 cu_seqlens=cu_seqlens, self=SimpleNamespace(block_size=64),
                                 FSA_topk_sparse_attention=fsa, topk_sparse_attention=nsa)
                    exec(dispatch, scope)
                    selected, unused = (fsa, nsa) if expected == "fsa" else (nsa, fsa)
                    selected.assert_called_once_with(q, k, v, topk_idx, 64, cu_seqlens, None)
                    unused.assert_not_called()
                    self.assertIs(scope["sparse_attn_output"], selected.return_value)


if __name__ == "__main__":
    unittest.main()
