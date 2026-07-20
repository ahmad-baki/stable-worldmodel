"""Eval entrypoint that installs the old-checkpoint backward-compat shim, then runs
the standard ``eval_ff.py`` hydra app UNCHANGED.

Use this INSTEAD of ``eval_ff.py`` when the policy is an OLD ``*_object.ckpt`` that
predates the ``pool_type`` / ``aggregation_layers`` additions to the Transformer
(see ``eval_compat_shim.py``). For current checkpoints, keep using ``eval_ff.py``.

Implementation: we do NOT ``import run`` from eval_ff and call it -- that makes
hydra resolve ``config_path='./config'`` relative to THIS file's package and fail
with "Primary config module '..config' not found". Instead we import the shim (to
patch Transformer, cached in sys.modules) and then execute eval_ff.py itself as
``__main__`` via runpy, so hydra sees eval_ff.py as the entry script and resolves
its config exactly as normal. Same CLI as eval_ff.py:
    python scripts/plan/eval_ff_compat.py --config-name pusht policy=... eval.goal_offset_steps=...
"""

import os
import runpy
import sys

import eval_compat_shim  # noqa: F401  -- installs the Transformer monkeypatch on import

_HERE = os.path.dirname(os.path.abspath(__file__))
_EVAL_FF = os.path.join(_HERE, "eval_ff.py")

# Make the launched-script identity look like eval_ff.py so hydra's config_path
# resolution matches a direct `python eval_ff.py` invocation.
sys.argv[0] = _EVAL_FF
runpy.run_path(_EVAL_FF, run_name="__main__")
