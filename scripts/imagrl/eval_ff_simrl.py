"""Eval entrypoint for simrl PPO checkpoints (GCBCMultiInputPolicy).

Installs the ``simrl_policy_shim`` (makes ``rl_zoo3.gcbc_policy`` importable
without the full rl-zoo dependency stack), then runs the standard ``eval_ff.py``
hydra app UNCHANGED via runpy so hydra resolves ``config_path='./config'`` exactly
as a direct ``python eval_ff.py`` would. Same CLI as eval_ff.py, e.g.:
    python scripts/plan/eval_ff_simrl.py --config-name pusht rl_algo=ppo policy=<zip>

imag-rl (stock MultiInputPolicy) does NOT need this -- use eval_ff.py directly.
Requires GCBC_WANDB_RUN (+ project/entity/alias) set so the GCBC-initialized
policy can resolve its checkpoint at construction.
"""

import os
import runpy
import sys

import simrl_policy_shim  # noqa: F401  -- installs bare rl_zoo3.gcbc_policy on import

_HERE = os.path.dirname(os.path.abspath(__file__))
_EVAL_FF = os.path.join(_HERE, 'eval_ff.py')

# Make the launched-script identity look like eval_ff.py so hydra's config_path
# resolution matches a direct `python eval_ff.py` invocation.
sys.argv[0] = _EVAL_FF
runpy.run_path(_EVAL_FF, run_name='__main__')
