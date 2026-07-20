"""Make ``rl_zoo3.gcbc_policy`` importable WITHOUT running ``rl_zoo3/__init__.py``.

simrl PPO checkpoints (e.g. runs/rl_pixels/<job>/ppo/swm-PushT-Pixels-v1_1/*.zip)
were saved with ``policy_class = rl_zoo3.gcbc_policy.GCBCMultiInputPolicy``
(the GCBC-initialized PPO policy). Unpickling them needs that class importable.

But ``import rl_zoo3`` runs ``rl_zoo3/__init__.py``, which pulls in the full rl-zoo
dependency stack (huggingface_sb3, sb3_contrib, optuna, ...) that the
stable-worldmodel eval venv does not have. ``gcbc_policy.py`` itself only needs
numpy / torch / gymnasium / stable_baselines3 / stable_worldmodel (all present in
the eval venv), so we build a *bare* ``rl_zoo3`` package object with no __init__
side effects and load ``gcbc_policy`` under it. After import, ``sys.modules`` has
``rl_zoo3.gcbc_policy`` so cloudpickle can resolve the class during PPO.load.

imag-rl policies use stock ``MultiInputPolicy`` and do NOT need this shim.
"""

import importlib.util
import sys
import types

_RLZOO_PKG = '/home/hk-project-p0024638/usjuy/worldbench/rl-baselines3-zoo/rl_zoo3'

if 'rl_zoo3.gcbc_policy' not in sys.modules:
    if 'rl_zoo3' not in sys.modules:
        pkg = types.ModuleType('rl_zoo3')
        pkg.__path__ = [_RLZOO_PKG]          # mark as a package (enables submodule import)
        sys.modules['rl_zoo3'] = pkg
    spec = importlib.util.spec_from_file_location(
        'rl_zoo3.gcbc_policy', f'{_RLZOO_PKG}/gcbc_policy.py'
    )
    mod = importlib.util.module_from_spec(spec)
    sys.modules['rl_zoo3.gcbc_policy'] = mod
    spec.loader.exec_module(mod)
    setattr(sys.modules['rl_zoo3'], 'gcbc_policy', mod)
    print(
        '[simrl_policy_shim] bare rl_zoo3.gcbc_policy installed:',
        mod.GCBCMultiInputPolicy.__name__,
    )
