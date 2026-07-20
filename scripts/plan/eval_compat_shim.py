"""Backward-compat shim for evaluating OLD dinogcbc ``*_object.ckpt`` checkpoints.

Old pickled ``Transformer`` objects (e.g.
``checkpoints/.../4170662/dinogcbc_epoch_0012_object.ckpt``) were saved before 
``aggregation_layers`` were added to ``stable_worldmodel.wm.gcrl.module.Transformer`` 
(commit de3b4ca, 2026-07-17). The current ``Transformer.forward`` references it, so unpickling an old object and
running it raises::

    AttributeError: 'Transformer' object has no attribute 'aggregation_layers'

Importing this module monkeypatches ``Transformer`` so that any instance restored
from a pickle that lacks these attributes gets correct defaults -- WITHOUT touching
the production ``module.py``. Import it *before* the checkpoint is unpickled (see
``eval_ff_compat.py``).

Semantics
---------
A model where ``aggregation_layers`` is empty. 
This exactly reproduces the old forward pass under today's code path.
"""

from torch import nn

from stable_worldmodel.wm.gcrl.module import Transformer

# nn.Module defines __setstate__ (backward-compat for _non_persistent_buffers_set);
# Transformer inherits it. Capture it so we can call through after restoring state.
_orig_setstate = Transformer.__setstate__


def _compat_setstate(self, state):
    _orig_setstate(self, state)  # restore the pickled __dict__ (incl. _modules)
    patched = []
    if not hasattr(self, "aggregation_layers"):
        # assigning an nn.Module registers it in self._modules (already restored)
        self.aggregation_layers = nn.ModuleList([])
        patched.append("aggregation_layers=[]")
    if patched:
        print(f"[eval_compat_shim] patched restored Transformer: {', '.join(patched)}")


Transformer.__setstate__ = _compat_setstate
print("[eval_compat_shim] Transformer backward-compat patch installed")
