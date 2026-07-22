"""Use a trained LeWM world model's encoder as the frozen encoder of a GCBC
(GCRL) policy -- selected via ``encoder_type: lewm`` in the gcbc config.

Why an adapter is needed
------------------------
GCBC's ``GCRL.encode`` reads ``encoder(pixels).last_hidden_state`` and drops the
CLS token (``[:, 1:, :]``) to get *patch* tokens. LeWM instead uses the *CLS*
token + a projector as its single per-frame latent (``wm/lewm/lewm.py``). To put
GCBC in LeWM's OWN latent space (so a GCBC checkpoint can later warm-start LeWM
imag-rl, symmetric to the DINO path), we feed GCBC the LeWM latent -- one token
per frame.

``LeWMEncoderAdapter.forward`` therefore returns a fake ``last_hidden_state`` of
shape ``(B, 2, D)`` = ``[dummy_cls, lewm_latent]`` so GCBC's existing
``[:, 1:, :]`` drop-CLS yields the single LeWM-latent token (``num_patches = 1``)
with ZERO change to ``gcrl.py``.
"""
from pathlib import Path
from types import SimpleNamespace

import torch
from torch import nn


class LeWMEncoderAdapter(nn.Module):
    """Wrap a LeWM ViT encoder + projector so it plugs into GCBC's encode path.

    Mirrors ``LeWM.encode``: CLS token -> projector -> latent. Presents the
    latent as a single non-CLS token in a fake ``last_hidden_state``.
    """

    def __init__(self, lewm_encoder: nn.Module, projector: nn.Module):
        super().__init__()
        self.encoder = lewm_encoder
        self.projector = projector

    def forward(self, pixels: torch.Tensor, **kwargs) -> SimpleNamespace:
        # pixels: (B, C, H, W), ImageNet-normalized (same preprocessing LeWM and
        # GCBC both use). interpolate_pos_encoding is always on (as in LeWM).
        pixels = pixels.to(next(self.encoder.parameters()).dtype)
        out = self.encoder(pixels, interpolate_pos_encoding=True)
        cls = out.last_hidden_state[:, 0]          # (B, hidden)
        emb = self.projector(cls)                  # (B, D) -- the LeWM latent
        # Fake last_hidden_state: index 0 is a throwaway "CLS" that GCBC drops,
        # index 1 is the LeWM latent -> GCBC sees exactly one patch token.
        lhs = torch.stack([torch.zeros_like(emb), emb], dim=1)  # (B, 2, D)
        return SimpleNamespace(last_hidden_state=lhs)


def _resolve_lewm_ckpt(artifact_name: str) -> Path:
    """Resolve a LeWM ``*_object.ckpt`` from a wandb reference artifact.

    No ``type=`` filter -- SaveCkptCallback logs WM artifacts as wandb type
    ``model`` (see world_envs lewm.py). file:// refs resolve only on the cluster
    that logged them.
    """
    import wandb

    api = wandb.Api()
    art = api.artifact(artifact_name)
    entry = next(iter(art.manifest.entries.values()))
    ref_path = Path(entry.ref.removeprefix('file://'))
    if not ref_path.name.endswith('_object.ckpt'):
        objs = sorted(
            ref_path.parent.glob('*_object.ckpt'),
            key=lambda p: p.stat().st_mtime,
        )
        if not objs:
            raise RuntimeError(
                f'Artifact {artifact_name} references {ref_path} and no sibling '
                f'*_object.ckpt was found.'
            )
        ref_path = objs[-1]
    if not ref_path.exists():
        raise FileNotFoundError(
            f'Artifact {artifact_name} -> {ref_path} is not reachable here '
            f'(file:// refs resolve only on the logging cluster).'
        )
    return ref_path


def build_lewm_encoder(
    artifact_name: str, device: str = 'cpu', image_size: int = 224
) -> tuple[nn.Module, int]:
    """Load the LeWM WM and return ``(frozen_adapter, embedding_dim)``.

    ``embedding_dim`` is probed from a dummy forward (= projector output dim).
    """
    ref_path = _resolve_lewm_ckpt(artifact_name)
    lewm = torch.load(ref_path, weights_only=False, map_location=device).eval()
    projector = getattr(lewm, 'projector', None) or nn.Identity()
    adapter = LeWMEncoderAdapter(lewm.encoder, projector).to(device).eval()
    adapter.requires_grad_(False)
    with torch.no_grad():
        dummy = torch.zeros(1, 3, image_size, image_size, device=device)
        emb_dim = int(adapter(dummy).last_hidden_state.shape[-1])
    return adapter, emb_dim
