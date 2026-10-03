"""MegaCreaFast must compute the same physics as MegaCrea (up to float rounding)."""

from pathlib import Path

import pytest
import torch

from champion import charger_champion
from megaVecto import MegaCrea
from megaVecto_fast import MegaCreaFast

REFERENCE = Path(__file__).resolve().parent.parent / "champion_raffine" / "reference.pt"
pytestmark = pytest.mark.skipif(not REFERENCE.exists(), reason="reference champion not available")


def _rollout(classe, dico, acts, frames=20, sub_step=20):
    torch.manual_seed(0)
    mega = classe(dico, acts.shape[2], device="cpu")
    dt = torch.tensor(1.0 / sub_step, dtype=mega.X.dtype)
    for frame in range(frames):
        if frame % 5 == 0:
            mega.apply_action(acts[frame // 5], frame)
        for _ in range(sub_step):
            mega.apply_physics(dt)
    return mega.X, mega.Y, mega.vX, mega.vY


def test_fast_physics_matches_reference_float64():
    torch.set_default_dtype(torch.float64)
    try:
        champ = charger_champion(str(REFERENCE), torch.device("cpu"))
        dico = {k: (v.double() if v.is_floating_point() else v) for k, v in champ.dico.items()}
        g = torch.Generator().manual_seed(1)
        acts = 0.8 * (2 * torch.rand(4, 1, 6, champ.action_size, generator=g, dtype=torch.float64) - 1)
        for a, b in zip(_rollout(MegaCrea, dico, acts), _rollout(MegaCreaFast, dico, acts)):
            assert (a - b).abs().max().item() < 1e-9
    finally:
        torch.set_default_dtype(torch.float32)
