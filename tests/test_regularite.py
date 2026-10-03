"""Action-smoothness penalty (E16): off by default, and it charges sign flips."""

import torch

from megaVecto import MegaCrea
from tests.test_reproducibility import _triangle_dico  # petite créature synthétique


def _recompense(coef, actions):
    torch.manual_seed(0)
    mega = MegaCrea(_triangle_dico(), 4, device="cpu")
    mega.coef_regularite = coef
    total = torch.zeros(1, 4)
    for k, a in enumerate(actions):
        mega.apply_action(a.expand(1, 4, -1).clone(), 5 * k)
        total = total + mega.get_reward(10000.0, 0.0)
    return total


def test_penalty_off_by_default_changes_nothing():
    acts = [torch.full((1, 1, 6), v) for v in (0.5, -0.5, 0.5)]
    assert torch.equal(_recompense(0.0, acts), _recompense(0.0, acts))
    torch.manual_seed(0)
    assert MegaCrea(_triangle_dico(), 4, device="cpu").coef_regularite == 0.0


def test_penalty_charges_flips_not_constant_commands():
    constant = [torch.full((1, 1, 6), 0.5)] * 3
    alterne = [torch.full((1, 1, 6), v) for v in (0.5, -0.5, 0.5)]
    # commande constante : aucun saut, aucune pénalité
    assert torch.allclose(_recompense(1.0, constant), _recompense(0.0, constant))
    # commande qui alterne : pénalisée
    assert (_recompense(1.0, alterne) < _recompense(0.0, alterne)).all()
