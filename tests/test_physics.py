"""Tests for the vectorized physics engine in megaVecto.py.

Uses a small hand-built creature (padded to a fixed size, like train2.py /
visualize.py do) rather than a real champion checkpoint, so these run fast
and need no GPU.
"""

import torch

from megaVecto import MegaCrea

DEVICE = torch.device("cpu")
MAX_NOEUDS = 6
MAX_MUSCLES = 6


def _pad(vec, size, dtype=torch.float32, pad_value=0.0):
    t = torch.tensor(vec, dtype=dtype)
    if len(vec) < size:
        t = torch.cat([t, torch.full((size - len(vec),), pad_value, dtype=dtype)])
    return t.unsqueeze(0)


def _triangle_dico():
    """3 real nodes (1 bone + 2 muscles), padded to MAX_NOEUDS / MAX_MUSCLES."""
    x = [0.0, 10.0, 5.0]
    y = [400.0, 400.0, 390.0]
    m1 = [0, 0, 1]
    m2 = [1, 2, 2]
    is_bone = [1.0, 0.0, 0.0]
    n_real, m_real = len(x), len(m1)

    return {
        "X": _pad(x, MAX_NOEUDS),
        "Y": _pad(y, MAX_NOEUDS),
        "m1": _pad(m1, MAX_MUSCLES, dtype=torch.long, pad_value=0),
        "m2": _pad(m2, MAX_MUSCLES, dtype=torch.long, pad_value=0),
        "stiffness": _pad([1.0 + 4 * b for b in is_bone], MAX_MUSCLES),
        "is_bone": _pad(is_bone, MAX_MUSCLES),
        "base_length": _pad([0.0] * m_real, MAX_MUSCLES),  # recomputed in __init__
        "masque_noeuds": (torch.arange(MAX_NOEUDS) < n_real).float().unsqueeze(0),
        "masque_muscles": (torch.arange(MAX_MUSCLES) < m_real).float().unsqueeze(0),
    }


def _make_mega(batch_size=4, seed=0):
    torch.manual_seed(seed)
    return MegaCrea(_triangle_dico(), batch_size, device=DEVICE)


def test_phantom_nodes_stay_immobile():
    mega = _make_mega()
    for _ in range(20):
        mega.apply_physics(torch.tensor(0.1))
    # nodes 3,4,5 are padding (phantom) -> velocity must be exactly masked to zero
    assert torch.all(mega.vX[:, :, 3:] == 0.0)
    assert torch.all(mega.vY[:, :, 3:] == 0.0)


def test_no_nan_under_zero_action():
    mega = _make_mega()
    for step in range(100):
        for _ in range(10):
            mega.apply_physics(torch.tensor(0.1))
        assert not torch.isnan(mega.X).any(), f"NaN in X at physics step {step}"
        assert not torch.isnan(mega.Y).any(), f"NaN in Y at physics step {step}"


def test_observation_and_reward_shapes():
    mega = _make_mega(batch_size=5)
    obs = mega.get_observation(frame=0)
    obs_size = MAX_NOEUDS * 4 + MAX_MUSCLES + 1
    assert obs.shape == (1, 5, obs_size)

    action = torch.zeros(1, 5, MAX_MUSCLES)
    mega.apply_action(action, frame=0)

    for _ in range(10):
        mega.apply_physics(torch.tensor(0.1))
    reward = mega.get_reward(coeff_energie=10000.0, coeff_hauteur=0.0)
    assert reward.shape == (1, 5)
    assert torch.isfinite(reward).all()


def test_energy_resets_after_get_reward():
    mega = _make_mega()
    action = torch.ones(1, mega.batch_size, MAX_MUSCLES) * 0.5
    mega.apply_action(action, frame=0)
    assert torch.any(mega.energy != 0.0)
    mega.get_reward(coeff_energie=10000.0, coeff_hauteur=0.0)
    assert torch.all(mega.energy == 0.0)


def test_apply_action_only_affects_real_muscles():
    mega = _make_mega()
    action = torch.ones(1, mega.batch_size, MAX_MUSCLES)
    before = mega.target_length.clone()
    mega.apply_action(action, frame=0)
    # padded muscle slots (index >= 3) must be untouched
    assert torch.allclose(mega.target_length[:, :, 3:], before[:, :, 3:])
    # real muscles (bone at 0 is excluded by mask_vrais_M_exp too)
    assert torch.allclose(mega.target_length[:, :, 0], before[:, :, 0])
