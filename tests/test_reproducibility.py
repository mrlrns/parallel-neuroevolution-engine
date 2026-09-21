"""Reproducibility tests.

verify_reproducibility.py's whole premise is that replaying a champion under
torch.manual_seed(rep) gives a deterministic rollout. These tests pin that
property down at the engine level with a small synthetic creature (no
checkpoint needed), plus a couple of the specific bugs documented in
EXPERIMENTS.md / README.md that a regression could silently reintroduce.
"""

import torch

from brain import Brain
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
        "base_length": _pad([0.0] * m_real, MAX_MUSCLES),
        "masque_noeuds": (torch.arange(MAX_NOEUDS) < n_real).float().unsqueeze(0),
        "masque_muscles": (torch.arange(MAX_MUSCLES) < m_real).float().unsqueeze(0),
    }


def _rollout(seed, n_frames=40, sub_step=10):
    torch.manual_seed(seed)
    dico = _triangle_dico()
    obs_size = MAX_NOEUDS * 4 + MAX_MUSCLES + 1
    action_size = MAX_MUSCLES
    brain = Brain(obs_size, action_size)
    brain.eval()

    mega = MegaCrea(dico, batch_size=3, device=DEVICE)
    dt = torch.tensor(1.0 / sub_step)
    with torch.no_grad():
        for frame in range(n_frames):
            if frame % 5 == 0:
                obs = mega.get_observation(frame)
                action = brain(obs)
                mega.apply_action(action, frame)
            for _ in range(sub_step):
                mega.apply_physics(dt)
    return mega.X.clone(), mega.Y.clone()


def test_same_seed_gives_identical_rollout():
    x1, y1 = _rollout(seed=123)
    x2, y2 = _rollout(seed=123)
    assert torch.equal(x1, x2)
    assert torch.equal(y1, y2)


def test_different_seed_gives_different_rollout():
    x1, _ = _rollout(seed=1)
    x2, _ = _rollout(seed=2)
    assert not torch.equal(x1, x2)


def test_rest_lengths_recomputed_on_noised_positions():
    """Regression test for the bug fixed in README: base_length must be
    measured on the noised spawn positions, not the un-noised topology, so a
    rollout starts at equilibrium and free initial impulse-from-relaxation
    cannot inflate the reward."""
    torch.manual_seed(0)
    dico = _triangle_dico()
    mega = MegaCrea(dico, batch_size=8, device=DEVICE)

    m1e = mega.muscle1.unsqueeze(1).expand(-1, mega.batch_size, -1)
    m2e = mega.muscle2.unsqueeze(1).expand(-1, mega.batch_size, -1)
    X1 = torch.gather(mega.X, dim=2, index=m1e)
    Y1 = torch.gather(mega.Y, dim=2, index=m1e)
    X2 = torch.gather(mega.X, dim=2, index=m2e)
    Y2 = torch.gather(mega.Y, dim=2, index=m2e)
    actual_dist = torch.sqrt((X2 - X1) ** 2 + (Y2 - Y1) ** 2 + 1e-8) * mega.mask_M_exp

    assert torch.allclose(mega.base_length, actual_dist, atol=1e-4)
    assert torch.allclose(mega.target_length, mega.base_length, atol=1e-4)


def test_weights_saved_before_optimizer_step_differ_from_after():
    """Regression test for the 'champion saved post-update' bug: weights
    snapshotted before an optimizer.step() must not equal the weights after
    it, for a nonzero gradient step."""
    torch.manual_seed(0)
    brain = Brain(10, 4)
    optimizer = torch.optim.Adam(brain.parameters(), lr=0.1)

    before = {k: v.clone() for k, v in brain.state_dict().items()}

    x = torch.randn(3, 10)
    loss = brain(x).sum()
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    after = brain.state_dict()
    changed = any(not torch.equal(before[k], after[k]) for k in before)
    assert changed
