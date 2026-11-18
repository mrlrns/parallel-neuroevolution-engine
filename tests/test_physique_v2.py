"""Physics v2, step 1 — links as XPBD constraints (physique_v2.py)."""

import math
import subprocess
import sys
from pathlib import Path

import pytest
import torch

from baselines.env import SwimEnv
from brain import Brain
from champion import charger_champion
from physique_v2 import MegaCreaXPBD, colorier_liens
from tools.rameur import commande, construire_rameur

RACINE = Path(__file__).resolve().parent.parent
REFERENCE = RACINE / "champion_raffine" / "reference.pt"
RAMEUR = RACINE / "champion_raffine" / "rameur.pt"
DEV = torch.device("cpu")

pytestmark = pytest.mark.skipif(not (REFERENCE.exists() and RAMEUR.exists()), reason="champions not available")


def _champ(chemin):
    return charger_champion(str(chemin), DEV)


def _barycentre(m):
    msk = m.mask_N_exp
    return (m.X * msk).sum(2) / msk.sum(2)


@pytest.mark.parametrize("chemin", [REFERENCE, RAMEUR])
def test_same_colour_links_share_no_node(chemin):
    c = _champ(chemin)
    couleurs = colorier_liens(c.dico["m1"], c.dico["m2"], c.dico["masque_muscles"])[0].tolist()
    for col in set(couleurs) - {-1}:
        noeuds = [n for m, cm in enumerate(couleurs) if cm == col for n in (c.muscle1[m], c.muscle2[m])]
        assert len(noeuds) == len(set(noeuds))


def test_rest_does_not_drift():
    torch.manual_seed(0)
    m = MegaCreaXPBD(_champ(REFERENCE).dico, 4, device="cpu")
    x0 = _barycentre(m).clone()
    for _ in range(100 * 20):
        m.apply_physics(torch.tensor(0.05))
    assert (_barycentre(m) - x0).abs().max() < 1e-3


def test_momentum_conserved_without_water():
    """Exact up to rounding: run in float64 (float32 rounding alone gives ~1e-4 relative)."""
    c = _champ(REFERENCE)
    defaut = torch.get_default_dtype()
    torch.set_default_dtype(torch.float64)
    try:
        dico = {k: (v.double() if v.is_floating_point() else v) for k, v in c.dico.items()}
        torch.manual_seed(1)
        m = MegaCreaXPBD(dico, 4, device="cpu")
        m.water_drag = 0.0
        for f in range(60):
            if f % 5 == 0:
                m.apply_action(torch.rand(1, 4, c.action_size) * 2 - 1, f)
            for _ in range(20):
                m.apply_physics(torch.tensor(0.05))
        p = (m.vX * m.mask_N_exp).sum(2)
        assert p.abs().max() < 1e-8
    finally:
        torch.set_default_dtype(defaut)


def _rameur_ouvert(sub, frames=300, **kw):
    d, muscles = construire_rameur()
    ch = _champ(RAMEUR)
    torch.manual_seed(0)
    m = MegaCreaXPBD(ch.dico, 2, device="cpu", **kw)
    x0 = _barycentre(m).clone()
    for f in range(frames):
        if f % 5 == 0:
            a = commande([(0.7, 1.0, 40, -math.pi / 2)], muscles, ch.action_size, f // 5).expand(2, -1)
            m.apply_action(a.unsqueeze(0).clone(), f)
        for _ in range(sub):
            m.apply_physics(torch.tensor(1.0 / sub))
    return (_barycentre(m) - x0).mean().item(), m


def test_converges_when_the_substep_shrinks():
    d40, _ = _rameur_ouvert(40, frames=200)
    d80, _ = _rameur_ouvert(80, frames=200)
    assert abs(d40 - d80) / abs(d80) < 0.05


def test_stiff_links_and_rigid_bones_stay_stable():
    d, m = _rameur_ouvert(20, frames=100, os_rigides=True, facteur_raideur=100.0)
    assert math.isfinite(d) and torch.isfinite(m.X).all()
    x1, y1, x2, y2 = m._paires(m.X, m.Y)
    longueur = torch.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
    ecart = ((longueur - m.target_length).abs() / m.target_length.clamp(min=1.0)) * m.is_bone_exp
    assert ecart.max() < 0.01                              # os : longueur tenue à 1 % près


def test_gradient_flows_for_train2():
    c = _champ(RAMEUR)
    torch.manual_seed(2)
    m = MegaCreaXPBD(c.dico, 4, device="cpu")
    cerveau = Brain(c.obs_size, c.action_size)
    for f in range(20):
        if f % 5 == 0:
            m.apply_action(cerveau(m.get_observation(f)), f)
        for _ in range(20):
            m.apply_physics(torch.tensor(0.05))
    (-m.get_reward(1e4, 0.0).sum()).backward()
    g = torch.sqrt(sum((p.grad ** 2).sum() for p in cerveau.parameters()))
    assert torch.isfinite(g) and g > 0


def test_swimenv_and_ppo_run_with_physics_v2(tmp_path):
    c = _champ(RAMEUR)
    env = SwimEnv(c.dico, 2, 20, 20, 1e4, 0.0, DEV, physique="v2", options_v2={"os_rigides": True})
    obs = env.reset()
    done = False
    while not done:
        obs, _, done = env.step(torch.zeros(2, c.action_size))
    assert isinstance(env.mega, MegaCreaXPBD)
    res = subprocess.run(
        [sys.executable, "-m", "baselines.ppo", "--chemin-champion", str(RAMEUR), "--poids-aleatoires",
         "--physique", "v2", "--os-rigides", "--ppo-nb-iterations", "2", "--ppo-n-envs", "8",
         "--ppo-minibatches", "2", "--ppo-eval-n-envs", "4", "--train2-frame-nb", "30",
         "--dossier-runs", str(tmp_path), "--dossier-baselines", str(tmp_path)],
        cwd=RACINE, capture_output=True, text=True, timeout=300)
    assert res.returncode == 0, res.stderr[-2000:]
