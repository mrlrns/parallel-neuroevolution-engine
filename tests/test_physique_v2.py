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


def test_compiled_v2_matches_eager():
    c = _champ(RAMEUR)
    etats = []
    for compile_ in (False, True):
        torch.manual_seed(3)
        m = MegaCreaXPBD(c.dico, 3, device="cpu", compile=compile_)
        for f in range(10):
            if f % 5 == 0:
                m.apply_action(torch.full((1, 3, c.action_size), 0.5), f)
            for _ in range(10):
                m.apply_physics(torch.tensor(0.1))
        etats.append(m.X.clone())
    assert torch.allclose(etats[0], etats[1], atol=1e-4)


# --- eau quadratique, unités physiques (étape 2) ----------------------------------------------

from megaVecto_fast import classe_physique  # noqa: E402
from physique_v2 import RHO_EAU  # noqa: E402


def _sans_bruit(m):
    """Remet la créature sur sa géométrie de base (MegaCrea ajoute un bruit de 2 mm)."""
    m.X = m.X_base.unsqueeze(1).repeat(1, m.X.shape[1], 1)
    m.Y = m.Y_base.unsqueeze(1).repeat(1, m.Y.shape[1], 1)
    x1, y1, x2, y2 = m._paires(m.X, m.Y)
    m.base_length = torch.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2 + 1e-8) * m.mask_M_exp
    m.target_length = m.base_length.clone()


def _sauver(tmp_path, x, y, m1, m2):
    d = {"x": x, "y": y, "muscle1": torch.tensor(m1), "muscle2": torch.tensor(m2), "is_bone": [1.0] * len(m1),
         "stiffness": torch.tensor([5.0] * len(m1)), "brain_weights": None, "max_noeuds": 6, "max_muscles": 6,
         "target_length": torch.tensor([math.hypot(x[a] - x[b], y[a] - y[b]) for a, b in zip(m1, m2)])}
    chemin = tmp_path / "c.pt"
    torch.save(d, chemin)
    return charger_champion(str(chemin), DEV)


def test_towed_plate_reaches_quadratic_terminal_velocity(tmp_path):
    """Plaque L × h tirée par F face au courant : v = √(2F / (ρ·C_n·L·h)). Schéma d'ordre 1 :
    l'erreur doit être petite et diminuer avec le sous-pas."""
    L, h, F = 60.0, 20.0, 5.0
    ch = _sauver(tmp_path, [0.0, L], [0.0, 0.0], [0], [1])
    erreurs = []
    for sub in (5, 20, 80):
        m = MegaCreaXPBD(ch.dico, 1, device="cpu", eau="quadratique", os_rigides=True, profondeur=h)
        _sans_bruit(m)
        assert torch.allclose(m.masses[0, 0, :2], torch.tensor([0.5 * RHO_EAU * h * L * 5.0] * 2))
        for _ in range(100 * sub):
            m.vY = m.vY + (1.0 / sub) * m._w * (F / 2) * m.mask_N_exp
            m.apply_physics(torch.tensor(1.0 / sub))
        v = m.vY[0, 0, :2].mean().item()
        erreurs.append(abs(v / math.sqrt(2 * F / (RHO_EAU * 2.0 * L * h)) - 1))
    assert erreurs[0] > erreurs[1] > erreurs[2] and erreurs[1] < 0.02 and erreurs[2] < 0.005, erreurs


def test_quadratic_drag_only_dissipates(tmp_path):
    """Structure toute en os rigides, sans muscle : l'énergie cinétique ne fait que décroître."""
    ch = _sauver(tmp_path, [0.0, 60.0, 30.0, 90.0], [0.0, 0.0, 30.0, 40.0], [0, 1, 2, 1], [1, 2, 0, 3])
    torch.manual_seed(0)
    m = MegaCreaXPBD(ch.dico, 64, device="cpu", eau="quadratique", os_rigides=True, iterations=20)
    m.vX = torch.randn_like(m.vX) * 20 * m.mask_N_exp
    m.vY = torch.randn_like(m.vY) * 20 * m.mask_N_exp

    def energie():
        return (0.5 * (m.vX ** 2 + m.vY ** 2) * m.masses * m.mask_N_exp).sum(2)

    e = e0 = energie()
    for _ in range(200):
        m.apply_physics(torch.tensor(1.0))
        e2 = energie()
        assert (e2 <= e * (1 + 1e-5) + 1e-9).all()
        e = e2
    assert (e < 0.05 * e0).all()                     # quadratique : v ∝ 1/t, décroissance lente


def test_rower_masses_and_wetted_links():
    ch = _champ(RAMEUR)
    m = MegaCreaXPBD(ch.dico, 1, device="cpu", eau="quadratique")
    x, y = ch.x, ch.y

    def aire(a, b, c):
        return 0.5 * abs((x[b] - x[a]) * (y[c] - y[a]) - (x[c] - x[a]) * (y[b] - y[a]))

    tronc = aire(0, 2, 3) + aire(1, 2, 3)                              # A-Su-Sd, C-Su-Sd
    membres = sum(math.hypot(x[a] - x[b], y[a] - y[b]) for a, b in zip(ch.muscle1, ch.muscle2)
                  if (a, b) in ((2, 4), (4, 6), (3, 5), (5, 7)))
    attendu = RHO_EAU * 20.0 * (tronc + 5.0 * membres)
    assert abs(m.masses[0, 0].sum().item() - attendu) < 1e-3
    mouille = (m._quad[4][0, 0] > 0).tolist()[:ch.num_muscles]
    assert mouille == [o == 1.0 and k != 2 for k, o in enumerate(ch.is_bone)]   # Su–Sd interne, muscles secs


def test_quadratic_water_needs_rule_1_and_physics_v2():
    with pytest.raises(ValueError, match="règle 1"):
        MegaCreaXPBD(_champ(REFERENCE).dico, 1, device="cpu", eau="quadratique")
    with pytest.raises(ValueError, match="v2"):
        classe_physique(False, "v1", eau="quadratique")


def _cap_et_distance(it, sub, frames=240, alt=True):
    _, muscles = construire_rameur()
    ch = _champ(RAMEUR)
    defaut = torch.get_default_dtype()
    torch.set_default_dtype(torch.float64)
    try:
        dico = {k: (v.double() if torch.is_tensor(v) and v.is_floating_point() else v) for k, v in ch.dico.items()}
        m = MegaCreaXPBD(dico, 1, device="cpu", eau="quadratique", iterations=it, ordre_alterne=alt)
        _sans_bruit(m)
        x0, pire = m.X[0, 0, :8].mean().item(), 0.0
        for f in range(frames):
            if f % 5 == 0:
                m.apply_action(commande([(1.0, 1.0, 40, -math.pi / 2)], muscles, ch.action_size, f // 5)
                               .unsqueeze(0).double(), f)
            for _ in range(sub):
                m.apply_physics(torch.tensor(1.0 / sub))
            X, Y = m.X[0, 0], m.Y[0, 0]
            pire = max(pire, abs(math.degrees(math.atan2(Y[1] - Y[0], X[1] - X[0]))))
        return pire, m.X[0, 0, :8].mean().item() - x0
    finally:
        torch.set_default_dtype(defaut)


def test_symmetric_rower_holds_its_course_when_the_solver_converges():
    """Sans bruit, le rameur est symétrique : il doit aller droit. Avec une seule passe de
    Gauss-Seidel le biais d'ordre le fait tourner ; 8 passes et l'ordre alterné suffisent."""
    cap, d10 = _cap_et_distance(8, 10)
    _, d20 = _cap_et_distance(8, 20)
    assert cap < 2.0 and d10 > 20
    assert abs(d10 - d20) / abs(d20) < 0.05


def test_compiled_matches_eager_in_quadratic_water():
    c = _champ(RAMEUR)
    etats = []
    for compile_ in (False, True):
        torch.manual_seed(3)
        m = MegaCreaXPBD(c.dico, 3, device="cpu", compile=compile_, eau="quadratique", iterations=2)
        for f in range(10):
            if f % 5 == 0:
                m.apply_action(torch.full((1, 3, c.action_size), 0.5), f)
            for _ in range(10):
                m.apply_physics(torch.tensor(0.1))
        etats.append(m.X.clone())
    assert torch.allclose(etats[0], etats[1], atol=1e-3)
