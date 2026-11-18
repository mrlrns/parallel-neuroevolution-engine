"""Morphology rules (1 and 2') and anchor points along a bone (physics v2)."""

import math
from pathlib import Path

import pytest
import torch

from champion import charger_champion
from megaVecto_fast import classe_physique
from morphologie import ajouter_ancrage, pieces_rigides, scinder_ancrages, valider_dict, valider_morphologie
from physique_v2 import MegaCreaXPBD

RACINE = Path(__file__).resolve().parent.parent
REFERENCE = RACINE / "champion_raffine" / "reference.pt"
RAMEUR = RACINE / "champion_raffine" / "rameur.pt"
DEV = torch.device("cpu")
ACTIONNE = 2                                   # indice du muscle A–E de la créature en T


def creature_t():
    """Tronc A–C d'un seul os, membre M–E attaché au milieu (M ancré à s = 0.5), muscle A–E."""
    d = {"x": [0.0, 140.0], "y": [400.0, 400.0], "muscle1": [0], "muscle2": [1], "is_bone": [1.0]}
    m = ajouter_ancrage(d, 0, 0.5)
    d["x"].append(70.0)
    d["y"].append(330.0)
    for a, b, o in ((m, 3, 1.0), (0, 3, 0.0)):
        d["muscle1"].append(a)
        d["muscle2"].append(b)
        d["is_bone"].append(o)
    x, y = d["x"], d["y"]
    longueurs = [math.hypot(x[a] - x[b], y[a] - y[b]) for a, b in zip(d["muscle1"], d["muscle2"])]
    d.update(muscle1=torch.tensor(d["muscle1"]), muscle2=torch.tensor(d["muscle2"]),
             stiffness=torch.tensor([1.0 + 4 * o for o in d["is_bone"]]), target_length=torch.tensor(longueurs),
             brain_weights=None, max_noeuds=20, max_muscles=30)
    return d


@pytest.fixture
def chemin_t(tmp_path):
    chemin = tmp_path / "creature_t.pt"
    torch.save(creature_t(), chemin)
    return str(chemin)


# --- règles -------------------------------------------------------------------------------

@pytest.mark.skipif(not RAMEUR.exists(), reason="champions not available")
def test_rower_is_valid_and_its_trunk_is_one_rigid_piece():
    d = torch.load(RAMEUR, map_location="cpu")
    assert valider_dict(d) == []
    m1, m2 = d["muscle1"].tolist(), d["muscle2"].tolist()
    pieces = pieces_rigides(m1, m2, d["is_bone"])
    assert len({pieces[m] for m in range(5)}) == 1             # deux triangles d'os soudés
    assert len(set(pieces.values())) == 5                      # tronc + 2 bras + 2 pales


@pytest.mark.skipif(not REFERENCE.exists(), reason="champions not available")
def test_reference_breaks_rules_1_and_2prime():
    erreurs = valider_dict(torch.load(REFERENCE, map_location="cpu"))
    assert sorted(e.split(" :")[0] for e in erreurs) == [
        "règle 1 — nœud 0", "règle 1 — nœud 5", "règle 1 — nœud 7", "règle 2' — nœud 3"]


def test_three_limbs_on_one_node_are_refused():
    # étoile : trois os indépendants sur le nœud 0
    assert any("règle 2'" in e for e in valider_morphologie([0, 0, 0], [1, 2, 3], [1.0] * 3, 4))
    # nœud relié seulement par un muscle
    assert any("règle 1" in e for e in valider_morphologie([0, 1], [1, 2], [1.0, 0.0], 3))


def test_anchor_rules():
    assert valider_dict(creature_t()) == []
    d = creature_t()
    d["ancrages"] = [[2, 2, 0.5]]                                      # ancré sur un muscle
    assert any("pas un os" in e for e in valider_dict(d))
    d["ancrages"] = [[2, 0, 1.0]]                                      # s hors de ]0, 1[
    assert any("hors de" in e for e in valider_dict(d))


def test_split_anchor_keeps_muscle_indices_and_breaks_rule_2prime():
    d = scinder_ancrages(creature_t())
    assert d["muscle1"].tolist() == [0, 2, 0, 2] and d["muscle2"].tolist() == [2, 3, 3, 1]
    assert d["is_bone"] == [1.0, 1.0, 0.0, 1.0]                        # muscle A–E toujours à l'indice 2
    assert torch.allclose(d["target_length"][[0, 3]], torch.tensor([70.0, 70.0]))
    assert any("règle 2' — nœud 2" in e for e in valider_dict(d))      # M devient une charnière à 3 pièces


def test_loader_splits_anchors_unless_asked(chemin_t):
    assert "ancrages" not in charger_champion(chemin_t, DEV).dico
    assert charger_champion(chemin_t, DEV).num_muscles == 4
    ch = charger_champion(chemin_t, DEV, ancrages=True)
    assert ch.dico["ancrages"] == [[(2, 0, 0.5)]] and ch.num_muscles == 3
    assert ch.to_save_dict(None)["ancrages"] == [[2, 0, 0.5]]


def test_anchors_need_the_option_and_physics_v2(chemin_t):
    ch = charger_champion(chemin_t, DEV, ancrages=True)
    with pytest.raises(ValueError, match="ancrage"):
        MegaCreaXPBD(ch.dico, 2, device="cpu")
    with pytest.raises(ValueError, match="v2"):
        classe_physique(False, "v1", ancrages=True)


def test_strict_morphology_refuses_invalid_creatures():
    if not REFERENCE.exists():
        pytest.skip("champions not available")
    with pytest.raises(ValueError, match="règle 1"):
        MegaCreaXPBD(charger_champion(str(REFERENCE), DEV).dico, 2, device="cpu", morphologie_stricte=True)


# --- physique ---------------------------------------------------------------------------

def _actionner(chemin, ancrages, frames=200, sub=10, dtype=None):
    ch = charger_champion(chemin, DEV, ancrages=ancrages)
    dico = ch.dico if dtype is None else {k: (v.to(dtype) if torch.is_tensor(v) and v.is_floating_point() else v)
                                          for k, v in ch.dico.items()}
    torch.manual_seed(0)
    m = MegaCreaXPBD(dico, 2, device="cpu", ancrages=ancrages)
    pli, angles = 0.0, []
    for f in range(frames):
        if f % 5 == 0:
            a = torch.zeros(1, 2, ch.action_size, dtype=m.X.dtype)
            a[..., ACTIONNE] = math.sin(2 * math.pi * f / 40)
            m.apply_action(a, f)
        for _ in range(sub):
            m.apply_physics(torch.tensor(1.0 / sub))
        X, Y = m.X[0, 0].tolist(), m.Y[0, 0].tolist()
        u = math.atan2(Y[0] - Y[2], X[0] - X[2])                       # M → A
        v = math.atan2(Y[1] - Y[2], X[1] - X[2])                       # M → C
        pli = max(pli, math.degrees(abs(((u - v) % (2 * math.pi)) - math.pi)))
        angles.append(math.degrees(math.atan2(Y[3] - Y[2], X[3] - X[2]) - math.atan2(Y[1] - Y[0], X[1] - X[0])))
    return pli, max(angles) - min(angles), m


def test_anchored_trunk_stays_straight_while_split_trunk_folds(chemin_t):
    pli, amplitude, m = _actionner(chemin_t, ancrages=True)
    assert pli < 1e-3 and amplitude > 30 and torch.isfinite(m.X).all()   # le membre bouge, le tronc non
    X, Y = m.X[0], m.Y[0]
    assert torch.allclose(X[:, 2], 0.5 * (X[:, 0] + X[:, 1])) and torch.allclose(Y[:, 2], 0.5 * (Y[:, 0] + Y[:, 1]))
    pli_scinde, _, _ = _actionner(chemin_t, ancrages=False)
    assert pli_scinde > 20                                              # charnière libre en M


def test_momentum_conserved_with_anchors(chemin_t):
    defaut = torch.get_default_dtype()
    torch.set_default_dtype(torch.float64)
    try:
        ch = charger_champion(chemin_t, DEV, ancrages=True)
        dico = {k: (v.double() if torch.is_tensor(v) and v.is_floating_point() else v) for k, v in ch.dico.items()}
        torch.manual_seed(1)
        m = MegaCreaXPBD(dico, 4, device="cpu", ancrages=True)
        m.water_drag = 0.0
        for f in range(60):
            if f % 5 == 0:
                m.apply_action(torch.rand(1, 4, ch.action_size) * 2 - 1, f)
            for _ in range(20):
                m.apply_physics(torch.tensor(0.05))
        reel = (m._w > 0).double()                                     # nœuds ancrés : sans masse propre
        p = (m.vX * m.masses * reel).sum(2), (m.vY * m.masses * reel).sum(2)
        assert p[0].abs().max() < 1e-8 and p[1].abs().max() < 1e-8
        assert m.vX.abs().max() > 0.1                                   # il s'est bien passé quelque chose
    finally:
        torch.set_default_dtype(defaut)


def test_gradient_flows_with_anchors(chemin_t):
    ch = charger_champion(chemin_t, DEV, ancrages=True)
    m = MegaCreaXPBD(ch.dico, 2, device="cpu", ancrages=True)
    a = torch.full((1, 2, ch.action_size), 0.3, requires_grad=True)
    for f in range(20):
        if f % 5 == 0:
            m.apply_action(a, f)
        for _ in range(10):
            m.apply_physics(torch.tensor(0.1))
    (-m.get_reward(1e4, 0.0).sum()).backward()
    assert torch.isfinite(a.grad).all() and a.grad[..., ACTIONNE].abs().sum() > 0


def test_compiled_matches_eager_with_anchors(chemin_t):
    ch = charger_champion(chemin_t, DEV, ancrages=True)
    etats = []
    for compile_ in (False, True):
        torch.manual_seed(3)
        m = MegaCreaXPBD(ch.dico, 3, device="cpu", compile=compile_, ancrages=True)
        for f in range(10):
            if f % 5 == 0:
                m.apply_action(torch.full((1, 3, ch.action_size), 0.5), f)
            for _ in range(10):
                m.apply_physics(torch.tensor(0.1))
        etats.append(m.X.clone())
    assert torch.allclose(etats[0], etats[1], atol=1e-4)
