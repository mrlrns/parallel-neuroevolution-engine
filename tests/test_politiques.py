"""E22 — symmetry and rhythmic policies (politiques.py, baselines/ppo.py options)."""

import subprocess
import sys
from pathlib import Path

import pytest
import torch

from baselines.ppo import Agent
from brain import Brain
from champion import charger_champion
from politiques import (
    BrainSymetrique,
    PolitiqueRythmique,
    Symetriseur,
    ajouter_phase,
    groupes_symetrie,
    politique_depuis_checkpoint,
)

RACINE = Path(__file__).resolve().parent.parent
REFERENCE = RACINE / "champion_raffine" / "reference.pt"
RAMEUR = RACINE / "champion_raffine" / "rameur.pt"
DEV = torch.device("cpu")

pytestmark = pytest.mark.skipif(not (REFERENCE.exists() and RAMEUR.exists()), reason="champions not available")


def _champ(chemin):
    return charger_champion(str(chemin), DEV)


def test_mirror_groups_found_by_geometry():
    r = _champ(RAMEUR)
    assert groupes_symetrie(r.x, r.y, r.muscle1, r.muscle2, r.is_bone) == [[7, 11], [8, 12]]
    ref = _champ(REFERENCE)
    groupes = groupes_symetrie(ref.x, ref.y, ref.muscle1, ref.muscle2, ref.is_bone)
    assert sorted(k for g in groupes for k in g) == [k for k, b in enumerate(ref.is_bone) if not b]


def test_asymmetric_body_is_rejected():
    r = _champ(RAMEUR)
    y = list(r.y)
    y[6] += 5.0                                   # casse la symétrie d'un nœud
    with pytest.raises(ValueError):
        groupes_symetrie(r.x, y, r.muscle1, r.muscle2, r.is_bone)


def test_symmetric_brain_saves_as_an_exact_plain_brain():
    c = _champ(REFERENCE)
    sym = Symetriseur(groupes_symetrie(c.x, c.y, c.muscle1, c.muscle2, c.is_bone), c.action_size)
    torch.manual_seed(0)
    bs = BrainSymetrique(c.obs_size, c.action_size, sym)
    plain = Brain(c.obs_size, c.action_size)
    plain.load_state_dict(bs.poids_brain())
    obs = torch.randn(5, c.obs_size)
    attendu = sym.etendre(bs(obs))
    m = c.masque_muscles_actifs.bool()
    assert torch.allclose(plain(obs)[:, m], attendu[:, m], atol=1e-6)
    for g in sym.groupes:                         # membres d'un groupe : même commande
        assert torch.allclose(plain(obs)[:, g[0]], plain(obs)[:, g[-1]], atol=1e-6)


def test_rhythmic_policy_is_periodic_and_ignores_state():
    torch.manual_seed(1)
    pol = PolitiqueRythmique(3, periode=40)
    o1, o2 = torch.randn(4, 10), torch.randn(4, 10)
    a0 = pol(ajouter_phase(o1, 15, 40))
    assert torch.allclose(a0, pol(ajouter_phase(o2, 15, 40)))          # l'état n'entre pas
    assert torch.allclose(a0, pol(ajouter_phase(o1, 55, 40)), atol=1e-5)  # période 40


@pytest.mark.parametrize("sym_on, periode", [(True, None), (False, 40), (True, 20)])
def test_checkpoint_roundtrip(tmp_path, sym_on, periode):
    c = _champ(RAMEUR)
    sym = Symetriseur(groupes_symetrie(c.x, c.y, c.muscle1, c.muscle2, c.is_bone), c.action_size) if sym_on else None
    torch.manual_seed(2)
    agent = Agent(c.obs_size, c.action_size, c.masque_muscles_actifs, 0.2, sym, periode)
    chemin = tmp_path / "ck.pt"
    torch.save(agent.checkpoint(c), chemin)
    f = politique_depuis_checkpoint(torch.load(chemin), charger_champion(str(chemin), DEV), DEV)
    obs = torch.randn(3, c.obs_size)
    with torch.no_grad():
        attendu = agent.vers_env(agent.acteur(agent.observer(obs, 25)))
    m = c.masque_muscles_actifs.bool()
    assert torch.allclose(f(obs, 25)[:, m], attendu[:, m], atol=1e-6)


@pytest.mark.parametrize("options", [["--symetrie"], ["--politique-rythmique", "--periode-rythme", "20"],
                                     ["--symetrie", "--politique-rythmique"]])
def test_ppo_runs_and_tools_read_the_checkpoint(tmp_path, options):
    res = subprocess.run(
        [sys.executable, "-m", "baselines.ppo", "--chemin-champion", str(RAMEUR), "--poids-aleatoires",
         "--ppo-nb-iterations", "2", "--ppo-n-envs", "8", "--ppo-minibatches", "2", "--ppo-eval-n-envs", "4",
         "--train2-frame-nb", "30", "--dossier-runs", str(tmp_path), "--dossier-baselines", str(tmp_path), *options],
        cwd=RACINE, capture_output=True, text=True, timeout=300)
    assert res.returncode == 0, res.stderr[-2000:]
    ck = tmp_path / "ppo_seed0_best.pt"
    res = subprocess.run([sys.executable, "-m", "tools.horizon_check", str(ck), "--frames", "60",
                          "--points", "30", "60", "--n", "2"], cwd=RACINE, capture_output=True, text=True, timeout=300)
    assert res.returncode == 0, res.stderr[-2000:]
