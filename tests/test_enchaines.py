"""E20 — chained episodes: an episode resumes from the final state of the previous one.

Checks: (1) chaining two chunks is the same physics as one long episode (re-centring is a
pure translation); (2) a new chunk keeps the body's shape and the observation clock;
(3) an exploded rollout is reset to rest without touching the others; (4) train2 and PPO
run end to end with the option.
"""

import subprocess
import sys
from pathlib import Path

import pytest
import torch

from baselines.env import SwimEnv, barycentre, recentrer, reinitialiser_explosions
from brain import Brain
from champion import charger_champion

RACINE = Path(__file__).resolve().parent.parent
REFERENCE = RACINE / "champion_raffine" / "reference.pt"
DEVICE = torch.device("cpu")

pytestmark = pytest.mark.skipif(not REFERENCE.exists(), reason="reference champion not available")


def _env(n, frames):
    champ = charger_champion(str(REFERENCE), DEVICE)
    cerveau = Brain(champ.obs_size, champ.action_size)
    cerveau.load_state_dict(champ.brain_weights)
    return champ, cerveau, SwimEnv(champ.dico, n, frames, 20, 10000.0, 0.0, DEVICE)


@torch.no_grad()
def test_two_chained_chunks_equal_one_long_episode():
    champ, cerveau, court = _env(4, 50)
    _, _, long_ = _env(4, 100)

    torch.manual_seed(0)
    obs = court.reset()
    total_court, done = 0.0, False
    while not done:
        obs, _, done = court.step(cerveau(obs))
    total_court = court.deplacement().clone()
    obs = court.nouvelle_tranche()
    done = False
    while not done:
        obs, _, done = court.step(cerveau(obs))
    total_court = total_court + court.deplacement()

    torch.manual_seed(0)
    obs = long_.reset()
    done = False
    while not done:
        obs, _, done = long_.step(cerveau(obs))

    assert torch.allclose(total_court, long_.deplacement(), atol=1e-3)


@torch.no_grad()
def test_new_chunk_keeps_shape_and_clock():
    _, cerveau, env = _env(3, 20)
    torch.manual_seed(1)
    obs = env.reset()
    done = False
    while not done:
        obs, _, done = env.step(cerveau(obs))
    forme_avant = env.mega.X - barycentre(env.mega)[0].unsqueeze(2)
    vitesses_avant = env.mega.vX.clone()
    env.nouvelle_tranche()
    m = env.mega.mask_N_exp
    assert torch.allclose(env.mega.X * m, forme_avant * m, atol=1e-4)    # même forme, recentrée
    assert torch.allclose(env.mega.vX, vitesses_avant)                    # mêmes vitesses
    assert torch.allclose(barycentre(env.mega)[0], torch.zeros(1, 3), atol=1e-4)
    assert env.frame == 0 and env.frame_globale == 20                     # l'horloge continue
    assert torch.allclose(env.deplacement(), torch.zeros(3), atol=1e-5)


@torch.no_grad()
def test_recentring_keeps_next_reward_exact():
    _, cerveau, env = _env(2, 10)
    torch.manual_seed(2)
    obs = env.reset()
    env.step(cerveau(obs))
    p_avant = env.mega.previous_distance - barycentre(env.mega)[0]
    recentrer(env.mega)
    p_apres = env.mega.previous_distance - barycentre(env.mega)[0]
    assert torch.allclose(p_avant, p_apres, atol=1e-4)


@torch.no_grad()
def test_exploded_rollout_is_reset_alone():
    _, cerveau, env = _env(3, 10)
    torch.manual_seed(3)
    obs = env.reset()
    env.step(cerveau(obs))
    sain_avant = env.mega.X[:, 0].clone()
    env.mega.X[0, 1, 0] = float("nan")
    casses = reinitialiser_explosions(env.mega, env._neuf)
    assert casses.tolist() == [False, True, False]
    assert torch.isfinite(env.mega.X).all()
    assert torch.equal(env.mega.X[:, 0], sain_avant)


def test_env_without_chaining_is_unchanged():
    """Default (reset every episode, no re-init): frame_globale == frame, as before E20."""
    _, cerveau, env = _env(2, 15)
    obs = env.reset()
    done = False
    while not done:
        obs, _, done = env.step(cerveau(obs))
    assert env.frame == env.frame_globale == 15
    obs = env.reset()
    assert env.frame == env.frame_globale == 0


@pytest.mark.parametrize("script", ["train2", "ppo"])
def test_chained_training_runs(tmp_path, script):
    commun = ["--chemin-champion", str(REFERENCE), "--poids-aleatoires", "--episodes-enchaines",
              "--frames-selection", "40", "--ppo-eval-n-envs", "4", "--ppo-eval-tous-les", "1",
              "--train2-frame-nb", "30", "--dossier-runs", str(tmp_path)]
    if script == "train2":
        cmd = [sys.executable, "train2.py", *commun, "--maj-par-fenetre", "--nb-episodes", "3",
               "--train2-batch-size", "4", "--dossier-champion-raffine", str(tmp_path)]
    else:
        cmd = [sys.executable, "-m", "baselines.ppo", *commun, "--ppo-nb-iterations", "3",
               "--ppo-n-envs", "8", "--ppo-minibatches", "2", "--dossier-baselines", str(tmp_path)]
    res = subprocess.run(cmd, cwd=RACINE, capture_output=True, text=True, timeout=300)
    assert res.returncode == 0, res.stderr[-2000:]
    logs = list(tmp_path.glob("*.jsonl"))
    assert logs and '"episodes_enchaines": true' in logs[0].read_text()
    assert '"eval_score_mean": ' in logs[0].read_text()


def test_train2_refuses_chaining_without_per_window_updates(tmp_path):
    res = subprocess.run(
        [sys.executable, "train2.py", "--chemin-champion", str(REFERENCE), "--episodes-enchaines",
         "--nb-episodes", "1", "--train2-batch-size", "2", "--dossier-runs", str(tmp_path),
         "--dossier-champion-raffine", str(tmp_path)],
        cwd=RACINE, capture_output=True, text=True, timeout=120)
    assert res.returncode != 0 and "maj-par-fenetre" in res.stderr


@torch.no_grad()
def test_return_to_rest_probability():
    """E20b: p = 0 keeps every rollout, p = 1 sends every rollout back to the rest pose."""
    _, cerveau, env = _env(4, 20)
    torch.manual_seed(4)
    obs = env.reset()
    done = False
    while not done:
        obs, _, done = env.step(cerveau(obs))
    vx = env.mega.vX.clone()
    env.nouvelle_tranche(0.0)
    assert not env.retours.any() and torch.equal(env.mega.vX, vx)
    env.nouvelle_tranche(1.0)
    assert env.retours.all()
    assert torch.count_nonzero(env.mega.vX) == 0                       # repos : vitesses nulles
    assert torch.allclose(env.mega.target_length, env.mega.base_length)
