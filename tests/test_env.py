"""SwimEnv must simulate exactly the same physics as the train2.py loop.

If this fails, any baseline trained in SwimEnv is not comparable to train2.py.
"""

from pathlib import Path

import pytest
import torch

from baselines.check_env import comparer
from baselines.env import SwimEnv
from brain import Brain
from champion import charger_champion

REFERENCE = Path(__file__).resolve().parent.parent / "champion_raffine" / "reference.pt"
DEVICE = torch.device("cpu")

pytestmark = pytest.mark.skipif(not REFERENCE.exists(), reason="reference champion not available")


def _champion_et_cerveau():
    champ = charger_champion(str(REFERENCE), DEVICE)
    cerveau = Brain(champ.obs_size, champ.action_size)
    cerveau.load_state_dict(champ.brain_weights)
    return champ, cerveau


def test_swimenv_reproduces_train2_loop():
    champ, cerveau = _champion_et_cerveau()
    ecart_depl, ecart_score, _ = comparer(champ.dico, cerveau, n=8, frame_nb=50, sub_step=20,
                                          coef_e=10000.0, coef_h=0.0, bruit=0.02, seed=0,
                                          device=DEVICE)
    assert ecart_depl < 1e-3
    assert ecart_score < 1e-3


def test_swimenv_episode_length_and_shapes():
    champ, cerveau = _champion_et_cerveau()
    env = SwimEnv(champ.dico, 4, 50, 10, 10000.0, 0.0, DEVICE)
    obs = env.reset()
    assert obs.shape == (4, champ.obs_size)
    n_steps, done = 0, False
    while not done:
        obs, r, done = env.step(torch.zeros(4, champ.action_size))
        n_steps += 1
        assert r.shape == (4,)
    assert n_steps == env.nb_decisions == 10
    assert env.sim_steps_par_episode == 4 * 50 * 10
