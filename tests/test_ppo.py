"""Smoke test: PPO runs end to end and writes a checkpoint loadable as a Brain."""

import dataclasses
from pathlib import Path

import pytest
import torch

from baselines.ppo import entrainer
from brain import Brain
from champion import charger_champion
from config import Config

REFERENCE = Path(__file__).resolve().parent.parent / "champion_raffine" / "reference.pt"

pytestmark = pytest.mark.skipif(not REFERENCE.exists(), reason="reference champion not available")


def test_ppo_smoke_run_produces_brain_checkpoint(tmp_path):
    cfg = dataclasses.replace(
        Config(), chemin_champion=str(REFERENCE), poids_aleatoires=True,
        dossier_runs=str(tmp_path / "runs"), dossier_baselines=str(tmp_path / "out"),
        train2_frame_nb=20, ppo_n_envs=8, ppo_nb_iterations=2, ppo_minibatches=2,
        ppo_epochs=1, ppo_eval_n_envs=4, ppo_eval_tous_les=1,
    )
    meilleur, chemin = entrainer(cfg)

    assert torch.isfinite(torch.tensor(meilleur))
    assert Path(chemin).exists()
    champ = charger_champion(chemin, torch.device("cpu"))
    Brain(champ.obs_size, champ.action_size).load_state_dict(champ.brain_weights)
    assert list((tmp_path / "runs").glob("ppo_seed0_*.jsonl"))
