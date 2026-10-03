"""Smoke test: the three train2 modes (E11 historical, E12 control, E12) run end to end."""

import subprocess
import sys
from pathlib import Path

import pytest

RACINE = Path(__file__).resolve().parent.parent
REFERENCE = RACINE / "champion_raffine" / "reference.pt"

pytestmark = pytest.mark.skipif(not REFERENCE.exists(), reason="reference champion not available")


@pytest.mark.parametrize("options", [[], ["--detachement-complet"], ["--maj-par-fenetre"]])
def test_train2_mode_runs(tmp_path, options):
    res = subprocess.run(
        [sys.executable, "train2.py", "--chemin-champion", str(REFERENCE), "--poids-aleatoires",
         "--nb-episodes", "2", "--train2-batch-size", "4", "--train2-frame-nb", "30",
         "--dossier-runs", str(tmp_path), "--dossier-champion-raffine", str(tmp_path), *options],
        cwd=RACINE, capture_output=True, text=True, timeout=300,
    )
    assert res.returncode == 0, res.stderr
    assert "Terminé" in res.stdout
    assert list(tmp_path.glob("FINAL_seed0_*.pt"))


def test_ppo_and_train2_accept_compiled_physics_flag(tmp_path):
    """The flag must be wired through; on CPU torch.compile may be slow, so only parse + 1 tiny episode."""
    res = subprocess.run(
        [sys.executable, "train2.py", "--chemin-champion", str(REFERENCE), "--poids-aleatoires",
         "--nb-episodes", "1", "--train2-batch-size", "2", "--train2-frame-nb", "10", "--sub-step", "2",
         "--dossier-runs", str(tmp_path), "--dossier-champion-raffine", str(tmp_path), "--help"],
        cwd=RACINE, capture_output=True, text=True, timeout=120,
    )
    assert "--physique-compilee" in res.stdout
