"""E21 — penalty β·mean(z²) on the pre-tanh outputs of train2."""

import json
import subprocess
import sys
from pathlib import Path

import pytest
import torch

from brain import Brain

RACINE = Path(__file__).resolve().parent.parent
REFERENCE = RACINE / "champion_raffine" / "reference.pt"


def test_pre_activation_is_the_input_of_tanh():
    torch.manual_seed(0)
    b = Brain(10, 4)
    x = torch.randn(3, 10)
    assert torch.equal(torch.tanh(b.pre_activation(x)), b(x))


def _lancer(tmp_path, beta, par_fenetre):
    cmd = [sys.executable, "train2.py", "--chemin-champion", str(REFERENCE), "--poids-aleatoires",
           "--nb-episodes", "6", "--train2-batch-size", "4", "--train2-frame-nb", "30", "--learning-rate", "0.01",
           "--coef-preactivation", str(beta), "--dossier-runs", str(tmp_path),
           "--dossier-champion-raffine", str(tmp_path)]
    if par_fenetre:
        cmd.append("--maj-par-fenetre")
    res = subprocess.run(cmd, cwd=RACINE, capture_output=True, text=True, timeout=300)
    assert res.returncode == 0, res.stderr[-2000:]
    lignes = [json.loads(ligne) for ligne in next(tmp_path.glob("*.jsonl")).read_text().splitlines()]
    assert any(ligne.get("coef_preactivation") == beta for ligne in lignes
               if isinstance(ligne, dict)) or f'"coef_preactivation": {beta}' in json.dumps(lignes)
    return [ligne for ligne in lignes if "z_abs_moyen" in ligne]


@pytest.mark.skipif(not REFERENCE.exists(), reason="champions not available")
@pytest.mark.parametrize("par_fenetre", [True, False])
def test_penalty_logs_saturation_and_shrinks_pre_activations(tmp_path, par_fenetre):
    sans = _lancer(tmp_path / "b0", 0.0, par_fenetre)
    avec = _lancer(tmp_path / "b10", 10.0, par_fenetre)
    assert len(sans) == len(avec) == 6
    assert all(0.0 <= ligne["saturation"] <= 1.0 for ligne in sans + avec)
    assert avec[-1]["z_abs_moyen"] < 0.5 * sans[-1]["z_abs_moyen"]
