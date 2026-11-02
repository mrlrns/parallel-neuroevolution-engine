"""
tools/horizon_check.py — la nage tient-elle au-delà de l'horizon d'entraînement ?

La récompense ne paie que le déplacement net sur `train2_frame_nb` frames (300) :
rien n'oblige une politique à continuer de nager après. E18 a montré qu'avec une
pénalité de régularité forte (λ = 1) les politiques apprennent une seule déformation
lente du corps qui avance puis se fige vers la fin de l'horizon. Ce script rejoue
chaque checkpoint bien au-delà et affiche le déplacement cumulé à plusieurs instants,
plus le rapport vitesse(fin) / vitesse(début) : ≈ 1 = nage entretenue, ≈ 0 = arrêt.

Mêmes conditions que l'évaluation (SwimEnv, sub_step et physique de la config,
bruit d'action `bruit_test`).

Usage :
    python -m tools.horizon_check results/E15/ppo_seed0_best.pt results/E18/B/ppo/*.pt
    python -m tools.horizon_check ckpt.pt --frames 1500 --points 150 300 600 1000 1500 --n 16
"""

import argparse

import torch

from baselines.env import SwimEnv
from brain import Brain
from champion import charger_champion
from config import Config


@torch.no_grad()
def deplacements(chemin, frames, points, n, seed, cfg):
    ch = charger_champion(chemin, torch.device("cpu"))
    cerveau = Brain(ch.obs_size, ch.action_size)
    cerveau.load_state_dict(ch.brain_weights)
    cerveau.eval()
    torch.manual_seed(seed)
    env = SwimEnv(ch.dico, n, frames, cfg.sub_step, cfg.coef_energie, cfg.coef_hauteur,
                  torch.device("cpu"), v_max=cfg.v_max, water_drag=cfg.water_drag)
    obs = env.reset()
    res = {}
    done = False
    while not done:
        a = cerveau(obs)
        obs, _, done = env.step(a + torch.randn_like(a) * cfg.bruit_test)
        if env.frame in points:
            res[env.frame] = env.deplacement().mean().item()
    return res


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("checkpoints", nargs="+")
    p.add_argument("--frames", type=int, default=1500)
    p.add_argument("--points", type=int, nargs="+", default=[150, 300, 600, 1000, 1500])
    p.add_argument("--n", type=int, default=16, help="rollouts par checkpoint")
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args()
    assert all(x % 5 == 0 and x <= a.frames for x in a.points), "points : multiples de 5, ≤ --frames"
    cfg = Config()

    largeur = max(len(c) for c in a.checkpoints)
    tete = "  ".join(f"{x:>7d}" for x in a.points)
    print(f"{'checkpoint':{largeur}s}  {tete}   v_fin/v_début")
    for c in a.checkpoints:
        r = deplacements(c, a.frames, set(a.points), a.n, a.seed, cfg)
        pts = sorted(r)
        v_debut = r[pts[0]] / pts[0]
        v_fin = (r[pts[-1]] - r[pts[-2]]) / (pts[-1] - pts[-2])
        ratio = v_fin / v_debut if abs(v_debut) > 1e-6 else float("nan")
        print(f"{c:{largeur}s}  " + "  ".join(f"{r[x]:7.1f}" for x in pts) + f"   {ratio:6.2f}", flush=True)


if __name__ == "__main__":
    main()
