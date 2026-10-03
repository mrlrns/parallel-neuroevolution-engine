"""
Test de convergence numérique de la physique, SANS contrôleur appris.

Si la simulation intègre correctement la dynamique, le déplacement produit par
une commande FIXE doit être (quasi) indépendant du nombre de sous-pas par frame :
sub_step change dt = 1/sub_step mais pas le temps physique simulé par frame.

Trois commandes en boucle ouverte, appliquées toutes les 5 frames comme en
entraînement, sur la morphologie de référence :
  - repos      : a = 0                         (doit donner ~0 partout)
  - onde lente : a_i = 0.5·sin(2π t/40 + i·π/4) (nage "raisonnable")
  - tout-ou-rien : a_i = ±0.8, signe inversé à chaque décision
                   (le motif saccadé observé chez PPO)

Mêmes conditions initiales pour tous les sub_step (même seed, MegaCrea ne
dépend pas de dt). Sortie : déplacement X et Y moyens, vitesse max, part des
nœuds à l'écrêtage ±20.

Usage :
    python -m tools.convergence_test
    python -m tools.convergence_test --frames 1000 --sub-steps 10 20 40 80
"""

import argparse
import math

import torch

from champion import charger_champion
from megaVecto import MegaCrea


def commande(nom, k, n_act, masque):
    i = torch.arange(n_act, dtype=torch.float32)
    t = 5 * k
    if nom == "repos":
        a = torch.zeros(n_act)
    elif nom == "onde lente":
        a = 0.5 * torch.sin(2 * math.pi * t / 40 + i * math.pi / 4)
    elif nom == "tout-ou-rien":
        a = 0.8 * (-1.0) ** k * torch.where(i % 2 == 0, 1.0, -1.0)
    else:
        raise ValueError(nom)
    return a * masque


@torch.no_grad()
def simuler(champ, nom, sub_step, frames, n, seed):
    torch.manual_seed(seed)
    mega = MegaCrea(champ.dico, n, device="cpu")
    dt = torch.tensor(1.0 / sub_step)
    m = mega.mask_N_exp
    nb = m.sum(dim=2)
    x0 = (mega.X * m).sum(dim=2) / nb
    y0 = (mega.Y * m).sum(dim=2) / nb
    masque = champ.masque_muscles_actifs
    vmax, sat, pas = 0.0, 0.0, 0
    for frame in range(frames):
        if frame % 5 == 0:
            a = commande(nom, frame // 5, champ.action_size, masque)
            mega.apply_action(a.expand(1, n, -1).clone(), frame)
        for _ in range(sub_step):
            mega.apply_physics(dt)
        v = torch.maximum(mega.vX.abs(), mega.vY.abs())[m.bool()]
        vmax = max(vmax, v.max().item())
        sat += (v >= 19.99).float().mean().item()
        pas += 1
    x1 = (mega.X * m).sum(dim=2) / nb
    y1 = (mega.Y * m).sum(dim=2) / nb
    return (x1 - x0).mean().item(), (y1 - y0).mean().item(), vmax, sat / pas


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--chemin-champion", default="champion_raffine/reference.pt")
    p.add_argument("--frames", type=int, default=300)
    p.add_argument("--n", type=int, default=8)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--sub-steps", type=int, nargs="+", default=[5, 10, 20, 40, 80])
    a = p.parse_args()

    champ = charger_champion(a.chemin_champion, torch.device("cpu"))
    print(f"{a.chemin_champion} — {a.frames} frames, {a.n} rollouts, commandes en boucle ouverte\n")
    print(f"{'commande':14s} {'sub_step':>8s} {'dt':>7s} {'ΔX':>10s} {'ΔY':>9s} {'|v| max':>8s} {'% à ±20':>8s}")
    for nom in ("repos", "onde lente", "tout-ou-rien"):
        for s in a.sub_steps:
            dx, dy, vmax, sat = simuler(champ, nom, s, a.frames, a.n, a.seed)
            print(f"{nom:14s} {s:8d} {1 / s:7.3f} {dx:10.2f} {dy:9.2f} {vmax:8.2f} {100 * sat:7.2f}%")
        print()


if __name__ == "__main__":
    main()
