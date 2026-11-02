"""
tools/rameur.py — morphologie conçue à la main pour la nage dans CETTE physique (E19).

Principes tirés du modèle de traînée de megaVecto (traînée par lien, os ×1, muscles
×0.3, tangentielle = 0.3 × normale ; liens reliés seulement aux nœuds) :
  - la poussée vient des OS balayés de face (traînée normale ×1) ;
  - le retour doit se faire de profil (traînée tangentielle ×0.3) : il faut donc une
    articulation qui replie la pale, sinon le mouvement est réciproque et n'avance pas ;
  - une articulation = un os en charnière + un muscle qui fixe l'angle (le muscle est un
    ressort à longueur cible : il pousse et tire, un seul suffit par degré de liberté) ;
  - corps rigide (deux triangles d'os) et membres symétriques haut/bas pour ne pas
    tourner sur soi-même.

        Eu───Tu          A : avant du corps, C : arrière
        │   (pale)       Su/Sd : épaules, Eu/Ed : coudes, Tu/Td : bouts des pales
    A ──Su──────── C     muscles : A–E (épaule, angle du bras)
      ╲ │       ╱                  S–T (poignet, repli de la pale)
        Sd
        │
        Ed───Td

8 nœuds, 13 liens (4 muscles) : du même ordre que la référence (9 nœuds, 12 liens).
Le .pt sauvegardé a le format champion (brain_weights=None) et passe dans baselines/ppo.py.

Usage :
    python -m tools.rameur                    # construit champion_raffine/rameur.pt + balayage
    python -m tools.rameur --gif docs/rameur_open_loop.gif
"""

import argparse
import itertools
import math

import torch

from baselines.env import SwimEnv
from champion import charger_champion


def construire_rameur(bras=60.0, pale=60.0, longueur_corps=140.0, epaisseur=20.0, repli=25.0,
                      max_noeuds=20, max_muscles=30):
    cx = 100.0
    profondeur = math.sqrt(pale ** 2 - repli ** 2)
    P = {
        "A": (cx - 60, 400.0), "C": (cx - 60 + longueur_corps, 400.0),
        "Su": (cx, 400 - epaisseur / 2), "Sd": (cx, 400 + epaisseur / 2),
        "Eu": (cx, 400 - epaisseur / 2 - bras), "Ed": (cx, 400 + epaisseur / 2 + bras),
        "Tu": (cx + repli, 400 - epaisseur / 2 - bras - profondeur),
        "Td": (cx + repli, 400 + epaisseur / 2 + bras + profondeur),
    }
    noms = list(P)
    idx = {n: i for i, n in enumerate(noms)}
    m1, m2, os_, muscles = [], [], [], {}

    def lien(a, b, est_os):
        m1.append(idx[a])
        m2.append(idx[b])
        os_.append(est_os)
        return len(m1) - 1

    for a, b in (("A", "Su"), ("A", "Sd"), ("Su", "Sd"), ("Su", "C"), ("Sd", "C")):
        lien(a, b, 1.0)
    for s in "ud":
        lien("S" + s, "E" + s, 1.0)
        lien("E" + s, "T" + s, 1.0)
        muscles["epaule_" + s] = lien("A", "E" + s, 0.0)
        muscles["poignet_" + s] = lien("S" + s, "T" + s, 0.0)

    x = [P[n][0] for n in noms]
    y = [P[n][1] for n in noms]
    base = [math.hypot(x[a] - x[b], y[a] - y[b]) for a, b in zip(m1, m2)]
    d = {
        "x": x, "y": y, "is_bone": os_,
        "muscle1": torch.tensor(m1, dtype=torch.long), "muscle2": torch.tensor(m2, dtype=torch.long),
        "stiffness": torch.tensor([1.0 + 4 * s for s in os_]), "target_length": torch.tensor(base),
        "brain_weights": None, "max_noeuds": max_noeuds, "max_muscles": max_muscles,
    }
    return d, muscles


def commande(combos, muscles, n_act, k):
    """Épaule A_e·sin(2πt/T), poignet A_p·sin(2πt/T + ψ), même commande haut et bas."""
    t = 5 * k
    a = torch.zeros(len(combos), n_act)
    for i, (ae, ap, T, psi) in enumerate(combos):
        for s in "ud":
            a[i, muscles["epaule_" + s]] = ae * math.sin(2 * math.pi * t / T)
            a[i, muscles["poignet_" + s]] = ap * math.sin(2 * math.pi * t / T + psi)
    return a


@torch.no_grad()
def balayer(champ, muscles, combos, frames, seed, enregistrer=False):
    torch.manual_seed(seed)
    env = SwimEnv(champ.dico, len(combos), frames, 20, 1e4, 0.0, torch.device("cpu"))
    env.reset()
    mi, images = {}, []
    for k in range(env.nb_decisions):
        env.step(commande(combos, muscles, champ.action_size, k))
        if env.frame == frames // 2:
            mi = env.deplacement().clone()
        if enregistrer:
            images.append((env.mega.X[0, 0].clone(), env.mega.Y[0, 0].clone()))
    return mi, env.deplacement().clone(), images


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--sauver", default="champion_raffine/rameur.pt")
    p.add_argument("--frames", type=int, default=600)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--gif", default=None, help="GIF de l'onde (0.7, 1.0, T=40, ψ=−π/2)")
    a = p.parse_args()

    d, muscles = construire_rameur()
    torch.save(d, a.sauver)
    champ = charger_champion(a.sauver, torch.device("cpu"))
    lc = max(champ.x) - min(champ.x)
    print(f"rameur : {champ.num_nodes} nœuds, {champ.num_muscles} liens, LC = {lc:.0f} → {a.sauver}\n")

    combos = [(0.0, 0.0, 40, 0.0)] + list(itertools.product(
        (0.4, 0.7, 1.0), (0.4, 0.7, 1.0), (20, 40, 80), (0.0, math.pi / 2, math.pi, -math.pi / 2)))
    mi, fin, _ = balayer(champ, muscles, combos, a.frames, a.seed)
    print(f"repos : {fin[0].item():.2f}")
    print(f"{'A_épaule':>8s} {'A_poignet':>9s} {'T':>4s} {'ψ':>6s} {a.frames // 2:>8d} {a.frames:>8d}   (déplacement)")
    for i in torch.argsort(fin, descending=True)[:6].tolist():
        ae, ap, T, psi = combos[i]
        print(f"{ae:8.1f} {ap:9.1f} {T:4d} {psi:6.2f} {mi[i].item():8.1f} {fin[i].item():8.1f}")
    i = torch.argmin(fin).item()
    print(f"le plus négatif : {combos[i]} → {fin[i].item():.1f} (changer ψ de signe inverse le sens : propulsion réelle)")

    if a.gif:
        from tools.eel_control import ecrire_gif
        _, _, images = balayer(champ, muscles, [(0.7, 1.0, 40, -math.pi / 2)], a.frames, a.seed, enregistrer=True)
        ecrire_gif(champ, images, a.gif, "rameur, boucle ouverte : épaule 0.7, poignet 1.0, T=40, ψ=−π/2")


if __name__ == "__main__":
    main()
