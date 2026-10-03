"""
tools/eel_control.py — morphologie TÉMOIN : une anguille en boucle ouverte (E17).

Question : la nage lente vient-elle de la physique (traînée) ou de la morphologie ?
On construit un nageur dont la stratégie est connue chez les vrais poissons
(onde progressive tête → queue) avec les MÊMES briques que les créatures
(nœuds, os, muscles) et on le simule dans la MÊME physique (SwimEnv).

Corps : une échelle de `nv` vertèbres
      o──o──o──o──o      ← muscles du haut (actifs)
      ║╲ ║╲ ║╲ ║╲ ║      ║ = os (barreaux), ╲ = os (empêche le cisaillement)
      o──o──o──o──o      ← muscles du bas (actifs, en opposition)
nv = 8 → 16 nœuds, 29 liens : tient dans max_noeuds=20 / max_muscles=30, donc
le .pt sauvegardé passe tel quel dans baselines/ppo.py.

Commande du segment k (décision toutes les 5 frames, comme en entraînement) :
    haut_k = A·sin(2π t/T + k·φ),  bas_k = −haut_k,   φ = ±2π/(w·nb_segments)
φ > 0 : l'onde descend vers les x décroissants → poussée vers +x.
Toutes les combinaisons (A, T, w) tournent en parallèle dans la dimension batch.

Sorties, pour chaque water_drag :
  - repos (A=0) : dérive, doit être ≈ 0
  - les 3 meilleures ondes + l'onde inversée de la meilleure (contrôle de signe :
    une vraie propulsion par onde change de signe quand l'onde s'inverse)
  - la meilleure onde de la PREMIÈRE traînée rejouée à cette traînée
    (cinématique de commande fixée, seule la traînée change)
  - optionnel `--controleur` : un checkpoint PPO/train2 rejoué à chaque traînée
Déplacements aussi exprimés en longueurs de corps (LC) pour comparer les formes.

Usage :
    python -m tools.eel_control
    python -m tools.eel_control --water-drags 0.005 0.02 0.05 \
        --controleur results/E16_lam1_long/ppo_seed0_best.pt --gif docs/eel.gif
"""

import argparse
import itertools
import math

import torch

from baselines.env import SwimEnv
from brain import Brain
from champion import charger_champion


def construire_anguille(nv=8, longueur=210.0, epaisseur=10.0, max_noeuds=20, max_muscles=30):
    """Dico au format des .pt de train.py (brain_weights=None) + indices haut/bas."""
    esp = longueur / (nv - 1)
    x = [50.0 + k * esp for k in range(nv)] * 2
    y = [400.0 - epaisseur / 2] * nv + [400.0 + epaisseur / 2] * nv
    m1, m2, os_, haut, bas = [], [], [], [], []

    def lien(a, b, est_os):
        m1.append(a)
        m2.append(b)
        os_.append(est_os)
        return len(m1) - 1

    for k in range(nv):                       # barreaux
        lien(k, nv + k, 1.0)
    for k in range(nv - 1):
        haut.append(lien(k, k + 1, 0.0))
        bas.append(lien(nv + k, nv + k + 1, 0.0))
        lien(k, nv + k + 1, 1.0)              # diagonale
    assert len(x) <= max_noeuds and len(m1) <= max_muscles, (len(x), len(m1))
    base = [math.hypot(x[a] - x[b], y[a] - y[b]) for a, b in zip(m1, m2)]
    d = {
        "x": x, "y": y, "is_bone": os_,
        "muscle1": torch.tensor(m1, dtype=torch.long),
        "muscle2": torch.tensor(m2, dtype=torch.long),
        "stiffness": torch.tensor([1.0 + 4 * s for s in os_]),
        "target_length": torch.tensor(base),
        "brain_weights": None, "max_noeuds": max_noeuds, "max_muscles": max_muscles,
    }
    return d, haut, bas


def longueur_corps(champ):
    return max(champ.x) - min(champ.x)


def commandes(combos, haut, bas, n_act, k):
    """[len(combos), n_act] pour la décision k (t = 5k frames)."""
    t = 5 * k
    a = torch.zeros(len(combos), n_act)
    ns = len(haut)
    s = torch.arange(ns, dtype=torch.float32)
    for i, (A, T, w) in enumerate(combos):
        phi = 2 * math.pi / (w * ns) if w != 0 else 0.0
        v = A * torch.sin(2 * math.pi * t / T + s * phi)
        a[i, haut] = v
        a[i, bas] = -v
    return a


@torch.no_grad()
def rouler_anguille(champ, combos, haut, bas, a, wd, seed, enregistrer=False):
    torch.manual_seed(seed)
    n = len(combos) * a.n
    env = SwimEnv(champ.dico, n, a.frames, a.sub_step, 1.0, 0.0, torch.device("cpu"), water_drag=wd)
    env.reset()
    images = []
    for k in range(env.nb_decisions):
        act = commandes(combos, haut, bas, champ.action_size, k).repeat_interleave(a.n, dim=0)
        env.step(act)
        if enregistrer:
            images.append((env.mega.X[0, 0].clone(), env.mega.Y[0, 0].clone()))
    d = env.deplacement().view(len(combos), a.n)
    return d.nan_to_num(nan=float("-inf")).mean(1), torch.isnan(env.deplacement()).sum().item(), images


@torch.no_grad()
def rouler_cerveau(champ, a, wd, seed, bruit=0.02):
    torch.manual_seed(seed)
    cerveau = Brain(champ.obs_size, champ.action_size)
    cerveau.load_state_dict(champ.brain_weights)
    env = SwimEnv(champ.dico, a.n, a.frames, a.sub_step, 1.0, 0.0, torch.device("cpu"), water_drag=wd)
    obs = env.reset()
    done = False
    while not done:
        act = cerveau(obs)
        obs, _, done = env.step(act + torch.randn_like(act) * bruit)
    return env.deplacement().mean().item()


def ecrire_gif(champ, images, chemin, titre):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation, PillowWriter

    m1, m2, os_ = champ.muscle1, champ.muscle2, champ.is_bone
    xs = torch.stack([im[0][:champ.num_nodes] for im in images]).numpy()
    ys = torch.stack([im[1][:champ.num_nodes] for im in images]).numpy()
    fig, ax = plt.subplots(figsize=(8, 3))
    ax.set_xlim(xs.min() - 30, xs.max() + 30)
    ax.set_ylim(ys.max() + 60, ys.min() - 60)    # y vers le bas comme pygame
    ax.set_aspect("equal")
    ax.set_title(titre, fontsize=9)
    lignes = [ax.plot([], [], color="black" if os_[i] else "tab:red", lw=2 if os_[i] else 1.5)[0]
              for i in range(len(m1))]

    def maj(f):
        for i, ln in enumerate(lignes):
            ln.set_data([xs[f, m1[i]], xs[f, m2[i]]], [ys[f, m1[i]], ys[f, m2[i]]])
        return lignes

    FuncAnimation(fig, maj, frames=len(images), blit=True).save(chemin, writer=PillowWriter(fps=12))
    plt.close(fig)
    print(f"🎞️  {chemin}")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--water-drags", type=float, nargs="+", default=[0.005, 0.02, 0.05])
    p.add_argument("--frames", type=int, default=300)
    p.add_argument("--sub-step", type=int, default=20)
    p.add_argument("--n", type=int, default=2, help="rollouts par onde (bruit initial différent)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--nv", type=int, default=8, help="nombre de vertèbres")
    p.add_argument("--epaisseur", type=float, default=10.0)
    p.add_argument("--reference", default="champion_raffine/reference.pt",
                   help="morphologie de référence, pour la longueur de corps")
    p.add_argument("--controleur", default=None,
                   help="checkpoint (.pt avec brain_weights) rejoué à chaque traînée")
    p.add_argument("--sauver", default="champion_raffine/anguille.pt",
                   help="où sauver l'anguille (format champion, passe dans baselines/ppo.py)")
    p.add_argument("--gif", default=None, help="GIF de la meilleure onde à la première traînée")
    a = p.parse_args()

    d, haut, bas = construire_anguille(a.nv, epaisseur=a.epaisseur)
    torch.save(d, a.sauver)
    champ = charger_champion(a.sauver, torch.device("cpu"))
    lc = longueur_corps(champ)
    lc_ref = longueur_corps(charger_champion(a.reference, torch.device("cpu")))
    print(f"anguille : {champ.num_nodes} nœuds, {champ.num_muscles} liens, LC = {lc:.0f}  → {a.sauver}")
    print(f"référence ({a.reference}) : LC = {lc_ref:.0f}")
    print(f"{a.frames} frames, sub_step {a.sub_step}, {a.n} rollouts par onde\n")

    grille = list(itertools.product((0.1, 0.2, 0.4, 0.7, 1.0), (20, 40, 80, 160), (1.0, 0.7, 2.0)))
    combos = [(0.0, 40, 1.0)] + grille + [(A, T, -w) for A, T, w in grille]
    meilleure_ref = None

    for wd in a.water_drags:
        dep, nb_nan, _ = rouler_anguille(champ, combos, haut, bas, a, wd, a.seed)
        print(f"=== water_drag = {wd}   (NaN : {nb_nan})")
        print(f"  repos                          ΔX = {dep[0].item():8.2f}")
        ordre = torch.argsort(dep, descending=True)
        for r, i in enumerate(ordre[:3].tolist()):
            A, T, w = combos[i]
            print(f"  #{r + 1} A={A:<4} T={T:<4} w={w:<5}       ΔX = {dep[i].item():8.2f}  ({dep[i].item() / lc:5.2f} LC)")
        A, T, w = combos[ordre[0].item()]
        if (A, T, -w) in combos:
            inv = combos.index((A, T, -w))
            print(f"  onde inversée du #1            ΔX = {dep[inv].item():8.2f}  (doit être ≈ opposé)")
        if meilleure_ref is None:
            meilleure_ref = combos[ordre[0].item()]
        else:
            j = combos.index(meilleure_ref)
            print(f"  meilleure onde de {a.water_drags[0]} {meilleure_ref}  ΔX = {dep[j].item():8.2f}")
        if a.controleur:
            dc = rouler_cerveau(charger_champion(a.controleur, torch.device("cpu")), a, wd, a.seed)
            print(f"  contrôleur {a.controleur} : ΔX = {dc:8.2f}  ({dc / lc_ref:5.2f} LC)")
        print()

    if a.gif:
        _, _, images = rouler_anguille(champ, [meilleure_ref], haut, bas,
                                       argparse.Namespace(**{**vars(a), "n": 1}),
                                       a.water_drags[0], a.seed, enregistrer=True)
        A, T, w = meilleure_ref
        ecrire_gif(champ, images, a.gif, f"anguille, water_drag={a.water_drags[0]}, A={A} T={T} w={w}")


if __name__ == "__main__":
    main()
