"""
baselines/compare.py — figure « score contre budget de simulation », une courbe
par méthode (moyenne ± écart-type sur les seeds d'entraînement).

Métriques tracées :
  - train2.py  : score_mean / distance_moyenne de l'épisode d'entraînement
                 (bruit d'exploration ≤ 0.03)
  - ppo.py     : eval_score_mean / eval_distance_moyenne (moyenne de la
                 politique + bruit 0.02), et NON le score stochastique
                 d'entraînement, pour comparer des politiques de même bruit.
Le verdict final reste evaluate_seeds.py sur les checkpoints (20 seeds × 1000 frames).

Usage :
    python -m baselines.compare \
        --diffsim "runs/train_phase2_seed*.jsonl" --ppo "runs/ppo_seed*.jsonl" \
        --sortie docs/baseline_ppo.png
"""

import argparse
import glob
import json

import matplotlib.pyplot as plt
import numpy as np

METRIQUES = {
    "diffsim": ("score_mean", "distance_moyenne"),
    "ppo": ("eval_score_mean", "eval_distance_moyenne"),
}
LABELS = {"diffsim": "Differentiable sim (BPTT)", "ppo": "PPO"}


def lire_run(chemin, cle):
    xs, ys = [], []
    with open(chemin, encoding="utf-8") as f:
        for ligne in f:
            rec = json.loads(ligne)
            if "_run_config" in rec or rec.get(cle) is None or rec.get("sim_steps") is None:
                continue
            xs.append(rec["sim_steps"])
            ys.append(rec[cle])
    return np.array(xs, dtype=float), np.array(ys, dtype=float)


def agreger(fichiers, cle, n_points=100):
    runs = [lire_run(f, cle) for f in fichiers]
    runs = [(x, y) for x, y in runs if len(x) > 1]
    if not runs:
        return None
    x_max = min(x.max() for x, _ in runs)            # grille commune : budget atteint par tous les runs
    x_min = max(x.min() for x, _ in runs)
    grille = np.linspace(x_min, x_max, n_points)
    ys = np.stack([np.interp(grille, x, y) for x, y in runs])
    return grille, ys


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--diffsim", required=True, help="glob des logs train2.py")
    p.add_argument("--ppo", required=True, help="glob des logs baselines/ppo.py")
    p.add_argument("--sortie", default="baseline_ppo.png")
    p.add_argument("--lineaire", action="store_true", help="axe vertical linéaire (logarithmique par défaut)")
    p.add_argument("--reference", type=float, default=None,
                   help="score du champion de référence, tracé en pointillés (même horizon)")
    a = p.parse_args()

    # Un seul panneau : la pénalité d'énergie est négligeable, récompense ≈ déplacement.
    # Pas de bande : avec 3 seeds, un écart-type n'est pas une estimation fiable ;
    # on montre chaque seed (trait fin) et leur moyenne (trait épais).
    fichiers = {"diffsim": sorted(glob.glob(a.diffsim)), "ppo": sorted(glob.glob(a.ppo))}
    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    for methode, fs in fichiers.items():
        res = agreger(fs, METRIQUES[methode][0])
        if res is None:
            print(f"⚠️  aucun point pour {methode}")
            continue
        x, ys = res
        ligne, = ax.plot(x, ys.mean(0), lw=2.5, label=f"{LABELS[methode]} — mean of {ys.shape[0]} seeds")
        for y in ys:
            ax.plot(x, y, lw=1, alpha=0.35, color=ligne.get_color())
    if a.reference is not None:
        ax.axhline(a.reference, ls="--", lw=1.2, color="0.4",
                   label="reference champion (evolution + refinement)")
    ax.set_xlabel("simulation budget (physics sub-steps × rollouts)")
    ax.set_ylabel("mean episode reward, 300 frames" + ("" if a.lineaire else " (log)"))
    if not a.lineaire:
        ax.set_yscale("log")
    ax.grid(alpha=0.3, which="both")
    ax.legend(frameon=False, loc="lower right")
    ax.set_title("Differentiable simulation vs PPO at equal simulation budget\n"
                 "random initial controller; thin lines = individual training seeds", fontsize=10)
    fig.tight_layout()
    fig.savefig(a.sortie, dpi=150)
    print(f"📈 {a.sortie}")


if __name__ == "__main__":
    main()
