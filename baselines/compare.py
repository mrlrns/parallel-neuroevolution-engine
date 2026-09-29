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
    return grille, ys.mean(0), ys.std(0), len(runs)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--diffsim", required=True, help="glob des logs train2.py")
    p.add_argument("--ppo", required=True, help="glob des logs baselines/ppo.py")
    p.add_argument("--sortie", default="baseline_ppo.png")
    a = p.parse_args()

    fichiers = {"diffsim": sorted(glob.glob(a.diffsim)), "ppo": sorted(glob.glob(a.ppo))}
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for i, titre in enumerate(("mean reward", "mean displacement")):
        ax = axes[i]
        for methode, fs in fichiers.items():
            res = agreger(fs, METRIQUES[methode][i])
            if res is None:
                print(f"⚠️  aucun point pour {methode} ({titre})")
                continue
            x, m, s, n = res
            ax.plot(x, m, label=f"{LABELS[methode]} (n={n} seeds)")
            ax.fill_between(x, m - s, m + s, alpha=0.2)
        ax.set_xlabel("physics sub-steps × rollouts")
        ax.set_ylabel(titre)
        ax.grid(alpha=0.3)
        ax.legend(frameon=False)
    fig.suptitle("Phase 2 — differentiable simulation vs PPO, equal simulation budget")
    fig.tight_layout()
    fig.savefig(a.sortie, dpi=150)
    print(f"📈 {a.sortie}")


if __name__ == "__main__":
    main()
