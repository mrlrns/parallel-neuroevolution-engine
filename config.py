"""
Configuration centrale du dépôt.

Un seul dataclass `Config` regroupe TOUS les paramètres qui étaient
auparavant en dur dans train.py, train2.py, visualize.py, test.py, test2.py
(chemins de fichiers, hyperparamètres, seeds, tailles de batch...).

Trois façons de fixer une valeur, par priorité croissante :
  1. La valeur par défaut du dataclass (celle qui était en dur avant)
  2. Le fichier YAML passé avec --config (config.yaml par défaut)
  3. Un argument CLI explicite (ex: --lr 0.001)

Usage dans un script :

    from config import Config, build_argparser, load_config

    def main():
        parser = build_argparser("train")          # "train" | "train2" | "visualize" | "test" | "test2"
        args = parser.parse_args()
        cfg = load_config(args)
        ...

    if __name__ == "__main__":
        main()
"""

from __future__ import annotations

import argparse
import dataclasses
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Optional

import yaml

# ==========================================================================
# 📦 LE DATACLASS : une seule source de vérité pour tous les paramètres
# ==========================================================================

@dataclass
class Config:
    # ---------- Chemins ----------
    # Dossier où train.py sauvegarde les champions de la phase 1 (évolution)
    dossier_elite_mutant: str = "elite_mutant"
    # Dossier où train2.py sauvegarde les champions raffinés de la phase 2
    dossier_champion_raffine: str = "champion_raffine"
    # Dossier des logs jsonl
    dossier_runs: str = "runs"
    # Dossier de sortie des vidéos (visualize.py)
    dossier_videos: str = "videos"

    # Fichier .pt d'entrée pour train2.py, visualize.py, test.py, test2.py
    # (le champion à charger, raffiner ou évaluer)
    chemin_champion: Optional[str] = None

    # ---------- Seed / reproductibilité ----------
    seed: int = 0

    # ---------- Simulation physique (commun à tous les scripts) ----------
    sub_step: int = 20          # nb de sous-pas physiques par frame (10 est instable, cf. E12)
    frame_nb: int = 200         # nb de frames par épisode / par run

    # ---------- Phase 1 : train.py (évolution de la population) ----------
    batch_size: int = 30
    max_noeuds_fixe: int = 20
    max_muscles_fixe: int = 30
    pop_size: int = 50
    nb_generations: int = 30
    lr: float = 1e-3
    rate_new_node: float = 0.1
    rate_mut_length: float = 0.3
    rate_change_bone: float = 0.2
    rate_pop_node: float = 0.15
    coef_energie: float = 10000.0
    coef_hauteur: float = 0.0
    bruit_action: float = 0.02

    # ---------- Phase 2 : train2.py (raffinement d'un champion figé) ----------
    train2_batch_size: int = 2000
    train2_frame_nb: int = 300  # horizon phase 2 (la référence a été raffinée sur 300 frames, cf. E2/E9)
    nb_episodes: int = 300
    learning_rate: float = 5e-5
    sauvegarde_tous_les: int = 25
    poids_aleatoires: bool = False  # True : ignore les poids du champion (comparaison de méthodes from scratch)
    fenetre_bptt: int = 10          # longueur (frames) de la fenêtre de troncature du gradient, multiple de 5
    maj_par_fenetre: bool = False   # E12 : une mise à jour Adam par fenêtre au lieu d'une par épisode
    detachement_complet: bool = False  # E12 contrôle : troncature complète en mode une-màj-par-épisode
    physique_compilee: bool = False    # MegaCreaFast + torch.compile (×3–4 sur GPU, même physique)

    # ---------- baselines/ppo.py (baseline PPO, même morphologie figée) ----------
    dossier_baselines: str = "baselines_out"
    ppo_nb_iterations: int = 300     # 1 itération = 1 épisode de ppo_n_envs rollouts (= 1 épisode de train2)
    ppo_n_envs: int = 2000
    ppo_lr: float = 3e-4
    ppo_gamma: float = 0.99
    ppo_gae_lambda: float = 0.95
    ppo_clip: float = 0.2
    ppo_epochs: int = 10
    ppo_minibatches: int = 8
    ppo_ent_coef: float = 0.0
    ppo_vf_coef: float = 0.5
    ppo_max_grad_norm: float = 0.5
    ppo_std_init: float = 0.2
    ppo_eval_tous_les: int = 5
    ppo_eval_n_envs: int = 500

    # ---------- visualize.py ----------
    largeur_ecran: int = 1200
    hauteur_ecran: int = 800
    fps_video: float = 60.0
    nb_frames_visualisation: int = 300
    bruit_visualisation: float = 0.02
    nom_video: str = "episode.mp4"

    # ---------- test.py (balayage de seeds) ----------
    nb_seeds: int = 50
    nb_frames_test: int = 1000
    bruit_test: float = 0.02

    # ---------- test2.py (reproduction des conditions de train.py) ----------
    nb_repetitions: int = 20

    def dossier_pour(self, phase: str) -> str:
        """Retourne le dossier de sauvegarde adapté à la phase ('phase1' ou 'phase2')."""
        return self.dossier_elite_mutant if phase == "phase1" else self.dossier_champion_raffine


# ==========================================================================
# 📄 CHARGEMENT YAML
# ==========================================================================

def _config_defaut_dict() -> dict:
    return dataclasses.asdict(Config())


def charger_yaml(chemin: Optional[str]) -> dict:
    """Charge un fichier YAML de config. Silencieux si le fichier n'existe pas
    (les valeurs par défaut du dataclass s'appliquent alors)."""
    if chemin is None:
        return {}
    p = Path(chemin)
    if not p.exists():
        print(f"⚠️  Fichier de config '{chemin}' introuvable, valeurs par défaut utilisées.")
        return {}
    with open(p, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    return data


# ==========================================================================
# 🖥️ ARGPARSE
# ==========================================================================

# Aide humaine pour chaque champ, affichée par --help.
_AIDE = {
    "dossier_elite_mutant": "Dossier de sortie des champions de la phase 1 (évolution)",
    "dossier_champion_raffine": "Dossier de sortie des champions raffinés (phase 2)",
    "dossier_runs": "Dossier des logs d'entraînement (.jsonl)",
    "dossier_videos": "Dossier de sortie des vidéos générées par visualize.py",
    "chemin_champion": "Chemin du fichier .pt du champion à charger",
    "seed": "Graine aléatoire",
    "sub_step": "Nombre de sous-pas physiques par frame",
    "frame_nb": "Nombre de frames par épisode",
    "batch_size": "Taille de batch (phase 1)",
    "max_noeuds_fixe": "Nombre max de nœuds par créature (padding)",
    "max_muscles_fixe": "Nombre max de muscles par créature (padding)",
    "pop_size": "Taille de la population (phase 1)",
    "nb_generations": "Nombre de générations (phase 1)",
    "lr": "Taux d'apprentissage (phase 1)",
    "rate_new_node": "Probabilité de mutation : ajout de nœud",
    "rate_mut_length": "Probabilité de mutation : longueur des muscles",
    "rate_change_bone": "Probabilité de mutation : muscle <-> os",
    "rate_pop_node": "Probabilité de mutation : suppression de nœud",
    "coef_energie": "Coefficient de pénalité d'énergie dans la récompense",
    "coef_hauteur": "Coefficient de pénalité de hauteur dans la récompense",
    "bruit_action": "Écart-type du bruit d'exploration sur les actions (phase 1)",
    "train2_batch_size": "Taille de batch (phase 2, une seule créature)",
    "train2_frame_nb": "Nombre de frames par épisode (phase 2)",
    "nb_episodes": "Nombre d'épisodes d'entraînement (phase 2)",
    "learning_rate": "Taux d'apprentissage (phase 2)",
    "sauvegarde_tous_les": "Sauvegarde un checkpoint tous les N épisodes (phase 2)",
    "poids_aleatoires": "Ignore les poids du champion et part d'un cerveau aléatoire",
    "fenetre_bptt": "Longueur de la fenêtre de troncature du gradient (frames, multiple de 5)",
    "maj_par_fenetre": "Une mise à jour de l'optimiseur par fenêtre de BPTT (E12) au lieu d'une par épisode",
    "detachement_complet": "Coupe aussi les chemins obs→action et previous_distance aux frontières de fenêtre",
    "physique_compilee": "Physique MegaCreaFast compilée par torch.compile (GPU ; même physique, ×3–4)",
    "dossier_baselines": "Dossier de sortie des checkpoints des baselines",
    "ppo_nb_iterations": "Nombre d'itérations PPO (1 itération = 1 épisode sur tous les envs)",
    "ppo_n_envs": "Nombre d'environnements parallèles (PPO)",
    "ppo_lr": "Taux d'apprentissage (PPO)",
    "ppo_gamma": "Facteur d'actualisation (PPO)",
    "ppo_gae_lambda": "Lambda de GAE (PPO)",
    "ppo_clip": "Clipping du ratio de probabilités (PPO)",
    "ppo_epochs": "Nombre de passes sur chaque rollout (PPO)",
    "ppo_minibatches": "Nombre de minibatchs par passe (PPO)",
    "ppo_ent_coef": "Coefficient du bonus d'entropie (PPO)",
    "ppo_vf_coef": "Coefficient de la loss du critique (PPO)",
    "ppo_max_grad_norm": "Clipping de la norme du gradient (PPO)",
    "ppo_std_init": "Écart-type initial de la politique gaussienne (PPO)",
    "ppo_eval_tous_les": "Évaluation quasi-déterministe toutes les N itérations (PPO)",
    "ppo_eval_n_envs": "Nombre de rollouts pour l'évaluation (PPO)",
    "largeur_ecran": "Largeur de la fenêtre / vidéo (pixels)",
    "hauteur_ecran": "Hauteur de la fenêtre / vidéo (pixels)",
    "fps_video": "Images par seconde de la vidéo exportée",
    "nb_frames_visualisation": "Nombre de frames simulées pour la vidéo",
    "bruit_visualisation": "Écart-type du bruit d'action pendant la visualisation",
    "nom_video": "Nom du fichier vidéo de sortie",
    "nb_seeds": "Nombre de seeds testées (test.py)",
    "nb_frames_test": "Nombre de frames par seed testée (test.py)",
    "bruit_test": "Écart-type du bruit d'action pendant le test",
    "nb_repetitions": "Nombre de répétitions de l'évaluation (test2.py)",
}

# Champs pertinents par script, pour ne pas noyer l'utilisateur sous 30 options
# quand il lance `python test.py --help`.
_CHAMPS_PAR_SCRIPT = {
    "train": [
        "dossier_elite_mutant", "dossier_runs", "seed", "sub_step", "frame_nb",
        "batch_size", "max_noeuds_fixe", "max_muscles_fixe", "pop_size",
        "nb_generations", "lr", "rate_new_node", "rate_mut_length",
        "rate_change_bone", "rate_pop_node", "coef_energie", "coef_hauteur",
        "bruit_action",
    ],
    "train2": [
        "chemin_champion", "dossier_champion_raffine", "dossier_runs", "seed",
        "sub_step", "train2_frame_nb", "train2_batch_size", "nb_episodes",
        "learning_rate", "coef_energie", "coef_hauteur", "sauvegarde_tous_les",
        "poids_aleatoires", "fenetre_bptt", "maj_par_fenetre",
        "detachement_complet", "physique_compilee",
    ],
    "ppo": [
        "chemin_champion", "dossier_baselines", "dossier_runs", "seed", "sub_step",
        "train2_frame_nb", "coef_energie", "coef_hauteur", "poids_aleatoires",
        "ppo_nb_iterations", "ppo_n_envs", "ppo_lr", "ppo_gamma", "ppo_gae_lambda",
        "ppo_clip", "ppo_epochs", "ppo_minibatches", "ppo_ent_coef", "ppo_vf_coef",
        "ppo_max_grad_norm", "ppo_std_init", "ppo_eval_tous_les", "ppo_eval_n_envs",
        "physique_compilee",
    ],
    "check_env": [
        "chemin_champion", "seed", "sub_step", "train2_frame_nb", "train2_batch_size",
        "coef_energie", "coef_hauteur", "bruit_action",
    ],
    "visualize": [
        "chemin_champion", "dossier_videos", "seed", "sub_step",
        "largeur_ecran", "hauteur_ecran", "fps_video",
        "nb_frames_visualisation", "bruit_visualisation", "nom_video",
    ],
    "test": [
        "chemin_champion", "sub_step", "nb_seeds", "nb_frames_test", "bruit_test",
    ],
    "test2": [
        "chemin_champion", "sub_step", "batch_size", "frame_nb",
        "coef_energie", "coef_hauteur", "bruit_action", "nb_repetitions",
    ],
}


def build_argparser(script: str) -> argparse.ArgumentParser:
    """Construit un parser CLI n'exposant que les options pertinentes pour `script`.

    `script` doit être l'un de : train, train2, visualize, test, test2.
    """
    if script not in _CHAMPS_PAR_SCRIPT:
        raise ValueError(f"script inconnu: {script!r}, attendu un de {list(_CHAMPS_PAR_SCRIPT)}")

    parser = argparse.ArgumentParser(
        description=f"{script}.py — voir config.yaml pour tous les paramètres par défaut.",
    )
    parser.add_argument(
        "--config", type=str, default="config.yaml",
        help="Chemin du fichier YAML de configuration",
    )

    defauts = _config_defaut_dict()
    # `from __future__ import annotations` fait que f.type est une chaîne
    # (ex: "int", "Optional[str]") plutôt qu'un vrai type -> on déduit le
    # type réel depuis la valeur par défaut, plus robuste.
    types_par_valeur = {}
    for f in fields(Config):
        val = defauts[f.name]
        if val is None:
            types_par_valeur[f.name] = str  # Optional[str] par défaut à None
        else:
            types_par_valeur[f.name] = type(val)

    for nom in _CHAMPS_PAR_SCRIPT[script]:
        type_effectif = types_par_valeur[nom]
        if type_effectif is bool:
            parser.add_argument(f"--{nom.replace('_', '-')}", dest=nom,
                                 action="store_true", default=None,
                                 help=_AIDE.get(nom, ""))
        else:
            parser.add_argument(f"--{nom.replace('_', '-')}", dest=nom,
                                 type=type_effectif, default=None,
                                 help=f"{_AIDE.get(nom, '')} (défaut: {defauts[nom]})")

    return parser


def load_config(args: argparse.Namespace) -> Config:
    """Fusionne défauts du dataclass <- YAML <- arguments CLI explicites."""
    fusion = _config_defaut_dict()
    fusion.update(charger_yaml(getattr(args, "config", None)))

    for cle, valeur in vars(args).items():
        if cle == "config":
            continue
        if valeur is not None:
            fusion[cle] = valeur

    # Ne garder que les clés connues du dataclass (un YAML avec une clé en trop
    # ne doit pas faire planter le programme)
    champs_connus = {f.name for f in fields(Config)}
    fusion_filtree = {k: v for k, v in fusion.items() if k in champs_connus}

    return Config(**fusion_filtree)
