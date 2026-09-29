"""
PHASE 2 : Raffinement du cerveau d'un champion sur topologie FIGÉE.

Contrairement à train.py :
  - Une seule créature (pas de population, pas de mutation)
  - Un seul optimiseur Adam, JAMAIS réinitialisé
  - Des centaines d'épisodes d'affilée pour laisser converger la nage

Usage :
    python train2.py --chemin-champion elite_mutant/champion_gen_17_score_231.4_family_1.pt
    python train2.py --config config.yaml --chemin-champion ... --nb-episodes 500
"""

import os
import random
from datetime import datetime

import torch

from brain import Brain
from champion import charger_champion
from config import Config, build_argparser, load_config
from logger import Logger
from megaVecto import MegaCrea


def main():
    parser = build_argparser("train2")
    args = parser.parse_args()
    cfg: Config = load_config(args)

    if not cfg.chemin_champion:
        parser.error("--chemin-champion est requis (ou renseigne 'chemin_champion' dans le YAML)")

    SEED = cfg.seed
    random.seed(SEED)
    torch.manual_seed(SEED)

    BATCH_SIZE = cfg.train2_batch_size          # plus gros qu'en phase 1 : une seule créature, donc on peut se le permettre
    NB_EPISODES = cfg.nb_episodes                # le coeur de la phase 2 : beaucoup de pas d'apprentissage
    SUB_STEP = cfg.sub_step
    FRAME_NB = cfg.train2_frame_nb          # horizon propre à la phase 2
    LEARNING_RATE = cfg.learning_rate

    COEF_ENERGIE = cfg.coef_energie
    COEF_HAUTEUR = cfg.coef_hauteur

    SAUVEGARDE_TOUS_LES = cfg.sauvegarde_tous_les  # sauvegarde un checkpoint tous les N épisodes

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device utilisé : {device}")
    dt = torch.tensor(1.0 / SUB_STEP, device=device)

    # ==========================================
    # 📂 CHARGEMENT DU CHAMPION
    # ==========================================
    champ = charger_champion(cfg.chemin_champion, device)
    dico = champ.dico
    print(f"✅ Champion chargé depuis {cfg.chemin_champion}")
    print(f"   Topologie : {champ.num_nodes} noeuds, {champ.num_muscles} liens "
          f"({int(champ.num_muscles - sum(champ.is_bone))} muscles, {int(sum(champ.is_bone))} os)")
    print(f"   Padding d'origine : max_noeuds={champ.max_noeuds}, max_muscles={champ.max_muscles}")

    # We only build one brain that will be updated with all the gradients
    cerveau = Brain(champ.obs_size, champ.action_size).to(device)
    if champ.brain_weights is not None:
        cerveau.load_state_dict(champ.brain_weights)

    # ⚡️ L'OPTIMISEUR EST CRÉÉ UNE SEULE FOIS, HORS DE TOUTE BOUCLE
    optimizer = torch.optim.Adam(cerveau.parameters(), lr=LEARNING_RATE)

    os.makedirs(cfg.dossier_champion_raffine, exist_ok=True)
    os.makedirs(cfg.dossier_runs, exist_ok=True)

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    logger = Logger(
        f"{cfg.dossier_runs}/train_phase2_{run_id}.jsonl",
        run_config={
            "chemin_champion": cfg.chemin_champion, "seed": SEED, "batch_size": BATCH_SIZE,
            "nb_episodes": NB_EPISODES, "sub_step": SUB_STEP, "frame_nb": FRAME_NB,
            "learning_rate": LEARNING_RATE, "coef_energie": COEF_ENERGIE,
            "coef_hauteur": COEF_HAUTEUR,
        },
    )

    meilleur_score_global = float('-inf')
    meilleurs_poids = None

    def sauvegarder(chemin):
        torch.save(champ.to_save_dict(meilleurs_poids), chemin)

    # ==========================================
    # 🏋️ BOUCLE D'ENTRAÎNEMENT
    # ==========================================
    for episode in range(NB_EPISODES):

        mega = MegaCrea(dico, BATCH_SIZE, device=device)
        nb_n = torch.clamp(torch.sum(mega.mask_N_exp, dim=2), min=1.0)
        pos_depart = (torch.sum(mega.X * mega.mask_N_exp, dim=2) / nb_n).detach().clone()

        rewards_accumulated = None
        explosion = False

        # Bruit d'exploration décroissant : beaucoup au début, précision à la fin
        bruit_scale = max(0.005, 0.03 * (1 - episode / NB_EPISODES))

        for frame in range(FRAME_NB):
            if frame % 5 == 0:
                obs = mega.get_observation(frame)                    # [1, BATCH, obs_size]
                action = cerveau(obs)                                # [1, BATCH, action_size]

                bruit = torch.randn_like(action) * bruit_scale
                mega.apply_action(action + bruit, frame)

                reward_step = mega.get_reward(COEF_ENERGIE, COEF_HAUTEUR)   # [1, BATCH]
                rewards_accumulated = reward_step if rewards_accumulated is None else rewards_accumulated + reward_step

            # Troncature du gradient : évite une chaîne de backprop de 3000 pas
            if frame % 10 == 0 and frame > 0:
                mega.X = mega.X.detach()
                mega.Y = mega.Y.detach()
                mega.vX = mega.vX.detach()
                mega.vY = mega.vY.detach()

            for _ in range(SUB_STEP):
                mega.apply_physics(dt)

            if frame % 20 == 0:
                if torch.isnan(mega.X).any() or torch.isnan(mega.Y).any():
                    explosion = True
                    break

        if explosion:
            print(f"  💥 Explosion à l'épisode {episode}, épisode ignoré.")
            logger.log(episode=episode, explosion=True, bruit_scale=bruit_scale)
            optimizer.zero_grad(set_to_none=True)
            if rewards_accumulated is not None:
                del rewards_accumulated
            if device.type == "cuda":
                torch.cuda.empty_cache()
            continue

        loss = -torch.sum(rewards_accumulated) / BATCH_SIZE
        # Le test NaN ne tourne que toutes les 20 frames : une explosion en fin
        # d'épisode passerait inaperçue et empoisonnerait les moments d'Adam.
        if not torch.isfinite(loss):
            logger.log(episode=episode, explosion=True, bruit_scale=bruit_scale)
            optimizer.zero_grad(set_to_none=True)
            continue
        # --- Suivi et sauvegarde du meilleur ---
        scores = rewards_accumulated[0]                      # [BATCH]
        score_moyen = scores.mean().item()

        if score_moyen > meilleur_score_global:
            meilleur_score_global = score_moyen
            meilleurs_poids = {k: v.detach().clone()
                        for k, v in cerveau.state_dict().items()}

        optimizer.zero_grad()
        loss.backward()
        norm=torch.nn.utils.clip_grad_norm_(cerveau.parameters(), max_norm=10.0)
        optimizer.step()
        nb_n = torch.clamp(torch.sum(mega.mask_N_exp, dim=2), min=1.0)
        pos_fin = torch.sum(mega.X * mega.mask_N_exp, dim=2) / nb_n
        distance_finale = (pos_fin - pos_depart).mean().item()

        logger.log(
            episode=episode, score_mean=score_moyen, score_std=scores.std().item(),
            grad_norm=float(norm), loss=float(loss.item()),
            distance_moyenne=distance_finale, bruit_scale=bruit_scale,
            meilleur_score_global=meilleur_score_global, explosion=False,
        )

        if episode % 10 == 0:
            print(f"Ép. {episode:4d} | moyenne: {score_moyen:8.2f} | "
                  f"écart-type: {scores.std().item():8.2f} | norme gradient {norm}  | "
                  f"distance moy: {distance_finale:7.2f} | bruit: {bruit_scale:.4f}")

        if episode % SAUVEGARDE_TOUS_LES == 0 and meilleurs_poids is not None:
            sauvegarder(f"{cfg.dossier_champion_raffine}/raffine_ep{episode}_score_{meilleur_score_global:.1f}.pt")

    # ==========================================
    # 💾 SAUVEGARDE FINALE
    # ==========================================
    if meilleurs_poids is not None:
        chemin_final = f"{cfg.dossier_champion_raffine}/FINAL_score_{meilleur_score_global:.1f}.pt"
        sauvegarder(chemin_final)
        print(f"\n🏆 Terminé — meilleur score : {meilleur_score_global:.2f}")
        print(f"💾 Sauvegardé dans {chemin_final}")
    else:
        print("\n⚠️ Aucun épisode valide (que des explosions) — vérifie la physique.")

    logger.close()


if __name__ == "__main__":
    main()
