"""
PHASE 2 : Raffinement du cerveau d'un champion sur topologie FIGÉE.

Contrairement à train.py :
  - Une seule créature (pas de population, pas de mutation)
  - Un seul optimiseur Adam, JAMAIS réinitialisé
  - Des centaines d'épisodes d'affilée pour laisser converger la nage

Usage :
    python train2.py --chemin-champion elite_mutant/champion_gen_17_score_231.4_family_1.pt
    python train2.py --config config.yaml --chemin-champion ... --nb-episodes 500
    python train2.py --chemin-champion ... --poids-aleatoires --seed 1   # comparaison de méthodes
    python train2.py --chemin-champion ... --poids-aleatoires --maj-par-fenetre   # E12
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


def detacher_etat(mega):
    """Coupe le graphe de calcul de TOUT l'état qui en porte un.

    Le mode historique (E9-E11) ne détache que X, Y, vX, vY : target_length,
    previous_distance et previous_height gardent alors un lien avec la fenêtre
    précédente. En mise à jour par fenêtre, ce lien ferait rétropropager dans un
    graphe déjà libéré par backward(), il faut donc tout couper."""
    for nom in ("X", "Y", "vX", "vY", "target_length", "previous_distance", "previous_height", "energy"):
        setattr(mega, nom, getattr(mega, nom).detach())


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
    FENETRE = cfg.fenetre_bptt
    if FENETRE % 5 != 0:
        parser.error("--fenetre-bptt doit être un multiple de 5 (une décision toutes les 5 frames)")
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
    if cfg.poids_aleatoires:
        print("   Cerveau RÉINITIALISÉ (poids aléatoires) — comparaison de méthodes")
    elif champ.brain_weights is not None:
        cerveau.load_state_dict(champ.brain_weights)

    # ⚡️ L'OPTIMISEUR EST CRÉÉ UNE SEULE FOIS, HORS DE TOUTE BOUCLE
    optimizer = torch.optim.Adam(cerveau.parameters(), lr=LEARNING_RATE)

    os.makedirs(cfg.dossier_champion_raffine, exist_ok=True)
    os.makedirs(cfg.dossier_runs, exist_ok=True)

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    logger = Logger(
        f"{cfg.dossier_runs}/train_phase2_seed{SEED}_{run_id}.jsonl",
        run_config={
            "method": ("diffsim_bptt_par_fenetre" if cfg.maj_par_fenetre
                       else "diffsim_bptt_detach_complet" if cfg.detachement_complet else "diffsim_bptt"),
            "chemin_champion": cfg.chemin_champion, "seed": SEED, "batch_size": BATCH_SIZE,
            "nb_episodes": NB_EPISODES, "sub_step": SUB_STEP, "frame_nb": FRAME_NB,
            "learning_rate": LEARNING_RATE, "coef_energie": COEF_ENERGIE,
            "coef_hauteur": COEF_HAUTEUR, "poids_aleatoires": cfg.poids_aleatoires,
            "fenetre_bptt": FENETRE, "maj_par_fenetre": cfg.maj_par_fenetre,
            "detachement_complet": cfg.detachement_complet,
        },
    )

    meilleur_score_global = float('-inf')
    meilleurs_poids = None
    # Budget de simulation consommé : appels à apply_physics × rollouts.
    # Unité commune avec les baselines (baselines/ppo.py) pour comparer à coût égal.
    sim_steps = 0

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

        if cfg.maj_par_fenetre:
            # ---------- E12 : une mise à jour par fenêtre de BPTT ----------
            # Chaque fenêtre est déjà un problème indépendant (graphe coupé) : on
            # met à jour à la fin de chacune au lieu d'accumuler tout l'épisode.
            # Récompense réalignée comme dans SwimEnv : le score d'épisode inclut le
            # mouvement des 5 dernières frames (~1 % de plus qu'en mode historique).
            # Les poids sauvegardés comme "meilleurs" sont ceux du DÉBUT de
            # l'épisode : le score de l'épisode est obtenu par une politique qui
            # évolue en cours de route ; le verdict reste evaluate_seeds.py.
            poids_debut = {k: v.detach().clone() for k, v in cerveau.state_dict().items()}
            score_total = torch.zeros(BATCH_SIZE, device=device)
            r_fenetre = None
            normes, nb_maj = [], 0

            def mise_a_jour():
                nonlocal r_fenetre, nb_maj
                loss_f = -torch.sum(r_fenetre) / BATCH_SIZE
                ok = bool(torch.isfinite(loss_f))
                if ok:
                    optimizer.zero_grad()
                    loss_f.backward()
                    normes.append(float(torch.nn.utils.clip_grad_norm_(cerveau.parameters(), max_norm=10.0)))
                    optimizer.step()
                    nb_maj += 1
                r_fenetre = None
                detacher_etat(mega)
                return ok

            def recompenser():
                # Mouvement depuis la décision précédente, moins l'énergie de l'action
                # qui l'a produit : chaque action est créditée de SON effet (comme SwimEnv).
                nonlocal r_fenetre
                r = mega.get_reward(COEF_ENERGIE, COEF_HAUTEUR)
                score_total.add_(r[0].detach())
                r_fenetre = r if r_fenetre is None else r_fenetre + r

            for frame in range(FRAME_NB):
                if frame % 5 == 0:
                    if frame > 0:
                        recompenser()
                        # La mise à jour se fait AVANT l'action suivante : couper le graphe
                        # après apply_action priverait cette action du gradient du
                        # mouvement qu'elle provoque (il ne resterait que son coût énergétique).
                        if frame % FENETRE == 0 and not mise_a_jour():
                            explosion = True
                            break
                    obs = mega.get_observation(frame)
                    action = cerveau(obs)
                    mega.apply_action(action + torch.randn_like(action) * bruit_scale, frame)
                for _ in range(SUB_STEP):
                    mega.apply_physics(dt)
                sim_steps += BATCH_SIZE * SUB_STEP
            if not explosion:
                recompenser()                      # mouvement des 5 dernières frames
                if not mise_a_jour():
                    explosion = True

            if explosion or not torch.isfinite(score_total).all():
                print(f"  💥 Explosion à l'épisode {episode}, fin d'épisode ignorée.")
                logger.log(episode=episode, explosion=True, bruit_scale=bruit_scale,
                           sim_steps=sim_steps, nb_maj=nb_maj)
                optimizer.zero_grad(set_to_none=True)
                continue

            score_moyen = score_total.mean().item()
            if score_moyen > meilleur_score_global:
                meilleur_score_global = score_moyen
                meilleurs_poids = poids_debut
            with torch.no_grad():
                nb_n = torch.clamp(torch.sum(mega.mask_N_exp, dim=2), min=1.0)
                distance_finale = ((torch.sum(mega.X * mega.mask_N_exp, dim=2) / nb_n) - pos_depart).mean().item()
            norm_moy = sum(normes) / max(1, len(normes))
            logger.log(
                episode=episode, sim_steps=sim_steps, score_mean=score_moyen,
                score_std=score_total.std().item(), grad_norm=norm_moy, grad_norm_max=max(normes, default=0.0),
                nb_maj=nb_maj, distance_moyenne=distance_finale, bruit_scale=bruit_scale,
                meilleur_score_global=meilleur_score_global, explosion=False,
            )
            if episode % 10 == 0:
                print(f"Ép. {episode:4d} | moyenne: {score_moyen:8.2f} | écart-type: {score_total.std().item():8.2f} | "
                      f"{nb_maj} màj, norme moy {norm_moy:.2f} | distance moy: {distance_finale:7.2f} | "
                      f"bruit: {bruit_scale:.4f}")
            if episode % SAUVEGARDE_TOUS_LES == 0 and meilleurs_poids is not None:
                sauvegarder(f"{cfg.dossier_champion_raffine}/raffine_seed{SEED}_ep{episode}_score_{meilleur_score_global:.1f}.pt")
            continue

        for frame in range(FRAME_NB):
            if frame % 5 == 0:
                obs = mega.get_observation(frame)                    # [1, BATCH, obs_size]
                if cfg.detachement_complet and frame % FENETRE == 0 and frame > 0:
                    # Chemin 1 : sans ce detach, l'action de la frontière dépend de l'état
                    # de la fenêtre précédente via l'observation (fuite de gradient).
                    obs = obs.detach()
                action = cerveau(obs)                                # [1, BATCH, action_size]

                bruit = torch.randn_like(action) * bruit_scale
                mega.apply_action(action + bruit, frame)

                reward_step = mega.get_reward(COEF_ENERGIE, COEF_HAUTEUR)   # [1, BATCH]
                rewards_accumulated = reward_step if rewards_accumulated is None else rewards_accumulated + reward_step

            # Troncature du gradient : évite une chaîne de backprop de 3000 pas
            if frame % FENETRE == 0 and frame > 0:
                mega.X = mega.X.detach()
                mega.Y = mega.Y.detach()
                mega.vX = mega.vX.detach()
                mega.vY = mega.vY.detach()
                if cfg.detachement_complet:
                    # Chemin 2 : la récompense suivante soustrait previous_distance,
                    # calculée avant ce detach. target_length n'est PAS détaché : il
                    # porte l'action de cette frame, qui appartient à la nouvelle fenêtre.
                    mega.previous_distance = mega.previous_distance.detach()
                    if torch.is_tensor(mega.previous_height):
                        mega.previous_height = mega.previous_height.detach()

            for _ in range(SUB_STEP):
                mega.apply_physics(dt)
            sim_steps += BATCH_SIZE * SUB_STEP

            if frame % 20 == 0:
                if torch.isnan(mega.X).any() or torch.isnan(mega.Y).any():
                    explosion = True
                    break

        if explosion:
            print(f"  💥 Explosion à l'épisode {episode}, épisode ignoré.")
            logger.log(episode=episode, explosion=True, bruit_scale=bruit_scale, sim_steps=sim_steps)
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
            logger.log(episode=episode, explosion=True, bruit_scale=bruit_scale, sim_steps=sim_steps)
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
            episode=episode, sim_steps=sim_steps, score_mean=score_moyen, score_std=scores.std().item(),
            grad_norm=float(norm), loss=float(loss.item()),
            distance_moyenne=distance_finale, bruit_scale=bruit_scale,
            meilleur_score_global=meilleur_score_global, explosion=False,
        )

        if episode % 10 == 0:
            print(f"Ép. {episode:4d} | moyenne: {score_moyen:8.2f} | "
                  f"écart-type: {scores.std().item():8.2f} | norme gradient {norm}  | "
                  f"distance moy: {distance_finale:7.2f} | bruit: {bruit_scale:.4f}")

        if episode % SAUVEGARDE_TOUS_LES == 0 and meilleurs_poids is not None:
            sauvegarder(f"{cfg.dossier_champion_raffine}/raffine_seed{SEED}_ep{episode}_score_{meilleur_score_global:.1f}.pt")

    # ==========================================
    # 💾 SAUVEGARDE FINALE
    # ==========================================
    if meilleurs_poids is not None:
        chemin_final = f"{cfg.dossier_champion_raffine}/FINAL_seed{SEED}_score_{meilleur_score_global:.1f}.pt"
        sauvegarder(chemin_final)
        print(f"\n🏆 Terminé — meilleur score : {meilleur_score_global:.2f}")
        print(f"💾 Sauvegardé dans {chemin_final}")
    else:
        print("\n⚠️ Aucun épisode valide (que des explosions) — vérifie la physique.")

    logger.close()


if __name__ == "__main__":
    main()
