"""
Reproduit EXACTEMENT les conditions d'evaluation de train.py pour un champion
sauvegarde, afin de verifier si son score annonce est reproductible.

Conditions repliquees depuis train.py (valeurs par defaut, voir config.yaml) :
  - batch_size = 30 (BATCH_SIZE de la phase 1)
  - frame_nb = 200
  - sub_step = 10, dt = 0.1
  - action toutes les 5 frames, avec bruit randn * bruit_action
  - reward accumule via get_reward(coef_energie, coef_hauteur)

Le score annonce est max(rewards) sur le batch. On affiche aussi la moyenne
et le deplacement, pour separer "score reproductible" de "statistique d'extreme".

Usage :
    python test2.py --chemin-champion champion_raffine/FINAL_score_222.9.pt
"""

import torch

from brain import Brain
from megaVecto import MegaCrea
from config import Config, build_argparser, load_config


def pad_1d(vec, size, dtype=torch.float32, pad_value=0.0):
    t = torch.tensor(vec, dtype=dtype)
    if len(vec) < size:
        t = torch.cat([t, torch.full((size - len(vec),), pad_value, dtype=dtype)])
    return t.unsqueeze(0)


def main():
    parser = build_argparser("test2")
    args = parser.parse_args()
    cfg: Config = load_config(args)

    if not cfg.chemin_champion:
        parser.error("--chemin-champion est requis (ou renseigne 'chemin_champion' dans le YAML)")

    # ---------- config identique a train.py (surchageable en CLI/YAML) ----------
    SUB_STEP = cfg.sub_step
    BATCH_SIZE = cfg.batch_size
    FRAME_NB = cfg.frame_nb
    COEF_ENERGIE = cfg.coef_energie
    COEF_HAUTEUR = cfg.coef_hauteur
    BRUIT = cfg.bruit_action

    CHEMIN = cfg.chemin_champion
    NB_REPETITIONS = cfg.nb_repetitions      # nombre d'"episodes" rejoues

    device = torch.device("cpu")
    dt = torch.tensor(1.0 / SUB_STEP)

    # ---------- chargement ----------
    donnees = torch.load(CHEMIN, map_location=device)

    x = donnees['x']
    y = donnees['y']
    is_bone = donnees['is_bone']
    muscle1 = donnees['muscle1'].tolist() if torch.is_tensor(donnees['muscle1']) else donnees['muscle1']
    muscle2 = donnees['muscle2'].tolist() if torch.is_tensor(donnees['muscle2']) else donnees['muscle2']
    target_length = donnees['target_length']
    max_noeuds = donnees['max_noeuds']
    max_muscles = donnees['max_muscles']

    num_nodes_reel = len(x)
    num_muscles_reel = len(muscle1)

    dico = {
        "X": pad_1d(x, max_noeuds),
        "Y": pad_1d(y, max_noeuds),
        "m1": pad_1d(muscle1, max_muscles, dtype=torch.long, pad_value=0),
        "m2": pad_1d(muscle2, max_muscles, dtype=torch.long, pad_value=0),
        "stiffness": pad_1d([1.0 + 4 * s for s in is_bone], max_muscles),
        "is_bone": pad_1d(is_bone, max_muscles),
        "base_length": pad_1d(target_length.tolist(), max_muscles),
        "masque_noeuds": (torch.arange(max_noeuds) < num_nodes_reel).float().unsqueeze(0),
        "masque_muscles": (torch.arange(max_muscles) < num_muscles_reel).float().unsqueeze(0),
    }

    obs_size = max_noeuds * 4 + max_muscles + 1
    action_size = max_muscles

    cerveau = Brain(obs_size, action_size)
    cerveau.load_state_dict(donnees['brain_weights'])
    cerveau.eval()

    print(CHEMIN)
    print(f"{num_nodes_reel} noeuds, {num_muscles_reel} liens")
    print(f"padding: max_noeuds={max_noeuds}, max_muscles={max_muscles}")
    print(f"conditions: batch={BATCH_SIZE}, frames={FRAME_NB}, bruit={BRUIT}, "
          f"coef_energie={COEF_ENERGIE}, coef_hauteur={COEF_HAUTEUR}\n")

    # ---------- rejeu ----------
    meilleur_global = float('-inf')

    with torch.no_grad():
        for rep in range(NB_REPETITIONS):
            torch.manual_seed(rep)
            mega = MegaCrea(dico, BATCH_SIZE, device=device)

            nb_n = torch.clamp(torch.sum(mega.mask_N_exp, dim=2), min=1.0)
            depart = (torch.sum(mega.X * mega.mask_N_exp, dim=2) / nb_n).clone()

            rewards = None

            for frame in range(FRAME_NB):
                if frame % 5 == 0:
                    obs = mega.get_observation(frame)                 # [1, BATCH, obs]
                    obs_flat = obs.reshape(BATCH_SIZE, obs_size)
                    action = cerveau(obs_flat).reshape(1, BATCH_SIZE, action_size)

                    action = action + torch.randn_like(action) * BRUIT
                    mega.apply_action(action, frame)

                    r = mega.get_reward(COEF_ENERGIE, COEF_HAUTEUR)   # [1, BATCH]
                    rewards = r if rewards is None else rewards + r

                for _ in range(SUB_STEP):
                    mega.apply_physics(dt)

            fin = torch.sum(mega.X * mega.mask_N_exp, dim=2) / nb_n
            deplacement = (fin - depart)[0]
            scores = rewards[0]

            meilleur_global = max(meilleur_global, scores.max().item())

            print(f"rep {rep:3d} | score max {scores.max().item():8.2f} | "
                  f"moyenne {scores.mean().item():8.2f} | "
                  f"depl. max {deplacement.max().item():8.2f} | "
                  f"depl. moyen {deplacement.mean().item():8.2f}")

    print(f"\nmeilleur score sur {NB_REPETITIONS} repetitions x {BATCH_SIZE} : "
          f"{meilleur_global:.2f}")


if __name__ == "__main__":
    main()