"""
PHASE 2 : Raffinement du cerveau d'un champion sur topologie FIGÉE.

Contrairement à train.py :
  - Une seule créature (pas de population, pas de mutation)
  - Un seul optimiseur Adam, JAMAIS réinitialisé
  - Des centaines d'épisodes d'affilée pour laisser converger la nage

Usage : modifier CHEMIN_CHAMPION ci-dessous, puis `python train2.py`
"""

import torch
import torch.nn as nn
import copy
import os
from torch.func import stack_module_state, functional_call, vmap
import random

from brain import Brain
from megaVecto import MegaCrea

# ==========================================
# ⚙️ CONFIGURATION
# ==========================================
CHEMIN_CHAMPION = "elite_mutant/champion_gen_14_score_240.8_family_1.pt"  # <-- À MODIFIER

SEED = 0
random.seed(SEED)
torch.manual_seed(SEED)

BATCH_SIZE = 2000          # plus gros qu'en phase 1 : une seule créature, donc on peut se le permettre
NB_EPISODES = 300        # le coeur de la phase 2 : beaucoup de pas d'apprentissage
SUB_STEP = 10
FRAME_NB = 300
LEARNING_RATE = 5*1e-5

COEF_ENERGIE = 3000.0
COEF_HAUTEUR = 10.0

SAUVEGARDE_TOUS_LES = 25  # sauvegarde un checkpoint tous les N épisodes

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device utilisé : {device}")
dt = torch.tensor(1.0 / SUB_STEP, device=device)


# ==========================================
# 📂 CHARGEMENT DU CHAMPION
# ==========================================
donnees = torch.load(CHEMIN_CHAMPION, map_location=device)
print(f"✅ Champion chargé depuis {CHEMIN_CHAMPION}")

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

print(f"   Topologie : {num_nodes_reel} noeuds, {num_muscles_reel} liens "
      f"({int(num_muscles_reel - sum(is_bone))} muscles, {int(sum(is_bone))} os)")
print(f"   Padding d'origine : max_noeuds={max_noeuds}, max_muscles={max_muscles}")


def pad_1d(vec, size, dtype=torch.float32, pad_value=0.0):
    """Pad un vecteur 1D à la taille voulue et ajoute la dimension POP_SIZE=1."""
    t = torch.tensor(vec, dtype=dtype)
    if len(vec) < size:
        pad = torch.full((size - len(vec),), pad_value, dtype=dtype)
        t = torch.cat([t, pad])
    return t.unsqueeze(0).to(device)


# Reconstruction du dico "méga-univers" pour UNE seule créature,
# avec exactement le même padding qu'à l'entraînement d'origine
base_length_list = target_length.tolist() if torch.is_tensor(target_length) else list(target_length)

dico = {
    "X": pad_1d(x, max_noeuds),
    "Y": pad_1d(y, max_noeuds),
    "m1": pad_1d(muscle1, max_muscles, dtype=torch.long, pad_value=0),
    "m2": pad_1d(muscle2, max_muscles, dtype=torch.long, pad_value=0),
    "stiffness": pad_1d([1.0 + 4 * s for s in is_bone], max_muscles),
    "is_bone": pad_1d(is_bone, max_muscles),
    "base_length": pad_1d(base_length_list, max_muscles),
    "masque_noeuds": (torch.arange(max_noeuds, device=device) < num_nodes_reel).float().unsqueeze(0),
    "masque_muscles": (torch.arange(max_muscles, device=device) < num_muscles_reel).float().unsqueeze(0),
}

obs_size = max_noeuds * 4 + max_muscles + 1
action_size = max_muscles


# ==========================================
# 🧠 CRÉATION DU BATCH DE CERVEAUX
# ==========================================
# Tous initialisés avec les poids du champion, puis légèrement perturbés
# pour créer de la diversité dans le batch (sinon les 200 sont identiques).
brains_liste = []
for i in range(BATCH_SIZE):
    b = Brain(obs_size, action_size).to(device)
    if donnees['brain_weights'] is not None:
        b.load_state_dict(donnees['brain_weights'])
        if i > 0:  # on garde le premier strictement identique au champion
            with torch.no_grad():
                for p in b.parameters():
                    p.add_(torch.randn_like(p) * 0.01)
    brains_liste.append(b)

params, buffers = stack_module_state(brains_liste)
params = {k: nn.Parameter(v) for k, v in params.items()}
brain_architecture = copy.deepcopy(brains_liste[0]).to(device)


def fmodel(parametres, tampons, observation):
    return functional_call(brain_architecture, (parametres, tampons), (observation,))


brain_batch = vmap(fmodel, in_dims=(0, 0, 0))

# ⚡️ L'OPTIMISEUR EST CRÉÉ UNE SEULE FOIS, HORS DE TOUTE BOUCLE
optimizer = torch.optim.Adam(params.values(), lr=LEARNING_RATE)

os.makedirs("champion_raffine", exist_ok=True)

meilleur_score_global = float('-inf')
meilleurs_poids = None


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
            obs_flat = obs.reshape(BATCH_SIZE, obs_size)

            action_flat = brain_batch(params, buffers, obs_flat)
            action = action_flat.reshape(1, BATCH_SIZE, action_size)

            bruit = torch.randn_like(action) * bruit_scale
            mega.apply_action(action + bruit, frame)

            reward_step = mega.get_reward(COEF_ENERGIE, COEF_HAUTEUR)   # [1, BATCH]
            rewards_accumulated = reward_step if rewards_accumulated is None else rewards_accumulated + reward_step

        # Troncature du gradient : évite une chaîne de backprop de 3000 pas
        if frame % 60 == 0 and frame > 0:
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
        optimizer.zero_grad(set_to_none=True)
        if rewards_accumulated is not None:
            del rewards_accumulated
        if device.type == "cuda":
            torch.cuda.empty_cache()
        continue

    loss = -torch.sum(rewards_accumulated) / BATCH_SIZE

    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(params.values(), max_norm=1.0)
    optimizer.step()

    # --- Suivi et sauvegarde du meilleur ---
    scores = rewards_accumulated[0]                      # [BATCH]
    best_idx = torch.argmax(scores).item()
    best_score = scores[best_idx].item()

    if best_score > meilleur_score_global:
        meilleur_score_global = best_score
        meilleurs_poids = {k: v[best_idx].detach().clone() for k, v in params.items()}

    if episode % 10 == 0:
        nb_n = torch.clamp(torch.sum(mega.mask_N_exp, dim=2), min=1.0)
        pos_fin = torch.sum(mega.X * mega.mask_N_exp, dim=2) / nb_n
        distance_finale = (pos_fin - pos_depart).mean().item()
        print(f"Ép. {episode:4d} | score max: {best_score:8.2f} | "
              f"moyenne: {scores.mean().item():8.2f} | "
              f"distance moy: {distance_finale:7.2f} | bruit: {bruit_scale:.4f}")

    if episode % SAUVEGARDE_TOUS_LES == 0 and meilleurs_poids is not None:
        torch.save({
            'x': x, 'y': y, 'is_bone': is_bone,
            'muscle1': torch.tensor(muscle1, dtype=torch.long),
            'muscle2': torch.tensor(muscle2, dtype=torch.long),
            'stiffness': torch.tensor([1.0 + 4 * s for s in is_bone], dtype=torch.float32),
            'target_length': target_length,
            'brain_weights': meilleurs_poids,
            'max_noeuds': max_noeuds, 'max_muscles': max_muscles
        }, f"champion_raffine/raffine_ep{episode}_score_{meilleur_score_global:.1f}.pt")


# ==========================================
# 💾 SAUVEGARDE FINALE
# ==========================================
if meilleurs_poids is not None:
    chemin_final = f"champion_raffine/FINAL_score_{meilleur_score_global:.1f}.pt"
    torch.save({
        'x': x, 'y': y, 'is_bone': is_bone,
        'muscle1': torch.tensor(muscle1, dtype=torch.long),
        'muscle2': torch.tensor(muscle2, dtype=torch.long),
        'stiffness': torch.tensor([1.0 + 4 * s for s in is_bone], dtype=torch.float32),
        'target_length': target_length,
        'brain_weights': meilleurs_poids,
        'max_noeuds': max_noeuds, 'max_muscles': max_muscles
    }, chemin_final)
    print(f"\n🏆 Terminé — meilleur score : {meilleur_score_global:.2f}")
    print(f"💾 Sauvegardé dans {chemin_final}")
else:
    print("\n⚠️ Aucun épisode valide (que des explosions) — vérifie la physique.")