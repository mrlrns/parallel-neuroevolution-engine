import os
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

import torch


from individu import Individu
from brain import Brain
from megaVecto import MegaCrea

import random
import hashlib
import copy
from torch.func import stack_module_state, functional_call, vmap
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device utilisé : {device}")

SUB_STEP = 10
BATCH_SIZE = 30
dt = torch.tensor(1.0 / SUB_STEP, device=device)


def generer_topologie(i):
    x = []
    y = []
    axe_central_y = 400.0

    # Colonne plus longue : permet une ondulation
    nb_centraux = random.randint(3, 5)
    for k in range(nb_centraux):
        x.append(50 + k * (200 / max(1, nb_centraux - 1)))  # espacement régulier
        y.append(axe_central_y)

    nb_haut = random.randint(2, 3)
    for _ in range(nb_haut):
        x.append(random.uniform(50, 250))
        y.append(random.uniform(250, 350))

    for i2 in range(nb_centraux, nb_centraux + nb_haut):
        x.append(x[i2])
        y.append(axe_central_y + (axe_central_y - y[i2]))

    links = set()
    is_bone_dict = {}

    # --- Colonne vertébrale : majoritairement en OS pour la structure ---
    for i2 in range(nb_centraux - 1):
        link = (i2, i2 + 1)
        links.add(link)
        is_bone_dict[link] = 1.0 if random.random() < 0.7 else 0.0

    # --- Membres : chaque nœud haut s'attache à DEUX centraux -> triangle ---
    for i2 in range(nb_centraux, nb_centraux + nb_haut):
        attache = random.randint(0, nb_centraux - 2)  # -2 pour pouvoir prendre attache+1

        if random.random() < 0.5:
            attaches = (attache, attache + 1)   # rigide
        else:
            attaches = (attache,)              # articulé
            
        for a in attaches:
            link = (min(a, i2), max(a, i2))
            links.add(link)
            is_bone_dict[link] = 1.0 if random.random() < 0.3 else 0.0
    # --- Miroir vers le bas ---
    liens_a_ajouter = {}
    for (n1, n2), type_os in is_bone_dict.items():
        bas1 = n1 if n1 < nb_centraux else n1 + nb_haut
        bas2 = n2 if n2 < nb_centraux else n2 + nb_haut
        if bas1 != n1 or bas2 != n2:
            link_bas = (min(bas1, bas2), max(bas1, bas2))
            liens_a_ajouter[link_bas] = type_os

    for lien, type_os in liens_a_ajouter.items():
        links.add(lien)
        is_bone_dict[lien] = type_os

    muscle1l = [l[0] for l in links]
    muscle2l = [l[1] for l in links]
    is_bone = [is_bone_dict[l] for l in links]

    # --- Garde-fous des DEUX côtés ---
    nb_os = sum(is_bone)
    if nb_os == len(is_bone):          # pas 100% os
        is_bone[random.randrange(len(is_bone))] = 0.0
    if nb_os < len(is_bone) * 0.2:     # au moins 20% d'os
        indices_muscles = [k for k, b in enumerate(is_bone) if b == 0.0]
        random.shuffle(indices_muscles)
        cible = int(len(is_bone) * 0.2) - int(nb_os)
        for k in indices_muscles[:max(0, cible)]:
            is_bone[k] = 1.0

    return Individu(x, y, muscle1l, muscle2l, is_bone, i)


def preparer_mega_univers(Population, device):
    liste_X, liste_Y = [], []
    liste_m1, liste_m2 = [], []
    liste_stiff, liste_is_bone, liste_base_length = [], [], []
    tailles_noeuds, tailles_muscles = [], []

    for crea in Population:
        x_t = torch.tensor(crea.x, dtype=torch.float32)
        y_t = torch.tensor(crea.y, dtype=torch.float32)
        m1_t = torch.tensor(crea.muscle1, dtype=torch.long)
        m2_t = torch.tensor(crea.muscle2, dtype=torch.long)
        is_bone_t = torch.tensor(crea.is_bone, dtype=torch.float32)

        liste_X.append(x_t)
        liste_Y.append(y_t)
        liste_m1.append(m1_t)
        liste_m2.append(m2_t)
        liste_is_bone.append(is_bone_t)
        liste_stiff.append(1.0 + 4 * is_bone_t)
        liste_base_length.append(
            torch.hypot(x_t[m1_t] - x_t[m2_t], y_t[m1_t] - y_t[m2_t])
        )

        tailles_noeuds.append(len(crea.x))
        tailles_muscles.append(len(crea.muscle1))

    Mega_X = pad_sequence(liste_X, batch_first=True, padding_value=0.0).to(device)
    Mega_Y = pad_sequence(liste_Y, batch_first=True, padding_value=0.0).to(device)
    Mega_m1 = pad_sequence(liste_m1, batch_first=True, padding_value=0).to(device)
    Mega_m2 = pad_sequence(liste_m2, batch_first=True, padding_value=0).to(device)
    Mega_stiff = pad_sequence(liste_stiff, batch_first=True, padding_value=0.0).to(device)
    Mega_is_bone = pad_sequence(liste_is_bone, batch_first=True, padding_value=0.0).to(device)
    Mega_base_length = pad_sequence(liste_base_length, batch_first=True, padding_value=0.0).to(device)

    tailles_noeuds_t = torch.tensor(tailles_noeuds, device=device)
    max_noeuds = Mega_X.shape[1]
    Masque_Noeuds = (torch.arange(max_noeuds, device=device) < tailles_noeuds_t.unsqueeze(1)).float()

    tailles_muscles_t = torch.tensor(tailles_muscles, device=device)
    max_muscles = Mega_m1.shape[1]
    Masque_Muscles = (torch.arange(max_muscles, device=device) < tailles_muscles_t.unsqueeze(1)).float()

    return {
        "X": Mega_X, "Y": Mega_Y, "m1": Mega_m1, "m2": Mega_m2,
        "stiffness": Mega_stiff, "is_bone": Mega_is_bone, "base_length": Mega_base_length,
        "masque_noeuds": Masque_Noeuds, "masque_muscles": Masque_Muscles
    }


if __name__ == '__main__':

    POP_SIZE = 50
    Population = [generer_topologie(i) for i in range(POP_SIZE)]
    os.makedirs("elite_mutant", exist_ok=True)

    for generation in range(30):
        print(f"\n🚀 GÉNÉRATION {generation}")

        # --- Préparation du méga-univers pour cette génération ---
        dico = preparer_mega_univers(Population, device)
        MAX_NOEUDS = dico["masque_noeuds"].shape[1]
        MAX_MUSCLES = dico["masque_muscles"].shape[1]
        obs_size = MAX_NOEUDS * 4 + MAX_MUSCLES + 1
        action_size = MAX_MUSCLES

        frame_nb = 200
        nb_episodes = 30 if generation < 3 else 20
        if generation < 15:
            # Phase d'exploration : Énergie gratuite et on tolère les rebonds
            coef_energie = 10000.0 
            coef_hauteur = 0.0      # Aucune punition si elle rebondit
        else:
            # Phase d'optimisation : On punit la dépense et le mal de mer
            coef_energie = 300.0 
            coef_hauteur = 10.0
        if generation == 15:
          for c in Population:
            c.best_score = float('-inf')
        # --- Création des POP_SIZE * BATCH_SIZE cerveaux ---
        brains_liste = []
        for creature in Population:
            creature.score_generation = float('-inf')
            # ⚡️ L'ADAPTATEUR DE CERVEAU ⚡️
            if creature.brain_weights is not None:
                old_w = creature.brain_weights
                old_obs = old_w['layer1.weight'].shape[1]
                old_act = old_w['layer3.bias'].shape[0]

                # Si la taille de l'univers a changé (mutation d'un individu)
                if old_obs != obs_size or old_act != action_size:
                    new_w = {}
                    for k, v in old_w.items():
                        if k == 'layer1.weight':
                            # On retaille la couche d'entrée (observations)
                            new_tensor = torch.zeros((64, obs_size), device=device)
                            min_o = min(old_obs, obs_size)
                            new_tensor[:, :min_o] = v[:, :min_o]
                            new_w[k] = new_tensor
                            
                        elif k == 'layer3.weight':
                            # On retaille la couche de sortie (actions)
                            new_tensor = torch.zeros((action_size, 64), device=device)
                            min_a = min(old_act, action_size)
                            new_tensor[:min_a, :] = v[:min_a, :]
                            new_w[k] = new_tensor
                            
                        elif k == 'layer3.bias':
                            # On retaille le biais de sortie
                            new_tensor = torch.zeros((action_size,), device=device)
                            min_a = min(old_act, action_size)
                            new_tensor[:min_a] = v[:min_a]
                            new_w[k] = new_tensor
                            
                        else:
                            # Les couches cachées ne changent pas de taille
                            new_w[k] = v.clone()
                            
                    creature.brain_weights = new_w
            # --------------------------------

            for _ in range(BATCH_SIZE):
                b = Brain(obs_size, action_size).to(device)
                if creature.brain_weights is not None:
                    b.load_state_dict(creature.brain_weights)
                brains_liste.append(b)

        params, buffers = stack_module_state(brains_liste)
        params = {k: nn.Parameter(v) for k, v in params.items()}
        brain_architecture = copy.deepcopy(brains_liste[0]).to(device)

        def fmodel(parametres, tampons, observation):
            return functional_call(brain_architecture, (parametres, tampons), (observation,))

        brain_batch = vmap(fmodel, in_dims=(0, 0, 0))
        
        optimizer = torch.optim.Adam(params.values(), lr=0.001)

        for episode in range(nb_episodes):
            mega = MegaCrea(dico, BATCH_SIZE, device=device)

            rewards_accumulated = None
            explosion = False

            for frame in range(frame_nb):
                if frame % 5 == 0:
                    obs = mega.get_observation(frame)  # [POP, BATCH, obs_size]
                    obs_flat = obs.reshape(POP_SIZE * BATCH_SIZE, obs_size)

                    action_flat = brain_batch(params, buffers, obs_flat)
                    action = action_flat.reshape(POP_SIZE, BATCH_SIZE, action_size)

                    bruit = torch.randn_like(action) * 0.02
                    mega.apply_action(action + bruit,frame)

                    reward_step = mega.get_reward(coef_energie,coef_hauteur)  # [POP, BATCH]
                    rewards_accumulated = reward_step if rewards_accumulated is None else rewards_accumulated + reward_step

                
                for _ in range(SUB_STEP):
                    mega.apply_physics(dt)

                if frame % 20 == 0:
                    if torch.isnan(mega.X).any() or torch.isnan(mega.Y).any():
                        explosion = True
                        break

                if frame % 20 == 0 and frame > 0:
                    mega.X = mega.X.detach()
                    mega.Y = mega.Y.detach()
                    mega.vX = mega.vX.detach()
                    mega.vY = mega.vY.detach()
            if explosion:
                print(f"  💥 Explosion détectée à l'épisode {episode}, on saute.")
                # set_to_none=True est plus efficace pour libérer la mémoire
                optimizer.zero_grad(set_to_none=True)
                
                # 🧹 LA SÉCURITÉ MÉMOIRE 🧹
                # On détruit le graphe de calcul accumulé pour libérer la VRAM
                if rewards_accumulated is not None:
                    del rewards_accumulated
                    rewards_accumulated = None
                torch.cuda.empty_cache() 
                
                continue

            loss = -torch.sum(rewards_accumulated) / (POP_SIZE * BATCH_SIZE)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params.values(), max_norm=1.0)
            optimizer.step()

            # --- Sélection du champion pour CHAQUE créature de la population ---
            params_reshaped = {
                k: v.reshape(POP_SIZE, BATCH_SIZE, *v.shape[1:]) for k, v in params.items()
            }

            for p, creature in enumerate(Population):
                scores_creature = rewards_accumulated[p]
                best_idx = torch.argmax(scores_creature).item()
                best_score = scores_creature[best_idx].item()

                # Suivi de la performance de CETTE génération (pour le tri)
                if best_score > creature.score_generation:
                    creature.score_generation = best_score

                if best_score > creature.best_score:
                    creature.best_score = best_score
                    creature.brain_weights = {
                        k: v[p, best_idx].detach().clone() for k, v in params_reshaped.items()
                    }

            if episode % 5 == 0:
                meilleur = max(c.best_score for c in Population)
                moyenne_episode = torch.mean(rewards_accumulated).item() 
                print(f"  Épisode {episode} — meilleur score population: {meilleur:.2f} | moyenne CET épisode: {moyenne_episode:.2f}")

        # --- Fin de génération : tri, sauvegarde, mutation ---
        Population.sort(key=lambda c: c.score_generation, reverse=True)
        champion = Population[0]
        print(f"✅ FIN GÉNÉRATION {generation} — 🏆 Champion score: {champion.best_score:.2f} (famille {champion.family})")

        x_base = torch.tensor(champion.x, dtype=torch.float32)
        y_base = torch.tensor(champion.y, dtype=torch.float32)
        m1 = torch.tensor(champion.muscle1, dtype=torch.long)
        m2 = torch.tensor(champion.muscle2, dtype=torch.long)
        stiff = torch.tensor([1.0 + 4 * s for s in champion.is_bone], dtype=torch.float32)
        t_len = torch.hypot(x_base[m1] - x_base[m2], y_base[m1] - y_base[m2])

        adn_texte = str(champion.x) + str(champion.y) + str(champion.is_bone)
        empreinte = hashlib.md5(adn_texte.encode()).hexdigest()[:6]
        nom_fichier = f"champion_gen_{generation}_score_{champion.best_score:.1f}_family_{champion.family}.pt"

        torch.save({
            'x': champion.x, 'y': champion.y, 'is_bone': champion.is_bone,
            'muscle1': m1, 'muscle2': m2, 'stiffness': stiff, 'target_length': t_len,
            'brain_weights': champion.brain_weights,
            'max_noeuds': MAX_NOEUDS, 'max_muscles': MAX_MUSCLES
        }, os.path.join("elite_mutant", nom_fichier))

        moitie = len(Population) // 2
        survivants = Population[:moitie]
        new_population = list(survivants)
        for creature in survivants:
            child = creature.cloner()
            child.mutate(rate_new_node=0.1, rate_mut_length=0.3, rate_change_bone=0.2)
            child.best_score = float('-inf')
            new_population.append(child)
        Population = new_population