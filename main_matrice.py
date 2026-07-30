import sys
import torch 
from brain import Brain
from crea_vecto2 import Creature_vecto 
import math

import numpy as np
import os
import random
import hashlib
import copy
from torch.func import stack_module_state, functional_call, vmap
import torch.nn as nn


# Vérifie si MPS est disponible (ce sera le cas sur ton M4)
if torch.backends.mps.is_available():
    device = torch.device("mps")
    print("🚀 Accélération Apple Silicon (MPS) activée !")
else:
    device = torch.device("cpu")
    print("⚠️ Attention, exécution sur CPU.")
    
SUB_STEP=10
BATCH_SIZE=100
dt=torch.tensor(1.0,device=device)/torch.tensor(SUB_STEP,device=device)





for creature in range(10):
    best_score = float('-inf')  # Initialisation du meilleur score à une valeur très basse
    
    score_max = 0.0

   

    num_nodes = random.randint(3, 6)
    x=[random.uniform(0,300) for i in range (num_nodes)]
    y=[random.uniform(800/2-150,800/2+150) for i in range (num_nodes)]





    connected_node=[0]
    unconnected_node=list(range(1,num_nodes))
    links=set()
    while len(unconnected_node)>0:
        n1=random.choice(connected_node)
        n2=random.choice(unconnected_node)
        links.add((min(n1, n2), max(n1, n2)))
        connected_node.append(n2)
        unconnected_node.remove(n2)

    # 1. On calcule le maximum de links physiquement possibles : N*(N-1)/2
    max_links_possibles = int(num_nodes * (num_nodes - 1) / 2)
    
    # 2. On choisit une cible réaliste
    # On veut ajouter entre 1 et num_nodes links, mais SANS DÉPASSER le max possible
    cible_totale = min(max_links_possibles, len(links) + random.randint(1, num_nodes))
    while len(links) < cible_totale:
        noeud_A = random.randint(0, num_nodes - 1)
        noeud_B = random.randint(0, num_nodes - 1)
        
        if noeud_A != noeud_B:
            link = (min(noeud_A, noeud_B), max(noeud_A, noeud_B))
            links.add(link)
    muscle1l = []
    muscle2l = []
    for link in links:
        muscle1l.append(link[0])
        muscle2l.append(link[1])



    is_bone=[random.choice([0.0,1.0]) for _ in range(len(muscle1l))]
    

    if sum(is_bone)==len(is_bone):
        is_bone[random.randint(0,len(is_bone)-1)]=0.0

    nb_muscles = int(len(is_bone) - sum(is_bone))

    obs_size = int(len(x) * 4 + nb_muscles + 1)

    brains_liste=[Brain(obs_size,nb_muscles) for _ in range(BATCH_SIZE)]
    params, buffers = stack_module_state(brains_liste)
    params = {k: nn.Parameter(v) for k, v in params.items()}
    cerveau_architecture = copy.deepcopy(brains_liste[0])


    optimizer=torch.optim.Adam(params.values(),lr=0.001)

        # Cette fonction fait un forward classique
    def fmodel(parametres, tampons, observation):
        return functional_call(cerveau_architecture, (parametres, tampons), (observation,))

 
    cerveau_batch = vmap(fmodel, in_dims=(0, 0, 0))
    os.makedirs("mutants_d_elite", exist_ok=True)
    for episode in range (1000):
        
        distance_parcourue = 0.0

        if episode==100 and score_max < 3.0 or episode==50 and score_max< 1.5 or episode==20 and score_max< 0.5 or episode==200 and score_max < 5.0:
            print("⚠️ Warning: Creature failed to reach score 1.0 after 200 episodes. Resetting with new topology!")
            break



    # --- CRÉATION DE NOTRE MONDE PHYSIQUE ---
    # On crée deux noeuds
        X = torch.tensor(x, dtype=torch.float32,device=device).unsqueeze(0).repeat(BATCH_SIZE, 1)+ torch.randn(BATCH_SIZE, num_nodes) * 2.0
        Y = torch.tensor(y, dtype=torch.float32,device=device).unsqueeze(0).repeat(BATCH_SIZE, 1)+ torch.randn(BATCH_SIZE, num_nodes) * 2.0
        muscle1=torch.tensor(muscle1l,dtype=torch.long,device=device)
        muscle2=torch.tensor(muscle2l,dtype=torch.long,device=device)
        siffness=torch.tensor([ 5.0-4*s for s in is_bone],dtype=torch.float32,device=device)
        x_base = torch.tensor(x, dtype=torch.float32,device=device)
        y_base = torch.tensor(y, dtype=torch.float32,device=device)
        length0=torch.hypot(x_base[muscle1]-x_base[muscle2],y_base[muscle1]-y_base[muscle2])
        bone_tensor=torch.tensor(is_bone,dtype=torch.float32,device=device)
    # On les relie avec un muscle (longueur cible: 150, siffness: 0.1)
        

        

        

        ma_creature=Creature_vecto(X,Y,muscle1,muscle2,siffness,length0,bone_tensor,None,device)
        pos_0=torch.mean(X)
        
        creature_explosee = False
        for frame in range (300):

            
            
                
        
        

            

            if frame%15==0:


                obs=ma_creature.get_observation(frame)
                action=cerveau_batch(params, buffers, obs)
                bruit = torch.randn_like(action) * 0.1
                ma_creature.apply_action(action+bruit)
            
            
            

            # --- 1. MISE À JOUR DE LA PHYSIQUE ---
            # Fausse gravité pour faire tomber le premier noeud
            for _ in range (SUB_STEP):
            
                ma_creature.appliquer_physique(dt)
                if torch.isnan(ma_creature.X).any() or torch.isnan(ma_creature.Y).any():
                    print("💥 NaN detected in physics simulation. Discarding episode!")
                    best_score = -9999.0 # On la disqualifie
                    creature_explosee = True
                    break # On casse la boucle des 300 frames pour arrêter les dégâts !
            if creature_explosee:
                break

        
                
                
            
        if creature_explosee:
            print("🛑 Episode cancelled to protect network weights.")
            optimizer.zero_grad() # Erasing corrupted memory
            continue # Starting next generation directly !
        
        mean_dist_batch = torch.mean(ma_creature.X) - pos_0
        print(f"Mean distance travelled by the batch : {mean_dist_batch.item():.2f}")
        rewards = ma_creature.get_reward()

        loss=-torch.sum(rewards)/100
        best_score_episode = torch.max(rewards).item()

        score_max = max(score_max, best_score_episode)
        print(f"Consumed energy : {torch.mean(ma_creature.energie).item():.2f}")

        if best_score_episode > best_score:
            best_score = best_score_episode
            
            if best_score > 7.0:
                print(f"Saving...")
                adn_texte = str(x) + str(y) + str(is_bone)
                empreinte = hashlib.md5(adn_texte.encode()).hexdigest()[:6]
                nom_fichier = f"champion_ADN_{empreinte}_score_.pt"
                chemin_complet = os.path.join("elite_mutant", nom_fichier)
            
                print(f"🌟 Elite mutant found ! Score: {best_score:.2f}")
                
                # On trouve QUI est le champion dans le batch (son numéro de ligne)
                index_champion = torch.argmax(rewards).item()
                
                # On extrait uniquement les poids de CE champion
                weight_champion = {k: v[index_champion].detach().clone() for k, v in params.items()}
                
                saving_box= {
                    'x': x, 'y': y, 'is_bone': is_bone, 'muscle1': muscle1,
                    'muscle2': muscle2, 'siffness': siffness, 'length0': length0,
                    'brain_weights': weight_champion # Saving the brain !
                }
                torch.save(saving_box, chemin_complet)

        # --- BACKPROPAGATION ---r
        optimizer.zero_grad()
        loss.backward()
        
        # On utilise params.values() et non plus cerveau.parameters()
        torch.nn.utils.clip_grad_norm_(params.values(), max_norm=1.0)
        
        # On vérifie les gradients directement dans le dictionnaire params !
        gradients = params['layer1.weight'].grad
        if gradients is None:
            print("🚨 ALARME : Le graphe est brisé ! L'IA n'apprend rien.")
        elif torch.sum(torch.abs(gradients)) == 0:
            print("⚠️ ATTENTION : Les gradients sont à zéro. La loss ne bouge pas.")
        else:
            print("✅ SUCCÈS : vmap backpropagation OK !")
            
        optimizer.step()
        print(f"Épisode {episode} terminé. Meilleur score actuel: {best_score:.2f}")