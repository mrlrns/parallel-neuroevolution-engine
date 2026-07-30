import pygame
import sys
import torch 

from brain import Brain
from creature_vecto import Creature_vecto # BOUM ! Voici ton import !
import math
import cv2
import numpy as np
import os
import random


SUB_STEP=10
dt=torch.tensor(1.0)/torch.tensor(SUB_STEP)
pygame.init()
ecran = pygame.display.set_mode((1200,800), pygame.RESIZABLE)
horloge = pygame.time.Clock()
largeur = ecran.get_width()
hauteur = ecran.get_height()


camera_x = 0
camera_y = 0
souris_clic_gauche = False
derniere_pos_souris = (0, 0)

NOIR = (30, 30, 30)
BLANC = (255, 255, 255)
ROUGE_MUSCLE = (200, 50, 50)
BLEU=(50, 50, 200)

chemin_fichier = "mutants_d_elite/champion_ADN_b75938_score_24.pt" 
donnees = torch.load(chemin_fichier)

print("✅ Sauvegarde chargée ! Reconstruction du mutant...")

# On récupère l'ADN
x = donnees['x']
y = donnees['y']
is_bone = donnees['is_bone']
muscle1 = donnees['muscle1']
muscle2 = donnees['muscle2']
stiffness = donnees['stiffness']
target_length = donnees['target_length']

# On reconstruit le cerveau à l'identique
cerveau = Brain(int(len(x)*4+(len(x)-sum(is_bone))+1),int(len(x)-sum(is_bone)))
cerveau.load_state_dict(donnees['brain_weights'])
cerveau.eval() # On dit à PyTorch : "Mode lecture seule, pas d'entraînement !"

# ==========================================
# 🧬 ÉTAPE 2 : LA RECONSTRUCTION PHYSIQUE
# ==========================================



tenseurs_os = torch.tensor(is_bone, dtype=torch.float32)

champion = Creature_vecto(x, y, muscle1, muscle2, stiffness, target_length, tenseurs_os, cerveau,torch.device("cpu"))
os.makedirs("videos_ia_(5FRAMES)", exist_ok=True)


fourcc = cv2.VideoWriter_fourcc(*'mp4v') # Format MP4
chemin_video = f"videos_ia_(5FRAMES)/episode_15FRAMES.mp4"
# Attention : La taille (1200, 800) doit correspondre EXACTEMENT à ta fenêtre Pygame !
video_writer = cv2.VideoWriter(chemin_video, fourcc, 60.0, (largeur, hauteur))
with torch.no_grad(): # Pas de calcul de gradients pendant l'affichage !
    for frame in range (1000):

        touches = pygame.key.get_pressed()
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                video_writer.release()
                pygame.quit()
                sys.exit()

            elif event.type == pygame.MOUSEBUTTONDOWN:
                souris_clic_gauche = True
                derniere_pos_souris = event.pos

            elif event.type == pygame.MOUSEBUTTONUP:
                souris_clic_gauche = False

            elif event.type == pygame.MOUSEMOTION :
                if souris_clic_gauche:
                    dx = event.pos[0] - derniere_pos_souris[0]
                    dy = event.pos[1] - derniere_pos_souris[1]
                    camera_x += dx
                    camera_y += dy
                    derniere_pos_souris = event.pos
    
    

        if touches[pygame.K_ESCAPE]:
            video_writer.release()
            pygame.quit()
            sys.exit()

        if frame%5==0:


            obs=champion.get_observation(frame)
            action=champion.cerveau.forward(obs)
            
        
        
        champion.apply_action(action)

        # --- 1. MISE À JOUR DE LA PHYSIQUE ---
        # Fausse gravité pour faire tomber le premier noeud
        for _ in range (SUB_STEP):
        
            champion.appliquer_physique(dt)
        

    # --- 2. AFFICHAGE (PYGAME) ---
        
        ecran.fill(NOIR)
        
        # 1. On extrait les tenseurs en listes Python classiques (pour Pygame)
        X_list = champion.X.tolist()
        Y_list = champion.Y.tolist()
        m1_list = champion.muscle1.tolist()
        m2_list = champion.muscle2.tolist()
        os_list = champion.is_bone.tolist()

        # 2. Dessin des Muscles et des Os
        # On boucle sur la quantité totale de liaisons
        for i in range(len(m1_list)):
            # On récupère les index des deux noeuds connectés par ce muscle
            idx1 = m1_list[i]
            idx2 = m2_list[i]
            
            # On calcule leurs positions (en int pour Pygame)
            pos1 = (int(X_list[idx1] + camera_x), int(Y_list[idx1] + camera_y))
            pos2 = (int(X_list[idx2] + camera_x), int(Y_list[idx2] + camera_y))
            
            # On choisit la couleur (Blanc si os_list vaut 1.0, Rouge sinon)
            couleur = BLANC if os_list[i] == 1.0 else ROUGE_MUSCLE
            
            pygame.draw.line(ecran, couleur, pos1, pos2, 4)

        # 3. Dessin des Noeuds par-dessus
        for i in range(len(X_list)):
            pos_noeud = (int(X_list[i] + camera_x), int(Y_list[i] + camera_y))
            pygame.draw.circle(ecran, BLANC, pos_noeud, 12)
    
        pygame.display.flip()
        horloge.tick(60)
        # 1. On aspire les pixels de l'écran Pygame dans une matrice Numpy
        matrice_ecran = pygame.surfarray.array3d(ecran)
        
        # 2. Pygame range les pixels en (X, Y, Rouge-Vert-Bleu). 
        # OpenCV veut du (Y, X, Bleu-Vert-Rouge). On doit donc pivoter l'image !
        matrice_ecran = np.transpose(matrice_ecran, (1, 0, 2))
        image_bgr = cv2.cvtColor(matrice_ecran, cv2.COLOR_RGB2BGR)
        
        # 3. On ajoute l'image au film
        video_writer.write(image_bgr)
    video_writer.release()
    print("🎬 Vidéo de démonstration sauvegardée avec succès !")
    

    

    
    