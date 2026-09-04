import os
##os.environ["SDL_VIDEODRIVER"] = "dummy"

import pygame
import sys
import torch
import torch.nn.functional as F

from brain import Brain
from megaVecto import MegaCrea
import cv2
import numpy as np
import os

SUB_STEP = 10
dt = torch.tensor(1.0 / SUB_STEP)
device = torch.device("cpu")

pygame.init()
ecran = pygame.display.set_mode((1200, 800), pygame.RESIZABLE)
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

# Champion issu de train.py     -> elite_mutant/
# Champion raffiné par train2.py -> champion_raffine/
chemin_fichier = "champion_raffine/raffine_ep100_score_427.9.pt"

donnees = torch.load(chemin_fichier, map_location=device)

print("✅ Sauvegarde chargée ! Reconstruction du mutant...")

x = donnees['x']
y = donnees['y']
is_bone = donnees['is_bone']
muscle1 = donnees['muscle1'].tolist() if torch.is_tensor(donnees['muscle1']) else donnees['muscle1']
muscle2 = donnees['muscle2'].tolist() if torch.is_tensor(donnees['muscle2']) else donnees['muscle2']
target_length = donnees['target_length']  # longueur de base réelle du champion
max_noeuds = donnees['max_noeuds']
max_muscles = donnees['max_muscles']

num_nodes_reel = len(x)
num_muscles_reel = len(muscle1)


def pad_1d(vec, size, dtype=torch.float32, pad_value=0.0):
    t = torch.tensor(vec, dtype=dtype)
    if len(vec) < size:
        pad = torch.full((size - len(vec),), pad_value, dtype=dtype)
        t = torch.cat([t, pad])
    return t.unsqueeze(0)  # ajoute la dimension POP_SIZE=1


# --- Reconstruction du dico "méga-univers" pour UNE seule créature ---
X_pad = pad_1d(x, max_noeuds)
Y_pad = pad_1d(y, max_noeuds)
m1_pad = pad_1d(muscle1, max_muscles, dtype=torch.long, pad_value=0)
m2_pad = pad_1d(muscle2, max_muscles, dtype=torch.long, pad_value=0)
is_bone_pad = pad_1d(is_bone, max_muscles, pad_value=0.0)
stiff_pad = pad_1d([1.0 + 4 * s for s in is_bone], max_muscles, pad_value=0.0)
base_length_pad = pad_1d(target_length.tolist(), max_muscles, pad_value=0.0)

masque_noeuds = (torch.arange(max_noeuds) < num_nodes_reel).float().unsqueeze(0)
masque_muscles = (torch.arange(max_muscles) < num_muscles_reel).float().unsqueeze(0)

dico = {
    "X": X_pad, "Y": Y_pad, "m1": m1_pad, "m2": m2_pad,
    "stiffness": stiff_pad, "is_bone": is_bone_pad, "base_length": base_length_pad,
    "masque_noeuds": masque_noeuds, "masque_muscles": masque_muscles
}

champion = MegaCrea(dico, batch_size=1, device=device)

# --- Reconstruction du cerveau avec les BONNES dimensions (celles de l'entraînement) ---
obs_size = max_noeuds * 4 + max_muscles + 1
action_size = max_muscles

cerveau = Brain(obs_size, action_size)
cerveau.load_state_dict(donnees['brain_weights'])
cerveau.eval()

os.makedirs("videos_ia_(5FRAMES)", exist_ok=True)
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
chemin_video = f"videos_ia_(5FRAMES)/episode_15FRAMES2.mp4"
video_writer = cv2.VideoWriter(chemin_video, fourcc, 60.0, (largeur, hauteur))

with torch.no_grad():
    for frame in range(1000):

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
            elif event.type == pygame.MOUSEMOTION:
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

        if frame % 5 == 0:
            obs = champion.get_observation(frame)          # [1, 1, obs_size]
            obs_flat = obs.reshape(1, obs_size)             # aplati pour le forward classique
            action = cerveau(obs_flat)  
            print (action.mean())                    # [1, action_size]
            action = action.reshape(1, 1, action_size)       # reforme pour apply_action
            champion.apply_action(action,frame)

        for _ in range(SUB_STEP):
            champion.apply_physics(dt)

        # --- AFFICHAGE (PYGAME) ---
        ecran.fill(NOIR)

        # On ne garde que les vrais noeuds/muscles (le padding est toujours en fin de tableau)
        X_list = champion.X[0, 0, :num_nodes_reel].tolist()
        Y_list = champion.Y[0, 0, :num_nodes_reel].tolist()
        m1_list = muscle1
        m2_list = muscle2
        os_list = is_bone

        for i in range(num_muscles_reel):
            idx1 = m1_list[i]
            idx2 = m2_list[i]

            pos1 = (int(X_list[idx1] + camera_x), int(Y_list[idx1] + camera_y))
            pos2 = (int(X_list[idx2] + camera_x), int(Y_list[idx2] + camera_y))

            couleur = BLANC if os_list[i] == 1.0 else ROUGE_MUSCLE
            pygame.draw.line(ecran, couleur, pos1, pos2, 4)

        for i in range(num_nodes_reel):
            pos_noeud = (int(X_list[i] + camera_x), int(Y_list[i] + camera_y))
            pygame.draw.circle(ecran, BLANC, pos_noeud, 12)

        pygame.display.flip()
        horloge.tick(60)

        matrice_ecran = pygame.surfarray.array3d(ecran)
        matrice_ecran = np.transpose(matrice_ecran, (1, 0, 2))
        image_bgr = cv2.cvtColor(matrice_ecran, cv2.COLOR_RGB2BGR)
        video_writer.write(image_bgr)

video_writer.release()
print("🎬 Vidéo de démonstration sauvegardée avec succès !")