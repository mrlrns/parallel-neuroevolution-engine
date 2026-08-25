import copy
import random

class Individu:
    def __init__(self, x, y, muscle1, muscle2, is_bone,family, brain_weights=None):
        # --- 1. L'ADN Morphologique ---
        self.x = x
        self.y = y
        self.muscle1 = muscle1
        self.muscle2 = muscle2
        self.is_bone = is_bone
        self.family=family
        
        # --- 2. L'Héritage Cérébral ---
        # Si on crée une créature de zéro, elle n'a pas de poids (None).
        # Si c'est un enfant, il recevra les poids de son parent.
        self.brain_weights = brain_weights
        
        # --- 3. L'Évaluation ---
        # On initialise le score à l'infini négatif pour que toute 
        # première tentative devienne le "meilleur" score.
        self.best_score = float('-inf')
        self.score_generation = float('-inf')

    def cloner(self):
        """
        Crée une copie stricte et indépendante de cet individu.
        C'est vital pour la mutation : on veut muter l'enfant, pas modifier le parent !
        """
        return copy.deepcopy(self)
    def mutate(self,rate_new_node,rate_mut_length,rate_change_bone):
        if random.random()<rate_new_node:
            xl=random.uniform(50,250)
            yl=random.uniform(200,350)
            self.x.append(xl)
            self.y.append(yl)
            self.x.append(xl)
            self.y.append(800-yl)

            # On enregistre leurs futurs index (qui seront à la fin de la liste)
            index_nouveau_haut = len(self.x)-2
            index_nouveau_bas = len(self.x) - 1

            # 2. Choisir une attache valide en haut (Y <= 400)
            noeuds_eligibles_haut = [i for i, y_val in enumerate(self.y) if y_val <= 400]
            attache_haut = random.choice(noeuds_eligibles_haut)

            # 3. Trouver l'attache du bas correspondante
            if self.y[attache_haut] == 400:
            # L'attache est sur l'axe central : le haut et le bas s'y connectent !
                attache_bas = attache_haut
            else:
                # L'attache est un noeud supérieur : on calcule les coordonnées de son miroir
                x_cible = self.x[attache_haut]
                y_cible = 400 + (400 - self.y[attache_haut])
    
                # On cherche le noeud existant qui possède ces coordonnées
                attache_bas = None
                for i in range(len(self.x)):
                # On utilise une petite marge (0.1) au cas où il y aurait des arrondis flottants
                    if abs(self.x[i] - x_cible) < 0.1 and abs(self.y[i] - y_cible) < 0.1:
                        attache_bas = i
                        break

            # 4. Créer les connexions physiques
            type_lien = random.choice([0.0, 1.0]) # 0.0 = Muscle, 1.0 = Os

            # Connecter le membre haut
            self.muscle1.append(attache_haut)
            self.muscle2.append(index_nouveau_haut)
            self.is_bone.append(type_lien)

            # Connecter le membre bas (seulement si on a bien trouvé le miroir)
            if attache_bas is not None:
                self.muscle1.append(attache_bas)
                self.muscle2.append(index_nouveau_bas)
                self.is_bone.append(type_lien)
            
        if random.random()<rate_mut_length:
            index_to_mutate_1=random.randint(0,len(self.x)-1)

            if self.y[index_to_mutate_1]!=400:
                target_y=800-self.y[index_to_mutate_1]

                target_x=self.x[index_to_mutate_1]

                for i in range (len(self.x)):
                    if abs(self.x[i] - target_x) < 0.1 and abs(self.y[i] - target_y) < 0.1:
                        index_to_mutate_2=i

                mut_x=random.uniform(-10,10)
                mut_y=random.uniform(-10,10)

                self.x[index_to_mutate_1]+=mut_x
                self.x[index_to_mutate_2]+=mut_x
                self.y[index_to_mutate_1]+=mut_y
                self.y[index_to_mutate_2]-=mut_y
            else:

                mut_x=random.uniform(-10,10)
                

                self.x[index_to_mutate_1]+=mut_x




        if random.random()<rate_change_bone:
            
            index=random.randint(0,len(self.muscle1)-1)

            if self.y[self.muscle1[index]] ==400 and self.y[self.muscle2[index]]==400 :
                self.is_bone[index] = (self.is_bone[index]+1)%2

            else:
                A_node=[self.x[self.muscle1[index]],self.y[self.muscle1[index]]]
                B_node=[self.x[self.muscle2[index]],self.y[self.muscle2[index]]]

                A_mirror=(A_node[0],800-A_node[1])
                B_mirror=(B_node[0],800-B_node[1])

                index_A = None
                index_B = None

                for i in range (len(self.x)):
                    if abs(self.x[i] - A_mirror[0]) < 0.1 and abs(self.y[i] - A_mirror[1]) < 0.1:
                        index_A=i
                        
                    if abs(self.x[i] - B_mirror[0]) < 0.1 and abs(self.y[i] - B_mirror[1]) < 0.1:
                        index_B=i
                        
                index_mirror=None
                if index_A is not None and index_B is not None:
                    for i in range (len(self.muscle1)):
                        if (min(self.muscle1[i],self.muscle2[i]),max(self.muscle1[i],self.muscle2[i]))==(min(index_A,index_B),max(index_A,index_B)):
                            index_mirror=i
                            break
                        
                    self.is_bone[index] = (self.is_bone[index]+1)%2
                    self.is_bone[index_mirror] = (self.is_bone[index_mirror]+1)%2

        