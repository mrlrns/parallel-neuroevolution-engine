import copy
import random
from collections import defaultdict

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
    def mutate(self,rate_new_node,rate_mut_length,rate_change_bone,rate_pop_node):
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

        if random.random()<rate_pop_node :
            test=self.cloner()
            to_be_removed=random.randint(0, len(self.x)-1)
            to_be_remove_mirror=None
            if abs(test.y[to_be_removed]-400)<=0.1 :
                return

            else :
                for i in range (len(test.x)):
                    if abs(test.x[to_be_removed]-test.x[i])<=0.1 and abs(test.y[to_be_removed]-(800-test.y[i]))<=0.1 :
                        to_be_remove_mirror=i
                        break
            if to_be_remove_mirror is None :
                return

            remove_list=[to_be_remove_mirror,to_be_removed]

            old_to_new = {}
            n = 0
            for i in range(len(self.x)):
                if i not in remove_list:
                    old_to_new[i] = n
                    n += 1

            test.x = [v for i, v in enumerate(test.x) if i not in remove_list]
            test.y = [v for i, v in enumerate(test.y) if i not in remove_list]


            muscle1_t=[]
            muscle2_t=[]
            is_bone_t=[]

            for k in range (len(test.muscle1)):
                if test.muscle1[k] not in remove_list and test.muscle2[k] not in remove_list:

                    muscle1_t.append(test.muscle1[k])
                    muscle2_t.append(test.muscle2[k])
                    is_bone_t.append(test.is_bone[k])
                    


            for k in range (len(muscle1_t)):
                muscle1_t[k]=old_to_new[muscle1_t[k]]
                muscle2_t[k]=old_to_new[muscle2_t[k]]

            voisins = defaultdict(list)
            for a, b in zip(muscle1_t, muscle2_t):
                voisins[a].append(b)
                voisins[b].append(a)

            vus = {0}
            file = [0]
            while file:
                n = file.pop()
                for voisin in voisins[n]:
                    if voisin not in vus:
                        vus.add(voisin)
                        file.append(voisin)
            connexe = (len(vus) == len(test.x))

            if len(test.x)< 5 or sum(1 for b in is_bone_t if b == 0.0) < 2:
                return
            if connexe :
                
                self.x = test.x
                self.y = test.y
                self.muscle1 = muscle1_t
                self.muscle2 = muscle2_t
                self.is_bone = is_bone_t

            
                                



        