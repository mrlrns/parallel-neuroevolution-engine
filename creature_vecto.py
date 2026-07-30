import torch

class Creature_vecto:
    def __init__(self,positionX,positionY,muscle1,muscle2,rigidite,longueur_base,is_os,cerveau):
        self.cerveau=cerveau
        
        self.X=torch.tensor(positionX,dtype=torch.float32)
        self.Y=torch.tensor(positionY,dtype=torch.float32)
        self.distance_precedente = torch.mean(self.X)
        self.vX=torch.zeros_like(self.X)
        self.vY=torch.zeros_like(self.Y)
        self.rigidite=torch.tensor(rigidite,dtype=torch.float32)
        self.masses=torch.ones_like(self.X)
        self.muscle1=torch.tensor(muscle1,dtype=torch.long)
        self.muscle2=torch.tensor(muscle2,dtype=torch.long)
        self.longueur0=torch.tensor(longueur_base,dtype=torch.float32)
        self.longueur_base=torch.tensor(longueur_base,dtype=torch.float32)
        self.c=2*torch.sqrt(self.rigidite)
        self.is_os = torch.tensor(is_os, dtype=torch.float32)
        self.energie=torch.tensor(0.0,dtype=torch.float32)
        self.hauteur_précédente = torch.mean(self.Y)

    def appliquer_physique(self,dt):
        X1 = self.X[self.muscle1] 
        X2 = self.X[self.muscle2]

        Y1 = self.Y[self.muscle1]
        Y2 = self.Y[self.muscle2]

        dx=X2-X1
        dy=Y2-Y1

        distances=torch.sqrt(torch.square(dx)+torch.square(dy)+1e-8)

        dirX=dx/distances
        dirY=dy/distances

        force_ressort=self.rigidite*(distances-self.longueur0)

        v_relat=(self.vX[self.muscle2]-self.vX[self.muscle1])*dirX+(self.vY[self.muscle2]-self.vY[self.muscle1])*dirY

        fx_totaleX=dirX*(force_ressort+self.c*v_relat)
        fx_totaleY=dirY*(force_ressort+self.c*v_relat)



        v_moyX=(self.vX[self.muscle1]+self.vX[self.muscle2])/2
        v_moyY=(self.vY[self.muscle1]+self.vY[self.muscle2])/2

        normale_X=-dy/distances
        normale_Y=dx/distances

        v_dot_normale = v_moyX * normale_X + v_moyY * normale_Y

        k_eau=0.005

        f_trainee_X = -distances * v_dot_normale * normale_X * k_eau * self.is_os
        f_trainee_Y = -distances * v_dot_normale * normale_Y * k_eau * self.is_os


        forces_noeuds_X = torch.zeros_like(self.X)
        forces_noeuds_Y = torch.zeros_like(self.Y)

        forces_noeuds_X.index_add_(0, self.muscle1, fx_totaleX+f_trainee_X/2)
        forces_noeuds_X.index_add_(0, self.muscle2, -fx_totaleX+f_trainee_X/2)

        forces_noeuds_Y.index_add_(0, self.muscle1, fx_totaleY+f_trainee_Y/2)
        forces_noeuds_Y.index_add_(0, self.muscle2, -fx_totaleY+f_trainee_Y/2)

        self.vX = self.vX + (forces_noeuds_X / self.masses) * dt
        self.vY = self.vY + (forces_noeuds_Y / self.masses) * dt

        self.X = self.X + self.vX * dt
        self.Y = self.Y + self.vY * dt

    def get_observation(self,frame):
        vitesses_normalisées_X=self.vX/10
        vitesses_normalisées_Y=self.vY/10
        bary_X=torch.mean(self.X)
        bary_Y=torch.mean(self.Y)
        pos_relat_X=(self.X-bary_X)/100
        pos_relat_Y=(self.Y-bary_Y)/100

        X1 = self.X[self.muscle1] 
        X2 = self.X[self.muscle2]

        Y1 = self.Y[self.muscle1]
        Y2 = self.Y[self.muscle2]

        dx=X2-X1
        dy=Y2-Y1

        distances=torch.sqrt(torch.square(dx)+torch.square(dy)+1e-8)

        index_vrais_muscles = (self.is_os == 0.0).nonzero(as_tuple=True)[0]

        contraction=(distances[index_vrais_muscles]-self.longueur0[index_vrais_muscles])/self.longueur0[index_vrais_muscles]

        rythme=torch.sin(torch.tensor([frame / 10.0], dtype=torch.float32))

        observations = torch.cat([
            pos_relat_X, 
            pos_relat_Y, 
            vitesses_normalisées_X, 
            vitesses_normalisées_Y, 
            contraction, 
            rythme
        ])
        return observations
    

    def apply_action(self, action):
        # On trouve l'index des vrais muscles (là où is_os vaut 0)
        index_vrais_muscles = (self.is_os == 0.0).nonzero(as_tuple=True)[0]
        
        # On calcule les nouvelles longueurs (ta fameuse formule !)
        base = self.longueur_base[index_vrais_muscles]
        nouvelles_longueurs = 0.7 * base - 0.6 * base * action
        
        # On met à jour self.longueur0 uniquement pour ces muscles-là
        self.longueur0 = self.longueur0.clone() # Petite sécurité pour PyTorch
        self.longueur0[index_vrais_muscles] = nouvelles_longueurs
        self.energie=self.energie+torch.sum(torch.abs(base*action))


    def get_reward(self):
        distance_actuelle = torch.mean(self.X)
        hauteur_actuelle = torch.mean(self.Y)
        reward = distance_actuelle - self.distance_precedente-self.energie/(100*sum(self.is_os))-(hauteur_actuelle-self.hauteur_précédente)*10
        
        self.distance_precedente = distance_actuelle
        self.hauteur_précédente = hauteur_actuelle

        return reward



















