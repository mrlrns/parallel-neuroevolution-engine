import torch

class Creature_vecto:
    def __init__(self,positionX,positionY,muscle1,muscle2,stiffness,base_length,is_bone,brain,device):
        self.brain=brain
        
        self.X=torch.tensor(positionX,dtype=torch.float32,device=device)
        self.Y=torch.tensor(positionY,dtype=torch.float32,device=device)
        self.previous_distance = torch.mean(self.X,dim=1,keepdim=True,device=device)
        self.vX=torch.zeros_like(self.X,device=device)
        self.vY=torch.zeros_like(self.Y,device=device)
        self.stiffness=torch.tensor(stiffness,dtype=torch.float32,device=device)
        self.masses=torch.ones_like(self.X,device=device)
        self.muscle1=torch.tensor(muscle1,dtype=torch.long,device=device)
        self.muscle2=torch.tensor(muscle2,dtype=torch.long,device=device)
        self.target_length=torch.tensor(base_length,dtype=torch.float32,device=device).unsqueeze(0).repeat(self.X.shape[0], 1)
        self.base_length=torch.tensor(base_length,dtype=torch.float32,device=device)
        self.c=2*torch.sqrt(self.stiffness)
        self.is_bone = torch.tensor(is_bone, dtype=torch.float32,device=device)
        self.energy = torch.zeros((self.X.shape[0], 1))
        self.previous_height = torch.mean(self.Y,dim=1,keepdim=True)

    def appliquer_physique(self,dt):
        X1 = self.X[:,self.muscle1] 
        X2 = self.X[:,self.muscle2]

        Y1 = self.Y[:,self.muscle1]
        Y2 = self.Y[:,self.muscle2]

        dx=X2-X1
        dy=Y2-Y1

        distances=torch.sqrt(torch.square(dx)+torch.square(dy)+1e-8)

        dirX=dx/distances
        dirY=dy/distances

        force_ressort=self.stiffness*(distances-self.target_length)

        v_relat=(self.vX[:,self.muscle2]-self.vX[:,self.muscle1])*dirX+(self.vY[:,self.muscle2]-self.vY[:,self.muscle1])*dirY

        fx_totaleX=dirX*(force_ressort+self.c*v_relat)
        fx_totaleY=dirY*(force_ressort+self.c*v_relat)



        v_moyX=(self.vX[:,self.muscle1]+self.vX[:,self.muscle2])/2
        v_moyY=(self.vY[:,self.muscle1]+self.vY[:,self.muscle2])/2

        normale_X=-dy/distances
        normale_Y=dx/distances

        v_dot_normale = v_moyX * normale_X + v_moyY * normale_Y

        water_drag=0.005

        f_trainee_X = -distances * v_dot_normale * normale_X * water_drag * self.is_bone
        f_trainee_Y = -distances * v_dot_normale * normale_Y * water_drag * self.is_bone


        forces_noeuds_X = torch.zeros_like(self.X)
        forces_noeuds_Y = torch.zeros_like(self.Y)

        forces_noeuds_X.index_add_(1, self.muscle1, fx_totaleX+f_trainee_X/2)
        forces_noeuds_X.index_add_(1, self.muscle2, -fx_totaleX+f_trainee_X/2)

        forces_noeuds_Y.index_add_(1, self.muscle1, fx_totaleY+f_trainee_Y/2)
        forces_noeuds_Y.index_add_(1, self.muscle2, -fx_totaleY+f_trainee_Y/2)

        self.vX = self.vX + (forces_noeuds_X / self.masses) * dt
        self.vY = self.vY + (forces_noeuds_Y / self.masses) * dt

        self.X = self.X + self.vX * dt
        self.Y = self.Y + self.vY * dt

    def get_observation(self,frame):
        normalized_velocities_X=self.vX/10
        normalized_velocities_Y=self.vY/10
        bary_X=torch.mean(self.X,dim=1,keepdim=True)
        bary_Y=torch.mean(self.Y,dim=1,keepdim=True)
        pos_relat_X=(self.X-bary_X)/100
        pos_relat_Y=(self.Y-bary_Y)/100

        X1 = self.X[:,self.muscle1] 
        X2 = self.X[:,self.muscle2]

        Y1 = self.Y[:,self.muscle1]
        Y2 = self.Y[:,self.muscle2]

        dx=X2-X1
        dy=Y2-Y1

        distances=torch.sqrt(torch.square(dx)+torch.square(dy)+1e-8)

        real_muscle_indices = (self.is_bone == 0.0).nonzero(as_tuple=True)[0]

        contraction=(distances[:,real_muscle_indices]-self.target_length[:,real_muscle_indices])/self.target_length[:,real_muscle_indices]

        rythme=torch.sin(torch.tensor([frame / 10.0], dtype=torch.float32)).unsqueeze(0).expand(self.X.shape[0], 1)

        observations = torch.cat([
            pos_relat_X, 
            pos_relat_Y, 
            normalized_velocities_X, 
            normalized_velocities_Y, 
            contraction, 
            rythme
        ],dim=1)
        return observations
    

    def apply_action(self, action):
        # On trouve l'index des vrais muscles (là où is_bone vaut 0)
        real_muscle_indices = (self.is_bone == 0.0).nonzero(as_tuple=True)[0]
        
        # On calcule les nouvelles longueurs (ta fameuse formule !)
        base = self.base_length[real_muscle_indices].unsqueeze(0) # On ajoute une dimension pour le batch
        new_lengths = 0.7 * base - 0.6 * base * action
        
        # On met à jour self.target_length uniquement pour ces muscles-là
        self.target_length = self.target_length.clone() # Petite sécurité pour PyTorch
        self.target_length[:,real_muscle_indices] = new_lengths
        self.energy=self.energy+torch.sum(torch.abs(base*action),dim=1,keepdim=True)


    def get_reward(self):
        distance_actuelle = torch.mean(self.X,dim=1,keepdim=True)
        hauteur_actuelle = torch.mean(self.Y,dim=1,keepdim=True)
        reward = distance_actuelle - self.previous_distance-self.energy/(100*sum(self.is_bone))-torch.abs((hauteur_actuelle-self.previous_height)/10)
        
        self.previous_distance = distance_actuelle
        self.previous_height = hauteur_actuelle

        return reward



















