import torch

class MegaCrea:
    def __init__(self, dico_mega_tenseurs, batch_size, device="cuda"):
        self.device = device
        self.batch_size = batch_size
        
        
        self.X_base = dico_mega_tenseurs["X"] 
        self.Y_base = dico_mega_tenseurs["Y"]
        self.masque_noeuds = dico_mega_tenseurs["masque_noeuds"]
        self.masque_muscles = dico_mega_tenseurs["masque_muscles"]
        self.muscle1 = dico_mega_tenseurs["m1"]
        self.muscle2 = dico_mega_tenseurs["m2"]
        self.stiffness = dico_mega_tenseurs["stiffness"]
        self.base_length = dico_mega_tenseurs["base_length"]
        self.is_bone = dico_mega_tenseurs["is_bone"]
        
        
        self.X = self.X_base.unsqueeze(1).repeat(1, batch_size, 1) + torch.randn(self.X_base.shape[0], batch_size, self.X_base.shape[1], device=device) * 2.0
        self.Y = self.Y_base.unsqueeze(1).repeat(1, batch_size, 1)+ torch.randn(self.X_base.shape[0], batch_size, self.X_base.shape[1], device=device) * 2.0
        self.vX = torch.zeros_like(self.X)
        self.vY = torch.zeros_like(self.Y)
        
        
        self.mask_N_exp = self.masque_noeuds.unsqueeze(1).expand(-1, batch_size, -1)
        self.mask_M_exp = self.masque_muscles.unsqueeze(1).expand(-1, batch_size, -1)
        
        self.is_bone_exp = self.is_bone.unsqueeze(1).expand(-1, batch_size, -1)
        self.mask_vrais_M_exp = (self.is_bone_exp == 0.0).float() * self.mask_M_exp

        self.target_length = self.base_length.unsqueeze(1).repeat(1, batch_size, 1)
        self.c = 2 * torch.sqrt(self.stiffness).unsqueeze(1).expand(-1, batch_size, -1)
        self.masses = torch.ones_like(self.X)
        
        self.energy = torch.zeros((self.X.shape[0], batch_size), device=device)
        self.total_energy = torch.zeros((self.X.shape[0], batch_size), device=device)
        
        nb_nodes = torch.clamp(torch.sum(self.mask_N_exp, dim=2), min=1.0)
        self.previous_distance = torch.sum(self.X * self.mask_N_exp, dim=2) / nb_nodes
        self.previous_height = torch.sum(self.Y * self.mask_N_exp, dim=2) / nb_nodes
        
        
        
        self.previous_vitesse = torch.zeros_like(self.previous_distance)

    def apply_physics(self, dt):
        
        # --- ÉTAPE 1 : LECTURE (gather) ---
        m1_exp = self.muscle1.unsqueeze(1).expand(-1, self.batch_size, -1)
        m2_exp = self.muscle2.unsqueeze(1).expand(-1, self.batch_size, -1)

        X1 = torch.gather(self.X, dim=2, index=m1_exp)
        Y1 = torch.gather(self.Y, dim=2, index=m1_exp)
        X2 = torch.gather(self.X, dim=2, index=m2_exp)
        Y2 = torch.gather(self.Y, dim=2, index=m2_exp)
        
        vX1 = torch.gather(self.vX, dim=2, index=m1_exp)
        vY1 = torch.gather(self.vY, dim=2, index=m1_exp)
        vX2 = torch.gather(self.vX, dim=2, index=m2_exp)
        vY2 = torch.gather(self.vY, dim=2, index=m2_exp)

        # --- ÉTAPE 2 : GÉOMÉTRIE ---
        dx = X2 - X1
        dy = Y2 - Y1
        distances = torch.sqrt(torch.square(dx) + torch.square(dy) + 1e-3)

        dirX = dx / distances
        dirY = dy / distances

        # --- ÉTAPE 3 : RESSORTS & AMORTISSEMENT ---
        stiff_exp = self.stiffness.unsqueeze(1).expand(-1, self.batch_size, -1)
        force_ressort = stiff_exp * (distances - self.target_length)
        
        v_relat = (vX2 - vX1) * dirX + (vY2 - vY1) * dirY
        
        fx_totaleX = dirX * (force_ressort + self.c * v_relat)
        fx_totaleY = dirY * (force_ressort + self.c * v_relat)

        # --- ÉTAPE 4 : WATER DRAG (Repris à 100% de ton code) ---
        v_moyX = (vX1 + vX2) / 2
        v_moyY = (vY1 + vY2) / 2

        normale_X = -dy / distances
        normale_Y = dx / distances
        v_dot_normale = v_moyX * normale_X + v_moyY * normale_Y

        water_drag = 0.005
        drag_factor = self.is_bone_exp + (1 - self.is_bone_exp) * 0.3
        
        f_trainee_X = -distances * v_dot_normale * normale_X * water_drag * drag_factor
        f_trainee_Y = -distances * v_dot_normale * normale_Y * water_drag * drag_factor

        v_dot_dir = v_moyX * dirX + v_moyY * dirY
        water_drag_tangentiel = water_drag * 0.3

        f_trainee_tang_X = -distances * v_dot_dir * dirX * water_drag_tangentiel * drag_factor
        f_trainee_tang_Y = -distances * v_dot_dir * dirY * water_drag_tangentiel * drag_factor

        f_trainee_X = f_trainee_X + f_trainee_tang_X
        f_trainee_Y = f_trainee_Y + f_trainee_tang_Y

        # ⚡️ COUP DE MASQUE MUSCLES ⚡️
        # On neutralise toutes les forces calculées sur les muscles fantômes
        fx_totaleX = fx_totaleX * self.mask_M_exp
        fx_totaleY = fx_totaleY * self.mask_M_exp
        f_trainee_X = f_trainee_X * self.mask_M_exp
        f_trainee_Y = f_trainee_Y * self.mask_M_exp

        # --- ÉTAPE 5 : DISTRIBUTION (scatter_add_) ---
        forces_noeuds_X = torch.zeros_like(self.X)
        forces_noeuds_Y = torch.zeros_like(self.Y)

        forces_noeuds_X.scatter_add_(2, m1_exp, fx_totaleX + f_trainee_X / 2)
        forces_noeuds_X.scatter_add_(2, m2_exp, -fx_totaleX + f_trainee_X / 2)

        forces_noeuds_Y.scatter_add_(2, m1_exp, fx_totaleY + f_trainee_Y / 2)
        forces_noeuds_Y.scatter_add_(2, m2_exp, -fx_totaleY + f_trainee_Y / 2)

        # --- ÉTAPE 6 : INTÉGRATION & MASQUE NOEUDS ---
        self.vX = torch.clamp(self.vX + (forces_noeuds_X / self.masses) * dt, -20, 20)
        self.vY = torch.clamp(self.vY + (forces_noeuds_Y / self.masses) * dt, -20, 20)
        
        # Les fantômes restent immobiles
        self.vX = self.vX * self.mask_N_exp
        self.vY = self.vY * self.mask_N_exp

        self.X = self.X + self.vX * dt
        self.Y = self.Y + self.vY * dt

    def get_observation(self, frame):
        # Vitesses normalisées
        normalized_velocities_X = self.vX / 10.0
        normalized_velocities_Y = self.vY / 10.0
        
        # Barycentres sécurisés avec keepdim=True pour le broadcasting
        nb_nodes = torch.clamp(torch.sum(self.mask_N_exp, dim=2, keepdim=True), min=1.0)
        bary_X = torch.sum(self.X * self.mask_N_exp, dim=2, keepdim=True) / nb_nodes
        bary_Y = torch.sum(self.Y * self.mask_N_exp, dim=2, keepdim=True) / nb_nodes
        
        pos_relat_X = ((self.X - bary_X) / 100.0) * self.mask_N_exp
        pos_relat_Y = ((self.Y - bary_Y) / 100.0) * self.mask_N_exp

        # Calcul des distances actuelles des muscles
        m1_exp = self.muscle1.unsqueeze(1).expand(-1, self.batch_size, -1)
        m2_exp = self.muscle2.unsqueeze(1).expand(-1, self.batch_size, -1)
        X1 = torch.gather(self.X, dim=2, index=m1_exp)
        Y1 = torch.gather(self.Y, dim=2, index=m1_exp)
        X2 = torch.gather(self.X, dim=2, index=m2_exp)
        Y2 = torch.gather(self.Y, dim=2, index=m2_exp)
        
        distances = torch.sqrt(torch.square(X2 - X1) + torch.square(Y2 - Y1) + 1e-8)

        # Contraction vectorisée sur TOUS les muscles (la taille reste MAX_MUSCLES)
        # On évite la division par zéro des target_length des fantômes
        safe_target_L = torch.clamp(self.target_length, min=1.0)
        contraction = (distances - self.target_length) / safe_target_L
        
        # On écrase la contraction des os et des fantômes (elle vaudra 0.0)
        contraction = contraction * self.mask_vrais_M_exp

        # Rythme [POP_SIZE, BATCH_SIZE, 1]
        val_rythme = torch.sin(torch.tensor([frame / 10.0], dtype=torch.float32, device=self.device))
        rythme = val_rythme.unsqueeze(1).expand(self.X.shape[0], self.batch_size).unsqueeze(2)

        # Assemblage final sur la dimension 2
        observations = torch.cat([
            pos_relat_X, 
            pos_relat_Y, 
            normalized_velocities_X, 
            normalized_velocities_Y, 
            contraction, 
            rythme
        ], dim=2)
        
        return observations
    
    def apply_action(self, action,frame):
        """
        L'action venant du réseau aura la forme [POP_SIZE, BATCH_SIZE, MAX_MUSCLES]
        """
        base = self.base_length.unsqueeze(1).expand(-1, self.batch_size, -1)
        new_lengths = torch.clamp(0.85 * base - 0.3 * base * action, min=0.3 * base)
        
        # torch.where permet de ne modifier QUE les vrais muscles, 
        # et de laisser la cible intacte pour les os / fantômes
        self.target_length = torch.where(
            self.mask_vrais_M_exp == 1.0, 
            new_lengths, 
            self.target_length
        )

        
        
        # L'énergie accumulée sur la dimension 2 (les muscles) pour obtenir [POP, BATCH]
        energy_step = torch.sum(torch.abs(base * action) * self.mask_vrais_M_exp, dim=2)
        self.energy = self.energy + energy_step
        self.total_energy = self.total_energy + energy_step

    def get_reward(self, coeff_energie, coeff_hauteur):
      nb_nodes = torch.clamp(torch.sum(self.mask_N_exp, dim=2), min=1.0)

      distance_actuelle = torch.sum(self.X * self.mask_N_exp, dim=2) / nb_nodes
      hauteur_actuelle = torch.sum(self.Y * self.mask_N_exp, dim=2) / nb_nodes

      progres = distance_actuelle - self.previous_distance
      penalite_energie = self.energy / coeff_energie

      if coeff_hauteur == 0.0:
          penalite_hauteur = 0.0
      else:
          penalite_hauteur = torch.abs(hauteur_actuelle - self.previous_height) / coeff_hauteur

      reward = progres - penalite_energie - penalite_hauteur

      self.previous_distance = distance_actuelle
      self.previous_height = hauteur_actuelle
      self.energy = torch.zeros_like(self.energy)

      return reward