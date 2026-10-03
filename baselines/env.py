"""
baselines/env.py — MegaCrea enveloppé en environnement RL vectorisé (sans gradient).

Une décision = 5 frames × sub_step sous-pas physiques, exactement comme dans
train2.py. Tous les environnements ont le même horizon fixe, donc ils se
terminent ensemble : un rollout = un épisode complet, sans auto-reset.

Récompense d'une décision k :
    progression du barycentre pendant les 5 frames qui suivent l'action k
    − énergie de l'action k
C'est la même récompense que train2.py (mêmes apply_action / get_reward),
seulement réalignée d'un pas pour que chaque action soit créditée de SON effet.
Conséquence : sur un épisode, train2.py somme le déplacement jusqu'à la frame
F−5 alors que SwimEnv le somme jusqu'à F. baselines/check_env.py vérifie que
les trajectoires physiques sont identiques et que c'est la seule différence.
"""

import torch

from megaVecto_fast import classe_physique

DECISION_TOUTES_LES = 5   # le cerveau est interrogé toutes les 5 frames (cf. train2.py)


class SwimEnv:
    def __init__(self, dico, n_envs, frame_nb, sub_step, coef_energie, coef_hauteur, device,
                 physique_compilee=False, v_max=float("inf"), coef_regularite=0.0):
        assert frame_nb % DECISION_TOUTES_LES == 0, "frame_nb doit être un multiple de 5"
        self.dico = dico
        self.n = n_envs
        self.frame_nb = frame_nb
        self.sub_step = sub_step
        self.coef_energie = coef_energie
        self.coef_hauteur = coef_hauteur
        self.device = device
        self.dt = torch.tensor(1.0 / sub_step, device=device)
        self.nb_decisions = frame_nb // DECISION_TOUTES_LES
        self.mega = None
        self._classe = classe_physique(physique_compilee)
        self.v_max = v_max
        self.coef_regularite = coef_regularite
        self.frame = 0

    @property
    def sim_steps_par_episode(self) -> int:
        """Appels à apply_physics × rollouts pour un épisode (même unité que train2.py)."""
        return self.n * self.frame_nb * self.sub_step

    def barycentre_x(self):
        m = self.mega.mask_N_exp
        return (torch.sum(self.mega.X * m, dim=2) / torch.clamp(m.sum(dim=2), min=1.0))[0]   # [n]

    @torch.no_grad()
    def reset(self):
        self.mega = self._classe(self.dico, self.n, device=self.device)
        self.mega.v_max = self.v_max
        self.mega.coef_regularite = self.coef_regularite
        self.frame = 0
        self.pos_depart = self.barycentre_x().clone()
        return self.mega.get_observation(0)[0]                        # [n, obs]

    @torch.no_grad()
    def step(self, action, clamp=True):
        """action : [n, act]. Retourne (obs [n, obs], reward [n], done bool)."""
        if clamp:
            action = action.clamp(-1.0, 1.0)
        self.mega.apply_action(action.unsqueeze(0), self.frame)
        for _ in range(DECISION_TOUTES_LES):
            for _ in range(self.sub_step):
                self.mega.apply_physics(self.dt)
            self.frame += 1
        r = self.mega.get_reward(self.coef_energie, self.coef_hauteur)[0]
        obs = self.mega.get_observation(self.frame)[0]
        # Une explosion numérique ne doit pas empoisonner l'apprentissage.
        r = torch.nan_to_num(r, nan=0.0, posinf=0.0, neginf=0.0)
        obs = torch.nan_to_num(obs, nan=0.0, posinf=0.0, neginf=0.0)
        done = self.frame >= self.frame_nb
        return obs, r, done

    @torch.no_grad()
    def deplacement(self):
        """Déplacement horizontal du barycentre depuis le reset : [n]."""
        return self.barycentre_x() - self.pos_depart
