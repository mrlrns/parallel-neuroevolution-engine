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

Épisodes enchaînés (E20, `nouvelle_tranche`) : au lieu de repartir du repos, un
épisode reprend l'état final du précédent (forme, vitesses, longueurs cibles,
dernière commande). Avec un horizon fini depuis une pose fixe, « une seule
déformation lente puis plus rien » est optimal (E18) ; en enchaînant, ce coup
unique ne rapporte qu'une fois et seule une nage qui se répète paie à chaque
tranche. La physique est invariante par translation : on recentre le corps sur
l'origine à chaque tranche pour éviter la perte de précision du float32 quand la
créature s'éloigne (x ≈ 10⁴ après des centaines de tranches).
"""

import torch

from megaVecto_fast import classe_physique

DECISION_TOUTES_LES = 5   # le cerveau est interrogé toutes les 5 frames (cf. train2.py)

# État propre à chaque rollout ([1, n, ...]) : ce qu'on recopie pour remettre un env au repos.
ETAT_PAR_ROLLOUT = ("X", "Y", "vX", "vY", "base_length", "target_length", "previous_distance",
                    "previous_height", "energy", "regularite", "action_prec")


def barycentre(mega):
    """Barycentre (x, y) de chaque rollout : deux tenseurs [1, n]."""
    m = mega.mask_N_exp
    nb = torch.clamp(m.sum(dim=2), min=1.0)
    return (mega.X * m).sum(dim=2) / nb, (mega.Y * m).sum(dim=2) / nb


@torch.no_grad()
def recentrer(mega):
    """Translate chaque corps pour que son barycentre soit à l'origine.

    Les forces ne dépendent que des différences de positions et des vitesses :
    la dynamique est inchangée. previous_distance / previous_height sont décalés
    d'autant pour que la prochaine récompense (différence de barycentre) soit exacte."""
    bx, by = barycentre(mega)
    m = mega.mask_N_exp
    mega.X = mega.X - bx.unsqueeze(2) * m
    mega.Y = mega.Y - by.unsqueeze(2) * m
    mega.previous_distance = mega.previous_distance - bx
    if torch.is_tensor(mega.previous_height):
        mega.previous_height = mega.previous_height - by


@torch.no_grad()
def reinitialiser_explosions(mega, fabrique):
    """Remet au repos les rollouts dont l'état n'est plus fini. Retourne leur masque [n] (bool).

    `fabrique()` construit un objet physique neuf de même taille : on recopie ses
    lignes (état au repos, bruit initial compris) dans les rollouts concernés."""
    fini = torch.isfinite(mega.X).all(dim=2) & torch.isfinite(mega.Y).all(dim=2) \
        & torch.isfinite(mega.vX).all(dim=2) & torch.isfinite(mega.vY).all(dim=2)   # [1, n]
    casses = ~fini[0]
    if not bool(casses.any()):
        return casses
    neuf = fabrique()
    for nom in ETAT_PAR_ROLLOUT:
        val, frais = getattr(mega, nom, None), getattr(neuf, nom, None)
        if not torch.is_tensor(val) or not torch.is_tensor(frais):
            if nom == "action_prec" and torch.is_tensor(val):
                val = val.clone()
                val[:, casses] = 0.0
                setattr(mega, nom, val)
            continue
        masque = casses.view(1, -1, *([1] * (val.dim() - 2)))
        setattr(mega, nom, torch.where(masque, frais, val))
    return casses


@torch.no_grad()
def evaluer_politique(politique, dico, n, frames, cfg, device):
    """Rollout quasi-déterministe depuis le repos (politique(obs) + bruit `bruit_test`).

    Sert à choisir le meilleur checkpoint, pour PPO comme pour train2 : avec
    frames > horizon d'entraînement, un coup unique ne peut plus l'emporter (E20).
    Retourne (récompense moyenne, déplacement moyen). Hors budget `sim_steps`."""
    env = SwimEnv(dico, n, frames, cfg.sub_step, cfg.coef_energie, cfg.coef_hauteur, device,
                  physique_compilee=cfg.physique_compilee, v_max=cfg.v_max,
                  coef_regularite=cfg.coef_regularite, water_drag=cfg.water_drag)
    obs = env.reset()
    total = torch.zeros(env.n, device=device)
    done = False
    while not done:
        a = politique(obs)
        obs, r, done = env.step(a + torch.randn_like(a) * cfg.bruit_test)
        total += r
    return total.mean().item(), env.deplacement().mean().item()


class SwimEnv:
    def __init__(self, dico, n_envs, frame_nb, sub_step, coef_energie, coef_hauteur, device,
                 physique_compilee=False, v_max=float("inf"), coef_regularite=0.0,
                 water_drag=0.005, reinit_explosions=False):
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
        self.water_drag = water_drag
        self.reinit_explosions = reinit_explosions   # E20 : remettre au repos un rollout qui explose
        self.frame = 0           # frame dans l'épisode (tranche) en cours
        self.frame_globale = 0   # frame depuis le dernier reset : horloge des observations (rythme)
        self.nb_reinit = 0       # rollouts remis au repos depuis le dernier reset/tranche

    @property
    def sim_steps_par_episode(self) -> int:
        """Appels à apply_physics × rollouts pour un épisode (même unité que train2.py)."""
        return self.n * self.frame_nb * self.sub_step

    def barycentre_x(self):
        m = self.mega.mask_N_exp
        return (torch.sum(self.mega.X * m, dim=2) / torch.clamp(m.sum(dim=2), min=1.0))[0]   # [n]

    def _neuf(self):
        mega = self._classe(self.dico, self.n, device=self.device)
        mega.v_max = self.v_max
        mega.coef_regularite = self.coef_regularite
        mega.water_drag = self.water_drag
        return mega

    @torch.no_grad()
    def reset(self):
        self.mega = self._neuf()
        self.frame = 0
        self.frame_globale = 0
        self.nb_reinit = 0
        self.pos_depart = self.barycentre_x().clone()
        return self.mega.get_observation(0)[0]                        # [n, obs]

    @torch.no_grad()
    def nouvelle_tranche(self):
        """E20 : nouvel épisode qui reprend l'état final du précédent (pas de retour au repos).

        Seuls le compteur de frames de l'épisode, le point de départ du déplacement et le
        compteur de remises au repos repartent de zéro ; l'horloge des observations continue."""
        if self.mega is None:
            return self.reset()
        recentrer(self.mega)
        self.frame = 0
        self.nb_reinit = 0
        self.pos_depart = self.barycentre_x().clone()
        return self.mega.get_observation(self.frame_globale)[0]

    @torch.no_grad()
    def step(self, action, clamp=True):
        """action : [n, act]. Retourne (obs [n, obs], reward [n], done bool)."""
        if clamp:
            action = action.clamp(-1.0, 1.0)
        self.mega.apply_action(action.unsqueeze(0), self.frame_globale)
        for _ in range(DECISION_TOUTES_LES):
            for _ in range(self.sub_step):
                self.mega.apply_physics(self.dt)
            self.frame += 1
            self.frame_globale += 1
        r = self.mega.get_reward(self.coef_energie, self.coef_hauteur)[0]
        if self.reinit_explosions:
            casses = reinitialiser_explosions(self.mega, self._neuf)
            if bool(casses.any()):
                self.nb_reinit += int(casses.sum().item())
                # le déplacement de ces rollouts repart de leur nouvelle position
                bx, _ = barycentre(self.mega)
                self.pos_depart = torch.where(casses, bx[0], self.pos_depart)
        obs = self.mega.get_observation(self.frame_globale)[0]
        # Une explosion numérique ne doit pas empoisonner l'apprentissage.
        r = torch.nan_to_num(r, nan=0.0, posinf=0.0, neginf=0.0)
        obs = torch.nan_to_num(obs, nan=0.0, posinf=0.0, neginf=0.0)
        done = self.frame >= self.frame_nb
        return obs, r, done

    @torch.no_grad()
    def deplacement(self):
        """Déplacement horizontal du barycentre depuis le reset : [n]."""
        return self.barycentre_x() - self.pos_depart
