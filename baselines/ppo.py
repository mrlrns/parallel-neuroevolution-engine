"""
baselines/ppo.py — baseline PPO sur la morphologie FIGÉE d'un champion.

Question posée : à budget de simulation égal, le gradient à travers la physique
(train2.py) apprend-il mieux qu'une méthode RL standard sans modèle ?

Choix pour une comparaison à une seule variable :
  - même morphologie, même physique, même récompense (baselines/env.py)
  - même horizon (train2_frame_nb), même nombre de rollouts par itération
    (ppo_n_envs = train2_batch_size) : 1 itération PPO = 1 épisode de train2.py
    = même nombre de sous-pas physiques, logué en `sim_steps`
  - l'acteur EST un `Brain` (64-64, sortie tanh) : le checkpoint sauvegardé a
    exactement le format des .pt de train2.py et passe tel quel dans
    evaluate_seeds.py, verify_reproducibility.py et visualize.py
  - politique gaussienne : moyenne = Brain(obs), log-std appris indépendant de
    l'état ; seuls les vrais muscles entrent dans la log-vraisemblance (les
    sorties sur os / fantômes n'ont aucun effet physique)

Implémentation d'après CleanRL `ppo_continuous_action.py`, tout reste sur le GPU.
Pas de normalisation d'observation : elle ferait partie de la politique et
casserait la compatibilité avec Brain ; les observations sont déjà
normalisées à la main dans megaVecto.get_observation.

Usage :
    python -m baselines.ppo --chemin-champion champion_raffine/reference.pt --poids-aleatoires --seed 0
"""

import math
import os
import random
from datetime import datetime

import torch
import torch.nn as nn

from baselines.env import SwimEnv
from brain import Brain
from champion import charger_champion
from config import Config, build_argparser, load_config
from logger import Logger


class Critique(nn.Module):
    def __init__(self, obs_size):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_size, 64), nn.Tanh(),
            nn.Linear(64, 64), nn.Tanh(),
            nn.Linear(64, 1),
        )
        # init CleanRL : orthogonale, petite échelle sur la sortie
        for i, m in enumerate(self.net):
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, 1.0 if i == 4 else math.sqrt(2))
                nn.init.zeros_(m.bias)

    def forward(self, x):
        return self.net(x).squeeze(-1)


class Agent(nn.Module):
    def __init__(self, obs_size, action_size, masque_actions, std_init):
        super().__init__()
        self.acteur = Brain(obs_size, action_size)
        self.critique = Critique(obs_size)
        self.log_std = nn.Parameter(torch.full((action_size,), math.log(std_init)))
        self.register_buffer("masque", masque_actions)   # [act], 1 = vrai muscle

    def distribution(self, obs):
        moyenne = self.acteur(obs)
        return torch.distributions.Normal(moyenne, self.log_std.exp().expand_as(moyenne))

    def log_prob(self, dist, action):
        return (dist.log_prob(action) * self.masque).sum(-1)

    def entropie(self, dist):
        return (dist.entropy() * self.masque).sum(-1)


@torch.no_grad()
def evaluer(agent, champ, cfg, device):
    """Rollout quasi-déterministe (moyenne + bruit 0.02, comme evaluate_seeds.py)
    sur un env séparé : n'entre pas dans le budget `sim_steps`."""
    env = SwimEnv(champ.dico, cfg.ppo_eval_n_envs, cfg.train2_frame_nb, cfg.sub_step,
                  cfg.coef_energie, cfg.coef_hauteur, device,
                  physique_compilee=cfg.physique_compilee, v_max=cfg.v_max)
    obs = env.reset()
    total = torch.zeros(env.n, device=device)
    done = False
    while not done:
        a = agent.acteur(obs)
        obs, r, done = env.step(a + torch.randn_like(a) * cfg.bruit_test)
        total += r
    return total.mean().item(), env.deplacement().mean().item()


def entrainer(cfg: Config):
    """Entraîne PPO selon `cfg`. Retourne (meilleur score d'évaluation, chemin du checkpoint)."""
    random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device utilisé : {device}")

    champ = charger_champion(cfg.chemin_champion, device)
    env = SwimEnv(champ.dico, cfg.ppo_n_envs, cfg.train2_frame_nb, cfg.sub_step,
                  cfg.coef_energie, cfg.coef_hauteur, device,
                  physique_compilee=cfg.physique_compilee, v_max=cfg.v_max)
    T, N = env.nb_decisions, env.n
    obs_size, act_size = champ.obs_size, champ.action_size

    agent = Agent(obs_size, act_size, champ.masque_muscles_actifs, cfg.ppo_std_init).to(device)
    if not cfg.poids_aleatoires and champ.brain_weights is not None:
        agent.acteur.load_state_dict(champ.brain_weights)
        print("Acteur initialisé avec les poids du champion")
    else:
        print("Acteur initialisé aléatoirement")
    optimizer = torch.optim.Adam(agent.parameters(), lr=cfg.ppo_lr, eps=1e-5)

    os.makedirs(cfg.dossier_baselines, exist_ok=True)
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    logger = Logger(
        f"{cfg.dossier_runs}/ppo_seed{cfg.seed}_{run_id}.jsonl",
        run_config={"method": "ppo", **{k: v for k, v in vars(cfg).items()
                                         if k.startswith("ppo_") or k in (
                                             "chemin_champion", "seed", "sub_step", "train2_frame_nb",
                                             "coef_energie", "coef_hauteur", "poids_aleatoires",
                                             "physique_compilee", "v_max")}},
    )

    # Tampons du rollout : [T, N, ...]
    b_obs = torch.zeros(T, N, obs_size, device=device)
    b_act = torch.zeros(T, N, act_size, device=device)
    b_logp = torch.zeros(T, N, device=device)
    b_rew = torch.zeros(T, N, device=device)
    b_val = torch.zeros(T, N, device=device)

    sim_steps = 0
    meilleur_eval = float("-inf")
    chemin_meilleur = None
    taille_mb = (T * N) // cfg.ppo_minibatches

    for it in range(cfg.ppo_nb_iterations):
        # Poids AVANT la mise à jour de cette itération (leçon de E6/E10) :
        # ce sont eux qui sont évalués ci-dessous.
        if it % cfg.ppo_eval_tous_les == 0:
            eval_score, eval_depl = evaluer(agent, champ, cfg, device)
            if eval_score > meilleur_eval:
                meilleur_eval = eval_score
                chemin_meilleur = f"{cfg.dossier_baselines}/ppo_seed{cfg.seed}_best.pt"
                torch.save(champ.to_save_dict({k: v.detach().clone()
                                               for k, v in agent.acteur.state_dict().items()}),
                           chemin_meilleur)
        else:
            eval_score = eval_depl = None

        # ---------- 1. Rollout : un épisode complet sur N envs ----------
        obs = env.reset()
        with torch.no_grad():
            for t in range(T):
                dist = agent.distribution(obs)
                action = dist.sample()
                b_obs[t], b_act[t] = obs, action
                b_logp[t] = agent.log_prob(dist, action)
                b_val[t] = agent.critique(obs)
                obs, r, _ = env.step(action)
                b_rew[t] = r
        sim_steps += env.sim_steps_par_episode
        score_episode = b_rew.sum(0)                      # [N]
        depl = env.deplacement().mean().item()

        # ---------- 2. GAE (horizon fixe : valeur terminale = 0) ----------
        with torch.no_grad():
            avantages = torch.zeros_like(b_rew)
            gae = torch.zeros(N, device=device)
            for t in reversed(range(T)):
                v_suiv = b_val[t + 1] if t + 1 < T else torch.zeros(N, device=device)
                delta = b_rew[t] + cfg.ppo_gamma * v_suiv - b_val[t]
                gae = delta + cfg.ppo_gamma * cfg.ppo_gae_lambda * gae
                avantages[t] = gae
            retours = avantages + b_val

        # ---------- 3. Mise à jour PPO ----------
        f_obs = b_obs.reshape(T * N, obs_size)
        f_act = b_act.reshape(T * N, act_size)
        f_logp, f_adv = b_logp.reshape(-1), avantages.reshape(-1)
        f_ret, f_val = retours.reshape(-1), b_val.reshape(-1)

        clipfracs, approx_kl = [], torch.tensor(0.0)
        pg_loss = v_loss = ent = torch.tensor(float("nan"))
        for _ in range(cfg.ppo_epochs):
            perm = torch.randperm(T * N, device=device)
            for d in range(0, T * N - taille_mb + 1, taille_mb):
                idx = perm[d:d + taille_mb]
                dist = agent.distribution(f_obs[idx])
                logp = agent.log_prob(dist, f_act[idx])
                log_ratio = logp - f_logp[idx]
                ratio = log_ratio.exp()
                with torch.no_grad():
                    approx_kl = ((ratio - 1) - log_ratio).mean()
                    clipfracs.append(((ratio - 1).abs() > cfg.ppo_clip).float().mean().item())

                adv = f_adv[idx]
                adv = (adv - adv.mean()) / (adv.std() + 1e-8)
                pg_loss = torch.max(-adv * ratio,
                                    -adv * ratio.clamp(1 - cfg.ppo_clip, 1 + cfg.ppo_clip)).mean()

                v = agent.critique(f_obs[idx])
                v_clip = f_val[idx] + (v - f_val[idx]).clamp(-cfg.ppo_clip, cfg.ppo_clip)
                v_loss = 0.5 * torch.max((v - f_ret[idx]) ** 2, (v_clip - f_ret[idx]) ** 2).mean()

                ent = agent.entropie(dist).mean()
                loss = pg_loss - cfg.ppo_ent_coef * ent + cfg.ppo_vf_coef * v_loss

                if not torch.isfinite(loss):
                    continue
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(agent.parameters(), cfg.ppo_max_grad_norm)
                optimizer.step()

        logger.log(
            iteration=it, episode=it, sim_steps=sim_steps,
            score_mean=score_episode.mean().item(), score_std=score_episode.std().item(),
            distance_moyenne=depl, eval_score_mean=eval_score, eval_distance_moyenne=eval_depl,
            pg_loss=pg_loss.item(), v_loss=v_loss.item(), entropie=ent.item(),
            approx_kl=approx_kl.item(), clipfrac=sum(clipfracs) / max(1, len(clipfracs)),
            std_moyen=(agent.log_std.exp() * agent.masque).sum().item() / agent.masque.sum().item(),
        )
        if it % 10 == 0:
            ev = f"{eval_score:8.2f}" if eval_score is not None else "     ---"
            print(f"It. {it:4d} | score (stochastique) {score_episode.mean().item():8.2f} | "
                  f"éval {ev} | distance {depl:7.2f} | kl {approx_kl.item():.4f} | "
                  f"std {agent.log_std.exp().mean().item():.3f}")

    # Évaluation des poids finaux (après la dernière mise à jour)
    eval_score, eval_depl = evaluer(agent, champ, cfg, device)
    logger.log(iteration=cfg.ppo_nb_iterations, episode=cfg.ppo_nb_iterations, sim_steps=sim_steps,
               eval_score_mean=eval_score, eval_distance_moyenne=eval_depl)
    if eval_score > meilleur_eval:
        meilleur_eval = eval_score
        chemin_meilleur = f"{cfg.dossier_baselines}/ppo_seed{cfg.seed}_best.pt"
        torch.save(champ.to_save_dict({k: v.detach().clone()
                                       for k, v in agent.acteur.state_dict().items()}), chemin_meilleur)
    torch.save(champ.to_save_dict({k: v.detach().clone() for k, v in agent.acteur.state_dict().items()}),
               f"{cfg.dossier_baselines}/ppo_seed{cfg.seed}_final.pt")

    logger.close()
    print(f"\n🏆 Meilleur score d'évaluation : {meilleur_eval:.2f}")
    print(f"💾 {chemin_meilleur}  (format Brain : passe dans evaluate_seeds.py)")
    return meilleur_eval, chemin_meilleur


def main():
    parser = build_argparser("ppo")
    args = parser.parse_args()
    cfg: Config = load_config(args)
    if not cfg.chemin_champion:
        parser.error("--chemin-champion est requis (ou renseigne 'chemin_champion' dans le YAML)")
    entrainer(cfg)


if __name__ == "__main__":
    main()
