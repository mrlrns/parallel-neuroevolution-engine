"""
baselines/check_env.py — vérifie que SwimEnv réplique EXACTEMENT la boucle de train2.py.

À lancer AVANT tout entraînement de baseline. Le cerveau du champion est rejoué
deux fois, avec la même seed donc le même tirage de bruit :
  (A) boucle copiée de train2.py (sans gradient)
  (B) SwimEnv

Critères :
  1. Déplacement final identique : les deux boucles simulent la même physique.
  2. score_A = score_B − progression des 5 dernières frames : seul le
     réalignement de la récompense diffère (voir env.py).

Usage :
    python -m baselines.check_env --chemin-champion champion_raffine/reference.pt
"""

import torch

from baselines.env import DECISION_TOUTES_LES, SwimEnv
from brain import Brain
from champion import charger_champion
from config import build_argparser, load_config
from megaVecto import MegaCrea


@torch.no_grad()
def boucle_train2(dico, cerveau, n, frame_nb, sub_step, coef_e, coef_h, bruit, device):
    """Copie conforme de la boucle de train2.py, avec un bruit fixe."""
    dt = torch.tensor(1.0 / sub_step, device=device)
    mega = MegaCrea(dico, n, device=device)
    m = mega.mask_N_exp
    nb_n = torch.clamp(m.sum(dim=2), min=1.0)
    depart = (torch.sum(mega.X * m, dim=2) / nb_n).clone()
    total = None
    for frame in range(frame_nb):
        if frame % DECISION_TOUTES_LES == 0:
            action = cerveau(mega.get_observation(frame))
            mega.apply_action(action + torch.randn_like(action) * bruit, frame)
            r = mega.get_reward(coef_e, coef_h)
            total = r if total is None else total + r
        for _ in range(sub_step):
            mega.apply_physics(dt)
    fin = torch.sum(mega.X * m, dim=2) / nb_n
    return total[0], (fin - depart)[0]


@torch.no_grad()
def boucle_env(dico, cerveau, n, frame_nb, sub_step, coef_e, coef_h, bruit, device):
    env = SwimEnv(dico, n, frame_nb, sub_step, coef_e, coef_h, device)
    obs = env.reset()
    total = torch.zeros(n, device=device)
    derniere_progression = torch.zeros(n, device=device)
    done = False
    while not done:
        action = cerveau(obs)
        x_avant = env.barycentre_x().clone()
        # clamp=False : train2.py n'écrête pas l'action bruitée
        obs, r, done = env.step(action + torch.randn_like(action) * bruit, clamp=False)
        derniere_progression = env.barycentre_x() - x_avant
        total += r
    return total, env.deplacement(), derniere_progression


def comparer(dico, cerveau, n, frame_nb, sub_step, coef_e, coef_h, bruit, seed, device):
    """Retourne (écart max de déplacement, écart max de score corrigé, détails)."""
    args = (dico, cerveau, n, frame_nb, sub_step, coef_e, coef_h, bruit, device)
    torch.manual_seed(seed)
    score_a, depl_a = boucle_train2(*args)
    torch.manual_seed(seed)
    score_b, depl_b, fin_b = boucle_env(*args)
    ecart_depl = (depl_a - depl_b).abs().max().item()
    ecart_score = (score_a - (score_b - fin_b)).abs().max().item()
    return ecart_depl, ecart_score, dict(score_a=score_a, depl_a=depl_a,
                                         score_b=score_b, depl_b=depl_b, fin_b=fin_b)


def main():
    parser = build_argparser("check_env")
    args = parser.parse_args()
    cfg = load_config(args)
    if not cfg.chemin_champion:
        parser.error("--chemin-champion est requis (ou renseigne 'chemin_champion' dans le YAML)")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    champ = charger_champion(cfg.chemin_champion, device)
    cerveau = Brain(champ.obs_size, champ.action_size).to(device)
    cerveau.load_state_dict(champ.brain_weights)
    cerveau.eval()
    n = cfg.train2_batch_size

    ecart_depl, ecart_score, d = comparer(
        champ.dico, cerveau, n, cfg.train2_frame_nb, cfg.sub_step,
        cfg.coef_energie, cfg.coef_hauteur, cfg.bruit_action, cfg.seed, device)

    print(f"{cfg.chemin_champion} | {n} rollouts × {cfg.train2_frame_nb} frames | device {device}")
    print(f"(A) train2  : score moyen {d['score_a'].mean().item():9.3f} | déplacement moyen {d['depl_a'].mean().item():9.3f}")
    print(f"(B) SwimEnv : score moyen {d['score_b'].mean().item():9.3f} | déplacement moyen {d['depl_b'].mean().item():9.3f}")
    print(f"    écart de score attendu (progression des 5 dernières frames) : {d['fin_b'].mean().item():.3f}")
    print(f"max |Δ déplacement|               = {ecart_depl:.2e}")
    print(f"max |score_A − (score_B − fin_B)| = {ecart_score:.2e}")

    tol = 1e-3 * max(1.0, d['depl_a'].abs().max().item())
    if ecart_depl < tol and ecart_score < tol:
        print("✅ SwimEnv réplique train2.py : la comparaison avec une baseline est valide.")
    else:
        print("❌ Divergence : NE PAS lancer de baseline avant d'avoir trouvé la différence.")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
