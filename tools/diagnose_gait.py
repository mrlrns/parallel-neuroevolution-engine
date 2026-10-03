"""
Diagnostic d'une allure : est-elle simplement brutale, ou exploite-t-elle le simulateur ?

Pour chaque contrôleur, sur N rollouts × F frames (même bruit d'action que
evaluate_seeds.py), mesure :
  - |a|            amplitude moyenne des commandes (vrais muscles)
  - |Δa|           saut moyen de commande entre deux décisions (saccades)
  - saturation a   fraction des commandes à |a| > 0.95 (tout-ou-rien)
  - sat. vitesse   fraction des nœuds à |v| ≥ 20 (seuil historique, même si --v-max diffère)
  - sat. longueur  fraction des muscles bloqués au plancher 0.3·base de apply_action
  - énergie / prog part de la récompense prise par la pénalité d'énergie
  - dérive Y       déplacement vertical du barycentre (question ouverte d'EXPERIMENTS)

Usage :
    python -m tools.diagnose_gait champion_raffine/ppo_seed1.pt champion_raffine/reference.pt
"""

import argparse

import torch

from baselines.env import SwimEnv
from brain import Brain
from champion import charger_champion


def moy(valeurs):
    return sum(valeurs) / max(1, len(valeurs))


@torch.no_grad()
def diagnostiquer(chemin, n, frames, bruit, coef_e, sub_step, seed, v_max=20.0):
    torch.manual_seed(seed)
    dev = torch.device("cpu")
    c = charger_champion(chemin, dev)
    cerveau = Brain(c.obs_size, c.action_size)
    cerveau.load_state_dict(c.brain_weights)
    m = c.masque_muscles_actifs.bool()
    env = SwimEnv(c.dico, n, frames, sub_step, coef_e, 0.0, dev)
    obs = env.reset()
    mega = env.mega
    mega.v_max = v_max
    nb_n = mega.mask_N_exp.sum(dim=2)
    y0 = (mega.Y * mega.mask_N_exp).sum(dim=2) / nb_n

    amp, saut, sat_a, sat_v, sat_l = [], [], [], [], []
    prog_tot, ener_tot = 0.0, 0.0
    a_prec, done = None, False
    while not done:
        a = cerveau(obs)
        a = (a + torch.randn_like(a) * bruit).clamp(-1, 1)
        am = a[:, m]
        amp.append(am.abs().mean().item())
        sat_a.append((am.abs() > 0.95).float().mean().item())
        if a_prec is not None:
            saut.append((am - a_prec).abs().mean().item())
        a_prec = am
        # énergie de cette action (même formule que apply_action), avant le step
        base = mega.base_length
        nl = torch.clamp(base - 0.3 * base * a.unsqueeze(0), min=0.3 * base)
        sat_l.append(((nl <= 0.3 * base + 1e-6) & (mega.mask_vrais_M_exp == 1)).float().sum().item()
                     / max(1.0, mega.mask_vrais_M_exp.sum().item()))
        ener_tot += (torch.abs(nl - base) * mega.mask_vrais_M_exp).sum(dim=2).mean().item() / coef_e
        x_av = env.barycentre_x().clone()
        obs, r, done = env.step(a, clamp=False)
        prog_tot += (env.barycentre_x() - x_av).mean().item()
        sat_v.append(((mega.vX.abs() >= 19.99) | (mega.vY.abs() >= 19.99))[mega.mask_N_exp.bool()].float().mean().item())

    y1 = (mega.Y * mega.mask_N_exp).sum(dim=2) / nb_n
    return {
        "déplacement X": env.deplacement().mean().item(),
        "dérive Y": (y1 - y0).mean().item(),
        "|a|": moy(amp), "|Δa|": moy(saut), "saturation a": moy(sat_a),
        "sat. vitesse": moy(sat_v), "sat. longueur": moy(sat_l),
        "énergie / progrès": ener_tot / max(1e-9, prog_tot),
    }


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("checkpoints", nargs="+")
    p.add_argument("--n", type=int, default=20)
    p.add_argument("--frames", type=int, default=300)
    p.add_argument("--bruit", type=float, default=0.02)
    p.add_argument("--coef-energie", type=float, default=10000.0)
    p.add_argument("--sub-step", type=int, default=20)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--v-max", type=float, default=20.0, help="écrêtage des vitesses (inf pour le retirer)")
    a = p.parse_args()

    res = {ck: diagnostiquer(ck, a.n, a.frames, a.bruit, a.coef_energie, a.sub_step, a.seed, a.v_max) for ck in a.checkpoints}
    noms = [ck.split("/")[-1] for ck in a.checkpoints]
    print(f"{'':20s}" + "".join(f"{n_:>26s}" for n_ in noms))
    for cle in next(iter(res.values())):
        fmt = "{:>26.1f}" if cle in ("déplacement X", "dérive Y") else "{:>26.4f}"
        print(f"{cle:20s}" + "".join(fmt.format(res[ck][cle]) for ck in a.checkpoints))


if __name__ == "__main__":
    main()
