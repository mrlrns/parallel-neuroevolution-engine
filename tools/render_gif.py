"""
tools/render_gif.py — GIF de démonstration d'un checkpoint, caméra qui suit la créature.

visualize.py filme avec une caméra fixe : sur un long horizon la créature sort de l'écran.
Ici la caméra suit le barycentre (lissé), une grille fixe dans le repère de l'eau montre le
déplacement, une traînée montre la trajectoire et un compteur affiche la distance parcourue.
Même physique et même boucle de contrôle que l'entraînement (SwimEnv : sub_step, v_max,
water_drag de la config, décision toutes les 5 frames, bruit d'action `--bruit`).

Usage :
    python -m tools.render_gif results/E16_lam0.3/ppo_seed0_best.pt docs/swim.gif --frames 900
"""

import argparse

import torch
from PIL import Image, ImageDraw

from baselines.env import SwimEnv
from brain import Brain
from champion import charger_champion
from config import Config

FOND, GRILLE, OS_, MUSCLE, NOEUD, TRACE, TEXTE = (
    (14, 30, 48), (32, 58, 82), (235, 235, 235), (225, 70, 70), (255, 255, 255), (90, 170, 220), (200, 215, 230))


@torch.no_grad()
def simuler(chemin, frames, seed, bruit, cfg):
    ch = charger_champion(chemin, torch.device("cpu"))
    cerveau = Brain(ch.obs_size, ch.action_size)
    cerveau.load_state_dict(ch.brain_weights)
    cerveau.eval()
    torch.manual_seed(seed)
    env = SwimEnv(ch.dico, 1, frames, cfg.sub_step, cfg.coef_energie, cfg.coef_hauteur,
                  torch.device("cpu"), v_max=cfg.v_max, water_drag=cfg.water_drag)
    obs = env.reset()
    n = ch.num_nodes
    etats = [(env.mega.X[0, 0, :n].clone(), env.mega.Y[0, 0, :n].clone())]
    done = False
    while not done:
        a = cerveau(obs)
        # une décision = 5 frames : on enregistre chaque frame en avançant la physique à la main
        env.mega.apply_action((a + torch.randn_like(a) * bruit).clamp(-1, 1).unsqueeze(0), env.frame_globale)
        for _ in range(5):
            for _ in range(env.sub_step):
                env.mega.apply_physics(env.dt)
            env.frame += 1
            env.frame_globale += 1
            etats.append((env.mega.X[0, 0, :n].clone(), env.mega.Y[0, 0, :n].clone()))
        env.mega.get_reward(env.coef_energie, env.coef_hauteur)
        obs = env.mega.get_observation(env.frame_globale)[0]
        done = env.frame >= env.frame_nb
    return ch, etats


def dessiner(ch, etats, sortie, largeur, hauteur, pas, echelle, fps, titre):
    bxs = [float(x.mean()) for x, _ in etats]
    bys = [float(y.mean()) for _, y in etats]
    x0 = bxs[0]
    cam_x, cam_y = bxs[0], bys[0]
    images = []
    m1, m2, ob = ch.muscle1, ch.muscle2, ch.is_bone
    for f in range(0, len(etats), pas):
        cam_x += 0.15 * (bxs[f] - cam_x) if f else 0.0     # caméra lissée
        cam_y += 0.15 * (bys[f] - cam_y) if f else 0.0
        im = Image.new("RGB", (largeur, hauteur), FOND)
        d = ImageDraw.Draw(im)

        def ecran(x, y):
            return (largeur / 2 + (x - cam_x) * echelle, hauteur / 2 + (y - cam_y) * echelle)

        pas_grille = 50
        gx = cam_x - (largeur / 2) / echelle
        k0 = int(gx // pas_grille) - 1
        for k in range(k0, k0 + int(largeur / (pas_grille * echelle)) + 3):
            xe, _ = ecran(k * pas_grille, 0)
            d.line([(xe, 0), (xe, hauteur)], fill=GRILLE, width=1)
        gy = cam_y - (hauteur / 2) / echelle
        k0 = int(gy // pas_grille) - 1
        for k in range(k0, k0 + int(hauteur / (pas_grille * echelle)) + 3):
            _, ye = ecran(0, k * pas_grille)
            d.line([(0, ye), (largeur, ye)], fill=GRILLE, width=1)
        trace = [ecran(bxs[i], bys[i]) for i in range(0, f + 1, 3)]
        if len(trace) > 1:
            d.line(trace, fill=TRACE, width=2)
        X, Y = etats[f]
        for i in range(len(m1)):
            p1, p2 = ecran(float(X[m1[i]]), float(Y[m1[i]])), ecran(float(X[m2[i]]), float(Y[m2[i]]))
            d.line([p1, p2], fill=OS_ if ob[i] else MUSCLE, width=4 if ob[i] else 3)
        for i in range(len(X)):
            xe, ye = ecran(float(X[i]), float(Y[i]))
            d.ellipse([xe - 4, ye - 4, xe + 4, ye + 4], fill=NOEUD)
        d.text((10, 8), titre, fill=TEXTE)
        d.text((10, hauteur - 20), f"frame {f:4d}   distance {bxs[f] - x0:6.0f}", fill=TEXTE)
        images.append(im)
    images[0].save(sortie, save_all=True, append_images=images[1:], duration=int(1000 / fps), loop=0, optimize=True)
    return len(images)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("checkpoint")
    p.add_argument("sortie")
    p.add_argument("--frames", type=int, default=900)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--bruit", type=float, default=0.02)
    p.add_argument("--largeur", type=int, default=720)
    p.add_argument("--hauteur", type=int, default=320)
    p.add_argument("--echelle", type=float, default=1.0, help="pixels par unité de la simulation")
    p.add_argument("--pas", type=int, default=3, help="une image du GIF toutes les N frames")
    p.add_argument("--fps", type=float, default=25.0)
    p.add_argument("--titre", default="")
    a = p.parse_args()
    ch, etats = simuler(a.checkpoint, a.frames, a.seed, a.bruit, Config())
    n = dessiner(ch, etats, a.sortie, a.largeur, a.hauteur, a.pas, a.echelle, a.fps, a.titre)
    print(f"🎞  {a.sortie} : {n} images, distance {float(etats[-1][0].mean() - etats[0][0].mean()):.0f}")


if __name__ == "__main__":
    main()
