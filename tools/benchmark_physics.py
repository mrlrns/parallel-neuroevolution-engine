"""
Compare megaVecto.MegaCrea (actuelle) et megaVecto_fast.MegaCreaFast (optimisée).

PARTIE 1 — la physique ne change pas (toujours sur CPU, déterministe)
  Les deux versions partent du même état et reçoivent les mêmes actions.
  a) écart des trajectoires après 1 sous-pas, 1 frame, 50 frames, 300 frames,
     en float32 puis en float64. Équations identiques ⇒ l'écart n'est que de
     l'arrondi : ~1e-7 relatif en float32, ~1e-15 en float64. S'il restait grand en
     float64, ce serait une vraie différence de physique.
  b) écart des gradients d/d(actions) d'une récompense sur une fenêtre de 10 frames
     (ce que train2 rétropropage).
  c) déplacement moyen sur 300 frames : identique à l'arrondi près.

PARTIE 2 — le temps gagné (sur le device demandé, GPU si disponible)
  ms par sous-pas, sans gradient (cas de PPO / SwimEnv) et avec gradient
  (forward + backward sur une fenêtre, cas de train2), pour : actuelle,
  optimisée, optimisée + torch.compile (si --compile).

Usage :
    python -m tools.benchmark_physics                     # CPU
    python -m tools.benchmark_physics --device cuda --compile
"""

import argparse
import time

import torch

from champion import charger_champion
from megaVecto import MegaCrea
from megaVecto_fast import MegaCreaFast

CHEMIN = "champion_raffine/reference.pt"


def dico_en(dico, dtype, device):
    return {k: (v.to(device=device, dtype=dtype) if v.is_floating_point() else v.to(device)) for k, v in dico.items()}


def actions(nb_decisions, n, a, seed):
    g = torch.Generator().manual_seed(seed)
    return 0.8 * (2 * torch.rand(nb_decisions, 1, n, a, generator=g) - 1)


def jouer(classe, dico, n, frames, sub_step, acts, seed, dtype, device, **kw):
    torch.manual_seed(seed)
    mega = classe(dico, n, device=device, **kw)
    dt = torch.tensor(1.0 / sub_step, dtype=dtype, device=device)
    traj = {}
    for frame in range(frames):
        if frame % 5 == 0:
            mega.apply_action(acts[frame // 5].to(device=device, dtype=dtype), frame)
        for s in range(sub_step):
            mega.apply_physics(dt)
            if frame == 0 and s == 0:
                traj[0] = (mega.X.clone(), mega.Y.clone())
        if frame + 1 in (1, 50, 300):
            traj[frame + 1] = (mega.X.clone(), mega.Y.clone())
    m = mega.mask_N_exp
    return traj, ((mega.X * m).sum(2) / m.sum(2)).mean().item()


def ecart(a, b):
    return max((a[0] - b[0]).abs().max().item(), (a[1] - b[1]).abs().max().item())


def gradient(classe, dico, n, sub_step, acts, seed, dtype):
    torch.manual_seed(seed)
    mega = classe(dico, n, device="cpu")
    dt = torch.tensor(1.0 / sub_step, dtype=dtype)
    a = acts[:2].to(dtype).clone().requires_grad_(True)
    total = 0
    for frame in range(10):
        if frame % 5 == 0:
            mega.apply_action(a[frame // 5], frame)
        for _ in range(sub_step):
            mega.apply_physics(dt)
        if frame % 5 == 4:
            total = total + mega.get_reward(10000.0, 0.0).sum()
    total.backward()
    return a.grad


def partie_equivalence(champ, sub_step):
    n, A = 16, champ.action_size
    acts = actions(60, n, A, seed=1)
    print("PARTIE 1 — même physique ? (CPU)\n")
    print(f"{'précision':10s} {'1 sous-pas':>12s} {'1 frame':>12s} {'50 frames':>12s} {'300 frames':>12s}"
          f" {'dépl. actuelle':>15s} {'dépl. optimisée':>16s}")
    for dtype in (torch.float32, torch.float64):
        torch.set_default_dtype(dtype)
        d = dico_en(champ.dico, dtype, "cpu")
        t0, x0 = jouer(MegaCrea, d, n, 300, sub_step, acts, 0, dtype, "cpu")
        t1, x1 = jouer(MegaCreaFast, d, n, 300, sub_step, acts, 0, dtype, "cpu")
        nom = "float32" if dtype == torch.float32 else "float64"
        print(f"{nom:10s} {ecart(t0[0], t1[0]):12.2e} {ecart(t0[1], t1[1]):12.2e} {ecart(t0[50], t1[50]):12.2e}"
              f" {ecart(t0[300], t1[300]):12.2e} {x0:15.4f} {x1:16.4f}")
    torch.set_default_dtype(torch.float64)
    d = dico_en(champ.dico, torch.float64, "cpu")
    g0 = gradient(MegaCrea, d, n, sub_step, acts, 0, torch.float64)
    g1 = gradient(MegaCreaFast, d, n, sub_step, acts, 0, torch.float64)
    rel = ((g0 - g1).norm() / g0.norm()).item()
    print(f"\ngradient d(récompense, 10 frames)/d(actions), float64 : écart relatif = {rel:.2e}")
    torch.set_default_dtype(torch.float32)
    print("(positions en unités de la scène, ~50–1000 ; un écart ≲ 1e-3 en float32 n'est que de l'arrondi)\n")


def chrono(fn, device, repetitions):
    for _ in range(3):
        fn()
    if device.type == "cuda":
        torch.cuda.synchronize()
    t = time.perf_counter()
    for _ in range(repetitions):
        fn()
    if device.type == "cuda":
        torch.cuda.synchronize()
    return (time.perf_counter() - t) / repetitions


def partie_vitesse(champ, device, n, sub_step, avec_compile):
    print(f"PARTIE 2 — vitesse ({device}, {n} rollouts, sub_step {sub_step})\n")
    d = dico_en(champ.dico, torch.float32, device)
    dt = torch.tensor(1.0 / sub_step, device=device)
    variantes = [("actuelle", MegaCrea, {}), ("optimisée", MegaCreaFast, {})]
    if avec_compile:
        variantes.append(("optimisée + compile", MegaCreaFast, {"compile": True}))
    a = torch.zeros(1, n, champ.action_size, device=device)
    ref = {}
    print(f"{'version':22s} {'sans gradient (ms/sous-pas)':>28s} {'avec gradient (ms/sous-pas)':>28s}")
    for nom, classe, kw in variantes:
        torch.manual_seed(0)
        mega = classe(d, n, device=device, **kw)
        mega.apply_action(a, 0)

        def sans_grad():
            with torch.no_grad():
                for _ in range(sub_step * 5):
                    mega.apply_physics(dt)

        def avec_grad():
            act = a.clone().requires_grad_(True)
            mega.X, mega.Y = mega.X.detach(), mega.Y.detach()
            mega.vX, mega.vY = mega.vX.detach(), mega.vY.detach()
            mega.apply_action(act, 0)
            for _ in range(sub_step * 10):
                mega.apply_physics(dt)
            (mega.X.sum() + mega.vX.sum()).backward()
            mega.target_length = mega.target_length.detach()

        t_ng = chrono(sans_grad, device, 5) / (sub_step * 5) * 1e3
        t_g = chrono(avec_grad, device, 3) / (sub_step * 10) * 1e3
        ref.setdefault("ng", t_ng)
        ref.setdefault("g", t_g)
        print(f"{nom:22s} {t_ng:20.3f}  (×{ref['ng'] / t_ng:4.1f}) {t_g:20.3f}  (×{ref['g'] / t_g:4.1f})")
    par_ep = ref["ng"] * sub_step * 300 / 1e3
    print(f"\nrepère : version actuelle sans gradient ≈ {par_ep:.1f} s de physique par épisode de 300 frames")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--chemin-champion", default=CHEMIN)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--rollouts", type=int, default=2000)
    p.add_argument("--sub-step", type=int, default=20)
    p.add_argument("--compile", action="store_true")
    p.add_argument("--sans-equivalence", action="store_true")
    a = p.parse_args()

    champ = charger_champion(a.chemin_champion, torch.device("cpu"))
    if not a.sans_equivalence:
        partie_equivalence(champ, a.sub_step)
    partie_vitesse(champ, torch.device(a.device), a.rollouts, a.sub_step, a.compile)


if __name__ == "__main__":
    main()
