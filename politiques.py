"""
politiques.py — structures de commande pour E22 : symétrie imposée et commande rythmique.

Deux options, indépendantes et désactivées par défaut (le Brain historique reste la règle) :

- Symétrie (`--symetrie`) : les muscles images l'un de l'autre par la symétrie miroir du
  corps reçoivent la même commande. Les paires sont trouvées par la géométrie (nœud i ↔
  nœud j si x_i = x_j et y_i + y_j = 2·y_axe). La politique produit une commande par
  GROUPE ; `Symetriseur.etendre` la recopie sur chaque muscle du groupe.
  Pour un Brain, la symétrie porte sur la pré-activation : z_groupe = moyenne des z des
  muscles du groupe, puis tanh. Comme la recopie commute avec tanh (élément par élément),
  ce Brain s'enregistre comme un Brain ordinaire dont les lignes de la dernière couche
  sont moyennées par groupe : le checkpoint reste lisible par tous les outils.

- Commande rythmique (`--politique-rythmique`) : le réseau ne reçoit que la phase d'une
  horloge de nage, φ = 2π·frame/T, sous la forme (sin φ, cos φ). La commande est donc une
  fonction lisse et T-périodique du temps : fluide par construction, sans pénalité à doser.
  Elle ne réagit pas à l'état du corps (boucle ouverte apprise).

Les checkpoints rythmiques portent une clé 'politique' ; `politique_depuis_checkpoint`
reconstruit, pour n'importe quel checkpoint, une fonction f(obs, frame) -> commande complète.
"""

import math

import torch
import torch.nn as nn

from brain import Brain


# ---------------------------------------------------------------- symétrie
def groupes_symetrie(x, y, muscle1, muscle2, is_bone, tol=1e-3):
    """Groupes d'indices de vrais muscles échangés par la symétrie miroir horizontale.

    Retourne une liste de listes (taille 1 pour un muscle sur l'axe, 2 pour une paire).
    Lève ValueError si le corps n'est pas exactement symétrique."""
    n = len(x)
    y_axe = (min(y) + max(y)) / 2
    miroir = []
    for i in range(n):
        cands = [j for j in range(n) if abs(x[i] - x[j]) < tol and abs(y[i] + y[j] - 2 * y_axe) < tol]
        if len(cands) != 1:
            raise ValueError(f"nœud {i} : pas d'image miroir unique (corps non symétrique ?)")
        miroir.append(cands[0])
    liens = {frozenset((a, b)): k for k, (a, b) in enumerate(zip(muscle1, muscle2))}
    vus, groupes = set(), []
    for k, (a, b) in enumerate(zip(muscle1, muscle2)):
        if is_bone[k] or k in vus:
            continue
        image = liens.get(frozenset((miroir[a], miroir[b])))
        if image is None or is_bone[image]:
            raise ValueError(f"muscle {k} : pas de muscle image")
        groupe = sorted({k, image})
        vus.update(groupe)
        groupes.append(groupe)
    return groupes


class Symetriseur(nn.Module):
    """Passage commande par groupe [.., g] <-> commande complète [.., action_size]."""

    def __init__(self, groupes, action_size):
        super().__init__()
        g = len(groupes)
        E = torch.zeros(action_size, g)          # recopie : a = E r
        P = torch.zeros(g, action_size)          # moyenne : r = P a
        for j, groupe in enumerate(groupes):
            for k in groupe:
                E[k, j] = 1.0
                P[j, k] = 1.0 / len(groupe)
        self.register_buffer("E", E)
        self.register_buffer("P", P)
        self.groupes = [list(gr) for gr in groupes]

    @property
    def n_groupes(self):
        return self.P.shape[0]

    def etendre(self, r):
        return r @ self.E.T

    def reduire(self, a):
        return a @ self.P.T


class BrainSymetrique(nn.Module):
    """Brain dont la sortie est une commande par groupe : tanh(moyenne par groupe des z)."""

    def __init__(self, obs_size, action_size, sym: Symetriseur):
        super().__init__()
        self.brain = Brain(obs_size, action_size)
        self.sym = sym

    def forward(self, obs):
        h = torch.relu(self.brain.layer1(obs))
        h = torch.relu(self.brain.layer2(h))
        return torch.tanh(self.sym.reduire(self.brain.layer3(h)))

    def poids_brain(self):
        """state_dict d'un Brain ordinaire qui donne EXACTEMENT la commande complète.

        z' = E P z (lignes de layer3 moyennées par groupe) puis tanh(z') = E tanh(P z),
        car E ne fait que recopier des coordonnées et tanh agit coordonnée par coordonnée.
        Les sorties hors muscles (os, fantômes) sont mises à 0 : elles n'ont aucun effet."""
        sd = {k: v.detach().clone() for k, v in self.brain.state_dict().items()}
        M = self.sym.E @ self.sym.P                       # [act, act]
        sd["layer3.weight"] = M @ sd["layer3.weight"]
        sd["layer3.bias"] = M @ sd["layer3.bias"]
        return sd


# ---------------------------------------------------------------- commande rythmique
def phase(frame, periode, device=None):
    """(sin φ, cos φ) avec φ = 2π·frame/periode : tenseur [2]."""
    phi = 2 * math.pi * float(frame) / periode
    return torch.tensor([math.sin(phi), math.cos(phi)], dtype=torch.float32, device=device)


class PolitiqueRythmique(nn.Module):
    """Commande = MLP(sin φ, cos φ) : lisse et périodique en temps. Ignore l'état du corps."""

    def __init__(self, n_sorties, periode, cache=32):
        super().__init__()
        self.periode = periode
        self.net = nn.Sequential(nn.Linear(2, cache), nn.Tanh(), nn.Linear(cache, cache), nn.Tanh(),
                                 nn.Linear(cache, n_sorties))

    def forward(self, obs_phase):
        """obs_phase : [..., obs_size + 2], la phase en dernier ; seule la phase est lue."""
        return torch.tanh(self.net(obs_phase[..., -2:]))


def ajouter_phase(obs, frame, periode):
    """Concatène (sin φ, cos φ) à chaque observation [..., obs] -> [..., obs + 2]."""
    p = phase(frame, periode, obs.device).expand(*obs.shape[:-1], 2)
    return torch.cat([obs, p], dim=-1)


# ---------------------------------------------------------------- chargement pour les outils
def politique_depuis_checkpoint(donnees, champ, device):
    """f(obs [n, obs_size], frame_globale) -> commande complète [n, action_size], sans gradient.

    Brain ordinaire (y compris symétrique, enregistré comme tel) ou politique rythmique."""
    meta = donnees.get("politique") if isinstance(donnees, dict) else None
    if meta and meta.get("type") == "rythmique":
        groupes = meta.get("groupes")
        n_sorties = len(groupes) if groupes else champ.action_size
        pol = PolitiqueRythmique(n_sorties, meta["periode"], meta.get("cache", 32)).to(device)
        pol.load_state_dict(meta["etat"])
        pol.eval()
        sym = Symetriseur(groupes, champ.action_size).to(device) if groupes else None

        @torch.no_grad()
        def f(obs, frame):
            r = pol(ajouter_phase(obs, frame, meta["periode"]))
            return sym.etendre(r) if sym is not None else r
        return f

    cerveau = Brain(champ.obs_size, champ.action_size).to(device)
    cerveau.load_state_dict(champ.brain_weights)
    cerveau.eval()

    @torch.no_grad()
    def g(obs, frame):
        return cerveau(obs)
    return g
