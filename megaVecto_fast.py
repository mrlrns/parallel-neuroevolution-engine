"""
megaVecto_fast.py — même physique que megaVecto.MegaCrea, calculée autrement.

MegaCrea.apply_physics lance ~80 petites opérations GPU par sous-pas (8 gather,
4 scatter_add_, 2 zeros_like, ~60 opérations élément par élément) sur des
tenseurs de ~30 000 nombres : le coût fixe de lancement domine le calcul.

MegaCreaFast hérite de MegaCrea et ne change QUE apply_physics :

1. Matrices d'incidence précalculées (la topologie est figée pendant un épisode)
       D[p, n, m] = +1 si n = m2[m], −1 si n = m1[m]       (différences)
       A[p, n, m] = ½ si n ∈ {m1[m], m2[m]}                  (moyennes)
   — 8 gather       → 1 produit (X, Y, vX, vY empilés) @ D  et  1 produit (vX, vY) @ A
   — 4 scatter_add_ → −F_axiale @ Dᵀ  +  F_traînée @ Aᵀ     (2 produits)
   Pour un lien fantôme (m1 = m2 = 0) la colonne de D est nulle, et sa traînée
   est masquée comme avant.
   Bonus : plus de scatter_add_ atomique, première source de non-déterminisme GPU (E11).
2. Constantes précalculées une fois (raideur étendue, coefficients de traînée
   avec le masque des liens déjà intégré) au lieu d'être recalculées à chaque sous-pas.
3. Option compile=True : torch.compile fusionne les opérations élément par
   élément en quelques kernels.

Les équations sont identiques ; seul l'ordre des additions flottantes change
(arrondi ~1e-7 en float32). tools/benchmark_physics.py le vérifie, et mesure le gain.
"""

import torch

from megaVecto import MegaCrea

WATER_DRAG = 0.005
FACTEUR_TANGENTIEL = 0.3
V_MAX = 20.0


def pas_physique(X, Y, vX, vY, D, A, target_length, stiff, c, kN, kT, mask_M, mask_N, masses, dt):
    """Un sous-pas de physique. Fonction pure : compilable par torch.compile."""
    S = torch.stack([X, Y, vX, vY], dim=2)                               # [P, B, 4, N]
    diff = torch.einsum("pbkn,pnm->pbkm", S, D)                          # x2 − x1 pour X, Y, vX, vY
    moy = torch.einsum("pbkn,pnm->pbkm", S[:, :, 2:], A)                 # (v1 + v2) / 2
    dx, dy, dvx, dvy = diff.unbind(2)
    vmx, vmy = moy.unbind(2)

    distances = torch.sqrt(dx * dx + dy * dy + 1e-3)
    dirX = dx / distances
    dirY = dy / distances

    # Ressort + amortissement le long du lien
    f_axiale = (stiff * (distances - target_length) + c * ((dvx * dirX + dvy * dirY))) * mask_M

    # Traînée : normale n = (−dirY, dirX), tangentielle le long de dir
    gN = -distances * (vmx * -dirY + vmy * dirX) * kN                    # kN = drag · facteur os/muscle · masque
    gT = -distances * (vmx * dirX + vmy * dirY) * kT
    ftrX = gN * -dirY + gT * dirX
    ftrY = gN * dirX + gT * dirY

    F_ax = torch.stack([f_axiale * dirX, f_axiale * dirY], dim=2)        # [P, B, 2, M]
    F_tr = torch.stack([ftrX, ftrY], dim=2)
    # m1 reçoit +F_axiale, m2 reçoit −F_axiale ; chaque extrémité reçoit la moitié de la traînée
    Fn = -torch.einsum("pbkm,pnm->pbkn", F_ax, D) + torch.einsum("pbkm,pnm->pbkn", F_tr, A)
    FX, FY = Fn.unbind(2)

    vX = torch.clamp(vX + (FX / masses) * dt, -V_MAX, V_MAX) * mask_N
    vY = torch.clamp(vY + (FY / masses) * dt, -V_MAX, V_MAX) * mask_N
    return X + vX * dt, Y + vY * dt, vX, vY


_PAS_COMPILE = None


def _pas_compile():
    """Une seule fonction compilée par processus, partagée par toutes les instances
    (évite de recompiler à chaque épisode)."""
    global _PAS_COMPILE
    if _PAS_COMPILE is None:
        _PAS_COMPILE = torch.compile(pas_physique)
    return _PAS_COMPILE


def classe_physique(compilee):
    """MegaCrea (référence) ou MegaCreaFast compilée, avec la même signature."""
    if not compilee:
        return MegaCrea
    return lambda dico, batch_size, device="cuda": MegaCreaFast(dico, batch_size, device=device, compile=True)


class MegaCreaFast(MegaCrea):
    def __init__(self, dico_mega_tenseurs, batch_size, device="cuda", compile=False):
        super().__init__(dico_mega_tenseurs, batch_size, device=device)
        P, _, N = self.X.shape
        M = self.muscle1.shape[1]
        dt_ = self.X.dtype
        un = torch.ones(P, 1, M, dtype=dt_, device=self.X.device)
        self._D = torch.zeros(P, N, M, dtype=dt_, device=self.X.device)
        self._D.scatter_add_(1, self.muscle2.unsqueeze(1), un)
        self._D.scatter_add_(1, self.muscle1.unsqueeze(1), -un)
        self._A = torch.zeros(P, N, M, dtype=dt_, device=self.X.device)
        self._A.scatter_add_(1, self.muscle1.unsqueeze(1), 0.5 * un)
        self._A.scatter_add_(1, self.muscle2.unsqueeze(1), 0.5 * un)

        self._stiff = self.stiffness.unsqueeze(1).expand(-1, batch_size, -1)
        facteur = self.is_bone_exp + (1 - self.is_bone_exp) * 0.3
        self._kN = WATER_DRAG * facteur * self.mask_M_exp
        self._kT = WATER_DRAG * FACTEUR_TANGENTIEL * facteur * self.mask_M_exp
        self._pas = _pas_compile() if compile else pas_physique

    def apply_physics(self, dt):
        self.X, self.Y, self.vX, self.vY = self._pas(
            self.X, self.Y, self.vX, self.vY, self._D, self._A, self.target_length,
            self._stiff, self.c, self._kN, self._kT, self.mask_M_exp, self.mask_N_exp, self.masses, dt)
