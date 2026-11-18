"""
physique_v2.py — physique v2, étape 1 : os et muscles en contraintes XPBD.

Même interface que megaVecto.MegaCrea (observations, actions, récompense) : seule
apply_physics change. La v1 reste intacte ; on choisit la v2 avec `--physique v2`.

Pourquoi : en v1, un lien est un ressort intégré explicitement, stable seulement si
ω·Δt ≤ 2 (ω² = k/m) : des muscles plus forts ou des os rigides exigent un pas plus petit.
XPBD (Macklin, Müller, Chentanez, 2016) écrit chaque lien comme une contrainte
C = |x_a − x_b| − L de souplesse α = 1/k et corrige les positions d'un pas d'Euler
implicite (fiche « XPBD sur un ressort ») : les liens ne peuvent pas faire exploser la
simulation, et un os peut être parfaitement rigide (α = 0).

Un sous-pas de durée Δt :
  1. traînée de l'eau (même modèle que la v1, explicite) :   v ← v + Δt·F_eau/m
  2. prédiction par inertie :                                 x̃ = x + Δt·v
  3. contraintes (`iterations` passes, λ repart de 0 à chaque sous-pas) :
       Δλ = (−C − α̃·λ) / (w_a + w_b + α̃),   α̃ = α/Δt²,   w = 1/m
       x_a += w_a·Δλ·∇_a C,   x_b += w_b·Δλ·∇_b C,   λ += Δλ
     Gauss-Seidel par couleurs : les liens sont coloriés (une couleur = des liens sans nœud
     commun) et corrigés couleur après couleur, sans conflit. (Jacobi — tous les liens en une
     passe, moyennés par nœud — a été essayé et écarté : avec 1 ou 2 passes il rend les
     liens bien trop mous et change la physique.)
  4. vitesses : v = (x_nouveau − x_début)/Δt
  Amortissement (c = 2√k comme en v1), intégré à la correction (Macklin et al. 2016, éq. 26) :
       Δλ = (−C − α̃·λ − γ·∇C·(x − x_début)) / ((1 + γ)(w_a + w_b) + α̃),   γ = α·c/Δt
     Une seule boucle sur les couleurs par sous-pas. Option `amort_integre=False` : ancien
     amortissement implicite séparé, v_rel' = v_rel / (1 + Δt·c·(w_a + w_b)), plus lent et
     plus dissipatif à grand pas.

Conservation : chaque correction déplace a et b de w_a·Δλ·n et −w_b·Δλ·n ; pondérées
par les masses, elles s'annulent (m_a·w_a = m_b·w_b = 1). Sans eau, la quantité de
mouvement totale est conservée (testé).
"""

import torch

from megaVecto import MegaCrea

FACTEUR_TANGENTIEL = 0.3


def colorier_liens(muscle1, muscle2, masque_muscles):
    """Couleur de chaque lien [P, M] (−1 pour un lien fantôme) : deux liens d'une même
    couleur n'ont aucun nœud commun. Coloration gloutonne, une fois par créature."""
    P, M = muscle1.shape
    couleurs = torch.full((P, M), -1, dtype=torch.long)
    m1, m2, mk = muscle1.cpu().tolist(), muscle2.cpu().tolist(), masque_muscles.cpu().tolist()
    for p in range(P):
        noeuds_par_couleur = []
        for m in range(M):
            if mk[p][m] == 0:
                continue
            a, b = m1[p][m], m2[p][m]
            c = 0
            while c < len(noeuds_par_couleur) and (a in noeuds_par_couleur[c] or b in noeuds_par_couleur[c]):
                c += 1
            if c == len(noeuds_par_couleur):
                noeuds_par_couleur.append(set())
            noeuds_par_couleur[c].update((a, b))
            couleurs[p, m] = c
    return couleurs.to(muscle1.device)


def pas_xpbd(X, Y, vX, vY, D, D1, D2, couleurs, target, alpha, c, kN, kT, w, mask_N, dt, v_max,
             iterations: int, amort_integre: bool = True):
    """Un sous-pas XPBD. Fonction pure (compilable par torch.compile).

    Matrices d'incidence [P, N, M] : D = +1 en m2, −1 en m1 (différences) ; D1, D2 = indicatrices
    de m1 et m2. Pour une grandeur de lien g [P, B, M], la somme sur les nœuds est D @ g :
    m2 reçoit +g, m1 reçoit −g, sans gather ni scatter. couleurs : [C, P, B, M] (0/1)."""
    def diff(Z):                                        # z_m2 − z_m1, [P, B, M]
        return torch.einsum("pbn,pnm->pbm", Z, D)

    def vers_noeuds(g):                                 # Σ_m D[n, m]·g[m], [P, B, N]
        return torch.einsum("pbm,pnm->pbn", g, D)

    w1 = torch.einsum("pbn,pnm->pbm", w, D1)
    w2 = torch.einsum("pbn,pnm->pbm", w, D2)
    wsum = w1 + w2

    # 1. traînée (modèle v1, explicite) — chaque extrémité reçoit la moitié
    dx, dy = diff(X), diff(Y)
    d = torch.sqrt(dx * dx + dy * dy + 1e-3)
    ux, uy = dx / d, dy / d
    vmx = 0.5 * torch.einsum("pbn,pnm->pbm", vX, D1 + D2)
    vmy = 0.5 * torch.einsum("pbn,pnm->pbm", vY, D1 + D2)
    gN = -d * (vmx * -uy + vmy * ux) * kN
    gT = -d * (vmx * ux + vmy * uy) * kT
    fx = gN * -uy + gT * ux
    fy = gN * ux + gT * uy
    vX = vX + dt * w * 0.5 * torch.einsum("pbm,pnm->pbn", fx, D1 + D2)
    vY = vY + dt * w * 0.5 * torch.einsum("pbm,pnm->pbn", fy, D1 + D2)

    # 2. prédiction par inertie
    X0, Y0 = X, Y
    X = X + dt * vX
    Y = Y + dt * vY

    # 3. contraintes
    alpha_t = alpha / (dt * dt)
    # amortissement intégré à la contrainte (Macklin et al. 2016, éq. 26) : γ = α·c/Δt
    gamma = alpha * c / dt if amort_integre else torch.zeros_like(alpha)
    dX0, dY0 = diff(X0), diff(Y0)
    lam = torch.zeros_like(target)
    for _ in range(iterations):
        for k in range(couleurs.shape[0]):
            mc = couleurs[k]
            dx, dy = diff(X), diff(Y)
            d = torch.sqrt(dx * dx + dy * dy + 1e-8)
            nx, ny = dx / d, dy / d
            vit = nx * (dx - dX0) + ny * (dy - dY0)          # ∇C·(x − x_début)
            dlam = (target - d - alpha_t * lam - gamma * vit) / ((1 + gamma) * wsum + alpha_t + 1e-12) * mc
            lam = lam + dlam
            cx, cy = vers_noeuds(dlam * nx), vers_noeuds(dlam * ny)     # m2 : +, m1 : −
            X = X + w * cx
            Y = Y + w * cy

    # 4. vitesses
    vX = (X - X0) / dt
    vY = (Y - Y0) / dt

    # 5. amortissement implicite le long des liens (sauf s'il est déjà intégré à l'étape 3)
    dx, dy = diff(X), diff(Y)
    d = torch.sqrt(dx * dx + dy * dy + 1e-8)
    nx, ny = dx / d, dy / d
    for k in range(0 if amort_integre else couleurs.shape[0]):
        mc = couleurs[k]
        vrel = diff(vX) * nx + diff(vY) * ny
        p = (vrel / (1 + dt * c * wsum) - vrel) / (wsum + 1e-12) * mc
        cx, cy = vers_noeuds(p * nx), vers_noeuds(p * ny)
        vX = vX + w * cx
        vY = vY + w * cy

    vX = torch.clamp(vX, -v_max, v_max) * mask_N
    vY = torch.clamp(vY, -v_max, v_max) * mask_N
    return X, Y, vX, vY, lam / (dt * dt)


_PAS_XPBD_COMPILE = None


def _pas_xpbd_compile():
    global _PAS_XPBD_COMPILE
    if _PAS_XPBD_COMPILE is None:
        _PAS_XPBD_COMPILE = torch.compile(pas_xpbd)
    return _PAS_XPBD_COMPILE


class MegaCreaXPBD(MegaCrea):
    def __init__(self, dico_mega_tenseurs, batch_size, device="cuda", iterations=1,
                 os_rigides=False, facteur_raideur=1.0, compile=False, amort_integre=True):
        super().__init__(dico_mega_tenseurs, batch_size, device=device)
        self.iterations = iterations
        self.amort_integre = amort_integre
        B = batch_size
        P, _, N = self.X.shape
        M = self.muscle1.shape[1]
        dev, dt_ = self.X.device, self.X.dtype
        k = self.stiffness.unsqueeze(1).expand(-1, B, -1) * facteur_raideur        # [P, B, M]
        os_ = self.is_bone_exp
        alpha = torch.where(k > 0, 1.0 / torch.clamp(k, min=1e-12), torch.zeros_like(k))
        if os_rigides:
            alpha = torch.where(os_ == 1.0, torch.zeros_like(alpha), alpha)
        self._alpha = alpha.contiguous()
        self._c = (2.0 * torch.sqrt(k)).contiguous()                                # amortissement, comme la v1
        # matrices d'incidence (liens fantômes : colonnes nulles)
        mk = self.masque_muscles.to(dt_).unsqueeze(1)                               # [P, 1, M]
        self._D1 = torch.zeros(P, N, M, dtype=dt_, device=dev).scatter_(1, self.muscle1.unsqueeze(1), 1.0) * mk
        self._D2 = torch.zeros(P, N, M, dtype=dt_, device=dev).scatter_(1, self.muscle2.unsqueeze(1), 1.0) * mk
        self._D = self._D2 - self._D1
        self._w = (self.mask_N_exp / self.masses).contiguous()                     # 0 pour un nœud fantôme
        col = colorier_liens(self.muscle1, self.muscle2, self.masque_muscles)
        self._couleurs = torch.stack([(col == c).to(dt_).unsqueeze(1).expand(-1, B, -1)
                                      for c in range(int(col.max().item()) + 1)]).contiguous()
        self._facteur_trainee = ((os_ + (1 - os_) * 0.3) * self.mask_M_exp).contiguous()
        self._pas = _pas_xpbd_compile() if compile else pas_xpbd
        self.force_liens = torch.zeros_like(k)                                      # λ/Δt² du dernier sous-pas

    def _paires(self, X, Y):
        """Positions des extrémités m1, m2 de chaque lien (pour les tests et les outils)."""
        x1 = torch.einsum("pbn,pnm->pbm", X, self._D1)
        y1 = torch.einsum("pbn,pnm->pbm", Y, self._D1)
        x2 = torch.einsum("pbn,pnm->pbm", X, self._D2)
        y2 = torch.einsum("pbn,pnm->pbm", Y, self._D2)
        return x1, y1, x2, y2

    def apply_physics(self, dt):
        kN = self.water_drag * self._facteur_trainee
        self.X, self.Y, self.vX, self.vY, self.force_liens = self._pas(
            self.X, self.Y, self.vX, self.vY, self._D, self._D1, self._D2, self._couleurs,
            self.target_length, self._alpha, self._c, kN, kN * FACTEUR_TANGENTIEL, self._w,
            self.mask_N_exp, float(dt), self.v_max, self.iterations, self.amort_integre)
