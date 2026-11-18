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
  3. contraintes (Gauss-Seidel par couleurs, `iterations` passes, λ repart de 0) :
       Δλ = (−C − α̃·λ) / (w_a + w_b + α̃),   α̃ = α/Δt²,   w = 1/m
       x_a += w_a·Δλ·∇_a C,   x_b += w_b·Δλ·∇_b C,   λ += Δλ
     Les liens sont coloriés (une couleur = des liens sans nœud commun) : tous les liens
     d'une couleur sont corrigés en parallèle sans conflit, couleur après couleur.
  4. vitesses : v = (x_nouveau − x_début)/Δt
  5. amortissement le long de chaque lien, implicite (donc stable pour tout c) :
       v_rel' = v_rel / (1 + Δt·c·(w_a + w_b)),   c = 2√k comme en v1.

Conservation : chaque correction déplace a et b de w_a·Δλ·n et −w_b·Δλ·n ; pondérées
par les masses, elles s'annulent (m_a·w_a = m_b·w_b = 1). Sans eau, la quantité de
mouvement totale est conservée (testé).
"""

import torch

from megaVecto import MegaCrea


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


class MegaCreaXPBD(MegaCrea):
    def __init__(self, dico_mega_tenseurs, batch_size, device="cuda", iterations=1,
                 os_rigides=False, facteur_raideur=1.0):
        super().__init__(dico_mega_tenseurs, batch_size, device=device)
        self.iterations = iterations
        B = batch_size
        k = self.stiffness.unsqueeze(1).expand(-1, B, -1) * facteur_raideur        # [P, B, M]
        os_ = self.is_bone_exp
        # souplesse α = 1/k (0 pour un os rigide) ; liens fantômes : sans effet (masqués)
        alpha = torch.where(k > 0, 1.0 / torch.clamp(k, min=1e-12), torch.zeros_like(k))
        if os_rigides:
            alpha = torch.where(os_ == 1.0, torch.zeros_like(alpha), alpha)
        self._alpha = alpha
        self._c = 2.0 * torch.sqrt(k)                                               # amortissement, comme la v1
        self._m1 = self.muscle1.unsqueeze(1).expand(-1, B, -1)
        self._m2 = self.muscle2.unsqueeze(1).expand(-1, B, -1)
        self._w = self.mask_N_exp / self.masses                                     # 0 pour un nœud fantôme
        couleurs = colorier_liens(self.muscle1, self.muscle2, self.masque_muscles)
        self._masques_couleur = [(couleurs == c).float().unsqueeze(1).expand(-1, B, -1)
                                 for c in range(int(couleurs.max().item()) + 1)]
        self._facteur_trainee = (os_ + (1 - os_) * 0.3) * self.mask_M_exp
        self.force_liens = torch.zeros_like(k)                                      # λ/Δt² du dernier sous-pas

    # ------------------------------------------------------------------ outils
    def _paires(self, X, Y):
        x1, y1 = X.gather(2, self._m1), Y.gather(2, self._m1)
        x2, y2 = X.gather(2, self._m2), Y.gather(2, self._m2)
        return x1, y1, x2, y2

    def _repartir(self, val1, val2):
        """Somme sur les nœuds de val1 (vers m1) et val2 (vers m2) : [P, B, N]."""
        out = torch.zeros_like(self.X)
        return out.scatter_add(2, self._m1, val1).scatter_add(2, self._m2, val2)

    # ------------------------------------------------------------------ un sous-pas
    def apply_physics(self, dt):
        dt = float(dt)
        X0, Y0 = self.X, self.Y
        m1w, m2w = self._w.gather(2, self._m1), self._w.gather(2, self._m2)

        # 1. traînée (modèle v1, explicite)
        x1, y1, x2, y2 = self._paires(X0, Y0)
        vx1, vy1, vx2, vy2 = self._paires(self.vX, self.vY)
        dx, dy = x2 - x1, y2 - y1
        d = torch.sqrt(dx * dx + dy * dy + 1e-3)
        ux, uy = dx / d, dy / d
        vmx, vmy = (vx1 + vx2) / 2, (vy1 + vy2) / 2
        kN = self.water_drag * self._facteur_trainee
        kT = kN * 0.3
        gN = -d * (vmx * -uy + vmy * ux) * kN
        gT = -d * (vmx * ux + vmy * uy) * kT
        fx, fy = gN * -uy + gT * ux, gN * ux + gT * uy
        FX, FY = self._repartir(fx / 2, fx / 2), self._repartir(fy / 2, fy / 2)
        vX = self.vX + dt * FX * self._w
        vY = self.vY + dt * FY * self._w

        # 2. prédiction par inertie
        X = X0 + dt * vX
        Y = Y0 + dt * vY

        # 3. contraintes, Gauss-Seidel par couleurs
        alpha_t = self._alpha / (dt * dt)
        lam = torch.zeros_like(self.target_length)
        for _ in range(self.iterations):
            for mc in self._masques_couleur:
                x1, y1, x2, y2 = self._paires(X, Y)
                dx, dy = x2 - x1, y2 - y1
                d = torch.sqrt(dx * dx + dy * dy + 1e-8)
                nx, ny = dx / d, dy / d                       # ∇_b C = n, ∇_a C = −n
                C = d - self.target_length
                dlam = (-C - alpha_t * lam) / (m1w + m2w + alpha_t + 1e-12) * mc
                lam = lam + dlam
                X = X + self._repartir(-m1w * dlam * nx, m2w * dlam * nx)
                Y = Y + self._repartir(-m1w * dlam * ny, m2w * dlam * ny)
        self.force_liens = lam / (dt * dt)

        # 4. vitesses
        vX = (X - X0) / dt
        vY = (Y - Y0) / dt

        # 5. amortissement implicite le long des liens
        for mc in self._masques_couleur:
            x1, y1, x2, y2 = self._paires(X, Y)
            dx, dy = x2 - x1, y2 - y1
            d = torch.sqrt(dx * dx + dy * dy + 1e-8)
            nx, ny = dx / d, dy / d
            vx1, vy1, vx2, vy2 = self._paires(vX, vY)
            vrel = (vx2 - vx1) * nx + (vy2 - vy1) * ny
            wsum = m1w + m2w
            p = (vrel / (1 + dt * self._c * wsum) - vrel) / (wsum + 1e-12) * mc
            vX = vX + self._repartir(-m1w * p * nx, m2w * p * nx)
            vY = vY + self._repartir(-m1w * p * ny, m2w * p * ny)

        self.vX = torch.clamp(vX, -self.v_max, self.v_max) * self.mask_N_exp
        self.vY = torch.clamp(vY, -self.v_max, self.v_max) * self.mask_N_exp
        self.X, self.Y = X, Y
