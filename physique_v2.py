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

Points d'ancrage (option `ancrages`, défaut désactivé ; voir morphologie.py) : un nœud ancré M
est le point x_M = Σ_j T[j, M]·x_j = (1−s)·x_a + s·x_b de l'os a→b, sans masse ni vitesse
propres. Toutes les matrices d'incidence sont composées par T une fois pour toutes
(D ← T·D) : un lien qui touche M agit sur a et b avec les poids (1−s) et s. Le dénominateur
de Δλ devient ∇C·M⁻¹·∇Cᵀ = Σ_j w_j·D[j, m]² (pour un lien sans ancrage, w_a + w_b). Les
colonnes de T somment à 1 et celles de D à 0, donc celles de T·D aussi : la conservation de
la quantité de mouvement tient toujours. Après chaque sous-pas, x_M est recalculé par T.
Sans ancrage, T = identité n'est pas appliquée : calcul identique à l'étape 1b.

Eau quadratique (option `eau="quadratique"`, défaut "lineaire" = modèle v1) et unités
physiques : 1 unité de longueur = 1 mm, 1 frame = 1/60 s, masse en g, ρ = 10⁻³ g/mm³ (eau).
La créature est une tranche de profondeur h (hors du plan) ; un os est une plaque L × h.
  - traînée sur les os seulement (les muscles ne sont que des moyens de plier les
    articulations : ni traînée ni masse), sauf les arêtes internes (partagées par deux
    triangles d'os, que l'eau ne touche pas) :
        dF_n = −½ρ·C_n·h·|v_n|·v_n·dl,   dF_t = −½ρ·C_t·h·|v_t|·v_t·dl
    intégrée sur l'os par Gauss à 2 points (s = ½ ∓ 1/(2√3), poids ½) : un os qui tourne sur
    lui-même freine aussi. C_n ≈ 2 (plaque face au courant, Re ~ 10⁴), C_t ≈ 0.05 (frottement).
  - implicite linéarisée en chaque point de Gauss, de masse inverse effective w_p :
        J = −a·v / (1 + a·w_p),   a = ½ρ·C·h·(L/2)·|v|·Δt
    (Euler implicite sur v' = −a·v à a figé) : la traînée ne peut pas inverser la vitesse,
    quel que soit le pas.
  - masses : flottabilité neutre (densité de l'eau). Triangle d'os : ρ·h·aire, un tiers par
    sommet ; os hors triangle : ρ·h·L·e (épaisseur e dans le plan), moitié par extrémité.
    Les nœuds ancrés passent leur masse aux extrémités de leur os (poids de T).
  - raideurs : k et c sont multipliés par la masse moyenne d'un nœud m̄, ce qui garde
    ω = √(k/m) et l'amortissement relatif de la v1 (on ne change que les unités).
"""

import warnings

import torch

from megaVecto import MegaCrea
from morphologie import aretes_internes, triangles_os, valider_morphologie

RHO_EAU = 1e-3                                  # g/mm³
GAUSS = (0.5 - 0.5 / 3 ** 0.5, 0.5 + 0.5 / 3 ** 0.5)

FACTEUR_TANGENTIEL = 0.3


def colorier_liens(muscle1, muscle2, masque_muscles, ancrages=None):
    """Couleur de chaque lien [P, M] (−1 pour un lien fantôme) : deux liens d'une même
    couleur n'ont aucun nœud commun. Coloration gloutonne, une fois par créature.
    Avec des ancrages, un nœud ancré compte comme les deux extrémités de son os."""
    P, M = muscle1.shape
    couleurs = torch.full((P, M), -1, dtype=torch.long)
    m1, m2, mk = muscle1.cpu().tolist(), muscle2.cpu().tolist(), masque_muscles.cpu().tolist()
    for p in range(P):
        support = {n: {m1[p][o], m2[p][o]} for n, o, _ in (ancrages[p] if ancrages else ())}
        noeuds_par_couleur = []
        for m in range(M):
            if mk[p][m] == 0:
                continue
            noeuds = support.get(m1[p][m], {m1[p][m]}) | support.get(m2[p][m], {m2[p][m]})
            c = 0
            while c < len(noeuds_par_couleur) and noeuds & noeuds_par_couleur[c]:
                c += 1
            if c == len(noeuds_par_couleur):
                noeuds_par_couleur.append(set())
            noeuds_par_couleur[c].update(noeuds)
            couleurs[p, m] = c
    return couleurs.to(muscle1.device)


def pas_xpbd(X, Y, vX, vY, D, A, wsum, couleurs, target, alpha, c, kN, kT, w, mask_N, dt, v_max,
             iterations: int, amort_integre: bool = True, T=None, quad=None):
    """Un sous-pas XPBD. Fonction pure (compilable par torch.compile).

    Matrices d'incidence [P, N, M] : D = +1 en m2, −1 en m1 (différences) ; A = D1 + D2
    (indicatrices des extrémités). Pour une grandeur de lien g [P, B, M], la somme sur les
    nœuds est D @ g : m2 reçoit +g, m1 reçoit −g, sans gather ni scatter. Avec des ancrages,
    D et A sont déjà composées par T [P, N, N] (x_virtuel = X @ T). wsum [P, B, M] :
    Σ_j w_j·D[j, m]². couleurs : [C, P, B, M] (0/1). quad : None (traînée linéaire v1) ou
    (G1, G2, wp1, wp2, an, at) pour l'eau quadratique : G = matrices des points de Gauss
    (x_p = X @ G), wp = masses inverses effectives, an/at = ½ρ·C·h par lien (0 hors os)."""
    def diff(Z):                                        # z_m2 − z_m1, [P, B, M]
        return torch.einsum("pbn,pnm->pbm", Z, D)

    def vers_noeuds(g):                                 # Σ_m D[n, m]·g[m], [P, B, N]
        return torch.einsum("pbm,pnm->pbn", g, D)

    def projeter(Z):                                    # recalcule les nœuds ancrés
        return Z if T is None else torch.einsum("pbj,pjk->pbk", Z, T)

    # 1. traînée
    if quad is None:                    # modèle v1, explicite — chaque extrémité reçoit la moitié
        dx, dy = diff(X), diff(Y)
        d = torch.sqrt(dx * dx + dy * dy + 1e-3)
        ux, uy = dx / d, dy / d
        vmx = 0.5 * torch.einsum("pbn,pnm->pbm", vX, A)
        vmy = 0.5 * torch.einsum("pbn,pnm->pbm", vY, A)
        gN = -d * (vmx * -uy + vmy * ux) * kN
        gT = -d * (vmx * ux + vmy * uy) * kT
        fx = gN * -uy + gT * ux
        fy = gN * ux + gT * uy
        vX = vX + dt * w * 0.5 * torch.einsum("pbm,pnm->pbn", fx, A)
        vY = vY + dt * w * 0.5 * torch.einsum("pbm,pnm->pbn", fy, A)
    else:                               # quadratique, 2 points de Gauss, implicite linéarisée
        G1, G2, wp1, wp2, an, at = quad
        dx, dy = diff(X), diff(Y)
        d = torch.sqrt(dx * dx + dy * dy + 1e-8)
        ux, uy = dx / d, dy / d
        ix = torch.zeros_like(vX)
        iy = torch.zeros_like(vY)
        for G, wp in ((G1, wp1), (G2, wp2)):
            px = torch.einsum("pbn,pnm->pbm", vX, G)
            py = torch.einsum("pbn,pnm->pbm", vY, G)
            vn = px * -uy + py * ux
            vt = px * ux + py * uy
            a_n = an * (0.5 * d) * torch.abs(vn) * dt
            a_t = at * (0.5 * d) * torch.abs(vt) * dt
            jn = -a_n * vn / (1 + a_n * wp)
            jt = -a_t * vt / (1 + a_t * wp)
            ix = ix + torch.einsum("pbm,pnm->pbn", jn * -uy + jt * ux, G)
            iy = iy + torch.einsum("pbm,pnm->pbn", jn * ux + jt * uy, G)
        vX = vX + w * ix
        vY = vY + w * iy

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
    X, Y = projeter(X), projeter(Y)
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
    if not amort_integre:
        vX, vY = projeter(vX), projeter(vY)

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
                 os_rigides=False, facteur_raideur=1.0, compile=False, amort_integre=True,
                 ancrages=False, morphologie_stricte=False, eau="lineaire", profondeur=20.0,
                 epaisseur_os=5.0, cn=2.0, ct=0.05, ordre_alterne=False):
        if eau not in ("lineaire", "quadratique"):
            raise ValueError(f"eau inconnue : {eau!r} (lineaire ou quadratique)")
        liste_ancrages = dico_mega_tenseurs.get("ancrages")
        if any(liste_ancrages or ()) and not ancrages:
            raise ValueError("créature avec points d'ancrage : activer l'option ancrages "
                             "(ou charger le champion sans, cf. morphologie.scinder_ancrages)")
        self._ancrages = liste_ancrages if ancrages and any(liste_ancrages or ()) else None
        super().__init__(dico_mega_tenseurs, batch_size, device=device)
        self.eau = eau
        # en eau quadratique, un nœud sans os n'aurait pas de masse : règle 1 obligatoire
        self._verifier_morphologie(morphologie_stricte, regle_1_obligatoire=eau == "quadratique")
        self.iterations = iterations
        self.amort_integre = amort_integre
        B = batch_size
        P, _, N = self.X.shape
        M = self.muscle1.shape[1]
        dev, dt_ = self.X.device, self.X.dtype
        T = self._matrice_ancrages(P, N, dev, dt_)
        m_ref = 1.0
        if eau == "quadratique":
            self.masses = self._masses_geometriques(T, profondeur, epaisseur_os).unsqueeze(1).expand(-1, B, -1)
            m_ref = (self.masses * (self.masses > 0)).sum(2, keepdim=True) / (self.masses > 0).sum(2, keepdim=True)
        k = self.stiffness.unsqueeze(1).expand(-1, B, -1) * facteur_raideur * m_ref  # [P, B, M]
        os_ = self.is_bone_exp
        alpha = torch.where(k > 0, 1.0 / torch.clamp(k, min=1e-12), torch.zeros_like(k))
        if os_rigides:
            alpha = torch.where(os_ == 1.0, torch.zeros_like(alpha), alpha)
        self._alpha = alpha.contiguous()
        self._c = (2.0 * torch.sqrt(k * m_ref)).contiguous()                       # amortissement, comme la v1
        # matrices d'incidence (liens fantômes : colonnes nulles)
        mk = self.masque_muscles.to(dt_).unsqueeze(1)                               # [P, 1, M]
        self._D1 = torch.zeros(P, N, M, dtype=dt_, device=dev).scatter_(1, self.muscle1.unsqueeze(1), 1.0) * mk
        self._D2 = torch.zeros(P, N, M, dtype=dt_, device=dev).scatter_(1, self.muscle2.unsqueeze(1), 1.0) * mk
        self._D = self._D2 - self._D1
        self._A = self._D1 + self._D2
        reel = self.mask_N_exp * (self.masses > 0)
        self._w = torch.where(reel > 0, 1.0 / torch.clamp(self.masses, min=1e-12), torch.zeros_like(reel)).contiguous()
        self._T = T
        D1e, D2e = self._D1, self._D2
        if T is not None:
            for p, liste in enumerate(self._ancrages):
                for n, _, _ in liste:
                    self._w[p, :, n] = 0.0                                          # pas de masse propre
            self._D = torch.einsum("pjk,pkm->pjm", T, self._D)
            self._A = torch.einsum("pjk,pkm->pjm", T, self._A)
            D1e = torch.einsum("pjk,pkm->pjm", T, self._D1)
            D2e = torch.einsum("pjk,pkm->pjm", T, self._D2)
            self._recalculer_repos()
        self._quad = None
        if eau == "quadratique":
            internes = torch.zeros(P, M, dtype=dt_, device=dev)
            m1l, m2l, os_l = self.muscle1.tolist(), self.muscle2.tolist(), self.is_bone.tolist()
            for p in range(P):
                for m in aretes_internes(m1l[p], m2l[p], os_l[p]):
                    internes[p, m] = 1.0
            mouille = (self.is_bone * self.masque_muscles * (1 - internes)).unsqueeze(1).expand(-1, B, -1)
            G = [(1 - g) * D1e + g * D2e for g in GAUSS]
            wp = [torch.einsum("pbn,pnm->pbm", self._w, Gi * Gi).contiguous() for Gi in G]
            self._quad = (G[0].contiguous(), G[1].contiguous(), wp[0], wp[1],
                          (0.5 * RHO_EAU * cn * profondeur * mouille).contiguous(),
                          (0.5 * RHO_EAU * ct * profondeur * mouille).contiguous())
        self._wsum = torch.einsum("pbn,pnm->pbm", self._w, self._D * self._D).contiguous()
        col = colorier_liens(self.muscle1, self.muscle2, self.masque_muscles, self._ancrages)
        self._couleurs = torch.stack([(col == c).to(dt_).unsqueeze(1).expand(-1, B, -1)
                                      for c in range(int(col.max().item()) + 1)]).contiguous()
        self._facteur_trainee = ((os_ + (1 - os_) * 0.3) * self.mask_M_exp).contiguous()
        # ordre_alterne : couleurs parcourues dans un sens puis dans l'autre d'un sous-pas au
        # suivant. Un seul passage de Gauss-Seidel corrige toujours le même lien en premier
        # (les deux muscles d'épaule du rameur partagent A, donc ont deux couleurs) : biais
        # haut/bas qui fait tourner une créature symétrique. Alterner l'annule en moyenne.
        self._couleurs_inv = self._couleurs.flip(0).contiguous() if ordre_alterne else None
        self._sous_pas = 0
        self._pas = _pas_xpbd_compile() if compile else pas_xpbd
        self.force_liens = torch.zeros_like(k)                                      # λ/Δt² du dernier sous-pas

    def _matrice_ancrages(self, P, N, dev, dt_):
        """T [P, N, N] (x_virtuel = X @ T), ou None sans ancrage."""
        if self._ancrages is None:
            return None
        T = torch.eye(N, dtype=dt_, device=dev).repeat(P, 1, 1)
        m1l, m2l = self.muscle1.tolist(), self.muscle2.tolist()
        for p, liste in enumerate(self._ancrages):
            for n, o, s in liste:
                T[p, n, n] = 0.0
                T[p, m1l[p][o], n] = 1.0 - s
                T[p, m2l[p][o], n] = s
        return T

    def _masses_geometriques(self, T, profondeur, epaisseur_os):
        """Masses [P, N] en g (flottabilité neutre), calculées sur la géométrie de base."""
        P, N = self.X_base.shape
        masses = torch.zeros(P, N, dtype=self.X_base.dtype, device=self.X_base.device)
        xs, ys = self.X_base.tolist(), self.Y_base.tolist()
        m1l, m2l, os_l, mk = self.muscle1.tolist(), self.muscle2.tolist(), self.is_bone.tolist(), \
            self.masque_muscles.tolist()
        for p in range(P):
            nm = int(sum(mk[p]))
            m1, m2, os_ = m1l[p][:nm], m2l[p][:nm], os_l[p][:nm]
            x, y = xs[p], ys[p]
            en_triangle = set()
            for a, b, c, liens in triangles_os(m1, m2, os_):
                aire = 0.5 * abs((x[b] - x[a]) * (y[c] - y[a]) - (x[c] - x[a]) * (y[b] - y[a]))
                for n in (a, b, c):
                    masses[p, n] += RHO_EAU * profondeur * aire / 3
                en_triangle.update(liens)
            for m in range(nm):
                if os_[m] >= 0.5 and m not in en_triangle:
                    longueur = ((x[m1[m]] - x[m2[m]]) ** 2 + (y[m1[m]] - y[m2[m]]) ** 2) ** 0.5
                    for n in (m1[m], m2[m]):
                        masses[p, n] += 0.5 * RHO_EAU * profondeur * longueur * epaisseur_os
        if T is not None:                                   # nœud ancré → extrémités de son os
            masses = torch.einsum("pk,pjk->pj", masses, T)
        return masses * self.masque_noeuds

    def _verifier_morphologie(self, stricte, regle_1_obligatoire=False):
        m1l, m2l = self.muscle1.tolist(), self.muscle2.tolist()
        os_l, mk = self.is_bone.tolist(), self.masque_muscles.tolist()
        nn = self.masque_noeuds.sum(dim=1).long().tolist()
        for p in range(len(m1l)):
            nm = int(sum(mk[p]))
            erreurs = valider_morphologie(m1l[p][:nm], m2l[p][:nm], os_l[p][:nm], nn[p],
                                          self._ancrages[p] if self._ancrages else ())
            if erreurs:
                msg = f"morphologie invalide (créature {p}) : " + " ; ".join(erreurs)
                if stricte or (regle_1_obligatoire and any("règle 1" in e for e in erreurs)):
                    raise ValueError(msg)
                warnings.warn(msg, stacklevel=3)

    def _recalculer_repos(self):
        """Ancrages : place les nœuds ancrés sur leur os (le bruit initial les en écarte), puis
        recalcule les longueurs au repos et le barycentre comme MegaCrea.__init__."""
        self.X = torch.einsum("pbj,pjk->pbk", self.X, self._T)
        self.Y = torch.einsum("pbj,pjk->pbk", self.Y, self._T)
        x1, y1, x2, y2 = self._paires(self.X, self.Y)
        self.base_length = torch.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2 + 1e-8) * self.mask_M_exp
        self.target_length = self.base_length.clone()
        nb = torch.clamp(torch.sum(self.mask_N_exp, dim=2), min=1.0)
        self.previous_distance = torch.sum(self.X * self.mask_N_exp, dim=2) / nb
        self.previous_height = torch.sum(self.Y * self.mask_N_exp, dim=2) / nb

    def _paires(self, X, Y):
        """Positions des extrémités m1, m2 de chaque lien (pour les tests et les outils)."""
        x1 = torch.einsum("pbn,pnm->pbm", X, self._D1)
        y1 = torch.einsum("pbn,pnm->pbm", Y, self._D1)
        x2 = torch.einsum("pbn,pnm->pbm", X, self._D2)
        y2 = torch.einsum("pbn,pnm->pbm", Y, self._D2)
        return x1, y1, x2, y2

    def apply_physics(self, dt):
        kN = self.water_drag * self._facteur_trainee
        couleurs = self._couleurs
        if self._couleurs_inv is not None:
            couleurs = self._couleurs_inv if self._sous_pas % 2 else self._couleurs
            self._sous_pas += 1
        self.X, self.Y, self.vX, self.vY, self.force_liens = self._pas(
            self.X, self.Y, self.vX, self.vY, self._D, self._A, self._wsum, couleurs,
            self.target_length, self._alpha, self._c, kN, kN * FACTEUR_TANGENTIEL, self._w,
            self.mask_N_exp, float(dt), self.v_max, self.iterations, self.amort_integre, self._T, self._quad)
