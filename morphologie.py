"""
morphologie.py — règles de morphologie (physique v2) et points d'ancrage.

Règles (voir EXPERIMENTS, physique v2) :
  1. tout nœud touche un os (sinon il n'a ni masse ni traînée en v2) ;
  2'. tout nœud appartient à au plus 2 pièces rigides. Une pièce rigide est un os seul,
     ou des os soudés par des chaînes de triangles d'os (le tronc du rameur). Une
     articulation relie deux pièces, comme le coude relie le bras et l'avant-bras ;
     un nœud où se rejoignent 3 membres articulés n'existe pas dans un vrai squelette.

Points d'ancrage (option `ancrages`, physique v2 seulement) : un nœud ancré n'est pas un
vrai nœud, c'est le point x_M = (1−s)·x_a + s·x_b de l'os a→b. On peut ainsi attacher un
membre au milieu d'un tronc d'un seul os, qui reste rigide. Format, dans le .pt :
    'ancrages': [[noeud, indice_os, s], ...]      s mesuré depuis muscle1 de l'os, 0 < s < 1
Sans l'option, `scinder_ancrages` remplace l'os a–b par a–M + M–b : M redevient une
articulation libre (le comportement historique, pour comparer).

Usage :  python -m morphologie champion_raffine/rameur.pt
"""

import math
import sys


def _union_find(n):
    parent = list(range(n))

    def trouver(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def unir(i, j):
        parent[trouver(i)] = trouver(j)

    return trouver, unir


def triangles_os(m1, m2, is_bone):
    """Triangles d'os : liste de (a, b, c, [indices des 3 os]), chacun une fois."""
    os_par_paire = {frozenset((m1[m], m2[m])): m for m in range(len(m1)) if is_bone[m] >= 0.5}
    voisins = {}
    for paire in os_par_paire:
        a, b = tuple(paire)
        voisins.setdefault(a, set()).add(b)
        voisins.setdefault(b, set()).add(a)
    triangles = []
    for paire in os_par_paire:
        a, b = sorted(paire)
        for c in voisins[a] & voisins[b]:
            if c > b:
                triangles.append((a, b, c, [os_par_paire[frozenset(p)] for p in ((a, b), (a, c), (b, c))]))
    return triangles


def aretes_internes(m1, m2, is_bone):
    """Os partagés par deux triangles d'os (l'arête Su–Sd du rameur) : à l'intérieur du corps,
    l'eau ne les touche pas."""
    compte = {}
    for *_, liens in triangles_os(m1, m2, is_bone):
        for m in liens:
            compte[m] = compte.get(m, 0) + 1
    return {m for m, n in compte.items() if n >= 2}


def pieces_rigides(m1, m2, is_bone):
    """{indice_os: indice_pièce} : deux os sont dans la même pièce s'ils sont reliés par
    une chaîne de triangles d'os (trois nœuds reliés deux à deux par des os)."""
    os_ = [m for m in range(len(m1)) if is_bone[m] >= 0.5]
    os_par_paire = {frozenset((m1[m], m2[m])): m for m in os_}
    voisins = {}
    for m in os_:
        voisins.setdefault(m1[m], set()).add(m2[m])
        voisins.setdefault(m2[m], set()).add(m1[m])
    trouver, unir = _union_find(len(m1))
    for m in os_:
        a, b = m1[m], m2[m]
        for c in voisins[a] & voisins[b]:
            unir(m, os_par_paire[frozenset((a, c))])
            unir(m, os_par_paire[frozenset((b, c))])
    racines = {}
    return {m: racines.setdefault(trouver(m), len(racines)) for m in os_}


def valider_morphologie(m1, m2, is_bone, n_noeuds, ancrages=()):
    """Liste des violations (vide si la créature est valide), une phrase par problème."""
    erreurs = []
    pieces = pieces_rigides(m1, m2, is_bone)
    ancre = {}
    for n, o, s in ancrages:
        if n in ancre:
            erreurs.append(f"nœud {n} : ancré deux fois")
        if not (0 <= o < len(m1)) or is_bone[o] < 0.5:
            erreurs.append(f"nœud {n} : ancré sur le lien {o}, qui n'est pas un os")
            continue
        if not 0.0 < s < 1.0:
            erreurs.append(f"nœud {n} : position s = {s} hors de ]0, 1[")
        if n in (m1[o], m2[o]):
            erreurs.append(f"nœud {n} : ancré sur un os dont il est l'extrémité")
        ancre[n] = (o, s)
    for n, (o, _) in ancre.items():
        for e in (m1[o], m2[o]):
            if e in ancre:
                erreurs.append(f"nœud {n} : ancré sur l'os {o}, dont l'extrémité {e} est elle-même ancrée")
    for m in range(len(m1)):
        for n, e in ((m1[m], m2[m]), (m2[m], m1[m])):
            if n in ancre and e in (m1[ancre[n][0]], m2[ancre[n][0]]):
                erreurs.append(f"lien {m} ({m1[m]}–{m2[m]}) : relie le nœud ancré {n} à une extrémité de "
                               "son propre os (longueur fixée par l'ancrage, lien inutile)")

    for n in range(n_noeuds):
        p = {pieces[m] for m in pieces if n in (m1[m], m2[m])}
        if n in ancre:
            p.add(pieces[ancre[n][0]])
        if not p:
            erreurs.append(f"règle 1 — nœud {n} : ne touche aucun os")
        elif len(p) > 2:
            erreurs.append(f"règle 2' — nœud {n} : appartient à {len(p)} pièces rigides (max 2)")
    return erreurs


def valider_dict(d):
    """Valide un dict au format .pt (champion / rameur)."""
    m1 = [int(v) for v in d["muscle1"]]
    m2 = [int(v) for v in d["muscle2"]]
    return valider_morphologie(m1, m2, [float(v) for v in d["is_bone"]], len(d["x"]),
                               [tuple(a) for a in d.get("ancrages") or ()])


def ajouter_ancrage(d, indice_os, s):
    """Ajoute à un dict .pt un nœud ancré à la fraction s de l'os `indice_os`. Retourne son indice.
    (Les liens qui partent de ce nœud sont à ajouter ensuite, comme pour un nœud ordinaire.)"""
    a, b = int(d["muscle1"][indice_os]), int(d["muscle2"][indice_os])
    d["x"] = list(d["x"]) + [(1 - s) * d["x"][a] + s * d["x"][b]]
    d["y"] = list(d["y"]) + [(1 - s) * d["y"][a] + s * d["y"][b]]
    d["ancrages"] = list(d.get("ancrages") or []) + [[len(d["x"]) - 1, int(indice_os), float(s)]]
    return len(d["x"]) - 1


def scinder_ancrages(d):
    """Copie de `d` sans ancrages : l'os a–b portant un nœud ancré M devient a–M (même indice)
    + M–b (ajouté à la fin), donc M devient une articulation libre. Les indices des muscles,
    et donc les sorties du cerveau, ne changent pas. Les ancrages sur un même os sont
    traités dans l'ordre de s."""
    import torch

    ancrages = sorted((tuple(a) for a in d.get("ancrages") or ()), key=lambda a: (a[1], a[2]))
    if not ancrages:
        return d
    d = dict(d)
    m1, m2 = [int(v) for v in d["muscle1"]], [int(v) for v in d["muscle2"]]
    os_ = [float(v) for v in d["is_bone"]]
    fin_courante = {}                       # os → (indice du dernier tronçon, s de son début)
    for n, o, s in ancrages:
        m, s0 = fin_courante.get(o, (o, 0.0))
        b = m2[m]
        m2[m] = n                           # tronçon courant : ... → M
        m1.append(n)
        m2.append(b)                        # nouveau tronçon : M → b
        os_.append(1.0)
        fin_courante[o] = (len(m1) - 1, s)
    if len(m1) > d["max_muscles"]:
        raise ValueError(f"scinder_ancrages : {len(m1)} liens > max_muscles = {d['max_muscles']}")
    x, y = d["x"], d["y"]
    d["muscle1"] = torch.tensor(m1, dtype=torch.long)
    d["muscle2"] = torch.tensor(m2, dtype=torch.long)
    d["is_bone"] = os_
    d["stiffness"] = torch.tensor([1.0 + 4 * v for v in os_])
    d["target_length"] = torch.tensor([math.hypot(x[a] - x[b], y[a] - y[b]) for a, b in zip(m1, m2)])
    d["ancrages"] = []
    return d


def main():
    import torch

    for chemin in sys.argv[1:]:
        erreurs = valider_dict(torch.load(chemin, map_location="cpu"))
        print(f"{chemin} : " + ("valide" if not erreurs else f"{len(erreurs)} problème(s)"))
        for e in erreurs:
            print("  - " + e)


if __name__ == "__main__":
    main()
