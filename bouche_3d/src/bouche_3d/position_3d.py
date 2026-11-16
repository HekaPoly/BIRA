"""TÂCHE « POSITION 3D » : position 3D de la bouche

Rôle
    Calculer la position réelle de la bouche dans l'espace (X, Y, Z, en mètres)
    à partir du nuage de points de la ZED Mini et du pixel de la bouche.
    Ce fichier contient seulement le calcul : il ne lit pas la caméra.
    La lecture de la caméra (webcam ou ZED) est faite dans camera.py.

Reçoit
    - Le nuage de points : un tableau NumPy de forme (hauteur, largeur, 3) qui donne
      X, Y, Z en mètres pour chaque pixel. Les mesures ratées valent NaN ou inf.
    - Le pixel de la bouche (u, v) : u = colonne, v = ligne.

Produit
    La position (X, Y, Z) en mètres dans le repère de la caméra
    (X vers la droite, Y vers le bas, Z vers l'avant),
    ou « invalide » (None) si la mesure n'est pas fiable.

Méthode
    1. Prendre une petite fenêtre de pixels autour de (u, v), sans sortir de l'image.
    2. Garder seulement les points valides (ni NaN ni inf).
    3. S'il y a trop peu de points valides : « invalide ».
    4. Sinon, prendre la médiane de X, de Y et de Z (décision D3).
    5. Si Z est hors de la plage de travail plausible : « invalide ».

Prérequis
    Pour développer et tester le calcul : seulement NumPy, avec des nuages inventés.
    Pour l'utiliser avec la vraie caméra : la ZED Mini (ou un fichier .svo), le ZED SDK
    et son module Python pyzed, qui exigent une carte graphique NVIDIA.

Dépend de
    Rien pour être développée : le pixel peut être choisi à la main.
    L'interface « détection → position 3D » ne sert qu'à l'assemblage.

Transmet à
    La tâche « stabilité et états » (stabilite_etats.py), par l'interface
    « position 3D → stabilité ».

Terminée quand
    - Les tests passent sur des nuages inventés : valeurs normales, point aberrant,
      fenêtre vide (NaN), pixel au bord de l'image, distance hors plage.
    - Avec la ZED, à la distance de travail, la position est cohérente avec une mesure
      à la règle, et le résultat est « invalide » plutôt qu'absurde quand la mesure
      est mauvaise.

À savoir
    La mesure est peu fiable dans les zones sombres, brillantes ou uniformes, comme
    l'intérieur d'une bouche ouverte : mieux vaut lire sur les lèvres ou le contour
    du visage. Dans un tableau NumPy, l'indice de ligne est v et celui de colonne est u :
    nuage[v, u].

Réglages à choisir (puis noter dans docs/decisions.md)
    Taille de la fenêtre, part minimale de points valides, plage de Z plausible.

Questions liées
    1, 5 et 8 de docs/mandat.md.
"""

from dataclasses import dataclass

import numpy as np

from bouche_3d.interfaces import Position3D


def position_bouche(nuage, u, v, rayon=5, part_min=0.5, z_min=0.2, z_max=1.0):
    """Donne la position (X, Y, Z) en mètres de la bouche, ou None si la mesure n'est pas fiable.

    nuage : tableau NumPy (hauteur, largeur, 3) avec X, Y, Z en mètres (NaN ou inf = mesure ratée).
    u, v : pixel de la bouche (u = colonne, v = ligne).
    rayon : demi-côté de la fenêtre (rayon=5 donne une fenêtre de 11×11 pixels).
    part_min : part minimale de points valides dans la fenêtre (entre 0 et 1).
    z_min, z_max : plage de distance Z acceptée, en mètres.
    """
    hauteur, largeur = nuage.shape[:2]

    # Refuser un pixel qui n'est pas dans l'image.
    if not (0 <= u < largeur and 0 <= v < hauteur):
        raise ValueError(
            f"Pixel (u={u}, v={v}) hors de l'image : "
            f"u doit être entre 0 et {largeur - 1}, v entre 0 et {hauteur - 1}."
        )

    # a. Bornes de la fenêtre autour de (v, u), coupées aux bords de l'image.
    #    La borne de fin est exclue, d'où le « + 1 ».
    v_min = max(v - rayon, 0)
    v_max = min(v + rayon + 1, hauteur)
    u_min = max(u - rayon, 0)
    u_max = min(u + rayon + 1, largeur)

    # b. Extraire la fenêtre, puis la mettre à plat : une ligne par point, colonnes X, Y, Z.
    fenetre = nuage[v_min:v_max, u_min:u_max].reshape(-1, 3)

    # c. Garder seulement les points dont X, Y et Z sont tous finis (ni NaN ni inf).
    valides = np.isfinite(fenetre).all(axis=1)
    points = fenetre[valides]

    # d. Trop peu de points valides : la mesure n'est pas fiable.
    if len(points) / len(fenetre) < part_min:
        return None

    # e. Médiane de X, de Y et de Z, chacune séparément (décision D3).
    x, y, z = np.median(points, axis=0)

    # f. Distance hors de la plage plausible : la mesure n'est pas fiable.
    if not (z_min <= z <= z_max):
        return None

    # g. Retourner la position en nombres Python ordinaires (float).
    return float(x), float(y), float(z)


def position_depuis_pixel(nuage: np.ndarray, u: float, v: float, heure: float, **reglages) -> Position3D:
    """Comme position_bouche, mais renvoie une Position3D (interface « position 3D → stabilité »).

    u, v peuvent être des float (sortie de la détection) : ils sont arrondis au pixel le plus proche.
    reglages : rayon, part_min, z_min, z_max, transmis tels quels à position_bouche.
    """
    resultat = position_bouche(nuage, int(round(u)), int(round(v)), **reglages)
    if resultat is None:
        return Position3D.invalide(heure)
    x, y, z = resultat
    return Position3D(x, y, z, True, heure)


@dataclass(frozen=True)
class DiagnosticMesure:
    """Détail d'une mesure, pour l'affichage (la conclusion est celle de position_bouche).

    part_valides : part des points mesurés dans la fenêtre (0 à 1).
    nb_points : nombre de points mesurés.
    z_median : médiane de Z en mètres, ou None s'il n'y a aucun point mesuré.
    raison : "ok", "trop_de_trous" ou "hors_portee".
    fenetre_valides : tableau booléen de la fenêtre (True = point mesuré).
    """

    part_valides: float
    nb_points: int
    z_median: float | None
    raison: str
    fenetre_valides: np.ndarray


def diagnostic_mesure(nuage: np.ndarray, u: int, v: int, rayon: int = 5, part_min: float = 0.5,
                      z_min: float = 0.2, z_max: float = 1.0) -> DiagnosticMesure:
    """Explique la mesure au pixel (u, v), avec exactement les règles de position_bouche."""
    hauteur, largeur = nuage.shape[:2]
    if not (0 <= u < largeur and 0 <= v < hauteur):
        raise ValueError(f"Pixel (u={u}, v={v}) hors de l'image.")

    # Même fenêtre que position_bouche, coupée aux bords de l'image.
    fenetre = nuage[max(v - rayon, 0):min(v + rayon + 1, hauteur),
                    max(u - rayon, 0):min(u + rayon + 1, largeur)]
    fenetre_valides = np.isfinite(fenetre).all(axis=2)
    points = fenetre[fenetre_valides]
    part = len(points) / fenetre_valides.size
    z_median = float(np.median(points[:, 2])) if len(points) else None

    # Mêmes tests, dans le même ordre : d'abord les trous, puis la distance.
    if part < part_min:
        raison = "trop_de_trous"
    elif not (z_min <= z_median <= z_max):
        raison = "hors_portee"
    else:
        raison = "ok"
    return DiagnosticMesure(part, len(points), z_median, raison, fenetre_valides)
