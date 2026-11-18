"""POSITIONS SIMULÉES POUR LA TÂCHE « STABILITÉ ET ÉTATS »

Rôle
    Fournir des suites de positions 3D inventées pour tester la tâche « stabilité et états »
    (stabilite_etats.py) sans caméra : bouche immobile, qui tremble, qui saute,
    qui disparaît puis revient.

Produit
    Chaque fonction renvoie une liste de couples (Position3D, ouverture), une par image,
    à 30 images par seconde, avec une graine fixe (mêmes données à chaque lancement).
    L'heure de la première image vaut 0. La bouche est à POSITION_BOUCHE (m, repère caméra).

Utilisé par
    tests/test_stabilite_etats.py et scripts/demo_stabilite.py (dictionnaire SCENARIOS).
"""

import numpy as np

from bouche_3d.interfaces import Position3D

IMAGES_PAR_SECONDE = 30
POSITION_BOUCHE = (0.0, 0.05, 0.50)  # m : centrée, 5 cm sous l'axe, à 50 cm
OUVERTURE_OUVERTE = 0.50
GRAINE = 0


def _heures(duree: float) -> np.ndarray:
    return np.arange(int(round(duree * IMAGES_PAR_SECONDE))) / IMAGES_PAR_SECONDE


def _suite(points, ouvertures, heures, valides=None) -> list[tuple[Position3D, float]]:
    """Assemble les couples (Position3D, ouverture) ; valides=False donne une position invalide."""
    suite = []
    for i, heure in enumerate(heures):
        heure = float(heure)
        if valides is not None and not valides[i]:
            position = Position3D.invalide(heure)
        else:
            x, y, z = (float(c) for c in points[i])
            position = Position3D(x, y, z, True, heure)
        suite.append((position, float(ouvertures[i])))
    return suite


def immobile(duree: float = 3.0, ouverture: float = OUVERTURE_OUVERTE) -> list[tuple[Position3D, float]]:
    """Bouche parfaitement immobile, ouverte."""
    heures = _heures(duree)
    points = np.tile(POSITION_BOUCHE, (len(heures), 1))
    return _suite(points, np.full(len(heures), ouverture), heures)


def tremblement(duree: float = 3.0, bruit: float = 0.003, graine: int = GRAINE) -> list[tuple[Position3D, float]]:
    """Bouche immobile, mesure bruitée : bruit gaussien d'écart-type `bruit` (m) sur chaque axe."""
    rng = np.random.default_rng(graine)
    heures = _heures(duree)
    points = np.array(POSITION_BOUCHE) + rng.normal(0.0, bruit, (len(heures), 3))
    return _suite(points, np.full(len(heures), OUVERTURE_OUVERTE), heures)


def saut_isole(duree: float = 3.0, heure_saut: float = 2.0, ecart_z: float = 0.20) -> list[tuple[Position3D, float]]:
    """Bouche immobile, avec une seule valeur aberrante à +20 cm en Z."""
    heures = _heures(duree)
    points = np.tile(POSITION_BOUCHE, (len(heures), 1))
    points[int(round(heure_saut * IMAGES_PAR_SECONDE)), 2] += ecart_z
    return _suite(points, np.full(len(heures), OUVERTURE_OUVERTE), heures)


def deplacement_reel(duree: float = 5.0, heure_depart: float = 2.0,
                     decalage_x: float = 0.10) -> list[tuple[Position3D, float]]:
    """La bouche se décale de 10 cm en X à heure_depart, et y reste."""
    heures = _heures(duree)
    points = np.tile(POSITION_BOUCHE, (len(heures), 1))
    points[int(round(heure_depart * IMAGES_PAR_SECONDE)):, 0] += decalage_x
    return _suite(points, np.full(len(heures), OUVERTURE_OUVERTE), heures)


def disparition(duree_absence: float, heure_debut: float = 2.0,
                duree_apres: float = 2.0) -> list[tuple[Position3D, float]]:
    """Bouche immobile qui disparaît (positions invalides) pendant duree_absence, puis revient."""
    heures = _heures(heure_debut + duree_absence + duree_apres)
    points = np.tile(POSITION_BOUCHE, (len(heures), 1))
    debut = int(round(heure_debut * IMAGES_PAR_SECONDE))
    fin = debut + int(round(duree_absence * IMAGES_PAR_SECONDE))
    valides = np.ones(len(heures), bool)
    valides[debut:fin] = False
    return _suite(points, np.full(len(heures), OUVERTURE_OUVERTE), heures, valides)


def disparition_courte() -> list[tuple[Position3D, float]]:
    """Disparition de 0,2 s (par exemple la cuillère passe devant)."""
    return disparition(0.2)


def disparition_longue() -> list[tuple[Position3D, float]]:
    """Disparition de 1 s, puis retour."""
    return disparition(1.0)


def disparition_tres_longue() -> list[tuple[Position3D, float]]:
    """Disparition de 5 s, puis retour."""
    return disparition(5.0)


def bouche_fermee(duree: float = 3.0) -> list[tuple[Position3D, float]]:
    """Bouche immobile mais fermée (ouverture 0,05)."""
    return immobile(duree, ouverture=0.05)


def ouverture_oscillante(duree: float = 6.0, graine: int = GRAINE) -> list[tuple[Position3D, float]]:
    """Bouche immobile, ouverture qui oscille autour de 0,30 (± 0,03, plus un bruit de 0,005).

    L'oscillation franchit le seuil 0,30 mais reste au-dessus de 0,25 (seuil moins hystérésis).
    """
    rng = np.random.default_rng(graine)
    heures = _heures(duree)
    points = np.tile(POSITION_BOUCHE, (len(heures), 1))
    ouvertures = 0.30 + 0.03 * np.sin(2 * np.pi * 0.5 * heures) + rng.normal(0.0, 0.005, len(heures))
    return _suite(points, ouvertures, heures)


# Scénarios par nom (démo et tests).
SCENARIOS = {
    "immobile": immobile,
    "tremblement": tremblement,
    "saut_isole": saut_isole,
    "deplacement_reel": deplacement_reel,
    "disparition_courte": disparition_courte,
    "disparition_longue": disparition_longue,
    "disparition_tres_longue": disparition_tres_longue,
    "bouche_fermee": bouche_fermee,
    "ouverture_oscillante": ouverture_oscillante,
}
