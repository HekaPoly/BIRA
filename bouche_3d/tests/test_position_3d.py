"""TESTS DE LA TÂCHE « POSITION 3D » (position_3d.py)

Ces tests utiliseront des cartes de profondeur fabriquées. Ils vérifieront
la lecture de la valeur médiane, le rejet des mesures invalides ou hors plage, la réponse
« invalide » quand trop de mesures manquent, et l'ordre ligne / colonne (v = ligne, u = colonne).
"""

import numpy as np
import pytest

from bouche_3d.camera import SourceSimulation
from bouche_3d.interfaces import Position3D
from bouche_3d.position_3d import diagnostic_mesure, position_bouche, position_depuis_pixel


def nuage_normal():
    """Nuage 100×100 où chaque point est à (0, 0, 0,5 m)."""
    return np.full((100, 100, 3), [0.0, 0.0, 0.5])


def test_cas_normal():
    # Tous les points sont à 0,5 m : on doit retrouver exactement cette position.
    assert position_bouche(nuage_normal(), 50, 50) == (0.0, 0.0, 0.5)


def test_point_aberrant():
    # Un seul point à 5 m dans la fenêtre : la médiane l'ignore.
    nuage = nuage_normal()
    nuage[50, 52, 2] = 5.0
    assert position_bouche(nuage, 50, 50) == (0.0, 0.0, 0.5)


def test_mesures_ratees():
    # Toute la fenêtre 11×11 autour de (50, 50) est ratée (NaN) : résultat invalide.
    nuage = nuage_normal()
    nuage[45:56, 45:56] = np.nan
    assert position_bouche(nuage, 50, 50) is None


def test_bord_de_l_image():
    # Pixel dans le coin : la fenêtre est coupée au bord, sans erreur.
    assert position_bouche(nuage_normal(), 0, 0) == (0.0, 0.0, 0.5)


def test_trop_loin():
    # Z = 3 m partout : hors de la plage [0,2 ; 1,0] m, résultat invalide.
    nuage = np.full((100, 100, 3), [0.0, 0.0, 3.0])
    assert position_bouche(nuage, 50, 50) is None


@pytest.mark.parametrize("u, v", [(100, 50), (50, 100), (-1, 50), (50, -1)])
def test_pixel_hors_image(u, v):
    # Un pixel hors de l'image est refusé avec une ValueError.
    with pytest.raises(ValueError):
        position_bouche(nuage_normal(), u, v)


def mettre_nan(nuage, nombre):
    """Met à NaN les `nombre` premiers points de la fenêtre 11×11 autour de (50, 50)."""
    # Numéro de ligne et de colonne, dans la fenêtre, de chacun des `nombre` premiers points.
    lignes, colonnes = np.unravel_index(np.arange(nombre), (11, 11))
    nuage[45 + lignes, 45 + colonnes] = np.nan


def test_seuil_trop_de_nan():
    # 60 % de la fenêtre ratée (73 points sur 121) : moins de 50 % valides, résultat invalide.
    nuage = nuage_normal()
    mettre_nan(nuage, 73)
    assert position_bouche(nuage, 50, 50) is None


def test_seuil_assez_de_points():
    # 40 % de la fenêtre ratée (48 points sur 121) : 60 % valides, la position est donnée.
    nuage = nuage_normal()
    mettre_nan(nuage, 48)
    assert position_bouche(nuage, 50, 50) == (0.0, 0.0, 0.5)


def test_resultat_en_float_python():
    # Le résultat doit être trois float Python, pas des types NumPy.
    resultat = position_bouche(nuage_normal(), 50, 50)
    assert isinstance(resultat, tuple) and len(resultat) == 3
    assert all(type(valeur) is float for valeur in resultat)


# --- position_depuis_pixel (format d'échange Position3D, décision D13) ---

def test_position_depuis_pixel_valide():
    position = position_depuis_pixel(nuage_normal(), 50, 50, heure=1.5)
    assert position == Position3D(0.0, 0.0, 0.5, True, 1.5)


def test_position_depuis_pixel_invalide():
    # Fenêtre entièrement ratée : valide=False et x, y, z à None.
    nuage = nuage_normal()
    nuage[45:56, 45:56] = np.nan
    position = position_depuis_pixel(nuage, 50, 50, heure=2.0)
    assert position.valide is False
    assert (position.x, position.y, position.z) == (None, None, None)
    assert position.heure == 2.0


def test_position_depuis_pixel_reglages_et_pixel_float():
    # Les réglages sont transmis, et un pixel en float (sortie de la détection) est accepté.
    nuage = np.full((100, 100, 3), [0.0, 0.0, 3.0])
    assert position_depuis_pixel(nuage, 50.4, 49.6, 0.0).valide is False
    assert position_depuis_pixel(nuage, 50.4, 49.6, 0.0, z_max=4.0).z == 3.0


def test_position3d_incoherente_refusee():
    with pytest.raises(ValueError):
        Position3D(0.0, 0.0, None, True, 0.0)
    with pytest.raises(ValueError):
        Position3D(0.0, 0.0, 0.5, False, 0.0)


# --- Intégration avec la simulation (camera.SourceSimulation) ---

@pytest.mark.parametrize("distance", [0.35, 0.55])
def test_simulation_bouche(distance):
    # Bouche trouée à 30 % : Z à moins de 5 mm de la vraie valeur.
    source = SourceSimulation(distance=distance, part_trous_bouche=0.3)
    trame = source.lire()
    u, v, z_reel = source.position_bouche_reelle
    position = position_depuis_pixel(trame.nuage, u, v, trame.heure)
    assert position.valide
    assert abs(position.z - z_reel) < 0.005


def test_simulation_bouche_trop_trouee():
    # Bouche trouée à 80 % : moins de 50 % de points valides, donc invalide.
    source = SourceSimulation(part_trous_bouche=0.8)
    trame = source.lire()
    u, v, _ = source.position_bouche_reelle
    assert position_depuis_pixel(trame.nuage, u, v, trame.heure).valide is False


def test_simulation_front():
    # Sur le front (aucun trou) : Z à moins de 5 mm de la vraie valeur.
    source = SourceSimulation()
    trame = source.lire()
    u, v, z_reel = source.position_front_reelle
    position = position_depuis_pixel(trame.nuage, u, v, trame.heure)
    assert position.valide
    assert abs(position.z - z_reel) < 0.005


def test_simulation_fond_invalide():
    # Le fond est à 1,2 m, hors de la plage plausible : invalide.
    source = SourceSimulation()
    trame = source.lire()
    assert position_depuis_pixel(trame.nuage, 20, 20, trame.heure).valide is False


# --- diagnostic_mesure (explication affichée par la démo) ---

def test_diagnostic_ok():
    diagnostic = diagnostic_mesure(nuage_normal(), 50, 50)
    assert diagnostic.raison == "ok"
    assert diagnostic.part_valides == 1.0 and diagnostic.nb_points == 121
    assert diagnostic.z_median == 0.5
    assert diagnostic.fenetre_valides.shape == (11, 11) and diagnostic.fenetre_valides.all()


def test_diagnostic_trop_de_trous():
    nuage = nuage_normal()
    mettre_nan(nuage, 73)  # 48 points valides sur 121
    diagnostic = diagnostic_mesure(nuage, 50, 50)
    assert diagnostic.raison == "trop_de_trous"
    assert diagnostic.nb_points == 48
    assert diagnostic.fenetre_valides.sum() == 48


def test_diagnostic_hors_portee():
    nuage = np.full((100, 100, 3), [0.0, 0.0, 3.0])
    diagnostic = diagnostic_mesure(nuage, 50, 50)
    assert diagnostic.raison == "hors_portee" and diagnostic.z_median == 3.0


def test_diagnostic_meme_conclusion_que_position_bouche():
    # Nuages aléatoires (trous, distances, pixels, réglages variés) : même conclusion partout.
    rng = np.random.default_rng(0)
    for _ in range(300):
        nuage = np.full((40, 40, 3), [0.0, 0.0, rng.uniform(0.0, 1.5)])
        nuage[..., 2] += rng.normal(0.0, 0.1, (40, 40))
        nuage[rng.random((40, 40)) < rng.uniform(0.0, 1.0)] = np.nan
        u, v = (int(i) for i in rng.integers(0, 40, 2))
        reglages = {"rayon": int(rng.integers(0, 8)), "part_min": rng.uniform(0.1, 0.9),
                    "z_min": 0.2, "z_max": 1.0}
        attendu = position_bouche(nuage, u, v, **reglages)
        diagnostic = diagnostic_mesure(nuage, u, v, **reglages)
        assert (diagnostic.raison == "ok") == (attendu is not None)
        if attendu is not None:
            assert diagnostic.z_median == attendu[2]
