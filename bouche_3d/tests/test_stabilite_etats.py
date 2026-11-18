"""TESTS DE LA TÂCHE « STABILITÉ ET ÉTATS » (stabilite_etats.py)

Ces tests utilisent les suites de positions simulées de tests/positions_simulees.py
(30 images par seconde, graine fixe). Ils vérifient le lissage, le rejet des sauts,
chaque passage d'un état à l'autre et les règles de sécurité sur le temps
(image déjà vue, mesure périmée, heure qui recule).
"""

import numpy as np
import pytest

import positions_simulees as ps
from bouche_3d.interfaces import CibleStabilisee, Etat, Position3D
from bouche_3d.stabilite_etats import (
    CONFIRMATIONS_SAUT,
    DELAI_OUBLI_S,
    DELAI_GRACE_PRET_S,
    DELAI_PERDU_S,
    DUREE_STABLE_S,
    MIN_VALIDES_SUIVI,
    SuiviBouche,
)

VRAIE = np.array(ps.POSITION_BOUCHE)
IPS = ps.IMAGES_PAR_SECONDE


def derouler(suite, suivi=None):
    """Passe toute la suite dans un SuiviBouche ; renvoie les cibles, image par image."""
    suivi = suivi or SuiviBouche()
    return [suivi.mettre_a_jour(position, ouverture) for position, ouverture in suite]


def etats(cibles):
    return [cible.etat for cible in cibles]


def premier_indice(cibles, etat, depuis=0):
    for i in range(depuis, len(cibles)):
        if cibles[i].etat == etat:
            return i
    return None


def changements(cibles):
    """Liste des (indice, ancien état, nouvel état)."""
    return [(i, a.etat, b.etat) for i, (a, b) in enumerate(zip(cibles, cibles[1:]), start=1)
            if a.etat != b.etat]


def verifier_jamais_pret_directement(cibles):
    # On n'entre en PRET que depuis SUIVI.
    for _, avant, apres in changements(cibles):
        if apres == Etat.PRET:
            assert avant == Etat.SUIVI


def verifier_coordonnees(cibles):
    # x, y, z valent None exactement en RECHERCHE et en PERDU.
    for cible in cibles:
        sans_position = cible.etat in (Etat.RECHERCHE, Etat.PERDU)
        assert (cible.x is None) == sans_position


# ---------------------------------------------------------------------------
# Format de sortie
# ---------------------------------------------------------------------------

def test_cible_incoherente_refusee():
    with pytest.raises(ValueError):
        CibleStabilisee(Etat.PERDU, 0.0, 0.0, 0.5, None, 0.0, 1.0)
    with pytest.raises(ValueError):
        CibleStabilisee(Etat.PRET, None, None, None, None, 0.0, 1.0)


# ---------------------------------------------------------------------------
# Scénarios
# ---------------------------------------------------------------------------

def test_immobile_recherche_suivi_puis_pret():
    cibles = derouler(ps.immobile())
    assert cibles[0].etat == Etat.RECHERCHE and cibles[0].x is None
    i_suivi = premier_indice(cibles, Etat.SUIVI)
    assert i_suivi == MIN_VALIDES_SUIVI - 1  # la 3e position valide
    i_pret = premier_indice(cibles, Etat.PRET)
    assert i_pret is not None
    assert cibles[i_pret].heure == pytest.approx(DUREE_STABLE_S, abs=0.1)  # environ 1 s
    assert set(etats(cibles[i_pret:])) == {Etat.PRET}
    np.testing.assert_allclose((cibles[-1].x, cibles[-1].y, cibles[-1].z), VRAIE)
    assert cibles[-1].stable_depuis >= DUREE_STABLE_S
    verifier_jamais_pret_directement(cibles)
    verifier_coordonnees(cibles)


def test_tremblement_lisse_et_pret():
    suite = ps.tremblement()
    cibles = derouler(suite)
    lissees = np.array([(c.x, c.y, c.z) for c in cibles if c.x is not None])
    brutes = np.array([(p.x, p.y, p.z) for p, _ in suite])
    erreur_lissee = np.linalg.norm(lissees - VRAIE, axis=1)
    erreur_brute = np.linalg.norm(brutes - VRAIE, axis=1)
    assert erreur_lissee.mean() < 0.003  # moins de 3 mm en moyenne
    assert np.abs(lissees - VRAIE).max() < 0.006  # jamais plus de 6 mm sur un axe
    assert erreur_lissee.mean() < erreur_brute.mean()  # le lissage aide vraiment
    assert Etat.PRET in etats(cibles)


def test_saut_isole_ignore():
    suite = ps.saut_isole()
    cibles = derouler(suite)
    i_saut = int(round(2.0 * IPS))
    for cible in cibles[i_saut:]:
        np.testing.assert_allclose((cible.x, cible.y, cible.z), VRAIE)  # la position ne bouge pas
    assert cibles[-1].etat == Etat.PRET
    verifier_coordonnees(cibles)


def test_deplacement_reel_suivi_apres_confirmations():
    cibles = derouler(ps.deplacement_reel())
    i_depart = int(round(2.0 * IPS))
    nouvelle_x = VRAIE[0] + 0.10
    assert cibles[i_depart - 1].etat == Etat.PRET
    # Pendant les confirmations (moins de DELAI_GRACE_PRET_S), la position lissée ne bouge pas
    # encore et le délai de grâce garde PRET.
    for i in range(i_depart, i_depart + CONFIRMATIONS_SAUT - 1):
        assert cibles[i].x == pytest.approx(VRAIE[0])
        assert cibles[i].etat == Etat.PRET
    # À la confirmation, la position lissée rejoint la nouvelle place : PRET perdu.
    i_confirme = i_depart + CONFIRMATIONS_SAUT - 1
    assert cibles[i_confirme].x == pytest.approx(nouvelle_x)
    assert cibles[i_confirme].etat == Etat.SUIVI
    # PRET retrouvé environ DUREE_STABLE_S plus tard.
    i_pret = premier_indice(cibles, Etat.PRET, depuis=i_confirme)
    assert i_pret is not None
    assert cibles[i_pret].heure - cibles[i_confirme].heure == pytest.approx(DUREE_STABLE_S, abs=0.1)
    assert cibles[-1].x == pytest.approx(nouvelle_x)


def test_disparition_courte_pas_de_perdu():
    cibles = derouler(ps.disparition_courte())
    assert Etat.PERDU not in etats(cibles)
    assert cibles[-1].etat == Etat.PRET
    verifier_coordonnees(cibles)


def test_disparition_longue_perdu_puis_suivi():
    cibles = derouler(ps.disparition_longue())
    i_debut = int(round(2.0 * IPS))
    i_perdu = premier_indice(cibles, Etat.PERDU)
    assert i_perdu is not None
    # PERDU un peu après DELAI_PERDU_S sans mesure.
    absence = cibles[i_perdu].heure - cibles[i_debut - 1].heure
    assert DELAI_PERDU_S < absence <= DELAI_PERDU_S + 1.5 / IPS
    perdus = [c for c in cibles if c.etat == Etat.PERDU]
    assert all(c.x is None and c.y is None and c.z is None for c in perdus)
    assert all(c.derniere_position_connue is not None for c in perdus)  # pour information
    # Au retour : SUIVI d'abord, jamais PRET directement.
    i_retour = premier_indice(cibles, Etat.SUIVI, depuis=i_perdu)
    assert i_retour is not None
    i_fin_absence = i_debut + IPS  # 1 s d'absence
    assert i_retour == i_fin_absence + MIN_VALIDES_SUIVI - 1
    verifier_jamais_pret_directement(cibles)
    # La stabilité est redemandée après une perte.
    i_pret = premier_indice(cibles, Etat.PRET, depuis=i_retour)
    assert cibles[i_pret].heure - cibles[i_fin_absence].heure >= DUREE_STABLE_S - 0.05
    verifier_coordonnees(cibles)


def test_disparition_tres_longue_retour_en_recherche():
    cibles = derouler(ps.disparition_tres_longue())
    i_debut = int(round(2.0 * IPS))
    i_recherche = premier_indice(cibles, Etat.RECHERCHE, depuis=i_debut)
    assert i_recherche is not None
    assert cibles[i_recherche].heure - cibles[i_debut - 1].heure > DELAI_OUBLI_S
    assert cibles[i_recherche].derniere_position_connue is None  # tout est oublié
    assert etats(cibles[i_debut:i_recherche]).count(Etat.PERDU) > 0
    verifier_jamais_pret_directement(cibles)
    verifier_coordonnees(cibles)


def test_bouche_fermee_jamais_pret():
    cibles = derouler(ps.bouche_fermee())
    assert Etat.PRET not in etats(cibles)
    assert cibles[-1].etat == Etat.SUIVI


def test_ouverture_inconnue_jamais_pret():
    suite = [(position, None) for position, _ in ps.immobile()]
    cibles = derouler(suite)
    assert Etat.PRET not in etats(cibles)


def test_ouverture_oscillante_hysteresis():
    suite = ps.ouverture_oscillante()
    ouvertures = np.array([o for _, o in suite])
    assert (ouvertures < 0.30).any() and (ouvertures >= 0.30).any()  # le seuil est bien franchi
    assert ouvertures.min() > 0.25  # mais jamais sous seuil - hystérésis
    cibles = derouler(suite)
    bascules = [c for c in changements(cibles) if {c[1], c[2]} == {Etat.SUIVI, Etat.PRET}]
    assert Etat.PRET in etats(cibles)
    assert len(bascules) <= 2


# ---------------------------------------------------------------------------
# Règles du temps
# ---------------------------------------------------------------------------

def test_position_heure_qui_recule_ignoree():
    suivi = SuiviBouche()
    suivi.mettre_a_jour(Position3D(0.0, 0.05, 0.5, True, 1.0), 0.5)
    # Mesure plus ancienne que la précédente : ignorée, sans exception.
    cible = suivi.mettre_a_jour(Position3D(0.0, 0.05, 0.5, True, 0.9), 0.5, heure_actuelle=1.05)
    assert suivi.nb_acceptees == 1 and cible.etat == Etat.RECHERCHE
    assert suivi.heure_derniere_recue == 1.0  # la mesure ignorée ne compte pas comme reçue
    # La suite normale reprend.
    suivi.mettre_a_jour(Position3D(0.0, 0.05, 0.5, True, 1.1), 0.5)
    assert suivi.nb_acceptees == 2


def test_heure_actuelle_qui_recule():
    suivi = SuiviBouche()
    suivi.mettre_a_jour(Position3D(0.0, 0.05, 0.5, True, 1.0), 0.5, heure_actuelle=1.2)
    with pytest.raises(ValueError, match="heure_actuelle recule"):
        suivi.mettre_a_jour(Position3D(0.0, 0.05, 0.5, True, 1.1), 0.5, heure_actuelle=1.1)


def test_mesure_un_peu_dans_le_futur_acceptee():
    suivi = SuiviBouche()
    suivi.mettre_a_jour(Position3D(0.0, 0.05, 0.5, True, 1.02), 0.5, heure_actuelle=1.0)
    assert suivi.nb_acceptees == 1  # 0,02 s d'avance : dans la tolérance d'horloge


def test_mesure_trop_dans_le_futur_ignoree():
    suivi = SuiviBouche()
    cible = suivi.mettre_a_jour(Position3D(0.0, 0.05, 0.5, True, 1.2), 0.5, heure_actuelle=1.0)
    assert suivi.nb_acceptees == 0 and cible.etat == Etat.RECHERCHE  # ignorée, sans exception
    assert suivi.heure_derniere_recue is None
    # Elle ne bloque pas les mesures normales qui suivent.
    suivi.mettre_a_jour(Position3D(0.0, 0.05, 0.5, True, 1.05), 0.5, heure_actuelle=1.05)
    assert suivi.nb_acceptees == 1


def test_delai_grace_doit_etre_inferieur_a_delai_perdu():
    with pytest.raises(ValueError, match="delai_grace_pret_s"):
        SuiviBouche(delai_grace_pret_s=0.3, delai_perdu_s=0.3)
    with pytest.raises(ValueError):
        SuiviBouche(delai_grace_pret_s=0.5)
    SuiviBouche(delai_grace_pret_s=0.2, delai_perdu_s=0.3)  # permis


def suivi_en_pret():
    """SuiviBouche amené en PRET par 2 s de bouche immobile ; renvoie (suivi, dernière heure)."""
    suivi = SuiviBouche()
    suite = ps.immobile(duree=2.0)
    cibles = derouler(suite, suivi)
    assert cibles[-1].etat == Etat.PRET
    return suivi, suite[-1][0].heure


def test_une_image_manquee_en_pret_reste_pret():
    suivi, t0 = suivi_en_pret()
    cible = suivi.mettre_a_jour(Position3D.invalide(t0 + 1 / IPS), 0.5)
    assert cible.etat == Etat.PRET
    np.testing.assert_allclose((cible.x, cible.y, cible.z), VRAIE)  # dernière position lissée
    cible = suivi.mettre_a_jour(Position3D(*ps.POSITION_BOUCHE, True, t0 + 2 / IPS), 0.5)
    assert cible.etat == Etat.PRET


def test_quatre_images_manquees_en_pret_donnent_suivi():
    suivi, t0 = suivi_en_pret()
    cibles = [suivi.mettre_a_jour(Position3D.invalide(t0 + k / IPS), 0.5) for k in range(1, 5)]
    assert cibles[0].etat == Etat.PRET and cibles[1].etat == Etat.PRET  # 0,033 et 0,067 s
    assert cibles[-1].etat == Etat.SUIVI  # 0,133 s > DELAI_GRACE_PRET_S
    assert cibles[-1].x is not None


def test_reinitialiser():
    suivi = SuiviBouche()
    cibles = derouler(ps.immobile(), suivi)
    assert cibles[-1].etat == Etat.PRET
    suivi.reinitialiser()
    assert suivi.etat == Etat.RECHERCHE
    # Après réinitialisation, l'heure peut repartir de zéro et tout recommence.
    cibles = derouler(ps.immobile(), suivi)
    assert cibles[0].etat == Etat.RECHERCHE and cibles[0].derniere_position_connue is None


def figer(suivi, position, ouverture, heure_depart, duree):
    """Répète la même Position3D (même heure) pendant `duree`, avec heure_actuelle qui avance."""
    cibles = []
    for k in range(1, int(round(duree * IPS)) + 1):
        cibles.append(suivi.mettre_a_jour(position, ouverture, heure_actuelle=heure_depart + k / IPS))
    return cibles


def test_image_figee_en_suivi_jamais_pret_puis_perdu():
    suivi = SuiviBouche()
    suite = ps.immobile(duree=0.5)  # assez pour SUIVI, pas pour PRET
    cibles = derouler(suite, suivi)
    assert cibles[-1].etat == Etat.SUIVI
    derniere, ouverture = suite[-1]
    figees = figer(suivi, derniere, ouverture, derniere.heure, 2.0)
    assert Etat.PRET not in etats(figees)
    i_perdu = premier_indice(figees, Etat.PERDU)
    assert i_perdu is not None
    assert figees[i_perdu].heure - derniere.heure > DELAI_PERDU_S
    assert figees[i_perdu - 1].heure - derniere.heure <= DELAI_PERDU_S
    assert set(etats(figees[i_perdu:])) == {Etat.PERDU}  # l'image figée ne fait jamais revenir
    verifier_coordonnees(figees)


def test_image_figee_en_pret_suivi_puis_perdu():
    suivi = SuiviBouche()
    suite = ps.immobile(duree=2.0)
    cibles = derouler(suite, suivi)
    assert cibles[-1].etat == Etat.PRET
    derniere, ouverture = suite[-1]
    figees = figer(suivi, derniere, ouverture, derniere.heure, 2.0)
    for cible in figees:
        ecoule = cible.heure - derniere.heure
        if ecoule <= DELAI_GRACE_PRET_S - 1e-9:
            assert cible.etat == Etat.PRET  # délai de grâce
        elif DELAI_GRACE_PRET_S + 1e-9 < ecoule <= DELAI_PERDU_S - 1e-9:
            assert cible.etat == Etat.SUIVI  # SUIVI après 0,1 s
        elif ecoule > DELAI_PERDU_S + 1e-9:
            assert cible.etat == Etat.PERDU  # PERDU après 0,3 s
            assert cible.x is None and cible.y is None and cible.z is None
    assert figees[-1].etat == Etat.PERDU


def test_image_figee_ne_compte_pas_dans_la_stabilite():
    suivi = SuiviBouche()
    suite = ps.immobile(duree=0.5)
    derouler(suite, suivi)
    derniere, ouverture = suite[-1]
    nb_acceptees = suivi.nb_acceptees
    avant = suivi.mettre_a_jour(derniere, ouverture, heure_actuelle=derniere.heure + 1 / IPS)
    apres = suivi.mettre_a_jour(derniere, ouverture, heure_actuelle=derniere.heure + 5 / IPS)
    assert apres.stable_depuis == avant.stable_depuis  # le temps de stabilité n'avance pas
    assert suivi.nb_acceptees == nb_acceptees  # l'image déjà vue n'est pas entrée dans la fenêtre
    assert suivi.heure_derniere_acceptee == derniere.heure


def test_mesure_vieille_de_05_s_traitee_comme_invalide():
    # Depuis le départ : des mesures valides mais vieilles de 0,5 s ne font jamais quitter RECHERCHE.
    suivi = SuiviBouche()
    cibles = [suivi.mettre_a_jour(position, ouverture, heure_actuelle=position.heure + 0.5)
              for position, ouverture in ps.immobile(duree=2.0)]
    assert set(etats(cibles)) == {Etat.RECHERCHE}
    assert len(suivi.fenetre) == 0


def test_mesures_qui_vieillissent_en_suivi_donnent_perdu():
    # En SUIVI ou PRET, des mesures qui arrivent avec 0,5 s de retard mènent à PERDU.
    suivi = SuiviBouche()
    suite = ps.immobile(duree=2.0)
    derouler(suite, suivi)
    assert suivi.etat == Etat.PRET
    t0 = suite[-1][0].heure
    cibles = []
    for k in range(1, 31):
        heure_mesure = t0 + k / IPS
        cibles.append(suivi.mettre_a_jour(Position3D(*ps.POSITION_BOUCHE, True, heure_mesure), 0.5,
                                          heure_actuelle=heure_mesure + 0.5))
    assert Etat.PRET not in etats(cibles)
    assert cibles[-1].etat == Etat.PERDU and cibles[-1].x is None
