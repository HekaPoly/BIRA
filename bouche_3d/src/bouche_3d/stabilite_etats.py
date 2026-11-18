"""TÂCHE « STABILITÉ ET ÉTATS » : stabilité et états de suivi

Rôle
    Rendre la position utilisable par le bras : sans tremblement, et accompagnée
    d'un état de fiabilité clair (recherche, suivi, prêt ou perdu).
    Les états sont décrits dans docs/mandat.md, section 3 ; les règles exactes sont ci-dessous.

Reçoit
    À chaque image : une Position3D (valide ou « invalide »), l'ouverture de la bouche
    (ou None si inconnue) et, si on la connaît, l'heure actuelle (heure_actuelle).

Produit
    Une CibleStabilisee (interfaces.py) : état, position lissée (None en recherche et en perdu),
    dernière position connue (pour information), durée de stabilité et heure.

Prérequis
    Aucun matériel : on travaille avec des positions simulées (tests/positions_simulees.py).

Dépend de
    Seulement de l'interface « position 3D → stabilité », pas du code de la tâche
    « position 3D ».

Transmet à
    La tâche « envoi au bras » (repere_bras.py), par l'interface
    « stabilité → envoi au bras ».

Lissage et sauts
    - Position lissée : médiane, axe par axe, des TAILLE_FENETRE dernières positions acceptées.
    - Une position valide à plus de SAUT_MAX_M de la position lissée est « suspecte » et ignorée.
      Si CONFIRMATIONS_SAUT positions suspectes consécutives sont proches entre elles (à moins de
      SAUT_MAX_M de la première), c'est un vrai déplacement : la fenêtre repart de ces positions.

Règles du temps (sécurité)
    - Le temps vient uniquement des données : position.heure et heure_actuelle (qui vaut
      position.heure si elle n'est pas donnée). Jamais de l'horloge : les tests sont reproductibles.
    - Image déjà vue : une position dont l'heure est égale à celle de la dernière position reçue
      n'est pas une nouvelle mesure. Elle n'entre pas dans la fenêtre et ne compte pas dans la
      durée de stabilité : elle est traitée comme une absence de mesure (le temps avance avec
      heure_actuelle). Une image figée finit donc en PERDU.
    - Mesure périmée : si heure_actuelle - position.heure > DELAI_PERDU_S, la mesure est traitée
      comme invalide.
    - Une donnée du capteur ne fait jamais planter le module :
      - mesure dans le futur de plus de TOLERANCE_HORLOGE_S (position.heure > heure_actuelle
        + TOLERANCE_HORLOGE_S) : ignorée, traitée comme invalide ; dans la tolérance, acceptée ;
      - position.heure qui recule par rapport à la dernière position reçue : ignorée.
      Une mesure ignorée ne change pas « la dernière position reçue ».
    - Seul heure_actuelle qui recule lève une ValueError (erreur de l'appelant).

États
    - Départ en RECHERCHE.
    - RECHERCHE → SUIVI : après MIN_VALIDES_SUIVI positions acceptées. Si plus aucune position
      n'est acceptée pendant DELAI_PERDU_S, le décompte et la fenêtre repartent de zéro.
    - SUIVI → PRET : la position lissée reste à moins de TOLERANCE_STABLE_M depuis au moins
      DUREE_STABLE_S (durée mesurée sur les heures des mesures acceptées), ET l'ouverture est
      ≥ OUVERTURE_PRET, ET l'image courante apporte une mesure acceptée.
    - PRET → SUIVI : mouvement au-delà de la tolérance, ou ouverture < OUVERTURE_PRET
      - HYSTERESIS_OUVERTURE (évite le clignotement autour du seuil), ou plus aucune mesure
      acceptée depuis plus de DELAI_GRACE_PRET_S.
    - Délai de grâce : en PRET, une image sans mesure acceptée (absence, image déjà vue, mesure
      périmée ou ignorée, saut ignoré) laisse en PRET, avec la dernière position lissée, tant que
      le temps écoulé depuis la dernière mesure acceptée est ≤ DELAI_GRACE_PRET_S.
      Il faut DELAI_GRACE_PRET_S < DELAI_PERDU_S (vérifié par le constructeur).
    - Ouverture None : jamais PRET.
    - SUIVI ou PRET → PERDU : aucune position acceptée depuis plus de DELAI_PERDU_S.
      La durée de stabilité repart de zéro.
    - PERDU → SUIVI : MIN_VALIDES_SUIVI nouvelles positions acceptées (jamais directement PRET).
      La fenêtre ne garde alors que ces nouvelles positions.
    - PERDU → RECHERCHE : aucune position acceptée depuis plus de DELAI_OUBLI_S ; tout est oublié.
    - On n'atteint jamais PRET directement depuis RECHERCHE ou PERDU.

Terminée quand
    Sur des suites de positions simulées (stable, qui saute, qui disparaît puis revient),
    la position ne tremble pas et les états suivent les règles.

À savoir
    Trop de lissage : la cible traîne derrière la vraie position. Pas assez : elle tremble.
    Un état « prêt » trop facile à obtenir présente un risque pour la personne : être prudent.

Questions liées
    4 et 7 de docs/mandat.md.
"""

from collections import deque

import numpy as np

from bouche_3d.interfaces import CibleStabilisee, Etat, Position3D

# Réglages : provisoires, à régler par mesure (décision D17).
TAILLE_FENETRE = 5            # nombre de positions acceptées dans la médiane
SAUT_MAX_M = 0.05             # au-delà, une position est suspecte (m)
CONFIRMATIONS_SAUT = 3        # positions suspectes proches qui confirment un vrai déplacement
TOLERANCE_STABLE_M = 0.01     # mouvement toléré de la position lissée pour rester « stable » (m)
DUREE_STABLE_S = 1.0          # durée de stabilité exigée pour PRET (s)
OUVERTURE_PRET = 0.30         # ouverture minimale pour PRET
HYSTERESIS_OUVERTURE = 0.05   # PRET n'est quitté que sous OUVERTURE_PRET - HYSTERESIS_OUVERTURE
MIN_VALIDES_SUIVI = 3         # positions acceptées pour passer en SUIVI
DELAI_PERDU_S = 0.3           # sans position acceptée depuis ce délai : PERDU ; aussi l'âge max d'une mesure (s)
DELAI_OUBLI_S = 3.0           # sans position acceptée depuis ce délai : RECHERCHE (s)
DELAI_GRACE_PRET_S = 0.1      # en PRET, images sans mesure tolérées pendant ce délai (s)
TOLERANCE_HORLOGE_S = 0.05    # avance maximale tolérée d'une mesure sur heure_actuelle (s)


def _distance(a, b) -> float:
    return float(np.linalg.norm(np.asarray(a, dtype=float) - np.asarray(b, dtype=float)))


class SuiviBouche:
    """Lisse les positions de la bouche et décide de l'état de fiabilité, image par image."""

    def __init__(self, taille_fenetre: int = TAILLE_FENETRE, saut_max_m: float = SAUT_MAX_M,
                 confirmations_saut: int = CONFIRMATIONS_SAUT,
                 tolerance_stable_m: float = TOLERANCE_STABLE_M, duree_stable_s: float = DUREE_STABLE_S,
                 ouverture_pret: float = OUVERTURE_PRET, hysteresis_ouverture: float = HYSTERESIS_OUVERTURE,
                 min_valides_suivi: int = MIN_VALIDES_SUIVI, delai_perdu_s: float = DELAI_PERDU_S,
                 delai_oubli_s: float = DELAI_OUBLI_S, delai_grace_pret_s: float = DELAI_GRACE_PRET_S,
                 tolerance_horloge_s: float = TOLERANCE_HORLOGE_S) -> None:
        if delai_grace_pret_s >= delai_perdu_s:
            raise ValueError(f"delai_grace_pret_s ({delai_grace_pret_s}) doit être inférieur à "
                             f"delai_perdu_s ({delai_perdu_s}).")
        self.delai_grace_pret_s = delai_grace_pret_s
        self.tolerance_horloge_s = tolerance_horloge_s
        self.taille_fenetre = taille_fenetre
        self.saut_max_m = saut_max_m
        self.confirmations_saut = confirmations_saut
        self.tolerance_stable_m = tolerance_stable_m
        self.duree_stable_s = duree_stable_s
        self.ouverture_pret = ouverture_pret
        self.hysteresis_ouverture = hysteresis_ouverture
        self.min_valides_suivi = min_valides_suivi
        self.delai_perdu_s = delai_perdu_s
        self.delai_oubli_s = delai_oubli_s
        self.reinitialiser()

    def reinitialiser(self) -> None:
        """Oublie tout : retour en RECHERCHE, comme au départ."""
        self.etat = Etat.RECHERCHE
        self.fenetre: deque = deque(maxlen=self.taille_fenetre)
        self.suspectes: list[tuple[float, float, float]] = []
        self.nb_acceptees = 0                # décompte vers SUIVI (depuis RECHERCHE ou PERDU)
        self.heure_derniere_acceptee = None  # position.heure de la dernière mesure acceptée
        self.heure_derniere_recue = None     # position.heure de la dernière position reçue
        self.heure_actuelle_precedente = None
        self.reference_stable = None         # position lissée au début de la période stable
        self.debut_stable = None             # heure de mesure du début de la période stable
        self.derniere_publiee = None         # dernière position publiée en SUIVI ou PRET

    # -- Calculs internes ----------------------------------------------------

    def _lissee(self):
        if not self.fenetre:
            return None
        x, y, z = np.median(np.array(self.fenetre, dtype=float), axis=0)
        return float(x), float(y), float(z)

    def _vider(self) -> None:
        self.fenetre.clear()
        self.suspectes.clear()
        self.nb_acceptees = 0
        self.reference_stable = None
        self.debut_stable = None

    def _stable_depuis(self) -> float:
        if self.debut_stable is None:
            return 0.0
        return self.heure_derniere_acceptee - self.debut_stable

    def _accepter(self, point, heure_mesure: float) -> bool:
        """Traite une mesure valide et fraîche. Renvoie True si elle est acceptée."""
        lissee = self._lissee()
        if lissee is None or _distance(point, lissee) <= self.saut_max_m:
            self.fenetre.append(point)
            self.suspectes.clear()
            nouvelles = 1
        else:
            # Position suspecte : ignorée, sauf si elle confirme un vrai déplacement.
            if self.suspectes and _distance(point, self.suspectes[0]) > self.saut_max_m:
                self.suspectes.clear()  # pas proche des précédentes : nouvelle série
            self.suspectes.append(point)
            if len(self.suspectes) < self.confirmations_saut:
                return False
            self.fenetre.clear()
            self.fenetre.extend(self.suspectes)
            nouvelles = len(self.suspectes)
            self.suspectes.clear()
        self.nb_acceptees += nouvelles
        self.heure_derniere_acceptee = heure_mesure
        # Stabilité : la position lissée doit rester près de la référence.
        lissee = self._lissee()
        if self.reference_stable is None or _distance(lissee, self.reference_stable) > self.tolerance_stable_m:
            self.reference_stable = lissee
            self.debut_stable = heure_mesure
        return True

    def _mesure_ignoree(self, position: Position3D, heure_actuelle: float) -> bool:
        """True si l'heure de la mesure est incohérente (futur, recul) : on l'ignore sans planter."""
        if position.heure > heure_actuelle + self.tolerance_horloge_s:
            return True  # dans le futur, au-delà de la tolérance d'horloge
        return self.heure_derniere_recue is not None and position.heure < self.heure_derniere_recue

    def _pret_permis(self, ouverture, seuil: float) -> bool:
        return (ouverture is not None and ouverture >= seuil
                and self._stable_depuis() >= self.duree_stable_s)

    # -- Mise à jour image par image -----------------------------------------

    def mettre_a_jour(self, position: Position3D, ouverture: float | None,
                      heure_actuelle: float | None = None) -> CibleStabilisee:
        """Traite une image et renvoie la cible stabilisée.

        position : position 3D de cette image (valide ou invalide).
        ouverture : ouverture de la bouche (hauteur / largeur), ou None si inconnue.
        heure_actuelle : heure de la décision, en secondes ; position.heure si None.
        """
        if heure_actuelle is None:
            heure_actuelle = position.heure
        # Seule erreur levée : l'appelant fait reculer son propre temps.
        if self.heure_actuelle_precedente is not None and heure_actuelle < self.heure_actuelle_precedente:
            raise ValueError(f"heure_actuelle recule : {heure_actuelle} < {self.heure_actuelle_precedente}.")
        self.heure_actuelle_precedente = heure_actuelle

        fraiche = False
        if not self._mesure_ignoree(position, heure_actuelle):
            deja_vue = self.heure_derniere_recue is not None and position.heure == self.heure_derniere_recue
            perimee = heure_actuelle - position.heure > self.delai_perdu_s
            self.heure_derniere_recue = position.heure
            if position.valide and not deja_vue and not perimee:
                fraiche = self._accepter((position.x, position.y, position.z), position.heure)

        absence = (None if self.heure_derniere_acceptee is None
                   else heure_actuelle - self.heure_derniere_acceptee)
        self._changer_etat(fraiche, absence, ouverture)
        return self._cible(heure_actuelle)

    def _changer_etat(self, fraiche: bool, absence, ouverture) -> None:
        etat = self.etat
        if etat == Etat.RECHERCHE:
            if not fraiche and absence is not None and absence > self.delai_perdu_s:
                self._vider()  # positions trop espacées : on recommence le décompte
                self.heure_derniere_acceptee = None
            elif self.nb_acceptees >= self.min_valides_suivi:
                self.etat = Etat.SUIVI
        elif etat in (Etat.SUIVI, Etat.PRET):
            if not fraiche:
                if absence > self.delai_perdu_s:
                    self.etat = Etat.PERDU
                    self.suspectes.clear()
                    self.nb_acceptees = 0
                    self.reference_stable = None  # la stabilité devra être retrouvée
                    self.debut_stable = None
                elif etat == Etat.PRET and absence > self.delai_grace_pret_s:
                    self.etat = Etat.SUIVI  # délai de grâce dépassé
            elif etat == Etat.SUIVI:
                if self._pret_permis(ouverture, self.ouverture_pret):
                    self.etat = Etat.PRET
            elif not self._pret_permis(ouverture, self.ouverture_pret - self.hysteresis_ouverture):
                self.etat = Etat.SUIVI
        elif etat == Etat.PERDU:
            if not fraiche and absence > self.delai_oubli_s:
                self._vider()
                self.heure_derniere_acceptee = None
                self.derniere_publiee = None
                self.etat = Etat.RECHERCHE
            elif self.nb_acceptees >= self.min_valides_suivi:
                # Ne garder que les positions vues depuis le retour.
                nouvelles = list(self.fenetre)[-self.nb_acceptees:]
                self.fenetre.clear()
                self.fenetre.extend(nouvelles)
                self.reference_stable = self._lissee()
                self.etat = Etat.SUIVI

    def _cible(self, heure_actuelle: float) -> CibleStabilisee:
        if self.etat in (Etat.SUIVI, Etat.PRET):
            x, y, z = self._lissee()
            self.derniere_publiee = (x, y, z)
            return CibleStabilisee(self.etat, x, y, z, self.derniere_publiee,
                                   self._stable_depuis(), heure_actuelle)
        return CibleStabilisee(self.etat, None, None, None, self.derniere_publiee, 0.0, heure_actuelle)
