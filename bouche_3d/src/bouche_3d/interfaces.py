"""INTERFACES : ce que les tâches s'échangent

Rôle
    Décrire le contrat entre les tâches. Tant que ces interfaces ne changent pas,
    chaque tâche peut être écrite et testée seule, sans attendre les autres.
    Ce fichier accueillera plus tard les types de données partagés par les tâches.

Règles communes
    Les distances sont en mètres et les temps en secondes.
    Pour un pixel, u est la colonne et v est la ligne.
    Les images sont au format BGR d'OpenCV. Le nuage de points est un tableau
    (hauteur, largeur, 3) en mètres ; les mesures ratées valent NaN ou inf.

Les quatre interfaces
    détection → position 3D : observation de la bouche
        Centre (u, v) en pixels, ouverture (rapport entre la hauteur et la largeur
        de la bouche), horodatage. S'il n'y a pas de bouche : « aucune bouche ».

    position 3D → stabilité : position 3D
        Position (X, Y, Z) en mètres dans le repère de la caméra, horodatage.
        Si la mesure n'est pas fiable : « invalide ».

    stabilité → envoi au bras : cible stabilisée
        Position stabilisée, état (recherche, suivi, prêt ou perdu), horodatage.
        À l'état « recherche » : pas de position.

    envoi au bras → bras : cible dans le repère du bras
        Position dans le repère du bras, état. Format à définir avec l'équipe
        (question 3 de docs/mandat.md).
"""

# ---------------------------------------------------------------------------
# PROPOSITION, à valider avec la binôme et l'équipe (décision D13).
# Types de données échangés entre les tâches. « frozen=True » : une fois créé,
# un objet ne peut plus être modifié, ce qui évite les erreurs de partage.
# ---------------------------------------------------------------------------

from dataclasses import dataclass


@dataclass(frozen=True)
class ObservationBouche:
    """Interface « détection → position 3D » : la bouche vue sur l'image.

    u, v : centre de la bouche en pixels (u = colonne, v = ligne).
    ouverture : rapport hauteur / largeur de la bouche.
    bouche_vue : False si aucune bouche n'a été trouvée (u, v et ouverture sont alors sans valeur utile).
    heure : horodatage de l'image, en secondes.
    """

    u: float
    v: float
    ouverture: float
    bouche_vue: bool
    heure: float


@dataclass(frozen=True)
class Position3D:
    """Interface « position 3D → stabilité » : la position de la bouche.

    x, y, z : position en mètres dans le repère de la caméra, ou None si invalide.
    valide : False si la mesure n'est pas fiable ; x, y et z valent alors None.
    heure : horodatage de l'image, en secondes.
    """

    x: float | None
    y: float | None
    z: float | None
    valide: bool
    heure: float

    def __post_init__(self) -> None:
        # Vérifier tôt la cohérence entre « valide » et les coordonnées.
        coordonnees = (self.x, self.y, self.z)
        if self.valide and any(c is None for c in coordonnees):
            raise ValueError("Position3D valide : x, y et z doivent être des nombres.")
        if not self.valide and any(c is not None for c in coordonnees):
            raise ValueError("Position3D invalide : x, y et z doivent valoir None.")

    @classmethod
    def invalide(cls, heure: float) -> "Position3D":
        """Crée une position « invalide » à l'heure donnée."""
        return cls(None, None, None, False, heure)
