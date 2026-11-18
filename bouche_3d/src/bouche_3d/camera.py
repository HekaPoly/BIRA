"""CAMÉRA : lecture des images et de la profondeur

Rôle
    Fournir aux autres fichiers les images de la caméra, et le nuage de points quand
    la caméra est une ZED. Toutes les sources s'utilisent de la même façon :
    - une webcam (pour développer la détection sans ZED) ;
    - la ZED Mini en direct ;
    - un enregistrement de la ZED (fichier .svo), pour rejouer les mêmes images ;
    - la caméra déjà ouverte par BIRA (source « bira », décision D14) : c'est la source
      à utiliser une fois le module branché, car une ZED ne s'ouvre qu'une fois.

Reçoit
    Le choix de la source (webcam, ZED, chemin d'un fichier .svo ou objet Camera de BIRA)
    et ses réglages.

Produit
    À chaque image :
    - l'image couleur (format BGR d'OpenCV) ;
    - le nuage de points (tableau NumPy de forme (hauteur, largeur, 3), X, Y, Z en mètres),
      seulement avec la ZED ; None avec une webcam ;
    - l'heure de l'image, en secondes.
    Signale aussi quand plus aucune image n'est disponible (fin du fichier, caméra débranchée).

Prérequis
    Webcam : OpenCV.
    ZED et fichier .svo : le ZED SDK et pyzed, qui exigent une carte graphique NVIDIA.
    pyzed ne doit être importé que si la ZED est utilisée, pour que la webcam fonctionne
    sur un PC sans ZED SDK.

Dépend de
    Rien dans le projet.

Transmet à
    La tâche « détection de la bouche » (l'image) et la tâche « position 3D »
    (le nuage de points), au moment de l'assemblage.

Terminée quand
    - La webcam fournit des images en direct.
    - La ZED fournit l'image et le nuage de points en mètres, et un pixel visé
      donne une distance cohérente avec une mesure à la règle.
    - Un fichier .svo se rejoue comme la caméra en direct.
    - La caméra est toujours libérée à la fin, même en cas d'erreur.

À savoir
    - Régler l'unité en mètres à l'ouverture de la ZED (coordinate_units = METER).
    - L'image de la ZED a 4 canaux (BGRA) : la convertir en BGR pour OpenCV et MediaPipe.
    - Le nuage de la ZED a 4 valeurs par pixel (X, Y, Z, couleur) : ne garder que X, Y, Z.
    - Les mesures ratées valent NaN ou inf : ne pas les corriger ici, position_3d.py s'en charge.
    - Utiliser l'image gauche de la ZED : c'est celle qui correspond au nuage de points.
    - Comparer le résultat de grab() à sl.ERROR_CODE.SUCCESS : un code d'erreur est
      « vrai » en Python, « if grab(): » ne détecte donc pas les échecs.
    - L'heure d'une trame ZED est l'horodatage de l'image (TIME_REFERENCE.IMAGE), en secondes.
    - Source « bira » : ne jamais ouvrir ni fermer la caméra, qui appartient à BIRA. BIRA l'ouvre
      avec les réglages par défaut (millimètres probables) : la source convertit en mètres
      (décision D15). Elle n'importe rien de BIRA : elle utilise seulement camera.get_camera()
      et camera.grab().

Réglages à choisir (puis noter dans docs/decisions.md)
    Résolution, nombre d'images par seconde, mode de profondeur de la ZED, distance minimale.

Questions liées
    1, 5 et 8 de docs/mandat.md.
"""

import sys
import time
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

# Message affiché quand pyzed manque.
MESSAGE_SANS_ZED = "ZED SDK non installé (carte NVIDIA nécessaire)"

# Noms acceptés par creer_source.
NOMS_SOURCES = ("webcam", "simulation", "npy", "zed", "svo", "bira")

# Nombre d'échecs de grab() consécutifs avant d'abandonner la lecture (SourceZed).
MAX_ECHECS_GRAB = 10

# Unités d'entrée acceptées par SourceBira : lue sur la caméra, millimètres ou mètres.
UNITES_ENTREE = ("auto", "mm", "m")


def _importer_pyzed():
    """Importe pyzed.sl au moment voulu, avec un message clair s'il manque."""
    try:
        import pyzed.sl as sl
    except ImportError as erreur:
        raise RuntimeError(MESSAGE_SANS_ZED) from erreur
    return sl


def _heure_image(sl, zed) -> float:
    """Horodatage de la dernière image de la ZED, en secondes."""
    return zed.get_timestamp(sl.TIME_REFERENCE.IMAGE).get_nanoseconds() / 1e9


@dataclass
class Trame:
    """Une image de la source.

    image : image couleur BGR (hauteur, largeur, 3), uint8.
    nuage : X, Y, Z en mètres (hauteur, largeur, 3), ou None (webcam).
    heure : horodatage en secondes.
    """

    image: np.ndarray
    nuage: np.ndarray | None
    heure: float


class Source:
    """Base commune à toutes les sources. S'utilise avec « with » :

        with creer_source("simulation") as source:
            trame = source.lire()
    """

    def ouvrir(self) -> None:
        """Prépare la source (rien à faire par défaut)."""

    def lire(self) -> Trame | None:
        """Renvoie la trame suivante, ou None s'il n'y a plus d'image."""
        raise NotImplementedError

    def fermer(self) -> None:
        """Libère la source (rien à faire par défaut)."""

    def __enter__(self) -> "Source":
        self.ouvrir()
        return self

    def __exit__(self, *erreur) -> None:
        # Appelé même en cas d'erreur : la source est toujours libérée.
        self.fermer()


# ---------------------------------------------------------------------------
# a. Webcam
# ---------------------------------------------------------------------------

class SourceWebcam(Source):
    """Webcam lue avec OpenCV. Pas de profondeur : nuage = None."""

    def __init__(self, indice: int = 0) -> None:
        self.indice = indice
        self.capture: cv2.VideoCapture | None = None

    def ouvrir(self) -> None:
        # CAP_DSHOW sous Windows : ouverture plus rapide et plus fiable.
        api = cv2.CAP_DSHOW if sys.platform == "win32" else cv2.CAP_ANY
        self.capture = cv2.VideoCapture(self.indice, api)
        if not self.capture.isOpened():
            self.fermer()
            raise RuntimeError(f"Impossible d'ouvrir la webcam {self.indice}.")

    def lire(self) -> Trame | None:
        if self.capture is None:
            return None
        ok, image = self.capture.read()
        if not ok:
            return None
        return Trame(image, None, time.time())

    def fermer(self) -> None:
        if self.capture is not None:
            self.capture.release()
            self.capture = None


# ---------------------------------------------------------------------------
# b. Simulation (aucun matériel)
# ---------------------------------------------------------------------------

# Caméra fictive et scène de la simulation.
FOCALE = 700.0          # fx = fy, en pixels
Z_FOND = 1.2            # distance du fond, en mètres
RELIEF = 0.02           # les bords du visage sont 2 cm plus loin que son centre
PART_ABERRANTS = 0.002  # part des pixels avec une valeur aberrante


class SourceSimulation(Source):
    """Faux visage devant un fond, avec bruit, trous et valeurs aberrantes.

    position_bouche_reelle : (u, v, Z) du centre de la bouche, sans bruit.
    position_front_reelle : (u, v, Z) d'un point du front, sans bruit.
    z_reel : carte de Z sans bruit ni trous.
    """

    def __init__(self, largeur: int = 1280, hauteur: int = 720, distance: float = 0.55,
                 bruit: float = 0.002, part_trous_bouche: float = 0.3, graine: int = 0) -> None:
        self.largeur = largeur
        self.hauteur = hauteur
        self.distance = distance
        self.bruit = bruit
        self.part_trous_bouche = part_trous_bouche
        self.cx = largeur / 2
        self.cy = hauteur / 2
        self.rng = np.random.default_rng(graine)
        self.numero = 0
        # Grille des pixels (v = ligne, u = colonne), calculée une seule fois.
        self.v_grille, self.u_grille = np.mgrid[0:hauteur, 0:largeur].astype(float)

        # Centres de la bouche (5,5 cm sous le centre du visage) et du front (5,5 cm au-dessus).
        u_c = int(round(self.cx))
        v_bouche = int(round(self.cy + self._en_pixels(0.055)))
        v_front = int(round(self.cy - self._en_pixels(0.055)))
        if v_front < 0 or v_bouche >= hauteur:
            raise ValueError("Image trop petite pour cette distance : agrandir l'image ou éloigner le visage.")
        self.z_reel, self.masque_bouche = self._construire_scene(u_c, v_bouche)
        self.position_bouche_reelle = (u_c, v_bouche, float(self.z_reel[v_bouche, u_c]))
        self.position_front_reelle = (u_c, v_front, float(self.z_reel[v_front, u_c]))

    def _en_pixels(self, longueur: float) -> float:
        """Taille en pixels d'une longueur (m) placée à la distance du visage."""
        return longueur * FOCALE / self.distance

    def _construire_scene(self, u_b: int, v_b: int) -> tuple[np.ndarray, np.ndarray]:
        """Carte de Z sans bruit, et masque de la bouche centrée en (u_b, v_b)."""
        u, v = self.u_grille, self.v_grille

        # Visage : ellipse de 15 × 20 cm centrée dans l'image, avec un léger relief.
        r2 = ((u - self.cx) / self._en_pixels(0.075)) ** 2 + ((v - self.cy) / self._en_pixels(0.10)) ** 2
        visage = r2 <= 1.0
        z = np.full((self.hauteur, self.largeur), Z_FOND)
        z[visage] = self.distance + RELIEF * r2[visage]

        # Bouche : petite ellipse de 5 × 2,4 cm.
        bouche = ((u - u_b) / self._en_pixels(0.025)) ** 2 + ((v - v_b) / self._en_pixels(0.012)) ** 2 <= 1.0
        return z, bouche

    def lire(self) -> Trame | None:
        # Nouvelle trame : nouveau bruit et nouveaux trous (la scène ne bouge pas).
        z = self.z_reel + self.rng.normal(0.0, self.bruit, self.z_reel.shape)

        # Quelques valeurs aberrantes (entre 0,3 et 3 fois la vraie distance).
        aberrants = self.rng.random(z.shape) < PART_ABERRANTS
        z[aberrants] *= self.rng.uniform(0.3, 3.0, int(aberrants.sum()))

        # Trous dans la bouche, comme une vraie bouche ouverte.
        trous = self.masque_bouche & (self.rng.random(z.shape) < self.part_trous_bouche)
        z[trous] = np.nan

        heure = self.numero / 30.0  # 30 images par seconde simulées
        self.numero += 1
        return Trame(self._image_depuis_z(z), self._nuage_depuis_z(z), heure)

    def _nuage_depuis_z(self, z: np.ndarray) -> np.ndarray:
        """X = (u - cx)·Z/fx, Y = (v - cy)·Z/fy (fx = fy = FOCALE)."""
        x = (self.u_grille - self.cx) * z / FOCALE
        y = (self.v_grille - self.cy) * z / FOCALE
        return np.stack([x, y, z], axis=-1)

    def _image_depuis_z(self, z: np.ndarray) -> np.ndarray:
        return coloriser_profondeur(z)


# Plage de la palette des cartes de profondeur (la légende de la démo utilise la même).
Z_COULEUR_PROCHE = 0.2  # rouge
Z_COULEUR_LOIN = 1.2    # bleu


def coloriser_profondeur(z: np.ndarray) -> np.ndarray:
    """Carte de Z (m) en fausses couleurs BGR : proche = rouge, loin = bleu ; NaN en noir."""
    niveaux = (z - Z_COULEUR_PROCHE) / (Z_COULEUR_LOIN - Z_COULEUR_PROCHE)
    niveaux = np.nan_to_num(np.clip(niveaux, 0.0, 1.0), nan=1.0)
    image = cv2.applyColorMap((255 * (1.0 - niveaux)).astype(np.uint8), cv2.COLORMAP_JET)
    image[np.isnan(z)] = 0
    return image


# ---------------------------------------------------------------------------
# c. Trames enregistrées (.npz)
# ---------------------------------------------------------------------------

class SourceNpy(Source):
    """Rejoue, dans l'ordre, les fichiers trame_XXXX.npz d'un dossier."""

    def __init__(self, dossier: str | Path = "data") -> None:
        self.dossier = Path(dossier)
        self.fichiers: list[Path] = []
        self.position = 0

    def ouvrir(self) -> None:
        self.fichiers = sorted(self.dossier.glob("trame_*.npz"))
        self.position = 0
        if not self.fichiers:
            raise FileNotFoundError(f"Aucun fichier trame_XXXX.npz dans {self.dossier}.")

    def lire(self) -> Trame | None:
        if self.position >= len(self.fichiers):
            return None  # plus d'image
        fichier = self.fichiers[self.position]
        self.position += 1
        with np.load(fichier, allow_pickle=False) as donnees:
            nuage = donnees["nuage"] if "nuage" in donnees.files else None
            return Trame(donnees["image"], nuage, float(donnees["heure"]))


def enregistrer_trame(trame: Trame, dossier: str | Path, numero: int) -> Path:
    """Sauvegarde la trame dans dossier/trame_XXXX.npz et renvoie le chemin du fichier."""
    dossier = Path(dossier)
    dossier.mkdir(parents=True, exist_ok=True)
    chemin = dossier / f"trame_{numero:04d}.npz"
    donnees = {"image": trame.image, "heure": np.float64(trame.heure)}
    if trame.nuage is not None:  # webcam : pas de nuage à sauvegarder
        donnees["nuage"] = trame.nuage
    np.savez_compressed(chemin, **donnees)
    return chemin


# ---------------------------------------------------------------------------
# d. ZED Mini (en direct ou fichier .svo)
# ---------------------------------------------------------------------------

class SourceZed(Source):
    """ZED en direct, ou rejeu d'un fichier .svo si fichier_svo est donné.

    pyzed n'est importé qu'à l'ouverture : le reste du fichier marche sans ZED SDK.
    """

    def __init__(self, fichier_svo: str | Path | None = None, mode_profondeur: str = "NEURAL") -> None:
        self.fichier_svo = fichier_svo
        self.mode_profondeur = mode_profondeur
        self.sl = None
        self.camera = None

    def ouvrir(self) -> None:
        sl = _importer_pyzed()
        self.sl = sl

        parametres = sl.InitParameters()
        parametres.coordinate_units = sl.UNIT.METER
        parametres.coordinate_system = sl.COORDINATE_SYSTEM.IMAGE  # X droite, Y bas, Z avant
        parametres.depth_mode = getattr(sl.DEPTH_MODE, self.mode_profondeur)
        if self.fichier_svo is not None:
            parametres.set_from_svo_file(str(self.fichier_svo))

        camera = sl.Camera()
        code = camera.open(parametres)
        if code != sl.ERROR_CODE.SUCCESS:
            raise RuntimeError(f"Impossible d'ouvrir la ZED : {code}")
        self.camera = camera
        self.runtime = sl.RuntimeParameters()
        self.mat_image = sl.Mat()
        self.mat_nuage = sl.Mat()

    def lire(self) -> Trame | None:
        if self.camera is None:
            return None
        sl = self.sl
        # Une erreur passagère ne termine pas la lecture : on réessaie.
        echecs = 0
        while True:
            code = self.camera.grab(self.runtime)
            if code == sl.ERROR_CODE.SUCCESS:
                break
            if code == sl.ERROR_CODE.END_OF_SVOFILE_REACHED:
                return None  # fin du fichier .svo
            echecs += 1
            if echecs >= MAX_ECHECS_GRAB:
                return None  # caméra débranchée ou panne durable
        self.camera.retrieve_image(self.mat_image, sl.VIEW.LEFT)
        self.camera.retrieve_measure(self.mat_nuage, sl.MEASURE.XYZRGBA)
        image = cv2.cvtColor(self.mat_image.get_data(), cv2.COLOR_BGRA2BGR)
        nuage = self.mat_nuage.get_data()[:, :, :3].copy()  # X, Y, Z seulement
        return Trame(image, nuage, _heure_image(sl, self.camera))

    def fermer(self) -> None:
        if self.camera is not None:
            self.camera.close()
            self.camera = None


# ---------------------------------------------------------------------------
# e. Caméra de BIRA (déjà ouverte par BIRA)
# ---------------------------------------------------------------------------

class SourceBira(Source):
    """Lit la ZED déjà ouverte par BIRA, sans jamais l'ouvrir ni la fermer (décision D14).

    camera : objet Camera de BIRA (src/bira_components/camera.py). Seules ses méthodes
        get_camera() et grab() sont utilisées ; rien n'est importé de BIRA.
    unite_entree : unité du nuage fourni par la caméra : "auto" (lue sur la caméra),
        "mm" ou "m". Le nuage rendu est toujours en mètres (décision D15).
    faire_grab : True si cette source doit appeler grab() elle-même ; False (défaut) si
        BIRA s'en charge et qu'on lit seulement la dernière image.
    """

    def __init__(self, camera, unite_entree: str = "auto", faire_grab: bool = False) -> None:
        if unite_entree not in UNITES_ENTREE:
            raise ValueError(f"unite_entree inconnue : {unite_entree!r}. Valeurs permises : "
                             f"{', '.join(UNITES_ENTREE)}.")
        self.camera = camera
        self.unite_entree = unite_entree
        self.faire_grab = faire_grab
        self.sl = None
        self.mat_image = None
        self.mat_nuage = None

    def ouvrir(self) -> None:
        sl = _importer_pyzed()
        if not self.camera.get_camera().is_opened():
            raise RuntimeError("Source bira : la caméra doit être ouverte par BIRA "
                               "(Camera.open()) avant ouvrir() ; cette source ne l'ouvre jamais.")
        self.sl = sl
        # Nos propres sl.Mat, créées une seule fois : on ne partage pas l'image de BIRA.
        if self.mat_image is None:
            self.mat_image = sl.Mat()
            self.mat_nuage = sl.Mat()

    def _facteur_metres(self, zed) -> float:
        """Facteur qui convertit le nuage de la caméra en mètres."""
        if self.unite_entree == "mm":
            return 0.001
        if self.unite_entree == "m":
            return 1.0
        unite = zed.get_init_parameters().coordinate_units
        if unite == self.sl.UNIT.MILLIMETER:
            return 0.001
        if unite == self.sl.UNIT.METER:
            return 1.0
        # Mieux vaut s'arrêter que deviner : une erreur d'unité fausse toute la position.
        raise ValueError(f"Unité de la caméra non prise en charge : {unite}. "
                         "Attendu : MILLIMETER ou METER (ou donner unite_entree).")

    def lire(self) -> Trame | None:
        if self.mat_image is None:
            return None  # source pas ouverte
        sl = self.sl
        if self.faire_grab and self.camera.grab() != sl.ERROR_CODE.SUCCESS:
            return None  # grab raté : pas d'image cette fois
        zed = self.camera.get_camera()
        if zed.retrieve_image(self.mat_image, sl.VIEW.LEFT) != sl.ERROR_CODE.SUCCESS:
            return None
        if zed.retrieve_measure(self.mat_nuage, sl.MEASURE.XYZRGBA) != sl.ERROR_CODE.SUCCESS:
            return None
        facteur = self._facteur_metres(zed)
        # Copies systématiques : les sl.Mat sont réécrites à l'image suivante.
        image = cv2.cvtColor(self.mat_image.get_data(), cv2.COLOR_BGRA2BGR).copy()
        nuage = self.mat_nuage.get_data()[:, :, :3].copy()  # X, Y, Z seulement
        if facteur != 1.0:
            nuage *= facteur
        return Trame(image, nuage, _heure_image(sl, zed))

    def fermer(self) -> None:
        # Libère seulement nos sl.Mat ; la caméra reste ouverte pour BIRA.
        for mat in (self.mat_image, self.mat_nuage):
            if mat is not None:
                mat.free()
        self.mat_image = None
        self.mat_nuage = None


# ---------------------------------------------------------------------------
# f. Choix de la source par son nom
# ---------------------------------------------------------------------------

def creer_source(nom: str, **options) -> Source:
    """Crée une source, sans l'ouvrir : "webcam", "simulation", "npy", "zed", "svo" ou "bira".

    Exemples : creer_source("npy", dossier="data") ; creer_source("svo", fichier_svo="essai.svo") ;
    creer_source("bira", camera=controller.camera).
    """
    if nom == "webcam":
        return SourceWebcam(**options)
    if nom == "simulation":
        return SourceSimulation(**options)
    if nom == "npy":
        return SourceNpy(**options)
    if nom == "zed":
        return SourceZed(**options)
    if nom == "svo":
        if options.get("fichier_svo") is None:
            raise ValueError("La source « svo » demande fichier_svo (chemin du fichier .svo).")
        return SourceZed(**options)
    if nom == "bira":
        if options.get("camera") is None:
            raise ValueError("La source « bira » demande camera (l'objet Camera de BIRA, déjà ouvert).")
        return SourceBira(**options)
    raise ValueError(f"Source inconnue : {nom!r}. Noms permis : {', '.join(NOMS_SOURCES)}.")
