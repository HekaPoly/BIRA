"""SCRIPT DE DÉMONSTRATION DE LA TÂCHE « POSITION 3D »

Rôle
    Afficher l'image d'une source et donner la position 3D du pixel cliqué,
    pour vérifier le calcul à l'œil (et à la règle avec la ZED, voir docs/mesures.md).
    L'écran doit se comprendre sans explication, même par une personne qui ne connaît pas le projet.

Reçoit
    Le choix de la source (--source simulation, webcam, npy, zed ou svo) et ses options :
    --fichier (fichier .svo), --dossier (trames .npz), --rayon (demi-côté de la zone de mesure).

Produit
    Une fenêtre OpenCV en deux parties :
    - à gauche, l'image de la caméra : carré jaune et point autour du clic, distance du pixel
      survolé près du curseur, et (simulation seulement) étiquettes « visage », « bouche », « mur » ;
    - à droite, un panneau : résultat en gros (vert = le bras peut viser, rouge = invalide,
      avec la raison), zoom de la zone de mesure (blanc = mesuré, noir = trou), légende des
      couleurs, repère X / Y / Z et liste des commandes.
    Touches : clic = mesurer ; + / - = taille de la zone ; l = afficher ou cacher la légende
    et les étiquettes ; s = enregistrer la trame dans data/ ; c = capture de toute la fenêtre
    dans data/captures/ ; espace = image suivante (npy et svo) ; q = quitter.

Prérequis
    numpy et opencv-python. Sources zed et svo : ZED SDK et pyzed (carte NVIDIA).

Dépend de
    camera.py (les sources, la palette) et position_3d.py (position_depuis_pixel, diagnostic_mesure).

Terminée quand
    Elle se lance avec --source simulation, un clic sur la bouche ou le front donne
    Z proche de la distance simulée, un clic sur le fond donne « INVALIDE » avec sa raison,
    et la source est toujours fermée à la fin, même en cas d'erreur.

À savoir
    Les polices d'OpenCV n'affichent pas les accents : le texte à l'écran est écrit sans accents.

Usage
    python scripts/demo_position_3d.py --source simulation
    python scripts/demo_position_3d.py --source npy --dossier data
    python scripts/demo_position_3d.py --source svo --fichier data/essai.svo
"""

import argparse
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from bouche_3d.camera import (
    FOCALE,
    NOMS_SOURCES,
    Z_COULEUR_LOIN,
    Z_COULEUR_PROCHE,
    Z_FOND,
    SourceSimulation,
    Trame,
    coloriser_profondeur,
    creer_source,
    enregistrer_trame,
)
from bouche_3d.position_3d import DiagnosticMesure, diagnostic_mesure, position_depuis_pixel

FENETRE = "Position 3D (q = quitter)"
DOSSIER_DATA = Path(__file__).resolve().parents[1] / "data"
SOURCES_PAS_A_PAS = ("npy", "svo")  # image suivante seulement sur « espace »

# Réglages du calcul (décision D12), affichés tels quels dans les messages.
PART_MIN, Z_MIN, Z_MAX = 0.5, 0.2, 1.0

# Mise en page (pixels).
LARGEUR_PANNEAU = 450
HAUTEUR_ECRAN = 900
LARGEUR_IMAGE_MAX = 1280
MARGE = 18
TAILLE_ZOOM = 150

# Couleurs BGR : vert = fiable, rouge = invalide, blanc = information.
BLANC, GRIS, JAUNE, NOIR = (255, 255, 255), (170, 170, 170), (0, 230, 255), (0, 0, 0)
FOND_PANNEAU, FOND_IMAGE = (45, 45, 45), (25, 25, 25)
FOND_VERT, FOND_ROUGE, FOND_NEUTRE = (40, 130, 40), (40, 40, 190), (90, 90, 90)
POLICE = cv2.FONT_HERSHEY_SIMPLEX


@dataclass
class Etat:
    """Ce que l'utilisateur a fait : clic, survol, taille de la zone, légende, message."""

    rayon: int = 5
    clic: tuple[int, int] | None = None     # pixel cliqué (u, v) dans l'image
    survol: tuple[int, int] | None = None   # pixel sous la souris
    legende: bool = True
    message: str = ""


@dataclass(frozen=True)
class Disposition:
    """Place de l'image dans l'écran : échelle et décalage."""

    echelle: float
    x0: int
    y0: int
    largeur: int  # largeur de la partie image
    hauteur: int  # hauteur de la partie image

    @property
    def taille_ecran(self) -> tuple[int, int]:
        return self.largeur + LARGEUR_PANNEAU, HAUTEUR_ECRAN

    def vers_image(self, x: int, y: int, forme: tuple) -> tuple[int, int] | None:
        """Point de l'écran → pixel de l'image, ou None hors de l'image."""
        u = int((x - self.x0) / self.echelle)
        v = int((y - self.y0) / self.echelle)
        return (u, v) if 0 <= u < forme[1] and 0 <= v < forme[0] else None

    def vers_ecran(self, u: float, v: float) -> tuple[int, int]:
        """Pixel de l'image → point de l'écran."""
        return int(self.x0 + u * self.echelle), int(self.y0 + v * self.echelle)


def disposition(forme: tuple) -> Disposition:
    """Réduit l'image si elle est trop grande, puis la centre verticalement."""
    hauteur, largeur = forme[:2]
    echelle = min(1.0, LARGEUR_IMAGE_MAX / largeur, HAUTEUR_ECRAN / hauteur)
    l_img, h_img = int(largeur * echelle), int(hauteur * echelle)
    return Disposition(echelle, 0, (HAUTEUR_ECRAN - h_img) // 2, l_img, HAUTEUR_ECRAN)


# ---------------------------------------------------------------------------
# Texte
# ---------------------------------------------------------------------------

def cm(metres: float) -> str:
    """0.558 → « 55,8 » (centimètres, une décimale, virgule)."""
    return f"{100 * metres:.1f}".replace(".", ",")


def texte(image: np.ndarray, chaine: str, x: int, y: int, echelle: float = 0.6,
          couleur: tuple = BLANC, epaisseur: int = 1, fond: bool = False) -> int:
    """Écrit une ligne (y = bas du texte) et renvoie sa hauteur. fond : petit cadre sombre (sur l'image)."""
    if fond:
        (l, h), bas = cv2.getTextSize(chaine, POLICE, echelle, epaisseur)
        cv2.rectangle(image, (x - 5, y - h - 6), (x + l + 5, y + bas + 2), FOND_IMAGE, -1)
    cv2.putText(image, chaine, (x, y), POLICE, echelle, couleur, epaisseur, cv2.LINE_AA)
    return cv2.getTextSize(chaine, POLICE, echelle, epaisseur)[0][1]


def couper(chaine: str, largeur_max: int, echelle: float, epaisseur: int) -> list[str]:
    """Coupe une phrase en lignes qui tiennent dans largeur_max pixels."""
    lignes, ligne = [], ""
    for mot in chaine.split():
        essai = f"{ligne} {mot}".strip()
        if ligne and cv2.getTextSize(essai, POLICE, echelle, epaisseur)[0][0] > largeur_max:
            lignes.append(ligne)
            ligne = mot
        else:
            ligne = essai
    return lignes + [ligne] if ligne else lignes


def paragraphe(image: np.ndarray, chaine: str, x: int, y: int, largeur_max: int,
               echelle: float = 0.55, couleur: tuple = BLANC, epaisseur: int = 1) -> int:
    """Écrit un texte sur plusieurs lignes à partir de y (haut) ; renvoie le y du dessous."""
    interligne = int(34 * echelle) + 6
    for ligne in couper(chaine, largeur_max, echelle, epaisseur):
        y += interligne
        texte(image, ligne, x, y, echelle, couleur, epaisseur)
    return y


# ---------------------------------------------------------------------------
# Mesure et phrases du résultat
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Mesure:
    diagnostic: DiagnosticMesure
    x: float | None
    y: float | None
    z: float | None
    valide: bool


def mesurer(trame: Trame, u: int, v: int, rayon: int) -> Mesure | None:
    """Position et diagnostic au pixel cliqué ; None sans profondeur (webcam)."""
    if trame.nuage is None:
        return None
    reglages = {"rayon": rayon, "part_min": PART_MIN, "z_min": Z_MIN, "z_max": Z_MAX}
    position = position_depuis_pixel(trame.nuage, u, v, trame.heure, **reglages)
    diagnostic = diagnostic_mesure(trame.nuage, u, v, **reglages)
    return Mesure(diagnostic, position.x, position.y, position.z, position.valide)


def phrases_resultat(etat: Etat, mesure: Mesure | None, avec_profondeur: bool) -> tuple[tuple, str, list[str]]:
    """Couleur de fond, titre et lignes du cadre « résultat »."""
    if etat.clic is None:
        return FOND_NEUTRE, "Cliquez sur l'image pour mesurer un point", []
    if not avec_profondeur:
        return FOND_ROUGE, "Pas de profondeur avec une webcam", ["Utiliser la simulation ou la ZED."]
    if mesure.valide:
        return FOND_VERT, "Bouche trouvee : le bras peut viser", [
            f"{cm(mesure.z)} cm devant la camera",
            f"{cm(abs(mesure.x))} cm {'a droite' if mesure.x >= 0 else 'a gauche'}",
            f"{cm(abs(mesure.y))} cm {'plus bas' if mesure.y >= 0 else 'plus haut'}",
        ]
    d = mesure.diagnostic
    if d.raison == "trop_de_trous":
        raison = (f"Trop de trous de mesure : {100 * d.part_valides:.0f} % de points valides "
                  f"(minimum {100 * PART_MIN:.0f} %)")
    else:
        raison = (f"Distance hors portee : {cm(d.z_median)} cm "
                  f"(accepte : {100 * Z_MIN:.0f} a {100 * Z_MAX:.0f} cm)")
    return FOND_ROUGE, "INVALIDE : le bras ne bouge pas", [raison]


# ---------------------------------------------------------------------------
# Panneau de droite
# ---------------------------------------------------------------------------

def dessiner_resultat(panneau: np.ndarray, y: int, fond: tuple, titre: str, lignes: list[str]) -> int:
    """Cadre coloré avec le titre et les lignes ; renvoie le y du dessous."""
    largeur = LARGEUR_PANNEAU - 2 * MARGE
    titre_coupe = couper(titre, largeur - 24, 0.75, 2)
    lignes_coupees = [l for ligne in lignes for l in couper(ligne, largeur - 24, 0.7, 2)]
    hauteur = 20 + 34 * len(titre_coupe) + (12 if lignes else 0) + 32 * len(lignes_coupees)
    cv2.rectangle(panneau, (MARGE, y), (MARGE + largeur, y + hauteur), fond, -1)
    yy = y + 6
    for ligne in titre_coupe:
        yy += 34
        texte(panneau, ligne, MARGE + 12, yy, 0.75, BLANC, 2)
    yy += 12 if lignes else 0
    for ligne in lignes_coupees:
        yy += 32
        texte(panneau, ligne, MARGE + 12, yy, 0.7, BLANC, 2)
    return y + hauteur


def dessiner_zoom(panneau: np.ndarray, y: int, mesure: Mesure | None) -> int:
    """Zoom de la zone de mesure : blanc = point mesuré, noir = trou."""
    texte(panneau, "Zone de mesure (zoom)", MARGE, y + 20, 0.6, BLANC, 1)
    y += 32
    x = MARGE
    if mesure is None:
        cv2.rectangle(panneau, (x, y), (x + TAILLE_ZOOM, y + TAILLE_ZOOM), FOND_NEUTRE, -1)
        texte(panneau, "aucune mesure", x + 18, y + TAILLE_ZOOM // 2 + 6, 0.55, BLANC)
        cote_h = TAILLE_ZOOM
        legende = "Blanc = point mesure, noir = trou"
    else:
        valides = mesure.diagnostic.fenetre_valides
        pas = max(1, TAILLE_ZOOM // max(valides.shape))
        zoom = cv2.resize((255 * valides).astype(np.uint8), (valides.shape[1] * pas, valides.shape[0] * pas),
                          interpolation=cv2.INTER_NEAREST)
        cote_h, cote_l = zoom.shape
        panneau[y:y + cote_h, x:x + cote_l] = cv2.cvtColor(zoom, cv2.COLOR_GRAY2BGR)
        cv2.rectangle(panneau, (x - 2, y - 2), (x + cote_l + 1, y + cote_h + 1), JAUNE, 2)
        legende = f"{100 * mesure.diagnostic.part_valides:.0f} % de points mesures (noir = trou)"
    return paragraphe(panneau, legende, MARGE, y + cote_h + 2, LARGEUR_PANNEAU - 2 * MARGE, 0.55)


def dessiner_legende(panneau: np.ndarray, y: int, avec_palette: bool) -> int:
    """Barre de couleurs (même palette que l'image) et explication des marques."""
    texte(panneau, "Legende", MARGE, y + 20, 0.6, BLANC, 1)
    y += 34
    x_texte = MARGE
    y_fin = y
    if avec_palette:
        # Barre verticale : Z de 20 cm (haut) à 120 cm (bas), coloriée comme l'image.
        hauteur_barre, largeur_barre = 170, 26
        z = np.linspace(Z_COULEUR_PROCHE, Z_COULEUR_LOIN, hauteur_barre)[:, None].repeat(largeur_barre, axis=1)
        panneau[y:y + hauteur_barre, MARGE:MARGE + largeur_barre] = coloriser_profondeur(z)
        for distance_cm in range(20, 121, 20):
            yy = y + int((distance_cm / 100 - Z_COULEUR_PROCHE) / (Z_COULEUR_LOIN - Z_COULEUR_PROCHE)
                         * (hauteur_barre - 1))
            cv2.line(panneau, (MARGE + largeur_barre, yy), (MARGE + largeur_barre + 6, yy), BLANC, 1)
            texte(panneau, f"{distance_cm} cm", MARGE + largeur_barre + 10, yy + 6, 0.5, BLANC)
        x_texte = MARGE + largeur_barre + 90
        y_fin = y + hauteur_barre

    largeur_texte = LARGEUR_PANNEAU - MARGE - x_texte - 26
    yy = y - 6
    lignes = [("noir", "Noir : la camera n'a rien pu mesurer"),
              ("carre", "Carre jaune : zone de mesure autour du clic"),
              ("points", "Points isoles : bruit de mesure, ignore grace a la mediane")]
    if not avec_palette:
        lignes = lignes[1:2]  # image en couleurs réelles : seul le carré jaune s'applique
    for marque, chaine in lignes:
        x_m, y_m = x_texte, yy + 14
        if marque == "noir":
            cv2.rectangle(panneau, (x_m, y_m), (x_m + 16, y_m + 16), NOIR, -1)
            cv2.rectangle(panneau, (x_m, y_m), (x_m + 16, y_m + 16), GRIS, 1)
        elif marque == "carre":
            cv2.rectangle(panneau, (x_m, y_m), (x_m + 16, y_m + 16), JAUNE, 2)
        else:
            for dx, dy in ((3, 4), (12, 2), (7, 13)):
                cv2.circle(panneau, (x_m + dx, y_m + dy), 1, (0, 200, 255), -1)
        yy = paragraphe(panneau, chaine, x_texte + 26, yy, largeur_texte, 0.5) + 10
    return max(y_fin, yy)


def dessiner_repere(panneau: np.ndarray, y: int) -> int:
    """Petit dessin des axes X, Y, Z de la caméra."""
    texte(panneau, "Repere de la camera", MARGE, y + 20, 0.6, BLANC, 1)
    o = (MARGE + 30, y + 62)
    for bout, nom, decalage in (((o[0] + 70, o[1]), "X", (8, 6)),
                                ((o[0], o[1] + 48), "Y", (-24, 6)),
                                ((o[0] + 40, o[1] - 20), "Z", (6, 0))):
        cv2.arrowedLine(panneau, o, bout, BLANC, 2, cv2.LINE_AA, tipLength=0.22)
        texte(panneau, nom, bout[0] + decalage[0], bout[1] + decalage[1], 0.55, BLANC, 2)
    x_t = MARGE + 140
    for i, chaine in enumerate(("X : vers la droite", "Y : vers le bas", "Z : devant la camera")):
        texte(panneau, chaine, x_t, y + 50 + 26 * i, 0.55, BLANC)
    return y + 120


def dessiner_panneau(panneau: np.ndarray, etat: Etat, mesure: Mesure | None,
                     nom_source: str, avec_profondeur: bool) -> None:
    """Remplit le panneau de droite, de haut en bas."""
    panneau[:] = FOND_PANNEAU
    largeur = LARGEUR_PANNEAU - 2 * MARGE
    texte(panneau, "Position 3D de la bouche", MARGE, 38, 0.8, BLANC, 2)
    cote = 2 * etat.rayon + 1
    texte(panneau, f"source : {nom_source}   zone : {cote} x {cote} pixels", MARGE, 64, 0.5, GRIS)

    y = dessiner_resultat(panneau, 80, *phrases_resultat(etat, mesure, avec_profondeur)) + 16
    y = dessiner_zoom(panneau, y, mesure) + 14
    if etat.legende:
        y = dessiner_legende(panneau, y, avec_palette=nom_source == "simulation") + 20
    else:
        y = paragraphe(panneau, "Legende cachee (touche l)", MARGE, y, largeur, 0.5, GRIS) + 14
    y = dessiner_repere(panneau, y)

    commandes = "clic : mesurer | + / - : taille de la zone | l : legende | s : enregistrer | c : capture | q : quitter"
    if nom_source in SOURCES_PAS_A_PAS:
        commandes += " | espace : image suivante"
    y_cmd = HAUTEUR_ECRAN - 30 - 22 * len(couper(commandes, largeur, 0.5, 1))
    cv2.line(panneau, (MARGE, y_cmd), (LARGEUR_PANNEAU - MARGE, y_cmd), GRIS, 1)
    paragraphe(panneau, commandes, MARGE, y_cmd, largeur, 0.5, BLANC)


# ---------------------------------------------------------------------------
# Image de gauche
# ---------------------------------------------------------------------------

def etiquettes_simulation(source: SourceSimulation) -> list[tuple[str, tuple, tuple]]:
    """Étiquettes (texte, position du texte, point visé), en pixels de l'image simulée."""
    cx, cy = source.cx, source.cy
    a = 0.075 * FOCALE / source.distance  # demi-largeur du visage
    b = 0.10 * FOCALE / source.distance   # demi-hauteur du visage
    u_b, v_b, _ = source.position_bouche_reelle
    a_b = 0.025 * FOCALE / source.distance
    return [
        (f"visage (~{100 * source.distance:.0f} cm)", (cx + a + 60, cy - 0.7 * b), (cx + 0.5 * a, cy - 0.35 * b)),
        ("bouche ouverte : trous de mesure", (cx + a + 60, v_b + 70), (u_b + 0.6 * a_b, v_b + 4)),
        (f"mur du fond ({100 * Z_FOND:.0f} cm)", (cx - a - 380, cy - 0.7 * b), (cx - a - 120, cy)),
    ]


def dessiner_sur_image(image: np.ndarray, trame: Trame, etat: Etat, d: Disposition,
                       etiquettes: list) -> None:
    """Étiquettes, carré de mesure et distance au survol, sur l'image réduite (coordonnées écran)."""
    if etat.legende:
        for chaine, (xt, yt), cible in etiquettes:
            x, y = d.vers_ecran(xt, yt)
            largeur, hauteur = cv2.getTextSize(chaine, POLICE, 0.6, 1)[0]
            depart = (x + largeur // 2, y + 8 if cible[1] * d.echelle + d.y0 > y else y - hauteur - 6)
            cv2.arrowedLine(image, depart, d.vers_ecran(*cible), BLANC, 1, cv2.LINE_AA, tipLength=0.08)
            texte(image, chaine, x, y, 0.6, BLANC, 1, fond=True)

    if etat.clic is not None:
        u, v = etat.clic
        r = etat.rayon
        coin1, coin2 = d.vers_ecran(u - r, v - r), d.vers_ecran(u + r + 1, v + r + 1)
        cv2.rectangle(image, coin1, coin2, NOIR, 4)  # contour noir : visible même sur du jaune
        cv2.rectangle(image, coin1, coin2, JAUNE, 2)
        centre = d.vers_ecran(u + 0.5, v + 0.5)
        cv2.circle(image, centre, 3, NOIR, -1)
        cv2.circle(image, centre, 2, JAUNE, -1)

    if etat.survol is not None:
        u, v = etat.survol
        if trame.nuage is None:
            chaine = "pas de profondeur"
        else:
            point = trame.nuage[v, u]
            chaine = f"{cm(point[2])} cm" if np.isfinite(point).all() else "pas de mesure"
        x, y = d.vers_ecran(u, v)
        texte(image, chaine, min(x + 14, d.largeur - 170), max(y - 12, 20), 0.55, BLANC, 1, fond=True)


def composer_ecran(trame: Trame, etat: Etat, nom_source: str, etiquettes: list) -> np.ndarray:
    """Image complète affichée : image de la caméra à gauche, panneau à droite."""
    d = disposition(trame.image.shape)
    largeur, hauteur = d.taille_ecran
    ecran = np.full((hauteur, largeur, 3), FOND_IMAGE, np.uint8)

    image = cv2.resize(trame.image, (d.largeur, int(trame.image.shape[0] * d.echelle)),
                       interpolation=cv2.INTER_AREA) if d.echelle < 1 else trame.image.copy()
    ecran[d.y0:d.y0 + image.shape[0], d.x0:d.x0 + image.shape[1]] = image
    dessiner_sur_image(ecran[:, :d.largeur], trame, etat, d, etiquettes)
    if etat.message:  # message de la dernière action (enregistrement, capture)
        texte(ecran, etat.message, 14, 36, 0.65, JAUNE, 1, fond=True)

    mesure = mesurer(trame, *etat.clic, etat.rayon) if etat.clic is not None else None
    dessiner_panneau(ecran[:, d.largeur:], etat, mesure, nom_source, trame.nuage is not None)
    return ecran


# ---------------------------------------------------------------------------
# Boucle principale
# ---------------------------------------------------------------------------

def lire_arguments() -> argparse.Namespace:
    parseur = argparse.ArgumentParser(description="Clic = position 3D du pixel.")
    parseur.add_argument("--source", choices=NOMS_SOURCES, default="simulation")
    parseur.add_argument("--fichier", help="fichier .svo (source svo)")
    parseur.add_argument("--dossier", default=str(DOSSIER_DATA), help="dossier des trames .npz (source npy)")
    parseur.add_argument("--rayon", type=int, default=5, help="demi-côté de la zone de mesure (pixels)")
    arguments = parseur.parse_args()
    if arguments.source == "svo" and arguments.fichier is None:
        parseur.error("--source svo demande --fichier CHEMIN.svo")
    return arguments


def options_source(arguments: argparse.Namespace) -> dict:
    """Options à donner à creer_source selon la source choisie."""
    if arguments.source == "npy":
        return {"dossier": arguments.dossier}
    if arguments.source == "svo":
        return {"fichier_svo": arguments.fichier}
    return {}


def prochain_numero(dossier: Path, motif: str) -> int:
    """Premier numéro libre pour un fichier nom_XXXX, pour ne rien écraser."""
    numeros = [int(f.stem.split("_")[1]) for f in dossier.glob(motif)]
    return max(numeros, default=-1) + 1


def taille_ecran_windows() -> tuple[int, int]:
    """Taille de l'écran (Windows) ; 1920 × 1080 si elle est inconnue."""
    try:
        import ctypes
        user32 = ctypes.windll.user32
        return user32.GetSystemMetrics(0), user32.GetSystemMetrics(1)
    except (AttributeError, OSError):
        return 1920, 1080


def ouvrir_fenetre(largeur: int, hauteur: int) -> None:
    """Fenêtre redimensionnable, réduite si besoin pour tenir à l'écran."""
    cv2.namedWindow(FENETRE, cv2.WINDOW_NORMAL)
    l_ecran, h_ecran = taille_ecran_windows()
    ratio = min(1.0, 0.95 * l_ecran / largeur, 0.88 * h_ecran / hauteur)
    cv2.resizeWindow(FENETRE, int(largeur * ratio), int(hauteur * ratio))


def boucle(source, nom_source: str, rayon: int) -> None:
    """Affiche les trames et réagit aux clics et aux touches, jusqu'à « q »."""
    etat = Etat(rayon=rayon)
    trame = source.lire()
    if trame is None:
        return
    forme = trame.image.shape
    d = disposition(forme)
    etiquettes = etiquettes_simulation(source) if isinstance(source, SourceSimulation) else []

    def sur_souris(evenement, x, y, *_):
        pixel = d.vers_image(x, y, forme)
        etat.survol = pixel
        if evenement == cv2.EVENT_LBUTTONDOWN and pixel is not None:
            etat.clic = pixel

    ouvrir_fenetre(*d.taille_ecran)
    cv2.setMouseCallback(FENETRE, sur_souris)
    pas_a_pas = nom_source in SOURCES_PAS_A_PAS

    while trame is not None:
        ecran = composer_ecran(trame, etat, nom_source, etiquettes)
        cv2.imshow(FENETRE, ecran)

        touche = cv2.waitKey(30 if pas_a_pas else 1) & 0xFF
        if touche == ord("q") or cv2.getWindowProperty(FENETRE, cv2.WND_PROP_VISIBLE) < 1:
            break
        if touche in (ord("+"), ord("=")):
            etat.rayon += 1
        elif touche == ord("-"):
            etat.rayon = max(etat.rayon - 1, 0)
        elif touche == ord("l"):
            etat.legende = not etat.legende
        elif touche == ord("s"):
            chemin = enregistrer_trame(trame, DOSSIER_DATA, prochain_numero(DOSSIER_DATA, "trame_*.npz"))
            etat.message = f"Trame enregistree : {chemin.name}"
            print(f"Trame enregistrée : {chemin}")
        elif touche == ord("c"):
            dossier = DOSSIER_DATA / "captures"
            dossier.mkdir(parents=True, exist_ok=True)
            chemin = dossier / f"capture_{prochain_numero(dossier, 'capture_*.png'):04d}.png"
            cv2.imwrite(str(chemin), ecran)
            etat.message = f"Capture enregistree : {chemin.name}"
            print(f"Capture enregistrée : {chemin}")

        # Sources en direct : image suivante à chaque tour ; npy et svo : sur « espace ».
        if not pas_a_pas or touche == ord(" "):
            suivante = source.lire()
            if suivante is not None:
                trame = suivante
            elif pas_a_pas:
                etat.message = "Derniere image atteinte"  # on garde la dernière image
            else:
                print("Plus d'image : fin de la source.")
                trame = None


def main() -> None:
    arguments = lire_arguments()
    try:
        source = creer_source(arguments.source, **options_source(arguments))
        with source:  # la source est fermée à la sortie, même en cas d'erreur
            boucle(source, arguments.source, arguments.rayon)
    except (RuntimeError, FileNotFoundError, ValueError) as erreur:
        raise SystemExit(f"Erreur : {erreur}") from erreur
    finally:
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
