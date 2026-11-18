"""TESTS DE camera.py

Ces tests vérifient, sans matériel :
- qu'une source inconnue est refusée avec un message clair ;
- que la simulation produit un nuage plausible et reproductible ;
- qu'une trame enregistrée se relit à l'identique, puis que la fin est signalée (None) ;
- que la webcam fonctionne sans que pyzed soit installé ;
- que la source « bira » et SourceZed se comportent bien, avec une fausse caméra BIRA et un
  faux module pyzed (aucun matériel) : caméra non ouverte, unités, copies, grab, fermeture.
Le test de la vraie ZED est ignoré si pyzed est absent. Les tests qui demandent une vraie
caméra sont lancés à la main et notés dans docs/mesures.md.
"""

import enum
import subprocess
import sys
import types

import numpy as np
import pytest

from bouche_3d import camera as module_camera
from bouche_3d.camera import (
    MAX_ECHECS_GRAB,
    SourceBira,
    SourceNpy,
    SourceSimulation,
    SourceWebcam,
    SourceZed,
    Trame,
    creer_source,
    enregistrer_trame,
)
from bouche_3d.position_3d import position_depuis_pixel


def test_source_inconnue():
    # Un nom inconnu est refusé, et le message donne les noms permis.
    with pytest.raises(ValueError, match="simulation"):
        creer_source("kinect")


def test_svo_sans_fichier():
    # La source « svo » sans fichier est refusée avec un message clair.
    with pytest.raises(ValueError, match="fichier_svo"):
        creer_source("svo")


def test_simulation_forme_du_nuage():
    source = SourceSimulation(largeur=320, hauteur=240)
    trame = source.lire()
    assert trame.nuage.shape == (240, 320, 3)
    assert trame.image.shape == (240, 320, 3) and trame.image.dtype == np.uint8


def test_simulation_nan_dans_la_bouche():
    # Environ 30 % des points de la bouche sont des trous (NaN).
    source = SourceSimulation(part_trous_bouche=0.3)
    trame = source.lire()
    z_bouche = trame.nuage[source.masque_bouche, 2]
    part_nan = np.isnan(z_bouche).mean()
    assert 0.2 < part_nan < 0.4


def test_simulation_z_proche_de_la_distance():
    # Au centre du visage, Z vaut « distance » (au bruit près).
    source = SourceSimulation(distance=0.45)
    trame = source.lire()
    centre = trame.nuage[355:366, 635:646, 2]
    assert abs(np.nanmedian(centre) - 0.45) < 0.002


def test_simulation_meme_graine_memes_donnees():
    a = SourceSimulation(graine=7).lire()
    b = SourceSimulation(graine=7).lire()
    c = SourceSimulation(graine=8).lire()
    np.testing.assert_array_equal(a.nuage, b.nuage)  # NaN égaux entre eux ici
    np.testing.assert_array_equal(a.image, b.image)
    assert not np.array_equal(a.nuage, c.nuage, equal_nan=True)


def test_enregistrer_puis_relire(tmp_path):
    source = SourceSimulation(largeur=320, hauteur=240)
    trames = [source.lire() for _ in range(3)]
    for numero, trame in enumerate(trames):
        enregistrer_trame(trame, tmp_path, numero)
    assert (tmp_path / "trame_0000.npz").exists()

    with SourceNpy(tmp_path) as relue:
        for originale in trames:
            copie = relue.lire()
            np.testing.assert_array_equal(copie.image, originale.image)
            np.testing.assert_array_equal(copie.nuage, originale.nuage)
            assert copie.heure == originale.heure
        assert relue.lire() is None  # fin des trames


def test_enregistrer_trame_sans_nuage(tmp_path):
    # Une trame de webcam (nuage = None) se sauvegarde et se relit aussi.
    trame = Trame(np.zeros((4, 5, 3), np.uint8), None, 12.5)
    enregistrer_trame(trame, tmp_path, 0)
    with SourceNpy(tmp_path) as relue:
        copie = relue.lire()
    assert copie.nuage is None and copie.heure == 12.5


def test_npy_dossier_vide(tmp_path):
    with pytest.raises(FileNotFoundError):
        SourceNpy(tmp_path).ouvrir()


def test_webcam_sans_pyzed():
    # Dans un Python neuf : importer camera et créer la webcam ne charge pas pyzed.
    code = (
        "import sys\n"
        "from bouche_3d.camera import creer_source\n"
        "creer_source('webcam')\n"
        "assert 'pyzed' not in sys.modules\n"
    )
    resultat = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert resultat.returncode == 0, resultat.stderr
    assert isinstance(creer_source("webcam"), SourceWebcam)


def test_source_zed():
    # Ne tourne que si le ZED SDK est installé ; sinon, test ignoré.
    pytest.importorskip("pyzed.sl")
    source = creer_source("zed")
    try:
        source.ouvrir()
    except RuntimeError as erreur:
        pytest.skip(f"ZED non branchée : {erreur}")
    try:
        trame = source.lire()
        assert trame is not None and trame.nuage.shape[2] == 3
    finally:
        source.fermer()


# ---------------------------------------------------------------------------
# Faux pyzed et fausse caméra BIRA (aucun matériel)
# ---------------------------------------------------------------------------

class FauxErrorCode(enum.Enum):
    # Comme dans pyzed : un membre d'enum est toujours « vrai », même un échec.
    SUCCESS = 0
    FAILURE = 1
    CAMERA_NOT_DETECTED = 2
    END_OF_SVOFILE_REACHED = 3


class FausseUnite(enum.Enum):
    MILLIMETER = 0
    METER = 1
    CENTIMETER = 2  # unité que le module ne prend pas en charge


class FausseMat:
    """sl.Mat : get_data() renvoie le tableau sans copie, comme la vraie."""

    def __init__(self):
        self.donnees = None
        self.liberee = False

    def get_data(self):
        return self.donnees

    def free(self):
        self.liberee = True


class FauxTimestamp:
    def __init__(self, nanosecondes):
        self.nanosecondes = nanosecondes

    def get_nanoseconds(self):
        return self.nanosecondes


class FauxZed:
    """sl.Camera : renvoie toujours la même image (BGRA) et le même nuage (XYZRGBA)."""

    def __init__(self, nuage_xyz, unite=FausseUnite.MILLIMETER, ouverte=True, codes_grab=()):
        hauteur, largeur = nuage_xyz.shape[:2]
        self.image_bgra = np.full((hauteur, largeur, 4), 100, np.uint8)
        self.image_bgra[..., 3] = 255
        self.nuage_xyzrgba = np.concatenate(
            [nuage_xyz.astype(np.float32), np.zeros((hauteur, largeur, 1), np.float32)], axis=2)
        self.unite = unite
        self.ouverte = ouverte
        self.codes_grab = list(codes_grab)  # SourceZed : codes renvoyés par grab, dans l'ordre
        self.nb_grab = 0
        self.nb_close = 0
        self.nanosecondes = 1_700_000_000_123_456_789

    def is_opened(self):
        return self.ouverte

    def open(self, parametres):
        self.ouverte = True
        return FauxErrorCode.SUCCESS

    def get_init_parameters(self):
        return types.SimpleNamespace(coordinate_units=self.unite)

    def grab(self, runtime=None):
        self.nb_grab += 1
        return self.codes_grab.pop(0) if self.codes_grab else FauxErrorCode.SUCCESS

    def retrieve_image(self, mat, vue):
        mat.donnees = self.image_bgra
        return FauxErrorCode.SUCCESS

    def retrieve_measure(self, mat, mesure):
        mat.donnees = self.nuage_xyzrgba
        return FauxErrorCode.SUCCESS

    def get_timestamp(self, reference):
        return FauxTimestamp(self.nanosecondes)

    def close(self):
        self.nb_close += 1
        self.ouverte = False


class FausseCameraBira:
    """Objet Camera de BIRA : seules get_camera() et grab() doivent être utilisées."""

    def __init__(self, zed, code_grab=FauxErrorCode.SUCCESS):
        self.zed = zed
        self.code_grab = code_grab
        self.nb_grab = 0
        self.nb_close = 0
        self.nb_open = 0

    def get_camera(self):
        return self.zed

    def grab(self):
        self.nb_grab += 1
        return self.code_grab

    def open(self, init_params=None):
        self.nb_open += 1

    def close(self):
        self.nb_close += 1


@pytest.fixture
def faux_sl(monkeypatch):
    """Installe un faux module pyzed.sl dans sys.modules, le temps d'un test."""
    sl = types.ModuleType("pyzed.sl")
    sl.ERROR_CODE = FauxErrorCode
    sl.UNIT = FausseUnite
    sl.VIEW = types.SimpleNamespace(LEFT="LEFT")
    sl.MEASURE = types.SimpleNamespace(XYZRGBA="XYZRGBA")
    sl.TIME_REFERENCE = types.SimpleNamespace(IMAGE="IMAGE")
    sl.COORDINATE_SYSTEM = types.SimpleNamespace(IMAGE="IMAGE")
    sl.DEPTH_MODE = types.SimpleNamespace(NEURAL="NEURAL", ULTRA="ULTRA")
    sl.Mat = FausseMat
    sl.InitParameters = types.SimpleNamespace
    sl.RuntimeParameters = types.SimpleNamespace
    sl.Camera = None  # remplacé par les tests de SourceZed
    paquet = types.ModuleType("pyzed")
    paquet.sl = sl
    monkeypatch.setitem(sys.modules, "pyzed", paquet)
    monkeypatch.setitem(sys.modules, "pyzed.sl", sl)
    return sl


def nuage_plat(z, hauteur=40, largeur=60):
    """Nuage où tous les points sont à la distance z (dans l'unité de z)."""
    nuage = np.zeros((hauteur, largeur, 3), np.float32)
    nuage[..., 2] = z
    return nuage


# ---------------------------------------------------------------------------
# Source « bira »
# ---------------------------------------------------------------------------

def test_bira_creer_source(faux_sl):
    camera = FausseCameraBira(FauxZed(nuage_plat(550.0)))
    assert isinstance(creer_source("bira", camera=camera), SourceBira)
    with pytest.raises(ValueError, match="camera"):
        creer_source("bira")


def test_bira_unite_entree_inconnue_refusee():
    with pytest.raises(ValueError, match="unite_entree"):
        SourceBira(FausseCameraBira(None), unite_entree="pouces")


def test_bira_camera_non_ouverte(faux_sl):
    camera = FausseCameraBira(FauxZed(nuage_plat(550.0), ouverte=False))
    source = SourceBira(camera)
    with pytest.raises(RuntimeError, match="ouverte par BIRA"):
        source.ouvrir()
    assert camera.nb_open == 0 and camera.zed.ouverte is False  # jamais ouverte par nous


def test_bira_millimetres_convertis_en_metres(faux_sl):
    camera = FausseCameraBira(FauxZed(nuage_plat(550.0), unite=FausseUnite.MILLIMETER))
    with SourceBira(camera) as source:
        trame = source.lire()
    assert trame.nuage.shape == (40, 60, 3)
    assert trame.image.shape == (40, 60, 3)  # BGRA -> BGR
    np.testing.assert_allclose(trame.nuage[..., 2], 0.55, atol=1e-6)
    position = position_depuis_pixel(trame.nuage, 30, 20, trame.heure)
    assert position.valide and position.z == pytest.approx(0.55, abs=1e-6)


def test_bira_millimetres_donnes_a_la_main(faux_sl):
    # unite_entree="mm" s'applique sans regarder l'unité annoncée par la caméra.
    camera = FausseCameraBira(FauxZed(nuage_plat(550.0), unite=FausseUnite.CENTIMETER))
    with SourceBira(camera, unite_entree="mm") as source:
        trame = source.lire()
    np.testing.assert_allclose(trame.nuage[..., 2], 0.55, atol=1e-6)


def test_bira_deja_en_metres_pas_converti(faux_sl):
    camera = FausseCameraBira(FauxZed(nuage_plat(0.55), unite=FausseUnite.METER))
    with SourceBira(camera) as source:
        trame = source.lire()
    np.testing.assert_allclose(trame.nuage[..., 2], 0.55, atol=1e-6)


def test_bira_unite_inconnue_erreur(faux_sl):
    camera = FausseCameraBira(FauxZed(nuage_plat(55.0), unite=FausseUnite.CENTIMETER))
    with SourceBira(camera) as source:
        with pytest.raises(ValueError, match="Unité"):
            source.lire()


def test_bira_fermer_ne_ferme_jamais_la_camera(faux_sl):
    zed = FauxZed(nuage_plat(550.0))
    camera = FausseCameraBira(zed)
    source = SourceBira(camera)
    with source:
        source.lire()
        mats = (source.mat_image, source.mat_nuage)
    source.fermer()  # deuxième fermeture : sans effet
    assert camera.nb_close == 0 and zed.nb_close == 0 and zed.ouverte
    assert all(mat.liberee for mat in mats)  # nos sl.Mat sont libérées
    assert source.lire() is None  # source fermée


def test_bira_sans_grab_par_defaut(faux_sl):
    zed = FauxZed(nuage_plat(550.0))
    camera = FausseCameraBira(zed)
    with SourceBira(camera) as source:
        for _ in range(3):
            assert source.lire() is not None
    assert camera.nb_grab == 0 and zed.nb_grab == 0


def test_bira_grab_rate_donne_none(faux_sl):
    camera = FausseCameraBira(FauxZed(nuage_plat(550.0)), code_grab=FauxErrorCode.FAILURE)
    with SourceBira(camera, faire_grab=True) as source:
        assert source.lire() is None  # FAILURE est « vrai » : il faut bien comparer à SUCCESS
    assert camera.nb_grab == 1


def test_bira_grab_reussi(faux_sl):
    camera = FausseCameraBira(FauxZed(nuage_plat(550.0)))
    with SourceBira(camera, faire_grab=True) as source:
        assert source.lire() is not None
    assert camera.nb_grab == 1


def test_bira_trames_sont_des_copies(faux_sl):
    zed = FauxZed(nuage_plat(550.0))
    with SourceBira(FausseCameraBira(zed)) as source:
        trame = source.lire()
    # La caméra réécrit ses tableaux à l'image suivante : la trame ne doit pas changer.
    zed.nuage_xyzrgba[...] = 999.0
    zed.image_bgra[...] = 0
    np.testing.assert_allclose(trame.nuage[..., 2], 0.55, atol=1e-6)
    assert (trame.image == 100).all()


def test_bira_heure_de_l_image(faux_sl):
    with SourceBira(FausseCameraBira(FauxZed(nuage_plat(550.0)))) as source:
        trame = source.lire()
    assert trame.heure == pytest.approx(1_700_000_000.123456789)


# ---------------------------------------------------------------------------
# SourceZed avec le faux pyzed
# ---------------------------------------------------------------------------

def ouvrir_fausse_zed(faux_sl, codes_grab):
    zed = FauxZed(nuage_plat(0.55), unite=FausseUnite.METER, ouverte=False, codes_grab=codes_grab)
    faux_sl.Camera = lambda: zed
    source = SourceZed()
    source.ouvrir()
    return source, zed


def test_zed_erreur_passagere_puis_succes(faux_sl):
    source, zed = ouvrir_fausse_zed(faux_sl, [FauxErrorCode.FAILURE, FauxErrorCode.SUCCESS])
    trame = source.lire()
    assert trame is not None and zed.nb_grab == 2
    np.testing.assert_allclose(trame.nuage[..., 2], 0.55, atol=1e-6)
    assert trame.heure == pytest.approx(1_700_000_000.123456789)  # horodatage de l'image
    source.fermer()


def test_zed_trop_d_echecs_consecutifs(faux_sl):
    source, zed = ouvrir_fausse_zed(faux_sl, [FauxErrorCode.FAILURE] * MAX_ECHECS_GRAB)
    assert source.lire() is None
    assert zed.nb_grab == MAX_ECHECS_GRAB
    source.fermer()


def test_zed_juste_sous_la_limite(faux_sl):
    codes = [FauxErrorCode.FAILURE] * (MAX_ECHECS_GRAB - 1) + [FauxErrorCode.SUCCESS]
    source, _ = ouvrir_fausse_zed(faux_sl, codes)
    assert source.lire() is not None
    source.fermer()


def test_zed_fin_du_fichier_svo(faux_sl):
    source, zed = ouvrir_fausse_zed(faux_sl, [FauxErrorCode.END_OF_SVOFILE_REACHED])
    assert source.lire() is None and zed.nb_grab == 1  # pas de nouvel essai
    source.fermer()


def test_max_echecs_grab_vaut_10():
    assert module_camera.MAX_ECHECS_GRAB == 10
