"""TESTS DE camera.py

Ces tests vérifient, sans matériel :
- qu'une source inconnue est refusée avec un message clair ;
- que la simulation produit un nuage plausible et reproductible ;
- qu'une trame enregistrée se relit à l'identique, puis que la fin est signalée (None) ;
- que la webcam fonctionne sans que pyzed soit installé.
Le test de la ZED est ignoré si pyzed est absent. Les tests qui demandent une vraie
caméra sont lancés à la main et notés dans docs/mesures.md.
"""

import subprocess
import sys

import numpy as np
import pytest

from bouche_3d.camera import (
    SourceNpy,
    SourceSimulation,
    SourceWebcam,
    Trame,
    creer_source,
    enregistrer_trame,
)


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
