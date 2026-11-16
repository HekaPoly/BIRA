# bouche_3d : position 3D de la bouche (projet Bira)

> **À propos de ce fichier**
> - **Rôle :** point d'entrée du dossier. Il explique le but du projet, son organisation et comment le reprendre.
> - **Pour qui :** toute personne qui découvre ce dossier, même sans connaître son historique.
> - **À mettre à jour :** chaque fois que l'organisation du dossier ou l'état d'avancement change.

## 1. Le projet en bref

Bira est un bras robotique développé par la société technique Heka ST (Polytechnique Montréal). Il aide des personnes à mobilité réduite à s'alimenter.

Ce dossier concerne un module précis : **trouver où se situe la bouche de la personne dans l'espace**. Le résultat est une position en trois dimensions (X, Y, Z, en mètres) que le bras utilise pour savoir où amener la cuillère.

## 2. État d'avancement

- Phase de cadrage terminée : objectifs, plan et questions à clarifier sont rédigés.
- Structure du dossier en place.
- **Tâche « position 3D » écrite et testée sans matériel :** calcul de la position 3D (`position_3d.py`), sources caméra (`camera.py` : webcam, simulation, trames enregistrées, ZED, `.svo`) et démonstration (`scripts/demo_position_3d.py`). **La ZED n'a pas encore été essayée.** Les autres fichiers de code ne contiennent encore que leur en-tête.
- Prochaine étape : la tâche « détection de la bouche » (détecter la bouche sur une image). Voir `docs/mandat.md`.

## 3. Principe de fonctionnement prévu

1. La caméra (ZED Mini) filme le visage et mesure la distance de chaque point de l'image.
2. Un modèle de vision repère la bouche dans l'image (sa position en pixels).
3. On lit la distance à cet endroit : on obtient la position 3D de la bouche, en mètres, dans le repère de la caméra.
4. On convertit cette position dans le repère du bras, puis on la lui transmet avec un état de fiabilité (recherche, suivi, prêt ou perdu).

Il s'agit d'un principe de travail : rien de tout cela n'est encore programmé.

Le travail est découpé en cinq tâches indépendantes. Chacune a son fichier de code dans `src/bouche_3d/`, dont l'en-tête décrit son rôle, ce qu'elle reçoit, ce qu'elle produit et de quoi elle dépend.

| Tâche | Rôle | Fichier |
|---|---|---|
| Détection de la bouche | Détecter la bouche sur l'image. | `src/bouche_3d/detection_2d.py` |
| Position 3D | Obtenir la position 3D de la bouche. | `src/bouche_3d/position_3d.py` |
| Stabilité et états | Stabiliser la position et gérer les états de fiabilité. | `src/bouche_3d/stabilite_etats.py` |
| Envoi au bras | Passer dans le repère du bras et lui transmettre la position. | `src/bouche_3d/repere_bras.py` |
| Assemblage et tests | Assembler, mesurer et faire tourner sur le Jetson. | `src/bouche_3d/assemblage.py` |

`camera.py` n'est pas une tâche : il fournit les images et la profondeur à la détection et à la position 3D.

Ce que les tâches s'échangent est décrit dans `src/bouche_3d/interfaces.py`.

### Qui fait quoi

Proposition de répartition, à valider avec la binôme (voir D9).

| Personne | Partie | Tâches |
|---|---|---|
| Membre 1 | Partie caméra | Détection de la bouche, position 3D |
| Membre 2 | Partie robot | Stabilité et états, envoi au bras, échanges avec l'équipe |
| Ensemble | | Assemblage et tests |

## 4. Contenu du dossier

| Élément | Rôle |
|---|---|
| `README.md` | Ce fichier : présentation et mode d'emploi du dossier. |
| `docs/mandat.md` | Fiche du mandat : objectif, plan, questions à clarifier, risques. |
| `docs/journal.md` | Journal de travail, session par session. |
| `docs/mesures.md` | Méthode de mesure et tableau des résultats. |
| `docs/decisions.md` | Registre des choix techniques et de leurs raisons. |
| `src/bouche_3d/` | Code du module : un fichier par tâche, un fichier d'interfaces et `camera.py`, qui lit les images et la profondeur de la caméra (webcam, ZED Mini ou fichier `.svo`). Chaque fichier ne contient pour l'instant que son en-tête. |
| `scripts/` | Programmes de démonstration et de lancement (en-têtes seulement pour l'instant). |
| `tests/` | Un fichier de tests par tâche (en-têtes seulement pour l'instant). |
| `tests/positions_simulees.py` | Suites de positions 3D inventées pour tester la tâche « stabilité et états » sans caméra (en-tête seulement pour l'instant). |
| `data/` | Enregistrements de la caméra, vidéos et captures. Jamais versionnés. |
| `requirements.txt` | Liste provisoire des bibliothèques Python prévues. |
| `pyproject.toml` | Description du paquet Python `bouche_3d` : nom, version, version minimale de Python et emplacement du code (`src/`). |
| `.gitignore` | Liste des fichiers que Git doit ignorer. |

Le dossier `data/` contient un fichier `.gitkeep`. C'est un simple marqueur qui permet à Git de conserver un dossier vide.

## 5. Si vous reprenez ce dossier

1. Lire `docs/mandat.md` pour comprendre l'objectif et le plan.
2. Lire la dernière entrée de `docs/journal.md` pour savoir où le travail s'est arrêté.
3. Consulter `docs/decisions.md` avant de remettre un choix en question.
4. Choisir une tâche dans le tableau de `docs/mandat.md`, puis ouvrir son fichier de code : son en-tête décrit ce qu'elle doit faire.

## 6. Matériel et prérequis

- **Caméra :** ZED Mini (Stereolabs).
- **Calculateur du robot :** Jetson (modèle à confirmer).
- **Langage :** Python 3.10 (à confirmer selon la version de JetPack du Jetson, voir D10).
- **ZED SDK :** logiciel de Stereolabs nécessaire pour lire la profondeur. Il exige une carte graphique NVIDIA. Sur le Jetson, sa version doit correspondre à celle de JetPack (le système fourni par NVIDIA). Son module Python, `pyzed`, s'installe avec le SDK et non avec `pip`.

## 7. Préparer l'environnement (quand le code sera présent)

Créer un environnement virtuel Python, l'activer, puis installer les bibliothèques.

Sous Windows (PowerShell) :

```
py -3.10 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -e .
pip install -r requirements.txt
```

Sous Linux (dont le Jetson) :

```
python3 -m venv .venv
source .venv/bin/activate
pip install -e .
pip install -r requirements.txt
```

Sous Windows, `py -3.10` choisit la bonne version de Python, même si plusieurs sont installées. Une fois le venv activé, `python` désigne celui du venv.

Lancer les tests : `pytest -v` (venv activé).

## 8. Essayer la position 3D

**Sans carte NVIDIA : sources simulation et npy seulement.** Les sources zed et svo demandent le ZED SDK (et donc une carte NVIDIA).

Toutes les commandes se lancent depuis le dossier `bouche_3d`, venv activé.

Tests (aucun matériel nécessaire ; le test de la ZED est ignoré si pyzed est absent) :

```
pytest -v
pytest -v tests/test_position_3d.py tests/test_camera.py
```

Démonstration : cliquer sur l'image donne la position 3D du pixel (X, Y, Z en mm, ou « INVALIDE »).
Touches : `q` quitter, `+` / `-` agrandir ou réduire la zone de mesure, `l` afficher ou cacher la légende et les étiquettes, `s` enregistrer la trame dans `data/`, `c` capture de toute la fenêtre dans `data/captures/`, espace image suivante (npy et svo).

**Comment lire l'écran.** À gauche, l'image de la caméra : en simulation, les couleurs donnent la distance (rouge = proche, bleu = loin, noir = rien mesuré) ; le carré jaune est la zone mesurée autour du clic, et la distance du pixel survolé s'affiche près du curseur.
À droite, le cadre du haut donne la conclusion : vert = « le bras peut viser », avec la distance devant la caméra et le décalage à droite ou à gauche, plus haut ou plus bas, en cm ; rouge = « INVALIDE », avec la raison (trop de trous, ou distance hors de 20 à 100 cm).
Le zoom montre la zone mesurée pixel par pixel (blanc = mesuré, noir = trou) ; dessous, la légende des couleurs et le repère X (droite), Y (bas), Z (devant la caméra).
Le texte à l'écran est écrit sans accents, car les polices d'OpenCV ne les affichent pas.

| Source | Commande | Matériel |
|---|---|---|
| Simulation | `python scripts/demo_position_3d.py --source simulation` | Aucun |
| Trames enregistrées | `python scripts/demo_position_3d.py --source npy --dossier data` | Aucun (trames `trame_XXXX.npz` enregistrées avec `s`) |
| Webcam | `python scripts/demo_position_3d.py --source webcam` | Webcam (image seulement, pas de profondeur) |
| ZED en direct | `python scripts/demo_position_3d.py --source zed` | ZED Mini, ZED SDK, carte NVIDIA |
| Fichier .svo | `python scripts/demo_position_3d.py --source svo --fichier data/essai.svo` | ZED SDK, carte NVIDIA |

Option commune : `--rayon 5` (demi-côté de la fenêtre de mesure, en pixels).

Pour travailler sans carte NVIDIA sur de vraies données : sur un PC NVIDIA, lancer la démo avec `--source zed` ou `--source svo` et appuyer sur `s` ; copier les fichiers `data/trame_XXXX.npz` sur l'autre PC, puis utiliser `--source npy`.

## 9. Règles de travail

- Travailler sur une seule tâche à la fois.
- À la fin de chaque session, ajouter une entrée dans `docs/journal.md`, en indiquant son auteur et la tâche concernée.
- Noter chaque mesure dans `docs/mesures.md`.
- Noter chaque choix technique important dans `docs/decisions.md`.
- Ne jamais enregistrer dans Git : enregistrements de la caméra, mots de passe, clés d'accès, informations personnelles.
- Avant de rattacher ce dossier au dépôt de l'équipe, confirmer avec elle où placer le code et quelles conventions suivre (branches, nommage).
