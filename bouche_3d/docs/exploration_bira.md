# Exploration du dépôt BIRA pour l'intégration de bouche_3d

> **À propos de ce fichier**
> - **Rôle :** état des lieux du dépôt BIRA (branche `feature/bouche3d`, identique à `main` + notre commit `ca5da98`), pour préparer le branchement du module `bouche_3d`.
> - **Date :** 2026-11-16. Branches distantes lues après `git fetch --all`.
> - **Convention de lecture :**
>   - **[LU]** = constaté dans le code ou les documents, avec le fichier et la ligne.
>   - **[DÉDUIT]** = interprétation ou connaissance externe (ZED SDK, Jetson), à vérifier.
> - Les chemins sont relatifs à la racine du dépôt BIRA. Les numéros de ligne sont ceux de `main` au commit `bfdb5b2`.

---

## 1. Résumé en 10 lignes

1. **Une seule caméra ZED, un singleton** : `src/bira_components/camera.py`, classe `Camera` (l. 5-16). Elle donne accès à l'objet `sl.Camera` brut via `get_camera()` (l. 28). C'est par là que notre module doit lire l'image et le nuage de points, **sans ouvrir sa propre ZED**. [LU]
2. **La caméra n'est ouverte que pendant une requête vocale** : `BiraController.vision()` ouvre, fait **un seul** `grab()`, puis ferme (`src/bira_orchestration/controller.py` l. 56-74). Il n'y a pas de flux continu. Pour suivre la bouche en continu, il faudra changer ce modèle d'utilisation. [LU]
3. **Unités incompatibles** : BIRA ouvre la ZED avec des `sl.InitParameters()` par défaut (`camera.py` l. 22 et 43). D'après le SDK, ce sont des **millimètres** en repère IMAGE. Notre `SourceZed` impose des **mètres** (`bouche_3d/src/bouche_3d/camera.py` l. 308). Si on partage la caméra, nos seuils `z_min=0.2, z_max=1.0` rejetteront tout. [LU pour le code, DÉDUIT pour les valeurs par défaut du SDK]
4. **Aucune détection de visage ni de bouche sur `main`** : la vision se limite à YOLOv8 COCO (objets) injecté dans le module d'objets de la ZED (`computer_vision.py` l. 198-228). MediaPipe n'apparaît dans **aucune** branche. [LU]
5. **Travail voisin réutilisable** : la branche `feature/face-recognition` contient dlib (68 points du visage) et `mouth_aspect_ratio()` (`src/utils/image_processing.py` l. 18-25). La branche `feature/glass-detection` contient `get_depth_from_camera()`, qui lit la profondeur via `Camera.get_camera()` (`src/glass_detection.py` l. 7-22). Aucune des deux n'est fusionnée. [LU]
6. **Machine à états vocale** (IDLE → LISTENING → PLANNING ↔ VISION → EXECUTING → EXIT, `manager.py` l. 10-17) avec un contexte partagé `BiraContext` (`context.py`). Aucun état ne concerne l'alimentation ni la bouche. Nos états (recherche, suivi, prêt, perdu) ne recoupent aucun enum existant. [LU]
7. **Il n'y a pas de vraie liaison avec le bras** : `UARTTransmitter` n'est pas instancié (`controller.py` l. 20-21), et `send_mechanical_command()` se contente d'un `sleep(2)` (l. 76-78). Le protocole UART envoie des **angles de moteurs** (32 bits), pas des positions XYZ. [LU]
8. **Aucune transformation caméra → bras** n'existe dans le code ni dans la documentation, sur aucune branche. La seule cinématique inverse (branche `feature/3ddl`, `src/robotic.py`) est un brouillon non utilisé. [LU]
9. **Conflits de dépendances** : BIRA impose `numpy==1.23.5` et `opencv-python==4.10.0.84` (`pyproject.toml` l. 11 et 13), et pyzed 4.1 exige `numpy<2.0`. Notre venv contient numpy 2.2.6, opencv 5.0 et opencv-contrib en double. mediapipe et pytest sont absents de BIRA. [LU]
10. **Conventions** : code et commentaires en anglais (`CONTRIBUTING.md` l. 19), `print("[Tag] ...")` plutôt que `logging`, aucun test automatisé ni linter sur `main`. Notre module est entièrement en français : il faut en discuter avec l'équipe avant la PR. [LU]

---

## 2. Architecture de BIRA

### 2.1 Arborescence (sans `.git`, `.venv`, contenu de `wheels/`, `models/`, `ffmpeg/`, `CTranslate2/`)

```
BIRA/
├── .gitattributes            # wheels/*.whl en Git LFS
├── .gitignore                # ignore .venv, logs/, *.egg-info, .pytest_cache…
├── CODEOWNERS                # * `@ryadbellit`
├── CONTRIBUTING.md           # conventions (snake_case, commentaires en anglais, branches)
├── LICENSE
├── Modelfile                 # modèle Ollama « bira-assistant » (SLM)
├── README.md                 # obsolète (cite algorithm.py, detector.py, ogl_viewer… qui n'existent plus)
├── poetry.lock / pyproject.toml
├── recording.wav             # enregistrement micro (artefact)
├── test_routing_context.py   # script manuel de test du routeur SLM (chemin codé en dur /Users/home/…)
├── CTranslate2/  ffmpeg/     # dossiers vides dans cette copie (submodule non déclaré : `git submodule status` échoue)
├── models/                   # yolov8m.pt, yolov8n.pt, yolov8n_resna.pt
├── wheels/                   # pyzed-4.1, torch-2.3.0, torchaudio-2.3.0, torchvision-0.18.0 (cp310, linux_aarch64)
├── bouche_3d/                # NOTRE module (déjà commité sur feature/bouche3d, ca5da98)
└── src/
    ├── main.py
    ├── utils.py
    ├── bira_components/
    │   ├── __init__.py
    │   ├── camera.py
    │   ├── computer_vision.py
    │   ├── COMPUTER_VISION_USAGE.md
    │   ├── history.py
    │   ├── micro.py
    │   ├── speech_to_text.py
    │   ├── text_to_speech.py
    │   ├── SLM_Manager.py
    │   ├── SLM_SIMPLIFICATION.md
    │   ├── slm_controller.py
    │   ├── slm_formatter.py
    │   ├── tensorrt_manager.py
    │   ├── uart_transmitter.py
    │   └── tensorRT/
    │       ├── __init__.py
    │       ├── benchmark_tensorrt.py
    │       ├── tensorrt_config.json
    │       ├── tensorrt_inference.py
    │       └── tensorrt_optimizer.py
    ├── bira_orchestration/
    │   ├── __init__.py
    │   ├── context.py
    │   ├── controller.py
    │   ├── enums.py
    │   ├── manager.py
    │   ├── mock_controller.py
    │   └── states/
    │       ├── __init__.py
    │       ├── base_state.py
    │       ├── idle_state.py
    │       ├── listening_state.py
    │       ├── vision_state.py
    │       ├── planning_state.py
    │       ├── executing_state.py
    │       └── exiting_state.py
    └── cv_viewer/
        ├── __init__.py
        ├── labels.py
        ├── tracking_viewer.py
        └── utils.py
```

### 2.2 Rôle de chaque fichier de `src/` [LU]

| Fichier | Rôle |
|---|---|
| `main.py` | Point d'entrée. Lit les arguments (`--mock`, `--mode local/cloud`, `--api-key`, `--slm-debug`, `--slm-stream`), crée `BiraManager`, installe le gestionnaire CTRL-C, puis `preload()` et `run()`. |
| `utils.py` | Table COCO `LABELS` (anglais/français), `HOLDABLE_OBJECT_LABELS` et `string_to_label()` : héritage de l'ancien pipeline, non importé par l'orchestration actuelle. |
| `bira_components/camera.py` | Singleton `Camera` qui enveloppe `sl.Camera` : `open`, `grab`, `get_frame`, `get_position`, `close`, gestionnaire de contexte. |
| `bira_components/computer_vision.py` | `ComputerVision` : YOLOv8 (ultralytics, CUDA). Ses boîtes sont injectées dans la ZED (`ingest_custom_box_objects`), qui renvoie des `sl.ObjectData` avec une position 3D. |
| `bira_components/COMPUTER_VISION_USAGE.md` | Guide d'utilisation de `ComputerVision` (structure de `sl.ObjectData`, repère, exemples). |
| `bira_components/history.py` | Journal des détections dans `logs/<horodatage>_history.txt`. Contient aussi `get_distance()` (moyenne de Z après rejet des valeurs aberrantes). |
| `bira_components/micro.py` | `Micro` (sounddevice) : attente d'un seuil de volume, enregistrement de `recording.wav`. |
| `bira_components/speech_to_text.py` | `SpeechToText` : Whisper « small » en local, sur CUDA, chargé paresseusement derrière un verrou. |
| `bira_components/text_to_speech.py` | `TextToSpeech` (pyttsx3). |
| `bira_components/SLM_Manager.py` | Routage et inférence du petit modèle de langage (Ollama local ou cloud, TensorRT en option), mise en forme des détections pour le prompt. |
| `bira_components/slm_controller.py` | Appels de chat Ollama (en flux ou non, avec schéma JSON). |
| `bira_components/slm_formatter.py` | Construction des prompts, analyse et nettoyage des réponses JSON du SLM. |
| `bira_components/tensorrt_manager.py` | `TensorRT_Manager` : disponibilité de TensorRT **pour le SLM uniquement**, avec repli sur Ollama. |
| `bira_components/tensorRT/*` | Outils TensorRT pour le LLM (qwen3:1.7b) : optimiseur, moteur d'inférence (lance une commande externe), benchmark, configuration Jetson Orin Nano. |
| `bira_components/uart_transmitter.py` | `UARTTransmitter` : envoie un mot de 32 bits (angle, moteur, vitesse) sur le port série. |
| `bira_components/SLM_SIMPLIFICATION.md` | Notes sur la simplification du SLM. |
| `bira_orchestration/manager.py` | `BiraManager` : boucle de la machine à états, contexte, table `STATE_CLASSES`. |
| `bira_orchestration/controller.py` | `BiraController` : instancie et appelle tous les composants (vision, voix, SLM, exécution). |
| `bira_orchestration/mock_controller.py` | `MockedBiraController` : même interface, avec des faux objets (positions en mètres) et le clavier à la place du micro. |
| `bira_orchestration/context.py` | `BiraContext` (dataclass) : données partagées entre les états et codes de résultat de chaque étape. |
| `bira_orchestration/enums.py` | `StateCode`, `ListeningCode`, `VisionCode`, `PlanificationCode`, `ExecutionCode` (des `IntEnum`). |
| `bira_orchestration/states/base_state.py` | Classe abstraite `State` : `handle()` = `_prepare()` → `_handle()` → `_decide_next_state()`. |
| `bira_orchestration/states/*_state.py` | Les six états concrets (voir §2.4). |
| `cv_viewer/tracking_viewer.py`, `utils.py` | Affichage 2D et vue de suivi (hérités des exemples Stereolabs). Non importés par l'orchestration actuelle. |
| `cv_viewer/labels.py` | Ancienne table de labels, déclarée obsolète (`COMPUTER_VISION_USAGE.md`, section « Deprecation Note »). |

### 2.3 Démarrage [LU]

1. `main.py` `main()` (l. 24-78) : analyse des arguments, puis `BiraManager(...)` (l. 69).
2. `BiraManager.__init__` (`manager.py` l. 20-48) :
   - sans `--mock` : `BiraController(...)`, qui construit dans l'ordre (`controller.py` l. 15-31) `Camera()` (singleton, **non ouverte**), `ComputerVision(camera=...)` (charge YOLO sur CUDA), `uart_transmitter = None`, `Micro()`, `TextToSpeech()`, `SpeechToText()` et `SLM_Manager(...)` ;
   - avec `--mock` : `MockedBiraController` ;
   - puis `self.state = IdleState(self)`, `self._context = BiraContext()` et `self.counter = 0`.
3. `setup_signal_handlers()` (`main.py` l. 10-21) : CTRL-C appelle `controller.destroy()`.
4. `preload()` → `BiraController.preload_components()` (`controller.py` l. 32-45) : TTS, STT, préchauffage YOLO, SLM. Les erreurs sont attrapées et affichées.
5. `run()` (`manager.py` l. 98-100) : `while True: self.state.handle()`. Boucle **synchrone, sur un seul fil d'exécution**, sans cadence fixe : chaque état bloque jusqu'à la fin de son travail (attente du micro, inférence du SLM…).

Les imports (`from bira_orchestration.manager import ...`) supposent que le programme est lancé depuis `src/` ou avec `src/` dans le `PYTHONPATH` (`pyproject.toml` l. 7 : `package-mode = false`). [LU / DÉDUIT]

### 2.4 Machine à états [LU]

`StateCode` (`enums.py` l. 83-95) : `IDLE=0`, `LISTENING=1`, `VISION=2`, `PLANNING=3`, `EXECUTING=4`, `EXIT=5`.

| État (fichier) | `_handle` | Transitions (`_decide_next_state`) |
|---|---|---|
| `IdleState` (`idle_state.py`) | `controller.sleep_mode()` (attend un seuil de volume) | toujours → LISTENING |
| `ListeningState` (`listening_state.py`) | `controller.listen()` (enregistrement + Whisper), code SUCCESS | SUCCESS → PLANNING ; NO_INPUT → IDLE ; ERROR / NO_RESPONSE / autre → EXIT |
| `PlanningState` (`planning_state.py`) | `route_request()`, puis `controller.vision()` **en ligne** si nécessaire (l. 64-87), puis `prompt_slm()` ; choisit `object_selected` | SUCCESS → EXECUTING ; UNDETECTED_OBJECT → VISION ; IDLE → IDLE ; UNCLEAR / INAPPROPRIATE / OUT_OF_SCOPE / REPEAT / NEED_MORE_INFO / REFORMULATE / CONVERSING → LISTENING ; ERROR → EXIT |
| `VisionState` (`vision_state.py`) | `controller.vision()`, remplit `objects_detected` et `detection_labels`, code **toujours** SUCCESS (l. 18) | SUCCESS → PLANNING ; NO_OBJECT_DETECTED → LISTENING ; ERROR → EXIT |
| `ExecutingState` (`executing_state.py`) | `controller.send_mechanical_command(object_selected)`, code **toujours** SUCCESS (l. 16) | SUCCESS → LISTENING ; UNABLE_TO_MOVE → IDLE ; UNREACHABLE_OBJECT → VISION ; OBJECT_DROPPED → LISTENING ; ERROR → EXIT |
| `ExitState` (`exiting_state.py`) | message vocal, puis `controller.destroy()` | `exit(0)` |

Flux réel : depuis LISTENING, on passe **toujours par PLANNING** (`listening_state.py` l. 33-34). PLANNING lance la vision lui-même. On n'entre dans `VisionState` que pour refaire une détection (UNDETECTED_OBJECT, UNREACHABLE_OBJECT).

**Garde-fou à connaître** (`manager.py` l. 87-93) : chaque transition vers un code plus petit (par exemple EXECUTING → LISTENING) incrémente `self.counter`. Au-delà de 6, le système passe de force en EXIT. Le compteur n'est **jamais remis à zéro**, sauf à la création du manager. [LU] Conséquence [DÉDUIT] : une session normale s'arrête après environ 6 allers-retours de dialogue ; une boucle d'alimentation répétée (plusieurs cuillerées) déclencherait aussi l'arrêt.

### 2.5 Comment un composant fournit une donnée à la machine à états [LU]

- Il n'y a ni file d'attente, ni bus de messages, ni fil d'exécution séparé. Un **état appelle directement une méthode du contrôleur** (`self.bira_manager.controller.vision()`), puis **écrit le résultat dans le contexte partagé** (`self.bira_manager.get_data()` → `BiraContext`). C'est le cas dans `vision_state.py` l. 14-18 et `planning_state.py` l. 67-69.
- `BiraContext` (`context.py` l. 10-24) : `objects_detected`, `detection_labels`, `user_inputs`, `feedbacks`, `object_selected`, les quatre codes, `skip_vision_for_current_input`, `route_mode_hint` et `feedback_source`. Méthodes de remise à zéro par étape (l. 52-91).
- Le commentaire de `controller.py` l. 17-19 (TODO) reconnaît que passer la caméra à la vision n'est pas idéal et évoque une mémoire partagée. La branche `feature/bira-manager` a essayé un médiateur `asyncio.Queue` (`src/bira_componants/mediator.py`), sans le fusionner.

### 2.6 Schéma du flux

```
                       ┌────────────────────────── BiraManager.run() : while True: state.handle() ──────────────────────────┐
                       │                                                                                                     │
 Micro ─► Whisper ─► LISTENING ─► PLANNING ──(si needs_vision)──► BiraController.vision()                                   │
                                     │                                 │  Camera.open()  (singleton, InitParameters par défaut)│
                                     │                                 │  with camera:  grab() ×1 ─► get_frame() (BGRA, gauche)│
                                     │                                 │  ComputerVision.detect_objects(frame)                 │
                                     │                                 │     YOLOv8 (COCO) ─► sl.CustomBoxObjectData           │
                                     │                                 │     zed.ingest_custom_box_objects / retrieve_objects  │
                                     │                                 │     ─► sl.ObjectData.position [x,y,z] (repère ZED)    │
                                     │                                 │  Camera.close()                                       │
                                     │◄────── BiraContext.objects_detected / detection_labels ◄─┘                              │
                                     │  SLM (Ollama/TensorRT) choisit l'objet ─► BiraContext.object_selected                   │
                                     ▼                                                                                         │
                                 EXECUTING ─► controller.send_mechanical_command(object_selected)                              │
                                                 = sleep(2)    (UARTTransmitter NON branché)                                   │
                                                 ╳ ··· UART 115200 bauds, mot 32 bits [angle|vitesse|moteur] ··· microcontrôleur│
                                                                                                                               │
 └──────────────────────────── retour LISTENING / IDLE ; EXIT si erreur ou plus de 6 retours en arrière ───────────────────────┘
```

---

## 3. Points de branchement de notre module

> Rappel : ce sont des **propositions** [DÉDUIT]. Rien n'a été modifié dans BIRA.

### 3.1 `camera.py` (sources) ↔ `src/bira_components/camera.py`

**Ce que BIRA fait déjà [LU]**

- `Camera` est un singleton à double vérification avec un `threading.Lock` de classe (l. 6-16). Le constructeur crée `sl.InitParameters()`, `sl.Camera()`, `sl.Pose()` et `sl.Mat()`, **sans rien régler** (l. 22-25).
- `open(init_params=None)` (l. 31-54) :
  - n'ouvre que si `is_opened()` est faux ;
  - utilise les `InitParameters` par défaut, sauf si on lui en passe d'autres ;
  - en cas d'échec, `print` puis `exit()` : le programme entier s'arrête ;
  - active **toujours** la détection d'objets `CUSTOM_BOX_OBJECTS`, avec `enable_tracking=False`.
- `grab(runtime_params=sl.RuntimeParameters())` (l. 56-63) : appelle directement `zed.grab`, **sans verrou**.
- `get_frame(view=LEFT, memory=CPU, resolution=(0,0))` (l. 65-78) : `retrieve_image` **sous verrou**, renvoie `self.__image.get_data()`. C'est une vue sur un `sl.Mat` réutilisé : l'image est écrasée au `retrieve` suivant [DÉDUIT : `get_data()` ne copie pas par défaut].
- `get_position()` (l. 80-88) : pose dans le repère WORLD. Le suivi de position n'est **jamais activé** sur `main` (aucun `enable_positional_tracking`), donc cette méthode n'est pas utilisable en l'état [LU pour l'absence, DÉDUIT pour la conséquence].
- `get_camera()` (l. 28-29) : renvoie le `sl.Camera` brut. C'est la porte d'entrée pour `retrieve_measure`.
- **Aucune lecture de profondeur ni de nuage de points** dans la classe : pas de `retrieve_measure`, ni `MEASURE.XYZ`, ni `DEPTH`.
- `__exit__` libère l'image et **ferme la caméra** (l. 110-112). `__del__` la ferme aussi (l. 102-104).
- Pas de réglage de résolution, de fps, de `depth_mode`, d'unité ni de repère sur `main`. Valeurs par défaut du SDK 4.x [DÉDUIT, à vérifier] : unité **MILLIMETER**, repère **IMAGE** (X droite, Y bas, Z avant), résolution AUTO (HD720 pour une ZED Mini), `depth_mode` par défaut du SDK 4.1 (ULTRA ou PERFORMANCE selon la version).
- Les anciennes versions (`feature/2ddl:src/detector.py` l. 104-109, `feature/depth-sensing:src/depth_sensing.py` l. 54-56) utilisaient **METER + RIGHT_HANDED_Y_UP**, avec ULTRA à 60 fps (ancien détecteur) ou NEURAL (démo de profondeur). Ce n'est plus le cas sur `main`. [LU]

**Qui ouvre la caméra, qui appelle `grab()`, à quelle fréquence [LU]**

- `BiraController.vision()` (`controller.py` l. 56-74) est **le seul** à ouvrir la caméra et à appeler `grab()`. Il le fait **une fois par requête vocale qui demande la vision** (`planning_state.py` l. 64-67) ou par passage dans `VisionState` (l. 14). La caméra est **refermée** après chaque requête (l. 59 et `__exit__`, puis l. 66, 70 et 73).
- `if self.camera.grab():` (l. 60) teste la vérité d'un `sl.ERROR_CODE`. [DÉDUIT, à vérifier] Avec pyzed, un membre d'enum est toujours « vrai », donc un `grab` raté n'est pas détecté. `face_recognition.py` compare correctement à `sl.ERROR_CODE.SUCCESS`.

**Comment réutiliser l'instance au lieu d'ouvrir la nôtre [DÉDUIT]**

- Ajouter dans `bouche_3d/camera.py` une nouvelle source, par exemple `SourceBira(Source)`. Elle recevrait l'instance BIRA (`Camera()` ou `controller.camera`) et, dans `lire()` :
  1. si notre module est le maître de la boucle : `cam.grab(runtime)` et comparaison explicite à `sl.ERROR_CODE.SUCCESS` ;
  2. `zed = cam.get_camera()` ;
  3. `zed.retrieve_image(mat_img, sl.VIEW.LEFT)` et `zed.retrieve_measure(mat_xyz, sl.MEASURE.XYZ)` (ou `XYZRGBA`, comme aujourd'hui) avec **nos propres** `sl.Mat`, pour ne pas partager le `__image` de BIRA ;
  4. convertir en BGR, garder `[:, :, :3]` et **convertir les mm en m** si BIRA reste en millimètres (`/1000`). Mieux : obtenir que BIRA ouvre la caméra avec `coordinate_units=METER`, en passant `init_params` à `Camera.open()` ;
  5. **ne jamais** appeler `close()` dans `fermer()` : la caméra appartient à BIRA.
- Notre `SourceZed` reste utile en autonome (démo, tests, `.svo`). Elle ne doit jamais tourner en même temps que BIRA.
- Problème de fond : BIRA ouvre et ferme la caméra à chaque requête. Pour un suivi continu à environ 30 Hz, il faut que **la caméra reste ouverte pendant la phase d'alimentation** et qu'**un seul code appelle `grab()`**. Deux possibilités :
  - (a) notre module devient le maître de `grab()` pendant un nouvel état d'alimentation ;
  - (b) un fil d'acquisition unique dans `Camera` publie la dernière trame, et la vision YOLO et la bouche la lisent.

  Le choix revient à l'équipe (voir §7).
- Le verrou de `Camera` ne protège que `get_frame`. Si on lit depuis un autre fil, il faudra un verrou commun autour de `grab` et des `retrieve_*`.

**Version de pyzed et du ZED SDK [LU / DÉDUIT]**

- `wheels/pyzed-4.1-cp310-cp310-linux_aarch64.whl` : métadonnées `Version: 4.1`, `Requires-Dist: numpy (<2.0,>=1.13)`, `cython>=3.0.0`, un seul binaire `pyzed/sl.cpython-310-aarch64-linux-gnu.so`. [LU]
- Il faut donc le **ZED SDK 4.1** installé sur le Jetson, Python **3.10**, aarch64. La wheel ne fonctionne ni sous Windows ni sur un PC x86. Sur PC, il faut installer le SDK et le pyzed correspondant (même version majeure.mineure de préférence, pour que les `.svo` et les comportements concordent). [DÉDUIT]
- `torch-2.3.0` aarch64 est la wheel NVIDIA pour **JetPack 6.0** (L4T R36) et `tensorrt_config.json` cite `"platform": "jetson_orin_nano"` avec 8 Go partagés. Le Jetson est donc probablement un **Orin Nano sous JetPack 6.0, Ubuntu 22.04, Python 3.10**. [DÉDUIT, cela répond en partie à notre question 5]
- `NEURAL` (notre valeur par défaut, `bouche_3d/.../camera.py` l. 294) existe en SDK 4.1 mais demande l'optimisation d'un modèle au premier lancement et coûte cher en GPU sur un Orin Nano. À arbitrer avec le budget GPU (question 6). [DÉDUIT]

### 3.2 `detection_2d.py` ↔ rien sur `main`

- [LU] Aucune détection de visage ni de bouche sur `main`. YOLO charge `models/yolov8n.pt` (`computer_vision.py` l. 28-33) : 80 classes COCO, dont `person` (0), **pas de visage**. `yolov8n_resna.pt` et `yolov8m.pt` sont aussi présents (contenu non inspecté : ce sont des binaires).
- [LU] `feature/face-recognition` (dernier commit 2026-04-12, `ydna985` ; « test on jetson » le 2026-04-01 par Marcelo Zevallos) :
  - `src/face_recognition.py` : `dlib.get_frontal_face_detector()` + `shape_predictor_68_face_landmarks.dat` (~100 Mo, dans `models/`) + classifieur d'émotions `emotion.joblib` ;
  - lit **une seule image** via `Camera()` (l. 34-41) ;
  - `src/utils/image_processing.py` : `mouth_aspect_ratio(mouth)` (l. 18-25), points 12 à 19 de la bouche dlib ; `Face.extract_part("mouth")` (l. 109-112) ;
  - aucune profondeur.
- Branchement proposé [DÉDUIT] :
  - `detection_2d` reçoit `Trame.image` (BGR) de `SourceBira` et produit `ObservationBouche` ;
  - l'image ZED est BGRA (`computer_vision.py` l. 217 la convertit) : notre conversion BGRA→BGR existe déjà dans `SourceZed.lire` (l. 334) ;
  - si MediaPipe ne tourne pas sur le Jetson, dlib 68 points est une solution de repli **déjà testée sur Jetson par l'équipe** (selon le message de commit, non vérifié). Son `mouth_aspect_ratio` correspond à notre « ouverture ».

### 3.3 `position_3d.py` ↔ `computer_vision.py` / `history.py` / glass-detection

- [LU] BIRA ne calcule **pas** de profondeur autour d'un pixel : la position 3D vient du module d'objets de la ZED (`ingest_custom_box_objects` puis `retrieve_objects`, `computer_vision.py` l. 222-224), c'est-à-dire `sl.ObjectData.position`, le centre 3D de la boîte calculé par le SDK.
- [LU] `history.get_distance(z_values)` (`history.py` l. 61-72) : moyenne, puis rejet de ±0,3 (unité implicite : le mètre), puis nouvelle moyenne. Ce code n'est appelé nulle part sur `main`. Notre médiane sur une fenêtre (`position_bouche`, l. 66-110) est plus robuste. On n'a rien à reprendre.
- [LU] `feature/glass-detection:src/glass_detection.py` `get_depth_from_camera(cam)` (l. 7-22) : `cam.get_camera().retrieve_measure(depth_mat, sl.MEASURE.DEPTH)`, puis `.get_data().copy()`. **C'est exactement le modèle d'accès à reproduire dans `SourceBira`.**
- À adapter [DÉDUIT] : `position_bouche` ne change pas (NumPy pur). Seule l'**unité** du nuage doit être garantie (voir §5). On peut aussi exposer les réglages (rayon, part_min, z_min, z_max) dans une configuration unique.

### 3.4 `stabilite_etats.py` ↔ `bira_orchestration/`

- [LU] Aucun état « alimentation », « approche » ou « suivi » n'existe. L'état qui consommerait la position de la bouche est **EXECUTING** (`executing_state.py` l. 14-16), aujourd'hui centré sur un **objet** (`object_selected`, un `sl.ObjectData`).
- [DÉDUIT] Nos quatre états sont des **sous-états internes au module** (fiabilité de la cible). Ce ne sont pas des états de la machine vocale. Il ne faut pas les ajouter à `StateCode`. Ils pourraient en revanche :
  - être publiés dans `BiraContext` (par exemple un nouveau champ `mouth_target`, avec position, état et horodatage) ;
  - alimenter un nouvel état BIRA, par exemple `FeedingState` ou `StateCode.FEEDING`, suivant la recette de `feature/integration:src/bira_orchestration/ORCHESTRATION.md` (section « Etendre L'Orchestration En Securite » : enum, classe d'état, `STATE_CLASSES`, transitions, logs) ;
  - se traduire en `ExecutionCode` à la sortie (« perdu » trop longtemps → `UNREACHABLE_OBJECT` ou un nouveau code).
- [DÉDUIT] Attention au modèle « un `handle()` bloquant par état » : un suivi continu doit tourner **dans** `_handle()` (boucle interne à cadence fixe jusqu'à « prêt », « perdu » ou un délai maximal), ou dans un fil séparé qui écrit dans le contexte. Il faut aussi tenir compte du compteur de retours en arrière (§2.4).

### 3.5 `repere_bras.py` ↔ `uart_transmitter.py` (+ `feature/3ddl`)

- [LU] `UARTTransmitter.__init__(baud_rate=115200, velocity_shift=3, angle_shift=16, bytes=4, uart_init_delay=2)` (`uart_transmitter.py` l. 11-16).
- [LU] `send_data_through_uart(angle, motor_id=0, velocity=75)` (l. 36-89) :
  - `angle = int(angle) % 360`, donc des **degrés entiers 0-359** ;
  - exige **exactement un** port série détecté (`get_serial_ports_list`, l. 18-34 : `COMx` sous Windows, `/dev/<nom>` sous Linux) ;
  - mot = `(angle << 16) + motor_id + (velocity << 3)`, envoyé sur **4 octets little-endian** ;
  - **ouvre le port à chaque appel**, attend **2 s** (`uart_init_delay`), écrit puis ferme ;
  - **aucun accusé de réception** (pas de lecture) ; erreurs affichées avec `print`.
- [DÉDUIT] Disposition des bits : moteur sur les bits 0-2 (si moins de 8 moteurs), vitesse sur les bits 3-15, angle sur les bits 16-31. Le débit effectif est d'environ **1 commande toutes les 2 s**, incompatible avec un asservissement sur la bouche. Le protocole parle en **angles articulaires**, pas en XYZ : une cinématique inverse est nécessaire entre `repere_bras` et l'UART.
- [LU] Pas instancié : `controller.py` l. 20-21 (`self.uart_transmitter = None`, ligne commentée). `send_mechanical_command` = `sleep(2)` (l. 76-78).
- [LU] `feature/3ddl:src/robotic.py` (2025-07-10, `nguyenm31`) : `is_reachable()` et `calculate_angles(base_height, arm1_length, arm2_length, target_position)`, cinématique inverse à 3 degrés de liberté par la loi des cosinus, repère « Y vertical, base tournant autour de Y » (l. 41-42). Exemples en **cm** (`main.py` de la branche : `get_distance(...) * 100`, longueurs 30, 23 et 15). `feature/2ddl:src/main.py` `find_angle()` : `atan(x/z)`, l'angle de base calculé **directement dans le repère caméra**, sans transformation.
- [LU] **Aucune transformation caméra → bras** (matrice, décalage, calibration main-œil) dans le code ou la documentation, sur aucune branche. Les seules `sl.Transform` servent à l'affichage OpenGL (`feature/2ddl:src/ogl_viewer/viewer.py` l. 693-701).
- À adapter [DÉDUIT] : `repere_bras` doit produire (position dans le repère du bras, état). Une couche séparée, propriété de l'équipe « robot », convertit en angles et en trames UART. Il faudra un protocole plus rapide et avec accusé de réception (port gardé ouvert, sans `sleep(2)`), ce qui relève de l'équipe.

### 3.6 `assemblage.py` ↔ `controller.py`, `mock_controller.py`, `main.py`

- [DÉDUIT] Suivre le modèle des composants :
  - instanciation dans `BiraController.__init__` (`controller.py` l. 14-31) : `self.mouth_tracker = MouthTracker(camera=self.camera)` ;
  - préchargement dans `preload_components()` (l. 32-45), avec une méthode `warmup()` comme pour `ComputerVision.warmup` (`computer_vision.py` l. 55-65) ;
  - nettoyage par `destroy()` (l. 131-148 ; seuls les composants qui ont une méthode `destroy` sont appelés) ;
  - une méthode du contrôleur, par exemple `track_mouth()`, appelée par l'état d'alimentation ;
  - un équivalent dans `MockedBiraController` (`mock_controller.py` l. 37-185) qui utilise **notre `SourceSimulation`** : cela permettra de tester toute la chaîne avec `python main.py --mock`.
- [LU] Notre paquet est un projet setuptools séparé (`bouche_3d/pyproject.toml`, `where = ["src"]`). BIRA est un projet Poetry sans paquet (`package-mode = false`). Il faudra décider : ajouter `bouche_3d = { path = "./bouche_3d", develop = true }` aux dépendances Poetry, ou déplacer le code dans `src/bira_components/`. Question pour l'équipe.

---

## 4. Ce qui existe déjà et peut être réutilisé

| Élément | Chemin, fonction ou ligne | Usage pour nous | Statut |
|---|---|---|---|
| Singleton caméra + accès brut | `src/bira_components/camera.py` `Camera`, `get_camera()` l. 28 | Partager la ZED ; `retrieve_measure` via `get_camera()` | `main` [LU] |
| `Camera.open(init_params)` | `camera.py` l. 31-49 | Ouvrir avec `METER`, `IMAGE`, `depth_mode` choisis, si l'équipe accepte | `main` [LU] |
| Lecture de profondeur sur la caméra partagée | `feature/glass-detection:src/glass_detection.py` `get_depth_from_camera()` l. 7-22 | Modèle exact pour `SourceBira.lire()` | non fusionné [LU] |
| Conversion BGRA→RGB/BGR | `computer_vision.py` l. 217 ; notre `SourceZed.lire` l. 334 | Déjà fait chez nous | [LU] |
| Points de la bouche + rapport d'ouverture | `feature/face-recognition:src/utils/image_processing.py` `mouth_aspect_ratio()` l. 18-25, `Face.extract_part("mouth")` l. 109-112 | Repli si MediaPipe ne marche pas ; calcul de l'« ouverture » | non fusionné ; modèle dlib de 100 Mo [LU] |
| Détecteur de visage dlib | `feature/face-recognition:src/face_recognition.py` l. 28-30, 47 | Repli de détection | non fusionné [LU] |
| Normalisation d'une position en `[x, y, z]` | `SLM_Manager._extract_position_xyz()` l. 89-100 | Si on donne la bouche au SLM (peu probable) | `main` [LU] |
| Contexte partagé | `bira_orchestration/context.py` `BiraContext` | Publier la cible bouche | `main` [LU] |
| Patron d'état | `states/base_state.py` `State` (`_prepare`, `_handle`, `_decide_next_state`, `log_state`, `emit_feedback`) | Écrire `FeedingState` | `main` [LU] |
| Procédure d'ajout d'un état | `feature/integration:src/bira_orchestration/ORCHESTRATION.md` | Liste des étapes | non fusionné [LU] |
| Journaux JSONL `log_event()` / `log_conversation()` | `feature/integration:src/bira_components/history.py` (ajout l. ~29-58) | Tracer les états de suivi et les latences | non fusionné [LU] |
| Contrôleur simulé | `bira_orchestration/mock_controller.py` | Y brancher notre `SourceSimulation` | `main` [LU] |
| UART (format des trames) | `uart_transmitter.py` `UARTTransmitter.send_data_through_uart()` | Dernier maillon, côté équipe robot | `main`, non branché [LU] |
| Cinématique inverse 3 DDL | `feature/3ddl:src/robotic.py` `calculate_angles()` l. 29-68 | Base possible pour position → angles | brouillon, en cm [LU] |
| Moteurs YOLO + TensorRT | — | **TensorRT de BIRA = SLM seulement** (`tensorrt_manager.py`, `tensorRT/tensorrt_config.json` : `qwen3:1.7b`) ; rien pour la vision | [LU] |

À propos de TensorRT [LU] : `TensorRTInferenceEngine` (`tensorRT/tensorrt_inference.py` l. 19-116) ne charge pas de moteur TensorRT lui-même. Il **lance une commande externe** (`TENSORRT_RUNNER_COMMAND`) avec une entrée et une sortie JSON. `TensorRT_Manager` (`tensorrt_manager.py`) se replie sur Ollama si ce moteur est indisponible. **Rien de tout cela n'est réutilisable pour un modèle de vision.** [DÉDUIT] Si MediaPipe ne tourne pas, la piste « ONNX + TensorRT » de notre décision D2 serait un chantier neuf. On pourrait plutôt passer par ultralytics 8.2.82 (déjà présent), qui exporte en `format="engine"` (TensorRT), avec un modèle de visage ou de points du visage.

---

## 5. Doublons et conflits

### 5.1 Caméra ouverte deux fois [LU / DÉDUIT]

- `bouche_3d/src/bouche_3d/camera.py` `SourceZed.ouvrir()` (l. 300-321) crée son propre `sl.Camera()` et l'ouvre. Si BIRA a déjà ouvert la ZED, la seconde ouverture échoue (`CAMERA_NOT_DETECTED` ou erreur équivalente). À l'inverse, si notre module l'ouvre d'abord, `Camera.open()` de BIRA échoue et appelle **`exit()`** (l. 47-49) : **tout BIRA s'arrête**. [DÉDUIT pour les codes d'erreur]
- `SourceZed.fermer()` ferme la caméra (l. 338-341). La même chose avec l'instance BIRA couperait la vision de BIRA. Inversement, `BiraController.vision()` ferme la caméra après chaque requête : notre suivi perdrait ses images.
- `Camera.open()` active toujours la détection d'objets (l. 51-54). Ce module tourne en tâche de fond du SDK pendant que nous lisons le nuage de points, ce qui coûte du GPU. [DÉDUIT]

### 5.2 Unités et repères

| Source | Unité | Repère | Référence |
|---|---|---|---|
| BIRA `main` (`Camera`) | **mm** (défaut SDK) | **IMAGE** (défaut SDK) : X droite, Y bas, Z avant | `camera.py` l. 22, 43 [LU] + défauts SDK [DÉDUIT] |
| Doc BIRA | « [x,y,z] in world coordinates », « Typically: x=right, y=down, z=forward » ; distances affichées en « m » | IMAGE | `COMPUTER_VISION_USAGE.md` l. 70-78, 202, 216-221 [LU] |
| Contrôleur simulé BIRA | **m** (`[0.35, -0.12, 1.05]`) | — | `mock_controller.py` l. 90, 97, 104 [LU] |
| Anciennes branches | **m** | **RIGHT_HANDED_Y_UP** (X droite, **Y haut**, **Z vers l'arrière**) | `feature/2ddl:src/detector.py` l. 105-107 ; `feature/depth-sensing` l. 54-56 [LU] |
| IK `feature/3ddl` | **cm** | Y vertical | `robotic.py` ; `main.py` de la branche (`* 100`) [LU] |
| Notre module | **m** | **IMAGE** | `bouche_3d/.../camera.py` l. 308-309 ; `interfaces.py` (« Règles communes ») [LU] |
| UART | **degrés entiers** + vitesse | articulaire | `uart_transmitter.py` l. 47-64 [LU] |

[DÉDUIT] Avec la caméra BIRA telle quelle, nous recevrions des **mm** : `position_bouche` rejetterait tout (Z de 200 à 1000 hors de `[0.2, 1.0]`). Le repère IMAGE est le même que le nôtre, ce qui est une bonne nouvelle. La documentation BIRA et le contrôleur simulé supposent des mètres alors que le matériel donne (probablement) des millimètres : une incohérence existe déjà chez eux.

### 5.3 États et enums [LU]

- Pas de collision de noms : BIRA utilise `StateCode`, `*Code` (`IntEnum`) ; nous avons prévu recherche, suivi, prêt, perdu (non codés). `PlanificationCode.IDLE` et `StateCode.IDLE` existent ; à éviter si on nomme nos états en anglais (`SEARCHING`, `TRACKING`, `READY`, `LOST` ne collisionnent pas).
- `ExecutionCode.UNREACHABLE_OBJECT` (« l'objet a bougé depuis la vision ») est le plus proche de notre « perdu » côté BIRA.
- `VisionCode.NO_OBJECT_DETECTED` n'est jamais produit (`vision_state.py` l. 18 met toujours SUCCESS).

### 5.4 Versions des dépendances (voir §8)

numpy, OpenCV (deux paquets `cv2` dans notre venv) et Python : détails au §8.

### 5.5 Autres anomalies relevées en passant (hors périmètre, mais utiles) [LU / DÉDUIT]

- `controller.vision()` renvoie `None` en cas d'échec (l. 67-71), mais `VisionState._handle` déballe directement `sl_object, detection_labels = ...` (l. 14) : il en résulterait un `TypeError` non attrapé. `PlanningState` attrape bien les exceptions (l. 84-87).
- `slm_formatter.py` l. 158-159 : `if position:` sur `obj.position`. Avec un vrai `sl.ObjectData`, position est un tableau NumPy, ce qui lève « truth value of an array is ambiguous ». Le correctif « Fix position extraction » (`d4889b0`, 2026-04-15) n'a corrigé que `SLM_Manager.set_detections`. [DÉDUIT]
- `README.md` racine obsolète (cite `algorithm.py`, `detector.py`, `ogl_viewer/` qui n'existent plus).
- `pyproject.toml` ne déclare pas `sounddevice` (`micro.py`), `pyttsx3` (`text_to_speech.py`) ni `ollama` (`SLM_Manager.py`). Ils sont absents de `poetry.lock` (vérifié par recherche). L'environnement du Jetson a donc été complété à la main. [LU]

---

## 6. Conventions à respecter

| Sujet | Observé | Référence |
|---|---|---|
| Langue | Code, noms, docstrings et commentaires **en anglais** (exigé pour les commentaires) ; messages des états en anglais ; `utils.py` garde des libellés français | `CONTRIBUTING.md` l. 17-19 ; tous les fichiers de `src/` [LU] |
| Nommage | `snake_case` pour fonctions et variables, `PascalCase` pour les classes, `SCREAMING_SNAKE_CASE` pour les constantes. Exceptions réelles : `SLM_Manager`, `TensorRT_Manager`, `SLM_Controller`, attribut `isInitialised` | `CONTRIBUTING.md` l. 7-11 ; `SLM_Manager.py`, `camera.py` l. 14 [LU] |
| Indentation | 4 espaces ; formatage « soigné » sans outil imposé | `CONTRIBUTING.md` l. 13-15 [LU] |
| Docstrings | Style libre « Parameters: / Returns: » (`camera.py`, `uart_transmitter.py`) ou Google « Args: / Returns: » (`computer_vision.py`, `history.py`) | [LU] |
| Typage | Partiel : récent dans l'orchestration (`str \| None`, `list[int]`, `from __future__ import annotations`), absent dans les composants anciens | `manager.py` l. 26 ; `context.py` [LU] |
| Logs | **`print` avec un préfixe entre crochets** : `[State log]`, `[Code]`, `[Main]`, `[MOCK:…]`, `[SLM_DEBUG]`. `logging` n'est utilisé que dans `tensorRT/` | `base_state.py` l. 23-24 ; `context.py` l. 26-30 ; `tensorrt_optimizer.py` l. 13 [LU] |
| Erreurs | `try/except Exception` + `print` ; `exit()` sur un échec matériel (`Camera.open`) ; garde-fou EXIT dans la machine à états | `controller.py` l. 32-45, 131-148 ; `camera.py` l. 47-49 [LU] |
| Tests | **Aucun test automatisé, pas de pytest** ; un script manuel `test_routing_context.py` (chemin macOS codé en dur l. 5) ; blocs `if __name__ == "__main__":` dans les composants | [LU] |
| Linter | Aucun sur `main`. `feature/add-linter` (2025-08-11, Fabrice Renard, non fusionnée) : GitHub Actions + flake8 (`E9,F63,F7,F82` bloquants, le reste en avertissement, `max-line-length=127`) | `.github/workflows/main.yml` de la branche [LU] |
| Organisation | Une classe principale par fichier dans `bira_components/` ; singleton seulement pour `Camera` (et le modèle Whisper global) ; **aucun fil d'exécution** (verrous seulement) ; un contrôleur unique qui possède les composants ; contrôleur simulé avec la même interface | [LU] |
| Branches | `CONTRIBUTING` demande `feat/...` et `fix/...`, mais l'usage réel est `feature/...` ; PR obligatoire vers `main` ; `CODEOWNERS` = `@ryadbellit` (relecteur automatique) | `CONTRIBUTING.md` l. 23-26 ; `git branch -r` [LU] |
| Commits | Libres (« try », « test test », « ca marche pas »). Le nôtre suit Conventional Commits (`feat(bouche3d): ...`) | `git log` [LU] |

[DÉDUIT] Pour que notre module s'intègre bien :
- façade en anglais (`MouthTracker` avec `warmup()`, `track()` ou `get_target()`, `destroy()`) ;
- docstrings « Parameters / Returns » ;
- `print("[Mouth3D] ...")` pour les logs ;
- un équivalent simulé dans `mock_controller.py`.

La traduction de nos noms et commentaires français est **à négocier** : soit traduire avant la PR, soit garder le français à l'intérieur de `bouche_3d/` avec une façade anglaise. Nos tests pytest seraient les premiers du dépôt : un argument en faveur de notre approche.

---

## 7. Questions à poser à l'équipe (par priorité)

**Priorité 1 : bloquantes pour l'intégration**

1. **Propriété de la caméra** : qui garde la ZED ouverte, et quand ? Accepte-t-on qu'elle reste ouverte pendant toute la phase d'alimentation (au lieu du modèle « ouvrir, une image, fermer » de `controller.vision()`) ? Qui appelle `grab()` : un fil d'acquisition unique dans `Camera`, ou le composant actif ?
2. **Paramètres d'ouverture** : peut-on fixer explicitement `coordinate_units=METER` et `coordinate_system=IMAGE` (plus résolution, fps et `depth_mode`) dans `Camera` ? Les positions YOLO de BIRA sont-elles aujourd'hui en mm sur le vrai matériel ? (La documentation et le contrôleur simulé disent mètres.)
3. **Repère et transformation caméra → bras** : existe-t-elle (même mesurée à la règle) ? Où la caméra est-elle montée (fixe ou sur le bras, notre question 1) ? Qui réalise la calibration ?
4. **Interface vers le bras** : le protocole UART actuel (angles, 2 s par commande, sans accusé) est-il celui qui sera utilisé ? Qui écrit la cinématique inverse ? Le bras accepte-t-il une cible XYZ quelque part (microcontrôleur, ROS) ?
5. **Où placer le code** : garder `bouche_3d/` comme paquet séparé (dépendance Poetry `path` + `develop`), ou le déplacer dans `src/bira_components/` ? Le français est-il accepté dans le module ?

**Priorité 2 : conception**

6. **Place dans la machine à états** : nouvel état `FEEDING` (et lequel le précède : PLANNING sur une commande « nourris-moi » ?) ou extension de EXECUTING ? Que faire du compteur de retours arrière (`manager.py` l. 87-93), qui coupera une boucle d'alimentation répétée ?
7. **Comportement à l'état « perdu »** (notre question 7) et traduction en `ExecutionCode`.
8. **Budget GPU** sur l'Orin Nano, partagé avec Whisper small, YOLO, la détection d'objets ZED et le SLM : peut-on désactiver `enable_object_detection` pendant l'alimentation ? Quel `depth_mode` (NEURAL est coûteux) ?
9. **Matériel** : confirmer Jetson Orin Nano, JetPack 6.0, ZED SDK 4.1 (ce que suggèrent les wheels). La caméra est-elle bien une ZED Mini ?

**Priorité 3 : outillage et suivi**

10. A-t-on le droit d'ajouter `mediapipe` et `pytest` aux dépendances ? MediaPipe a-t-il déjà été essayé sur le Jetson ? Sinon, dlib (`feature/face-recognition`, « test on jetson ») est-il installé et utilisable ?
11. Un fichier `.svo` d'une personne face à la caméra est-il disponible (notre question 8) ?
12. `feature/add-linter` (flake8) et `feature/integration` (journaux JSONL, `ORCHESTRATION.md`) seront-elles fusionnées ? Faut-il s'y conformer dès maintenant ?
13. Qui relit la PR (`CODEOWNERS` = `@ryadbellit` pour 2026-2027, commit `b1cef27`) ? Nom de branche attendu (`feature/bouche3d` existe déjà sur `origin`) ?

---

## 8. Dépendances : `bouche_3d/requirements.txt` et `pyproject.toml` racine

| Paquet | BIRA (`pyproject.toml`, `poetry.lock`) | bouche_3d (`requirements.txt`) | Notre `.venv` Windows (dist-info) | Remarque |
|---|---|---|---|---|
| Python | `>=3.9,<3.11` (l. 10) ; en réalité **3.10** (wheels cp310, `match` dans les états, `str \| None` à l'exécution dans `manager.py` l. 26) | `>=3.10` (`bouche_3d/pyproject.toml` l. 8) | 3.10.11 | Compatible : **3.10** des deux côtés [LU] |
| numpy | **`1.23.5`** imposé (l. 11) ; pyzed 4.1 exige `<2.0` | non épinglé | **2.2.6** | **Conflit.** Notre code doit fonctionner avec numpy 1.23.5. À vérifier : `np.isfinite`, `np.median`, `default_rng` et `float \| None` existent en 1.23 [DÉDUIT : oui] |
| OpenCV | `opencv-python==4.10.0.84` (l. 13) | `opencv-python` non épinglé | **opencv-python 5.0.0.93 + opencv-contrib-python 5.0.0.93** | Deux paquets qui fournissent `cv2` dans le même environnement : à éviter. MediaPipe tire généralement `opencv-contrib-python` [DÉDUIT] : conflit probable avec le 4.10 imposé |
| mediapipe | absent (ni pyproject ni lock) | prévu | 1.2.3 (Windows) | Sur le Jetson (aarch64, JetPack 6), PyPI ne garantit pas une wheel avec accélération GPU ; souvent CPU seulement ou compilation manuelle [DÉDUIT, à tester tôt, cf. notre D2] |
| pytest | absent | prévu | 9.2.0 | À ajouter en dépendance de développement (`[tool.poetry.group.dev.dependencies]`) [DÉDUIT] |
| pyzed | wheel locale `pyzed-4.1` aarch64 (l. 17) | « fourni avec le SDK » (commentaire) | absent (normal sous Windows sans GPU NVIDIA) | Sur Jetson : la wheel BIRA. Sur PC : le pyzed 4.1 du SDK correspondant [LU / DÉDUIT] |
| torch / torchvision | wheels NVIDIA aarch64 2.3.0 / 0.18.0 | — | — | Lourd en mémoire partagée ; à prendre en compte pour le budget (Orin Nano 8 Go) [DÉDUIT] |
| ultralytics | 8.2.82 | — | — | Possible piste de repli pour la détection (modèle de visage exporté en TensorRT) [DÉDUIT] |
| scipy | 1.8.0 | — | — | Utilisé par `feature/face-recognition` (`scipy.spatial.distance`) [LU] |
| dlib, imutils, joblib, scikit-learn | absents | — | — | Requis par `feature/face-recognition` si on la réutilise [LU] |

Conclusion [DÉDUIT] : notre `requirements.txt` doit épingler `numpy==1.23.5` et `opencv-python==4.10.0.84`, ou au moins être testé avec ces versions. Il faut recréer le venv sans `opencv-contrib-python` et vérifier que la version de mediapipe choisie accepte numpy 1.23.5. Le vrai test d'intégration doit se faire sur le Jetson avec l'environnement Poetry de BIRA.

---

## 9. Branches distantes (lecture seule)

36 branches distantes (`git branch -r`). Les « en avance » et « en retard » sont comptés par rapport à `origin/main` (`git rev-list --left-right --count origin/main...<branche>`). Les listes de fichiers viennent de `git diff --stat origin/main...<branche>`.

| Branche | Dernier commit | Auteur | Retard / avance sur main | Fichiers modifiés (vs base commune) | Ce qu'elle apporte |
|---|---|---|---|---|---|
| `feature/depth-sensing` | 2025-11-05 | Marly | 103 / 1 | `src/depth_sensing.py` (+107) | Exemple Stereolabs de nuage de points (NEURAL, METER, RIGHT_HANDED_Y_UP, rendu OpenGL), « not tested ». Faute de frappe `MEASURE.XYZRBGA` l. 71. Dépend d'`ogl_viewer`, absent de `main`. **Seule branche « profondeur »** ; rien sur la bouche. |
| `feature/face-recognition` | 2026-04-12 | ydna985 (+ Marcelo Zevallos) | 82 / 14 | `src/face_recognition.py`, `src/utils/image_processing.py`, `src/utils/modelmanager.py`, `models/shape_predictor_68_face_landmarks.dat` (~100 Mo), `models/emotion.joblib` | Détection de visage dlib + 68 points + classification d'émotions ; **`mouth_aspect_ratio`** ; « test on jetson » (2026-04-01). Une seule image ; pas de 3D. **Travail le plus proche du nôtre.** |
| `feature/computer-vision` | 2026-01-02 | nguyenm31 | 114 / 0 | aucun (déjà dans main) | Contenait le revert de depth-sensing ; entièrement intégrée. |
| `feature/3ddl` | 2025-07-10 | nguyenm31 | 236 / 3 | `src/UART.py`, `src/main.py`, `src/robotic.py`, `src/outputs/retrieve_data.py` | Cinématique inverse 3 DDL (`calculate_angles`, en cm), envoi d'angles à 3 moteurs (`send_data_to_all_motors`) ; `main` saisit x, y, z à la main. Ancienne architecture. |
| `feature/2ddl` | 2025-08-06 | nguyenm31 | 236 / 5 | `src/UART.py`, `src/main.py`, `src/outputs/retrieve_data.py` | Deux angles calculés directement en repère caméra (`find_angle` = `atan(x/z)`, `find_second_angle` = `asin(y/d)`) → UART. Ancien `detector.py` : METER, ULTRA, RIGHT_HANDED_Y_UP, 60 fps, nuage XYZRGBA. |
| `feature/integration` | 2026-04-17 | Marcelo Zevallos | 18 / 8 | `SLM_Manager.py`, `history.py` (`log_event`, `log_conversation`), `micro.py`, `tensorrt_inference.py`, dossiers `tensorRT/tensorrt_models/*`, `controller.py`, `mock_controller.py`, `ORCHESTRATION.md`, `Modelfile`, `README.md` | Journaux JSONL et résumé du préchargement ; corrections TensorRT « TO BE TESTED » ; documentation de l'orchestration. Rien sur la vision ni la profondeur. |
| `feature/final-integration` | 2026-04-15 | rayanajakane | 3 / 0 | aucun (fusionnée par la PR #73) | Contient « Fix position extraction for computer vision » (`d4889b0`) : ajout de `SLM_Manager._extract_position_xyz()` et tests `is not None` (`SLM_Manager.py`, +17/-8). Ne concerne que le passage des positions au SLM. |
| `feature/tensorRT` | 2026-01-07 | zakikero | 154 / 28 | `src/SLM/**` (18 fichiers, dont `tensorRT/jetson_utils.py`, `setup_jetson.sh`, `tensorrt_requirements.txt`) | TensorRT pour le **LLM** (Jetson), ancienne arborescence `src/SLM/`. `setup_jetson.sh` et `jetson_utils.py` peuvent renseigner sur la configuration du Jetson (non lus en détail). |
| `feature/bira-manager` | 2026-01-26 | Michellen Nguyen | 82 / 4 | `src/bira_componants/*` (camera, computer_vision, mediator, bira_manager…), `src/main.py` | Essai d'architecture par **médiateur `asyncio.Queue`** (`mediator.py`) avec `Camera.receive()` asynchrone. Abandonnée au profit de `bira_orchestration`. |
| `feature/add-linter` | 2025-08-11 | Fabrice Renard | 173 / 1 | `.github/workflows/main.yml` | CI flake8 (erreurs de syntaxe et noms indéfinis bloquants). Non fusionnée. |
| `feature/bouche3d` | 2026-11-16 | AloysRussel1 | 0 / 1 | `bouche_3d/**` (28 fichiers, +2127) | **Notre module**, déjà poussé sur `origin` (commit `ca5da98`). |
| `feature/glass-detection` (en plus) | 2026-04-07 | Marcelo Zevallos | 82 / 5 | `src/glass_detection.py`, `src/camera.py` | `get_depth_from_camera()` via `Camera.get_camera()` + `MEASURE.DEPTH` ; détection de verre par heuristiques de profondeur et de couleur. **Exemple de lecture de profondeur sur la caméra partagée.** |

Recherches faites sur **toutes** les branches distantes (`git grep`, fichiers `*.py` et `*.md`, hors `bouche_3d/`) [LU] :
- `mediapipe`, `face_mesh`, `bouche` : **aucun résultat** ;
- `mouth` : uniquement `feature/face-recognition` ;
- `retrieve_measure` : `depth-sensing`, anciens `detector.py` (2ddl, docs, optimized_prototype, setup-poetry, folie-technique-app, add-linter), `glass-detection` ;
- `transform`, `calib` : uniquement l'affichage (OpenGL, `TrackingViewer.set_camera_calibration`) et des bibliothèques (transformers, sklearn) : **aucune transformation caméra → bras**.

**Travail déjà fait sur la bouche, le visage ou la profondeur :**
- **Visage et bouche 2D** : `feature/face-recognition` (dlib, rapport d'ouverture de la bouche), sans 3D.
- **Profondeur** : `feature/depth-sensing` (démo de nuage de points, non testée), `feature/glass-detection` (carte de profondeur pour le verre), anciens `detector.py` (nuage de points pour la vue).
- **Position 3D de la bouche** : rien d'autre que notre module.

---

## 10. Risques pour l'intégration

| # | Risque | Gravité | Origine | Mesure proposée [DÉDUIT] |
|---|---|---|---|---|
| R1 | Double ouverture de la ZED, ou fermeture par l'autre composant ; `exit()` dans `Camera.open` arrête tout BIRA | Haute | `camera.py` l. 47-49, 99-112 ; `controller.py` l. 59-73 ; notre `SourceZed` | `SourceBira` sans `open` ni `close` ; un seul propriétaire de `grab()` ; caméra ouverte pendant la phase d'alimentation |
| R2 | Unités mm / m : positions rejetées ou fausses d'un facteur 1000 | Haute | `InitParameters()` par défaut (`camera.py` l. 22) vs `METER` chez nous | Fixer `METER` dans BIRA, ou convertir dans l'adaptateur et l'affirmer par un test |
| R3 | Repère caméra vs bras inexistant ; anciennes branches en RIGHT_HANDED_Y_UP, IK en cm, UART en degrés | Haute (sécurité) | §3.5, §5.2 | Tâche « repère_bras » prioritaire ; tests sur des points connus ; unités dans les noms de champs |
| R4 | Protocole UART lent (2 s par commande, port rouvert à chaque fois, pas d'accusé) : impossible de suivre une bouche en mouvement ; « prêt » envoyé mais non confirmé | Haute | `uart_transmitter.py` l. 66-89 | À traiter par l'équipe robot ; notre module doit fonctionner sans supposer un envoi à 30 Hz |
| R5 | Machine à états synchrone et bloquante + compteur de retours arrière jamais remis à zéro : la boucle d'alimentation se ferait couper | Moyenne | `manager.py` l. 87-100 | Boucle interne dans `FeedingState._handle` avec délai maximal ; demander la remise à zéro du compteur |
| R6 | Conflits numpy 2 / 1.23.5, OpenCV 5 / 4.10, opencv-contrib en double ; mediapipe peut-être indisponible ou CPU seulement sur aarch64 | Moyenne | §8 | Recréer le venv aux versions BIRA ; tester MediaPipe sur le Jetson tout de suite ; repli dlib ou ultralytics |
| R7 | Budget GPU / mémoire de l'Orin Nano (Whisper, YOLO, détection d'objets ZED, SLM, profondeur NEURAL, MediaPipe) | Moyenne | `tensorrt_config.json` (8 Go partagés) ; `camera.py` l. 51-54 | Désactiver la détection d'objets pendant l'alimentation ; essayer NEURAL vs ULTRA ; mesurer la latence (`docs/mesures.md`) |
| R8 | Image partagée écrasée (`get_frame` renvoie une vue sur un `sl.Mat` réutilisé) et verrou partiel (pas sur `grab`) | Moyenne | `camera.py` l. 56-78 | Nos propres `sl.Mat` ; `.copy()` systématique (déjà fait pour le nuage, l. 335) |
| R9 | `if self.camera.grab():` ne détecte pas les échecs ; `vision()` peut renvoyer `None`, et `VisionState` plante alors | Faible pour nous, réelle pour BIRA | `controller.py` l. 60 ; `vision_state.py` l. 14 | Comparer à `sl.ERROR_CODE.SUCCESS` dans notre code ; signaler à l'équipe |
| R10 | Langue et style (français vs anglais imposé), absence de linter et de tests chez eux : friction en revue | Faible | `CONTRIBUTING.md` l. 19 | Façade anglaise ; décider avec le relecteur avant la PR |
| R11 | Code de référence non fusionné (face-recognition, glass-detection, integration) : il peut changer ou disparaître | Faible | §9 | Copier les idées, pas dépendre des branches |

---

*Fin du rapport. Aucun fichier de BIRA n'a été modifié. Seules les références distantes ont été mises à jour par `git fetch --all`.*
