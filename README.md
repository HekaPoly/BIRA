# BRAs_VoiceAndVision

## Vue d'ensemble

**BRAs_VoiceAndVision** est un système d'interface avancé conçu pour un exosquelette de bras qui intègre des commandes vocales et une vision par ordinateur afin d'améliorer l'interaction et le contrôle de l'utilisateur. Ce projet utilise des technologies de pointe pour créer une expérience fluide et intuitive. Les composants principaux de ce système sont :

- [**pyzed**](https://www.stereolabs.com/docs/object-detection) : Un wrapper Python pour la caméra stéréo ZED de Stereolabs, offrant une perception de profondeur de haute qualité et une conscience spatiale.
- [**openai-whisper**](https://openai.com/index/whisper/) : Un puissant modèle de reconnaissance vocale développé par OpenAI, permettant une reconnaissance précise des commandes vocales.
- [**yolov8**](https://yolov8.com/) : La dernière itération du modèle de détection d'objets YOLO (You Only Look Once), offrant des capacités de détection d'objets en temps réel et précises.

## Fonctionnalités

- **Reconnaissance des Commandes Vocales** : Utilise `openai-whisper` pour transcrire avec précision les commandes vocales en texte.
- **Détection d'Objets en Temps Réel** : Utilise `yolov8` pour identifier et suivre les objets dans l'environnement.
- **Détection de Profondeur** : Utilise `pyzed` pour la conscience spatiale et les informations de profondeur.

## Prérequis

Assurez-vous d'avoir les exigences logicielles et matérielles suivantes :

- Python 3.7 ou supérieur
- Caméra stéréo ZED (pour `pyzed`)
- GPU compatible pour YOLOv8 et OpenAI Whisper
- Bibliothèques Python requises

## Installation ** À METTRE À JOUR **

### 1. Clonez le Répertoire

```bash
git clone https://github.com/your-username/BRAs_VoiceAndVision.git
cd BRAs_VoiceAndVision
```

### 2. Transcription des Commandes Vocales
Pour transcrire une commande vocale, exécutez le script principal :
```bash
python main.py
```

Le système enregistrera votre commande vocale, la transcrira en texte, et utilisera ce texte pour identifier un objet.

### 3. Détection d'Objets

Le projet utilise YOLOv8 pour détecter des objets en temps réel. Assurez-vous que votre caméra est connectée et exécutez :

```
python [main.py] --weights models/yolov8n.pt
```

Vous pouvez changer le modèle en modifiant l'argument `--weights`.

### 4. Détection de Profondeur
Pour activer la détection de profondeur, connectez une caméra ZED et suivez les instructions dans le fichier `src/ogl_viewer/viewer.py`.

## **Structure du Projet**
- **models/** : Contient les modèles YOLOv8 utilisés pour la détection d'objets.
- **src/** : Contient le code source principal.
  - **algorithm.py** : Convertit les commandes vocales en labels d'objets.
  - **detector.py** : Implémente la détection d'objets.
  - **label.py** : Définit les labels d'objets et leurs traductions.
  - **record.py** : Implémente la transcription vocale avec OpenAI Whisper.
  - **cv_viewer/** : Contient les outils pour afficher les résultats de détection.
  - **ogl_viewer/** : Contient les outils pour la visualisation 3D avec la caméra ZED.
  - **SLM/** : Contient les outils pour l'interaction avec les modèles de langage (Small Language Models).
- **outputs/** : Contient les scripts pour récupérer les données générées.
  