# Modèles de Langage (SLM)

Le projet BIRA intègre des modèles de langage pour permettre une interaction naturelle avec le bras robotique. Cette section utilise **Ollama** pour exécuter des modèles localement.

## Prérequis SLM

1. **Installer Ollama** : téléchargez et installez depuis [https://ollama.ai](https://ollama.ai). Vérifiez avec `ollama --version`.

## Configuration

Créez un fichier `.env` à la racine du projet :

```env
SOURCE_MODEL_NAME=llama3.2:1b
TARGET_MODEL_NAME=BIRA
NEW_MODEL_FILE=True
TEST_RESPONSE_TIME=False
OLLAMA_API_URL=http://localhost:11434
```

- `SOURCE_MODEL_NAME` : modèle de base (défaut : llama3.2:1b, ou une variante quantifiée comme `llama3.2:1b-q4_0`).
- `TARGET_MODEL_NAME` : nom du modèle personnalisé (défaut : BIRA).
- `NEW_MODEL_FILE` : reconstruire le modèle personnalisé depuis `Modelfile` (défaut : False).
- `TEST_RESPONSE_TIME` : activer les tests de performance (défaut : True).
- `OLLAMA_API_URL` : URL de l’API Ollama (défaut : http://localhost:11434).

Le modèle BIRA est basé sur Llama 3.2 avec un prompt système défini dans `src/SLM/Modelfile`.

## Scripts SLM

### 🔧 configure_test_ollama.py — Configuration et tests

Script principal pour configurer et tester l’environnement Ollama. **⚠️ Exécutez-le avant `run_model_example.py` ou `SLM_Manager.py`.**

**Utilisation :**
```bash
cd src
python ./SLM/configure_test_ollama.py
```

**Fonctionnalités :**
- ✅ Vérification de l’installation d’Ollama
- 📥 Téléchargement automatique du modèle source (`SOURCE_MODEL_NAME`)
- 🔍 Vérification de compatibilité des modèles
- 🛠️ Construction du modèle `TARGET_MODEL_NAME` à partir du `Modelfile`
- ⏱️ Tests de performance et temps de réponse

### 💬 run_model_example.py — Interface de chat

Exemple interactif pour converser avec le modèle BIRA. Lancez `configure_test_ollama.py` au préalable.

**Utilisation :**
```bash
cd src
python ./SLM/run_model_example.py
```

**Fonctionnalités :**
- 🤖 Démarrage du modèle BIRA
- 💬 Chat temps réel dans le terminal
- 🧠 Historique de conversation conservé
- ⏱️ Affichage des métriques de performance
- 🛑 Arrêt propre avec Ctrl+C

### 🧭 SLM_Manager.py — Extraction structurée

Gestionnaire qui formate un prompt, envoie la requête au modèle et retourne un objet `Extraction` (response, target_object, obstacles, status, confidence). Supporte un mode “chat:” pour du texte libre.

**Utilisation windows:**
```bash
cd src
python ./SLM/SLM_Manager.py
```

**Utilisation Jetson:**
```bash
cd src
python -m SLM.SLM_Manager
```

## Personnalisation du modèle

### Modification du Modelfile
Éditez `src/SLM/Modelfile` pour changer le modèle de base ou le prompt système :
```
FROM llama3.2:1b
PARAMETER temperature 0.3
SYSTEM """
Tu es BIRA, un bras robotique...
"""
```
Consultez la doc Ollama Modelfile : https://docs.ollama.com/modelfile

### Modification du .env

- Changer le modèle source :
```env
SOURCE_MODEL_NAME=llama3.2:1b-q4_0
```
N’oubliez pas d’aligner le `FROM` dans `Modelfile`.

- Changer le modèle cible :
```env
TARGET_MODEL_NAME=BIRA_V2
```
Reconstruisez avec `NEW_MODEL_FILE=True`.

## Dépannage

- **“Model BIRA not found”** : exécutez `python ./SLM/configure_test_ollama.py` pour le construire.
- **Modèle lent / VRAM insuffisante** : utilisez une variante plus petite ou quantifiée (`llama3.2:1b-q4_0`), réduisez `num_predict` côté client. 
