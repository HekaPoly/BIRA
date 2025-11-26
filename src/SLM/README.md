# Modèles de Langage (SLM)

Le projet BIRA intègre des modèles de langage pour permettre une interaction naturelle avec le bras robotique. Cette section utilise **Ollama** pour exécuter des modèles de langage localement.

## Prérequis SLM

1. **Installer Ollama** : Téléchargez et installez Ollama depuis [https://ollama.ai](https://ollama.ai)

## Configuration

1. **Créer un fichier .env** dans la racine du projet :
   ```env
    SOURCE_MODEL_NAME=llama3.2
    TARGET_MODEL_NAME=BIRA
    NEW_MODEL_FILE=True
    TEST_RESPONSE_TIME=False
    OLLAMA_API_URL=http://localhost:11434
   ```
    - `SOURCE_MODEL_NAME` : Modèle de base (défaut: llama3.2)
    - `TARGET_MODEL_NAME` : Nom du modèle personnalisé (défaut: BIRA)
    - `NEW_MODEL_FILE` : Forcer la reconstruction du modèle personalisé à partir du Modelfile (défaut: False)
    - `TEST_RESPONSE_TIME` : Activer les tests de performance (défaut: True)
    - `OLLAMA_API_URL` : URL pour l'API d'Ollama (défaut: http://localhost:11434)


2. **Modèle BIRA** : Le modèle BIRA est basé sur Llama 3.2 avec un prompt système personnalisé défini dans `src/SLM/Modelfile`

## Scripts SLM

### 🔧 configure_test_ollama.py - Configuration et Tests

Script principal pour configurer et tester l'environnement Ollama. Exécutez ce script avant d'exécuter `run_model_example.py` afin de vous assurer que votre environnement est correctement configuré.

**Utilisation :**
```bash
cd src
python ./SLM/configure_test_ollama.py
```

**Fonctionnalités :**
- ✅ Vérification de l'installation d'Ollama
- 📥 Téléchargement automatique des modèles source (llama3.2)
- 🔍 Vérification de la compatibilité des modèles
- 🛠️ Construction du modèle BIRA à partir du Modelfile
- ⏱️ Tests de performance et temps de réponse

### 💬 run_model_example.py - Interface de Chat

Script interactif pour converser avec le modèle BIRA. Exécuter le script `configure_test_ollama.py` avant afin de vous assurer que votre environnement est correctement configuré.

**Utilisation :**
```bash
cd src
python ./SLM/run_model_example.py
```

**Fonctionnalités :**
- 🤖 Démarrage automatique du modèle BIRA
- 💬 Interface de chat en temps réel sur le terminal
- 🧠 Conservation de l'historique de conversation
- ⏱️ Affichage des métriques de performance
- 🛑 Arrêt propre du modèle avec Ctrl+C

**Exemple d'interaction :**
```
Votre prompt: Peux-tu me passer la banane sur la table ?
🤖BIRA: Bien sûr ! Je vais identifier la banane sur la table et utiliser mon bras robotique pour la saisir et vous la passer. Laissez-moi localiser l'objet...

Total Duration: 2.34s
Load Duration: 0.12s
```

## Workflow Recommandé

1. **Premier lancement - Configuration complète :**
   ```bash
   cd src
   python ./SLM/test_ollama.py
   ```

2. **Utilisation - Chat interactif :**
   ```bash
   python ./SLM/run_model_example.py
   ```

## Personnalisation du Modèle

**⚠️ IMPORTANT : Après toute modification, définissez `NEW_MODEL_FILE=True` dans votre `.env` et relancez `configure_test_ollama.py` pour que vos changements soient correctement appliqués.**

### Modification du Modelfile
Pour modifier le comportement de BIRA, éditez le fichier `src/SLM/Modelfile` :

```
FROM llama3.2
PARAMETER temperature 1
SYSTEM """
Ton nom est BIRA, tu est le cerveau d'un bras robotique autonome...
"""
```
**Consultez la documentation d'Ollama sur les Modelfile ([https://docs.ollama.com/modelfile](https://docs.ollama.com/modelfile)) afin de personnaliser le comportement du modèle.**

### Modification du .env

**Vous pouvez modifier le modèle utilisé pour bâtir BIRA en changeant la variable `SOURCE_MODEL_NAME` :**

```env
SOURCE_MODEL_NAME=llama3.1  # Au lieu de llama3.2
```
Ne pas oublier de changer le modèle source dans `Modelfile` également
```
FROM llama3.1 # Au lieu de llama3.2
...
```

- ✅ **Téléchargement automatique** : Le script téléchargera automatiquement le nouveau modèle source (ex: llama3.1)
- 🔍 **Vérification de compatibilité** : Validation que le TARGET_MODEL est bien basé sur le nouveau SOURCE_MODEL
- ⚠️ **Reconstruction requise** : Si le TARGET_MODEL existe déjà avec un autre source, il sera reconstruit
- 📊 **Impact performance** : Modèles différents = performances différentes (taille, vitesse, qualité)

**Modification de `TARGET_MODEL_NAME` :**
```env
TARGET_MODEL_NAME=BIRA_V2  # Au lieu de BIRA
```
- 🆕 **Nouveau modèle personnalisé** : Création d'un modèle avec le nouveau nom
- 📁 **Coexistence** : L'ancien modèle (BIRA) reste disponible
- 🔄 **Reconstruction systématique** : Le nouveau modèle sera construit à partir du Modelfile
- 💬 **Chat indépendant** : Conversations séparées entre les différents modèles

## Dépannage

**Problème : "Model BIRA not found"**
- Solution : Exécutez `python ./SLM/test_ollama.py` pour construire le modèle

**Problème : Modèle lent**
- Vérifiez que vous avez suffisamment de RAM
- Considérez utiliser un modèle plus petit (llama3.1 au lieu de llama3.2)