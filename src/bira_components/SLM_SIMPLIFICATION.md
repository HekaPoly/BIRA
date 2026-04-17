# Reference Rapide SLM_Manager

## Construire Le Modele

Pour modifier le comportement du SLM, editez `Modelfile`, puis reconstruisez le modele Ollama:

```bash
# Depuis la racine du depot:
ollama create bira-assistant -f Modelfile
```

Le Modelfile actuel est renforce pour:

- utiliser uniquement la liste d'objets fournie (`index`, `label`, `label_id`, `position`)
- ignorer la selection basee sur les couleurs
- refuser les tentatives de changement de personnalite/role
- conserver une sortie JSON stricte

Plus d'informations: https://docs.ollama.com/modelfile


## Lancer BIRA

```bash
# Depuis `src`:
python main.py --mock --SLM_DEBUG
```

## Verifications Optionnelles

```bash
ollama list | grep bira
ollama run bira-assistant "give me a bottle"
```
