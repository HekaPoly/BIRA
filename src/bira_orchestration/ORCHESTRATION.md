# Orchestration BIRA

Ce document decrit l'orchestration d'execution utilisee par `BiraManager`.

## Machine D'Etats

L'ordre d'execution suit cette boucle:

1. `IDLE`
2. `LISTENING`
3. `VISION`
4. `PLANNING`
5. `EXECUTING`
6. retour vers `LISTENING` ou `IDLE`

En cas d'echec non recuperable, le gestionnaire passe a `EXIT`.

## Contexte Partage

L'objet de contexte (`BiraContext`) est la source de verite echangee entre les etats.

- `objects_detected`: liste des objets ZED provenant de la vision.
- `detection_labels`: identifiants de classes issus des detections YOLO.
- `user_inputs`: historique des entrees utilisateur.
- `feedbacks`: historique des retours assistant.
- `object_selected`: objet cible resolu pendant la planification.

Chaque etape ecrit aussi un code d'etat:

- `listening_code`
- `vision_code`
- `planification_code`
- `execution_code`

Ces codes pilotent la transition suivante.

## Responsabilites Des Sous-Equipes

Le controleur consolide quatre groupes de sous-systemes:

1. Equipe sortie audio: `TextToSpeech`
2. Equipe entree audio: `Micro` + `SpeechToText`
3. Equipe vision: `Camera` + `ComputerVision`
4. Equipe langage/planification: `SLM_Manager` (+ backend TensorRT/Ollama)

`BiraController.preload_components()` enregistre maintenant un resume de disponibilite pour chaque sous-systeme dans les logs d'evenements (`component_preload_result`, `component_preload_summary`).

## Logs Et Tracabilite

Les traces d'execution sont persistees via `bira_components.history`:

- `*_conversation.jsonl`: entrees conversationnelles (`user`, `assistant`, `assistant_planning`)
- `*_events.jsonl`: evenements techniques et de cycle de vie (preload, enregistrement, routage SLM, fallback TensorRT)
- `*_history.txt`: instantanes d'objets produits par la vision

Ces traces rendent les decisions d'orchestration reproductibles pour le debogage.

## Etendre L'Orchestration En Securite

Pour ajouter un nouvel etat:

1. Ajouter une nouvelle valeur d'enum dans `StateCode`.
2. Implementer une classe d'etat sous `src/bira_orchestration/states/`.
3. L'enregistrer dans `STATE_CLASSES` dans `manager.py`.
4. Definir les transitions avec les codes d'etat dans `_decide_next_state`.
5. Ajouter des evenements de log pour l'entree, la decision, et les chemins d'erreur.
