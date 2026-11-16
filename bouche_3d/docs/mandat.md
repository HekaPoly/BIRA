# Fiche du mandat : position 3D de la bouche

> **À propos de ce fichier**
> - **Rôle :** définir clairement ce que le module doit faire, comment on compte y arriver et ce qu'il reste à clarifier. C'est le document de référence du projet.
> - **Pour qui :** toute personne qui travaille sur le module ou qui doit comprendre son but.
> - **À mettre à jour :** quand un objectif, une étape ou une question change. Cocher les étapes terminées et remplir les réponses de l'équipe dans le tableau des questions.

## 1. Objectif

Fournir au bras robotique, en continu, la position 3D (X, Y, Z) de la bouche de la personne à nourrir. La position est exprimée en mètres, dans le repère du bras.

Ce travail correspond au mandat 1 de l'échéancier « BIRA LOG/IA », estimé à 7 semaines. L'approche retenue ici : obtenir d'abord un prototype fonctionnel rapidement, puis le perfectionner.

## 2. Ce que le module reçoit et produit

- **Entrée :** le flux de la caméra ZED Mini (image et carte de profondeur).
- **Sortie :** pour chaque image, une position (X, Y, Z) en mètres dans le repère du bras, un état de fiabilité et un horodatage.

## 3. Les quatre états du suivi

Le bras ne doit avancer que si la position est fiable. Le module indique donc son état en plus de la position.

| État | Quand | Ce que le bras reçoit |
|---|---|---|
| Recherche | Aucun visage valide n'est détecté. | « Pas de cible ». |
| Suivi | Une position valide existe, mais le moment n'est pas encore propice. | La position, marquée valide. |
| Prêt | La bouche est ouverte et la position est stable. | La position, marquée « prêt à avancer ». |
| Perdu | Le suivi vient d'être interrompu (par exemple, la cuillère masque la bouche). | La dernière bonne position, ou un ordre d'arrêt. À confirmer avec l'équipe. |

## 4. Matériel

- **Caméra :** ZED Mini (Stereolabs). C'est une caméra stéréo : elle calcule la distance en comparant deux images, comme le font les deux yeux. Elle mesure de 0,1 m à 15 m, avec une plage idéale de 0,1 m à 9 m.
- **Calculateur du robot :** Jetson (modèle et version de JetPack à confirmer).
- **Développement :** d'abord sur un PC. La détection de la bouche sur image se met au point avec une webcam. La profondeur se teste avec un enregistrement de la ZED, ou directement sur le Jetson.

## 5. Plan de travail

Le travail est découpé en cinq tâches indépendantes : détection de la bouche, position 3D, stabilité et états, envoi au bras, assemblage et tests. Chacune a son fichier de code dans `src/bouche_3d/`, dont l'en-tête décrit son rôle, ce qu'elle reçoit, ce qu'elle produit et de quoi elle dépend. Ce que les tâches s'échangent est décrit dans `src/bouche_3d/interfaces.py`.

Statuts possibles : À faire, En cours, Terminée, Bloquée (en attente de l'équipe). Ce tableau est la référence pour suivre l'avancement.

| Tâche | Objectif | Résultat attendu | Fichier de code | Statut |
|---|---|---|---|---|
| Détection de la bouche | Repérer la bouche dans l'image (position en pixels et degré d'ouverture). | La bouche détectée s'affiche en direct sur une webcam. | `src/bouche_3d/detection_2d.py` | À faire |
| Position 3D | Obtenir la position 3D de la bouche avec la ZED Mini. | Une position (X, Y, Z) en mètres, repère de la caméra. | `src/bouche_3d/position_3d.py` | À faire |
| Stabilité et états | Rendre la position stable et gérer les cas d'échec. | Pas de tremblement ; les quatre états fonctionnent. | `src/bouche_3d/stabilite_etats.py` | À faire |
| Envoi au bras | Exprimer la position dans le repère du bras et la transmettre. | Le bras reçoit une position utilisable. | `src/bouche_3d/repere_bras.py` | À faire |
| Assemblage et tests | Relier les quatre autres tâches, mesurer la qualité et faire tourner l'ensemble sur le Jetson. | Module de bout en bout ; erreur, latence et taux de suivi perdu notés dans `mesures.md`. | `src/bouche_3d/assemblage.py` | À faire |

`src/bouche_3d/camera.py` n'est pas une tâche : il fournit les images et la profondeur à la détection et à la position 3D.

À traiter plus tard, hors urgence : calibration fine, conditions d'éclairage, bouche masquée par la cuillère, optimisation de la vitesse.

## 6. Questions ouvertes

Questions à poser à l'équipe. Noter la réponse et la date dès qu'elle arrive.

| N° | Question | Pourquoi c'est important | Réponse | Date |
|---|---|---|---|---|
| 1 | La caméra est-elle fixe ou montée sur le bras ? | Détermine le risque que la cuillère masque la bouche, et si l'image peut être déformée par le mouvement. | À confirmer | |
| 2 | Existe-t-il une transformation entre le repère de la caméra et celui du bras ? Sinon, qui la réalise ? | Sans elle, les positions ne sont pas utilisables par le bras. Elle fixe aussi la précision maximale du système. | À confirmer | |
| 3 | Quelle interface de sortie vers le bras (ROS ou autre) ? | Définit le format dans lequel la position est envoyée. | À confirmer | |
| 4 | Quelle erreur de position le bras peut-il tolérer ? | Fixe l'objectif de précision à atteindre. | À confirmer | |
| 5 | Quel est le modèle exact du Jetson et la version de JetPack ? | Le ZED SDK doit correspondre à cette version. | À confirmer | |
| 6 | Quel budget de calcul (GPU, mémoire) est alloué à ce module ? | Le Jetson peut être partagé avec d'autres modules du projet. | À confirmer | |
| 7 | Que doit faire le bras à l'état « perdu » : garder la dernière position ou s'arrêter ? | C'est une question de sécurité. | À confirmer | |
| 8 | Peut-on obtenir un enregistrement de la ZED (fichier .svo) d'une personne face à la caméra ? | Permet de développer sans avoir la caméra sous la main. | À confirmer | |

## 7. Risques connus

| Risque | Explication | Piste |
|---|---|---|
| Bouche masquée | La cuillère ou le bras peut cacher la bouche pendant l'approche. | Conserver la dernière bonne position ou signaler l'état « perdu ». |
| Mesure de distance peu fiable | La caméra mesure mal dans les zones sombres, brillantes ou uniformes, comme l'intérieur d'une bouche ouverte. | Lire la distance sur les lèvres ou le contour du visage. |
| Image déformée par le mouvement | La caméra lit l'image ligne par ligne (« rolling shutter ») : si elle bouge, l'image se déforme. | À évaluer une fois la question 1 réglée. |
| MediaPipe sur Jetson | Cette bibliothèque de détection n'est pas garantie sur l'architecture du Jetson. | La tester tôt. Alternative : un modèle exporté au format ONNX et accéléré avec TensorRT. |
| Mauvais changement de repère | Une erreur de calibration entre la caméra et le bras ruine la précision, même avec une excellente détection. | En faire une tâche à part entière (tâche « envoi au bras »). |

## 8. Critères d'acceptation

À définir avec l'équipe : erreur de position maximale, latence maximale, part d'images avec suivi perdu tolérée. Valeurs à confirmer.

## 9. Documents liés

- `journal.md` : où en est le travail.
- `mesures.md` : comment la qualité est mesurée et résultats obtenus.
- `decisions.md` : choix techniques et leurs raisons.
