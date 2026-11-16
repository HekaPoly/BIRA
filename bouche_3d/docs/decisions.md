# Registre des décisions

> **À propos de ce fichier**
> - **Rôle :** conserver les choix techniques importants, avec les alternatives envisagées et la raison du choix. Cela évite de refaire les mêmes débats et aide une personne qui arrive à comprendre pourquoi les choses sont faites ainsi.
> - **Pour qui :** toute personne qui travaille sur le module ou le reprend.
> - **À mettre à jour :** à chaque choix technique important. Ne jamais effacer une décision : changer son statut.

## Comment l'utiliser

- Ajouter une ligne au tableau pour chaque nouvelle décision.
- Si une décision est remise en cause, ne pas l'effacer. Passer son statut à « Abandonnée » et ajouter une nouvelle ligne qui explique le changement.

Statuts possibles :
- **Retenue** : la décision est appliquée.
- **À valider** : la décision est provisoire, une vérification est nécessaire.
- **Abandonnée** : la décision a été remplacée.

## Décisions

| N° | Date | Décision | Alternatives envisagées | Raison | Statut |
|---|---|---|---|---|---|
| D1 | 2026-10-07 | Lire la profondeur directement avec la ZED Mini. | Estimer la distance à partir d'une image ordinaire, sans capteur de profondeur. | La ZED Mini fait partie du matériel du projet. La mesure directe est plus précise et plus simple. | Retenue |
| D2 | 2026-10-07 | Détecter la bouche avec MediaPipe (choix provisoire). | Entraîner un modèle maison. Utiliser un modèle exporté au format ONNX et accéléré avec TensorRT. | Aucun entraînement n'est nécessaire et c'est rapide à essayer. | À valider : disponibilité sur le Jetson à vérifier. |
| D3 | 2026-10-07 | Lire la distance sur une petite zone autour de la bouche, avec la valeur médiane, de préférence sur les lèvres. | Lire un seul pixel. | La mesure d'un seul pixel est bruitée. L'intérieur d'une bouche ouverte est sombre et mal mesuré. La médiane (valeur du milieu) est peu sensible aux mesures aberrantes. | À valider : par mesure. |
| D4 | 2026-10-07 | Développer en local, dans un dossier autonome. | Attendre l'accès au dépôt de l'équipe. | Avancer sans attendre. Le dossier pourra être rattaché plus tard sans conflit de fichiers. | Retenue |
| D5 | 2026-10-07 | Découper le futur code en modules séparés : caméra, détection, profondeur, filtrage, états, changement de repère. | Un seul programme. | Remplacer une pièce sans toucher au reste. Tester sans avoir le matériel. | Retenue |
| D6 | 2026-10-07 | Transmettre au bras un état de fiabilité en plus de la position. | Transmettre seulement la position. | « Je ne vois plus la bouche » doit être signalé explicitement. C'est une question de sécurité. | Retenue : comportement à l'état « perdu » à valider avec l'équipe. |
| D7 | 2026-10-07 | Découper le travail en cinq parties indépendantes (A à E), chacune dans son propre fichier de code, dont l'en-tête décrit le rôle, les entrées et les sorties. | Des fiches de documentation séparées. Un seul programme. | L'information reste dans le fichier concerné et ne peut pas être oubliée. Chaque partie peut être travaillée et testée seule. Le découpage organise le travail : le code est écrit ensuite, partie par partie. | Retenue |
| D8 | 2026-10-14 | Travailler dans un dépôt GitHub privé temporaire (binôme), une branche par tâche, puis le rattacher au dépôt de l'équipe avec `git subtree add --prefix=...`. | Copier le dossier à la main le jour venu. Attendre l'accès au dépôt de l'équipe. | Collaborer dès maintenant, garder l'historique, intégration sans conflit car le dossier est autonome. | Retenue : dépôt pas encore créé ; responsable Heka à prévenir ; chemin et nom de branche à confirmer avec l'équipe. |
| D9 | 2026-10-14 | Répartition entre les deux membres de la binôme décrite dans le README : partie caméra (détection de la bouche, position 3D) pour le membre 1 ; partie robot (stabilité et états, envoi au bras, échanges avec l'équipe) pour le membre 2 ; assemblage et tests ensemble. Ordre : format d'échange → détection et stabilité en parallèle → premier test branché sans ZED → position 3D et envoi au bras → assemblage. | Répartir tâche par tâche, sans ordre fixe. | Suites logiques, charge équilibrée, une seule dépendance entre les deux parties (position 3D → stabilité). | À valider avec la binôme. |
| D10 | 2026-10-14 | Utiliser Python 3.10, venv créé avec `py -3.10`. | Python 3.14 (déjà installé). Python de MSYS2. | MediaPipe et pyzed ne prennent pas en charge les versions trop récentes ; le Jetson sous JetPack 6 utilise Python 3.10 ; le Python de MSYS2 donne un venv à structure Linux, qui risque de bloquer OpenCV, MediaPipe et le ZED SDK. | À valider : version de JetPack à confirmer avec l'équipe. |
| D11 | 2026-10-18 | Séparer la lecture de la caméra (camera.py) du calcul de la position 3D (position_3d.py). | Lire la caméra directement dans position_3d.py. | Le calcul se développe et se teste sans ZED ni carte NVIDIA, avec des tableaux inventés ; on change de source (webcam, ZED, .svo) sans toucher au calcul ; applique D5. | Retenue |
| D12 | 2026-10-21 | Réglages provisoires du calcul de position : fenêtre de 11×11 pixels (rayon 5), au moins 50 % de points valides, Z accepté entre 0,2 et 1,0 m. | Lire un seul pixel. Accepter toutes les distances. | Valeurs de départ raisonnables pour un visage à 30–60 cm ; à ajuster après les mesures réelles. | À valider : par mesure. |
| D13 | 2026-11-04 | Format d'échange proposé : ObservationBouche (détection → position 3D) et Position3D (position 3D → stabilité), dataclasses figées dans interfaces.py. | Tuples ou dictionnaires. | Noms explicites, erreurs détectées tôt, même format pour tout le monde. | À valider avec la binôme et l'équipe. |
