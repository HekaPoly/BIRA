"""TÂCHE « DÉTECTION DE LA BOUCHE » : détection de la bouche sur image

Rôle
    Repérer la bouche dans une image couleur et en tirer sa position (le centre)
    et son degré d'ouverture.

Reçoit
    Une image couleur : l'image de gauche de la ZED Mini, ou celle d'une webcam
    pendant le développement.

Produit
    Le centre de la bouche en pixels (u, v), son ouverture (rapport entre la hauteur
    et la largeur de la bouche) et un horodatage. Si aucune bouche n'est trouvée :
    « aucune bouche ».

Prérequis
    Un PC avec une webcam. La ZED Mini n'est pas nécessaire.

Dépend de
    Rien.

Transmet à
    La tâche « position 3D » (position_3d.py), par l'interface « détection → position 3D » (voir interfaces.py).

Terminée quand
    La bouche est détectée de façon stable à plusieurs distances et sous plusieurs
    angles, et l'ouverture distingue clairement une bouche ouverte d'une bouche fermée.

À savoir
    Méthode provisoire : MediaPipe (décision D2 de docs/decisions.md), à valider
    sur le Jetson. Les points des lèvres à utiliser doivent être vérifiés à l'écran.
"""
