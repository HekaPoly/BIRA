"""TÂCHE « ENVOI AU BRAS » : repère et interface du bras

Rôle
    Exprimer la position dans le repère du bras, puis la lui transmettre dans le bon format.

Reçoit
    La cible stabilisée : position dans le repère de la caméra et état.

Produit
    La position dans le repère du bras et l'état, envoyés au bras.

Prérequis
    Les réponses de l'équipe aux questions 2, 3 et 7 de docs/mandat.md, et la vraie
    transformation entre le repère de la caméra et celui du bras. Le calcul du changement
    de repère peut être écrit et testé sans l'équipe, avec une transformation d'exemple.

Dépend de
    Seulement de l'interface « stabilité → envoi au bras ».

Transmet à
    Le bras, par l'interface « envoi au bras → bras ».

Terminée quand
    Des points connus sont convertis avec une erreur que l'équipe juge acceptable
    (question 4), et le bras reçoit la position et l'état dans le format convenu.

À savoir
    Une erreur dans la transformation ruine la précision de tout le système.
    Attention aux unités (mètres ou millimètres) et aux axes. Ne jamais envoyer
    une position sans son état.

Questions liées
    1, 2, 3, 4 et 7 de docs/mandat.md.
"""
