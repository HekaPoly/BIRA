"""TÂCHE « ASSEMBLAGE ET TESTS » : assemblage, mesures et Jetson

Rôle
    Relier les tâches « détection de la bouche », « position 3D », « stabilité et états »
    et « envoi au bras » en un module qui fonctionne de bout en bout,
    mesurer sa qualité, puis le faire tourner sur le Jetson.

Reçoit
    Les quatre autres tâches.

Produit
    Le module assemblé et trois mesures (erreur de position, latence, suivi perdu),
    notées dans docs/mesures.md.

Prérequis
    Les quatre autres tâches terminées, au moins dans une version simple. Le Jetson, la ZED Mini
    et une cible de mesure. Les réponses aux questions 5 et 6 de docs/mandat.md.

Dépend de
    Les tâches « détection de la bouche », « position 3D », « stabilité et états »
    et « envoi au bras ».

Terminée quand
    Le module tourne de bout en bout sur le Jetson, les trois mesures sont notées dans
    docs/mesures.md et les critères d'acceptation (docs/mandat.md, section 8) sont vérifiés.

À savoir
    Les mesures faites sur un PC ne valent pas pour le Jetson. Changer une seule chose
    à la fois entre deux mesures, pour savoir ce qui a produit la différence.

Questions liées
    4, 5 et 6 de docs/mandat.md.
"""
