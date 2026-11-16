"""TÂCHE « STABILITÉ ET ÉTATS » : stabilité et états de suivi

Rôle
    Rendre la position utilisable par le bras : sans tremblement, et accompagnée
    d'un état de fiabilité clair (recherche, suivi, prêt ou perdu).
    Les règles des états sont décrites dans docs/mandat.md, section 3.

Reçoit
    Des positions successives (valides ou « invalides »), l'ouverture de la bouche
    et le temps.

Produit
    Une position stabilisée, un état et un horodatage.

Prérequis
    Aucun matériel : on travaille avec des positions simulées.

Dépend de
    Seulement de l'interface « position 3D → stabilité », pas du code de la tâche
    « position 3D ».

Transmet à
    La tâche « envoi au bras » (repere_bras.py), par l'interface
    « stabilité → envoi au bras ».

Terminée quand
    Sur des suites de positions simulées (stable, qui saute, qui disparaît puis revient),
    la position ne tremble pas et les états suivent les règles.

À savoir
    Trop de lissage : la cible traîne derrière la vraie position. Pas assez : elle tremble.
    Un état « prêt » trop facile à obtenir présente un risque pour la personne : être prudent.

Questions liées
    4 et 7 de docs/mandat.md.
"""
