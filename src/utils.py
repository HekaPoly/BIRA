import string

HOLDABLE_OBJECT_LABELS = [24,26,27,29,32,35,39,40,41,42,43,44,45,46,47,48,49,50,51,52,53,54,55,58,62,64,65,66,67,73,74,75,76,77,79]

LABELS = {
    0: ('personne', 'person'),
    1: ('bicycle', 'bicyclette'),
    2: ('car', 'voiture'),
    3: ('motorcycle', 'moto'),
    4: ('airplane', 'avion'),
    5: ('bus', 'bus'),
    6: ('train', 'train'),
    7: ('truck', 'camion'),
    8: ('boat', 'bateau'),
    9: ('traffic light', 'feu de signalisation'),
    10: ('fire hydrant', 'borne d\'incendie'),
    11: ('stop sign', 'panneau stop'),
    12: ('parking meter', 'parcomètre'),
    13: ('bench', 'banc'),
    14: ('bird', 'oiseau'),
    15: ('cat', 'chat'),
    16: ('dog', 'chien'),
    17: ('horse', 'cheval'),
    18: ('sheep', 'mouton'),
    19: ('cow', 'vache'),
    20: ('elephant', 'éléphant'),
    21: ('bear', 'ours'),
    22: ('zebra', 'zèbre'),
    23: ('giraffe', 'girafe'),
    24: ('backpack', 'sac à dos'),
    25: ('umbrella', 'parapluie'),
    26: ('handbag', 'sac à main'),
    27: ('tie', 'cravate'),
    28: ('suitcase', 'valise'),
    29: ('frisbee', 'frisbee'),
    30: ('skis', 'skis'),
    31: ('snowboard', 'snowboard'),
    32: ('sports ball', 'ballon de sport'),
    33: ('kite', 'cerf-volant'),
    34: ('baseball bat', 'batte de baseball'),
    35: ('baseball glove', 'gant de baseball'),
    36: ('skateboard', 'skateboard'),
    37: ('surfboard', 'planche de surf'),
    38: ('tennis racket', 'raquette de tennis'),
    39: ('bottle', 'bouteille'),
    40: ('wine glass', 'verre à vin'),
    41: ('cup', 'tasse'),
    42: ('fork', 'fourchette'),
    43: ('knife', 'couteau'),
    44: ('spoon', 'cuillère'),
    45: ('bowl', 'bol'),
    46: ('banana', 'banane'),
    47: ('apple', 'pomme'),
    48: ('sandwich', 'sandwich'),
    49: ('orange', 'orange'),
    50: ('broccoli', 'brocoli'),
    51: ('carrot', 'carotte'),
    52: ('hot dog', 'hot-dog'),
    53: ('pizza', 'pizza'),
    54: ('donut', 'beignet'),
    55: ('cake', 'gâteau'),
    56: ('chair', 'chaise'),
    57: ('couch', 'canapé'),
    58: ('potted plant', 'plante en pot'),
    59: ('bed', 'lit'),
    60: ('dining table', 'table à manger'),
    61: ('toilet', 'toilette'),
    62: ('tv', 'télévision'),
    63: ('laptop', 'ordinateur portable'),
    64: ('mouse', 'souris'),
    65: ('remote', 'télécommande'),
    66: ('keyboard', 'clavier'),
    67: ('cell phone', 'téléphone portable'),
    68: ('microwave', 'micro-ondes'),
    69: ('oven', 'four'),
    70: ('toaster', 'grille-pain'),
    71: ('sink', 'évier'),
    72: ('refrigerator', 'réfrigérateur'),
    73: ('book', 'livre'),
    74: ('clock', 'horloge'),
    75: ('vase', 'vase'),
    76: ('scissors', 'ciseaux'),
    77: ('teddy bear', 'ours en peluche'),
    78: ('hair drier', 'sèche-cheveux'),
    79: ('toothbrush', 'brosse à dents')
}

def string_to_label(sentence: str) -> int:
    """
    Converts a sentence into a label based on predefined English or French values.

    Args:
        sentence (str): The input sentence to be processed and analyzed.

    Returns:
        int: The key corresponding to the recognized object, or 80 if no match is found.
    """
    sentence = sentence.strip(string.punctuation)
    words = sentence.lower().split()
    labels = []

    for word in words:
        for key, values in LABELS.items():
            if word in values:
                labels.append(key)
                print(f"Key of recognized object: {key} , {LABELS[key]}")

    if len(labels) != 1:
        print("Error, there are", len(labels), "objects.")
        return 80

    return labels[0]
