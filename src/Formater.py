from typing import Dict, Any
from dataclasses import dataclass
import json

# --------------------------------------------------------------------------------------
#  Classe Formater
# --------------------------------------------------------------------------------------
@dataclass
class Formater:
    """
    Le `Formatter` sert à préparer le prompt envoyé au modèle, à extraire la
    première structure JSON renvoyée, puis à normaliser les données pour produire
    un dictionnaire cohérent avec les clés attendues par la classe `Extraction`.

    Étapes :
      1) build_prompt(user_cmd) : formate une consigne claire pour le SLM.
      2) parse(raw_text) : isole le premier bloc JSON de la sortie texte brute.
      3) postprocess(raw, user_cmd) : normalise et complète les champs manquants.
    """

    def build_prompt(self, user_cmd: str) -> str:
        """
        Construit un prompt explicite demandant au modèle d’extraire uniquement un JSON.

        Parameters
        ----------
        user_cmd : str
            Commande ou phrase en langage naturel saisie par l’utilisateur.

        Returns
        -------
        str
            Prompt à envoyer au modèle Ollama.
        """
        return (
            "Tu es un extracteur d'informations. Retourne UNIQUEMENT un JSON compact "
            'avec les clés: response (str), target_object (str ou null), obstacles (liste de str), '
            'status ("ok" | "missing_target" | "ambiguous" | "empty"), confidence (0..1). '
            "Règles STRICTES: "
            "1) 'response' = courte phrase polie à la 1re personne qui confirme l’action (ex: « Je te donne la pomme bleue. »). "
            "2) NE JAMAIS répéter la phrase de l’utilisateur ni commencer par « BIRA, ... ». "
            "3) 'obstacles' = objets à éviter (noms), jamais des couleurs/adjectifs. "
            "4) JSON valide et auto-contenu, aucun texte autour. "
            "5) Lis plusieurs exemples ci-dessous mais RÉPONDS UNIQUEMENT pour la ligne marquée TEXTE_CIBLE. "
            "6) N’emploie pas mot-pour-mot les formulations des exemples.\n"
            # --- Exemples variés ---
            'Texte: "Attrape la voiture rouge et évite le bus à gauche"\n'
            '{"response":"Je prends la voiture rouge.","target_object":"voiture rouge","obstacles":["bus"],"status":"ok","confidence":0.86}\n'
            'Texte: "Apporte-moi la balle jaune"\n'
            '{"response":"Je t’apporte la balle jaune.","target_object":"balle jaune","obstacles":[],"status":"ok","confidence":0.85}\n'
            'Texte: "Donne la pomme bleue devant l’ordinateur"\n'
            '{"response":"Je te donne la pomme bleue.","target_object":"pomme bleue","obstacles":["ordinateur"],"status":"ok","confidence":0.9}\n'
            # --- Cible ---
            f"TEXTE_CIBLE: {user_cmd}\n"
            "JSON:"
        )

    def parse(self, raw_text: str) -> Dict[str, Any]:
        """
        Extrait le premier objet JSON valide contenu dans la sortie du modèle.

        Parameters
        ----------
        raw_text : str
            Texte brut renvoyé par le modèle.

        Returns
        -------
        dict
            Dictionnaire Python décodé à partir du JSON extrait.
        """
        start = raw_text.find("{")
        end = raw_text.rfind("}")
        if start == -1 or end == -1 or end <= start:
            raise ValueError("Aucun JSON détecté dans la sortie du modèle.")
        return json.loads(raw_text[start:end + 1])

    def postprocess(self, raw: Dict[str, Any], user_cmd: str) -> Dict[str, Any]:
        """
        Normalise les champs du dictionnaire brut pour assurer la cohérence du schéma.

        Returns
        -------
        dict
            Dictionnaire contenant les champs normalisés : response, target_object,
            obstacles, status et confidence.
        """
        return {
            "response": raw.get("response", "D'accord."),
            "target_object": raw.get("target_object"),
            "obstacles": raw.get("obstacles", []),
            "status": raw.get("status", "ok"),
            "confidence": float(raw.get("confidence", 0.5)),
        }
