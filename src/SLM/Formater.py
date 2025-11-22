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
        prompt = f"""Retourne UNIQUEMENT du JSON.
            Règles: target_object=objet à saisir (+couleur), obstacles=objets physiques (PAS directions seules), 
            response=confirmation amicale 1ère pers. (pas répéter commande), confidence=0.7-1.0 si clair.

            Exemples:
            "Attrape voiture rouge et évite bus"
            {{"response":"Avec plaisir, je prends la voiture rouge !","target_object":"voiture rouge","obstacles":["bus"],"status":"ok","confidence":0.85}}
            "Prends clavier à droite"
            {{"response":"D'accord, je saisis le clavier pour toi.","target_object":"clavier","obstacles":[],"status":"ok","confidence":0.7}}
            "Donne pomme bleue devant ordinateur"
            {{"response":"Compris, je te donne la pomme bleue !","target_object":"pomme bleue","obstacles":["ordinateur"],"status":"ok","confidence":0.9}}
            "Prends le truc rouge"
            {{"response":"Hmm, quel objet rouge veux-tu exactement ?","target_object":null,"obstacles":[],"status":"ambiguous","confidence":0.35}}

            Commande: {user_cmd}
            JSON:"""

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

        REFORMULATION_MSG = (
            "Je ne suis pas sûr de bien comprendre (confiance ≈{pct}%). "
            "Peux-tu reformuler en précisant l'objet cible (nom + couleur), "
            "sa position (ex.: « devant l'ordinateur, derrière toi ») et, s'il y en a, "
            "les obstacles à éviter ?"
        )

        THRESHOLD = 0.50

        response = (raw.get("response") or "").strip()
        target_object = raw.get("target_object", None)
        obstacles = raw.get("obstacles") or []
        status = (raw.get("status") or "ok").strip()
        confidence = float(raw.get("confidence", 0.5))
        confidence = max(0.0, min(1.0, confidence))

        # Forcer obstacles à être une liste de strings propres
        if not isinstance(obstacles, list):
            obstacles = []
        obstacles = [str(o).strip() for o in obstacles if str(o).strip()]

        # 2) Appliquer la règle de reformulation
        if (confidence < THRESHOLD) or (status in {"missing_target", "ambiguous", "empty"}):
            # Si le modèle disait "ok" mais confiance faible, on bascule en "ambiguous"
            if status == "ok":
                status = "ambiguous"
            pct = int(round(confidence * 100))
            response = REFORMULATION_MSG.format(pct=pct)

        # 3) Valeur par défaut pour response si vide et status ok (rare)
        if not response and status == "ok":
            response = "D’accord."

        return {
            "response": response,
            "target_object": target_object,
            "obstacles": obstacles,
            "status": status,
            "confidence": confidence,
        }
