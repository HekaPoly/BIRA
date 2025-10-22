"""
------------------------------------------------------------------------------------
SLM Manager
------------------------------------------------------------------------------------
2025-10-22 v1.0 Marcelo Zevallos Gavidia, Ryan Ajakane et Zakaria Kerouani, création
------------------------------------------------------------------------------------
DESCRIPTION GÉNÉRALE
------------------------------------------------------------------------------------
Le module **SLM Manager** encapsule l’interaction avec un *Small Language Model*
(SLM) local exécuté via Ollama. Il permet d’analyser une commande exprimée en
langage naturel et d’en extraire les informations essentielles nécessaires au
contrôle d’un système robotique ou d’une interface intelligente.

Le module effectue automatiquement les étapes suivantes :
    1. Construction d’un *prompt* explicite à partir de la commande utilisateur.
    2. Envoi de ce *prompt* au modèle via l’API HTTP d’Ollama.
    3. Extraction et validation du premier objet JSON renvoyé par le modèle.
    4. Normalisation des données (labels, statut, confiance).
    5. Encapsulation du résultat dans la dataclass `Extraction`.

Champs extraits :
    - **response** : message ou feedback textuel à renvoyer à l’utilisateur.
    - **target_object** : étiquette de l’objet cible (doit exister dans `labelDict`).
    - **obstacles** : liste des objets à éviter.
    - **status** : état global de l’analyse ("ok", "missing_target", "ambiguous", etc.).
    - **confidence** : score de confiance (0.0 à 1.0).

------------------------------------------------------------------------------------
NOTES
------------------------------------------------------------------------------------
- Le `Formatter` gère la construction du prompt, le parsing JSON et la
  normalisation des résultats.
- Le `SLM_Manager` se charge de la communication avec Ollama et de
  l’orchestration complète du flux d’analyse.
- Ce module est conçu pour être intégré à un système plus large de perception
  et de commande (ex. : bras robotisé ou agent conversationnel).
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Any, Optional, List
import json
import argparse
import requests

# --------------------------------------------------------------------------------------
#  Classe Formatter
# --------------------------------------------------------------------------------------
class Formatter:
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
            "Tu es un extracteur d'informations. "
            "Retourne UNIQUEMENT un JSON compact avec les clés suivantes : "
            "response (str), target_object (str ou null), obstacles (liste de str), "
            "status (ok/missing_target/ambiguous/empty), confidence (0..1). "
            f"Texte : {user_cmd}\nJSON :"
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

# --------------------------------------------------------------------------------------
#  Classe Extraction
# --------------------------------------------------------------------------------------
@dataclass
class Extraction:
    """
    Représente la sortie structurée d'une analyse de commande.

    Attributs
    ---------
    response : str
        Message vocal ou textuel à retourner à l’utilisateur.
    target_object : Optional[str]
        Nom exact de l’objet cible (ou None si non détecté).
    obstacles : List[str]
        Liste des obstacles à éviter.
    status : str
        Statut global de l’analyse : "ok", "missing_target", "ambiguous", "empty".
    confidence : float
        Niveau de confiance entre 0 et 1.
    """

    response: str
    target_object: Optional[str]
    obstacles: List[str]
    status: str
    confidence: float = 0.5

    def to_payload(self) -> Dict[str, Any]:
        """
        Retourne une version simplifiée du résultat pour un usage en aval (API, robot…)
        """
        return {
            "response": self.response,
            "target_object": self.target_object,
            "obstacles": self.obstacles,
        }

# --------------------------------------------------------------------------------------
#  Classe SLM_Manager
# --------------------------------------------------------------------------------------
class SLM_Manager:
    """
    Gère la communication avec le modèle Ollama et orchestre le pipeline complet :
      1) Création du prompt via Formatter
      2) Appel HTTP à Ollama pour générer la réponse
      3) Parsing et post-traitement du JSON
      4) Conversion en objet `Extraction`

    Exemple d’utilisation :
    -----------------------
    >>> mgr = SLM_Manager()
    >>> mgr.load_model()
    >>> ext = mgr.analyze_command("Attrape la voiture et évite le bus.")
    >>> print(ext.to_payload())
    """

    def __init__(
        self,
        model_name: str = "llama3.2",                       # Nom du modèle Ollama local
        api_url: str = "http://localhost:11434/api/generate",
        formatter: Optional[Formatter] = None,
        max_new_tokens: int = 192,
        temperature: float = 0.4,
    ):
        self.model_name = model_name
        self.api_url = api_url
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.formatter = formatter or Formatter()

    # ------------------------------------------------------------------
    def load_model(self):
        """
        Vérifie que le serveur Ollama est accessible (aucun téléchargement HF).
        """
        r = requests.post(
            self.api_url,
            json={"model": self.model_name, "prompt": "ping", "stream": False, "num_predict": 1},
            timeout=15,
        )
        r.raise_for_status()
        print("Ollama accessible et prêt.")

    # ------------------------------------------------------------------
    def generate_response(self, prompt: str) -> str:
        """
        Envoie un prompt au modèle Ollama et récupère la réponse textuelle complète.

        Parameters
        ----------
        prompt : str
            Instruction ou question à traiter.

        Returns
        -------
        str
            Sortie textuelle complète produite par le modèle.
        """
        r = requests.post(
            self.api_url,
            json={
                "model": self.model_name,
                "prompt": prompt,
                "stream": False,
                "num_predict": self.max_new_tokens,
                "temperature": self.temperature,
            },
            timeout=60,
        )
        r.raise_for_status()
        data = r.json()
        return data.get("response", "")

    # ------------------------------------------------------------------
    def analyze_command(self, user_cmd: str) -> Extraction:
        """
        Analyse une commande en langage naturel pour en extraire les informations clés.

        Parameters
        ----------
        user_cmd : str
            Commande donnée par l’utilisateur (ex. "Attrape la voiture et évite le bus.")

        Returns
        -------
        Extraction
            Résultat structuré de l’analyse.
        """
        prompt = self.formatter.build_prompt(user_cmd)
        raw_text = self.generate_response(prompt)
        try:
            raw = self.formatter.parse(raw_text)
        except Exception:
            raw = {
                "response": "Peux-tu reformuler en précisant l’objet cible ?",
                "target_object": None,
                "obstacles": [],
                "status": "ambiguous",
                "confidence": 0.3,
            }
        data = self.formatter.postprocess(raw, user_cmd)
        return Extraction(**data)

# --------------------------------------------------------------------------------------
#  Exécution directe du script
# --------------------------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyse de commandes NL -> labels connus (labelDict)")
    parser.add_argument("command", type=str, nargs="*", help="Commande en langage naturel")
    parser.add_argument("--model", dest="model", default="llama3.2")
    args = parser.parse_args()

    cmd = " ".join(args.command).strip()
    mgr = SLM_Manager(model_name=args.model)
    mgr.load_model()

    if not cmd:
        print('Exemple : python slm_manager.py "Attrape la voiture et évite le bus."')
    else:
        extraction = mgr.analyze_command(cmd)
        print(json.dumps(extraction.to_payload(), ensure_ascii=False))
        print(f"status={extraction.status} confidence={extraction.confidence}")
