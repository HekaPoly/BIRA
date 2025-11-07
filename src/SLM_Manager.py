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
- Le `Formater` gère la construction du prompt, le parsing JSON et la
  normalisation des résultats.
- Le `SLM_Manager` se charge de la communication avec Ollama et de
  l’orchestration complète du flux d’analyse.
- Ce module est conçu pour être intégré à un système plus large de perception
  et de commande (ex. : bras robotisé ou agent conversationnel).
"""
from __future__ import annotations
from typing import Optional
from Formater import Formater
from Extraction import Extraction
import json
import argparse
import requests

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
        formater: Optional[Formater] = None,
        max_new_tokens: int = 192,
        temperature: float = 0.4,
    ):
        self.model_name = model_name
        self.api_url = api_url
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.formater = formater or Formater()
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
                "format": "json",
                "options": {
                    "num_predict": self.max_new_tokens,
                    "temperature": self.temperature,
                },
                "stream": False,
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
        prompt = self.formater.build_prompt(user_cmd)
        raw_text = self.generate_response(prompt)
        try:
            raw = self.formater.parse(raw_text)
        except Exception:
            raw = {
                "response": "Peux-tu reformuler en précisant l’objet cible ?",
                "target_object": None,
                "obstacles": [],
                "status": "ambiguous",
                "confidence": 0.3,
            }
        data = self.formater.postprocess(raw, user_cmd)
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
        print('Exemple : python SLM_Manager.py "BIRA, donne moi la pomme bleu"')
    else:
        extraction = mgr.analyze_command(cmd)
        print(json.dumps(extraction.to_payload(), ensure_ascii=False))
        print(f"status={extraction.status} confidence={extraction.confidence}")
