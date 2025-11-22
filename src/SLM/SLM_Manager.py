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
from SLM.Formater import Formater
from SLM.Extraction import Extraction
import json
import argparse
from ollama import Client
import subprocess

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
        model_name: str = "BIRA",                       
        formater: Optional[Formater] = None,
        max_new_tokens: int = 128,
        temperature: float = 0.3,
    ):
        self.model_name = model_name
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.formater = formater or Formater()
        self.client = Client(
            # USE LOCAL OLLAMA SERVER
            host='http://localhost:11434'
            
            # USE API OLLAMA SERVER
            # host='https://ollama.com',
            # headers={'Authorization': 'Bearer ' + OLLAMA_API_KEY}
            )
    # ------------------------------------------------------------------
    def load_model(self):
        """
        Vérifie qu’Ollama tourne, puis lance le modèle avec subprocess
        pour le 'réveiller', puis crée le client Python.
        """

        # 1. Vérifier que Ollama tourne
        try:
            subprocess.run(
                ["ollama", "list"],
                check=True,
                capture_output=True,
                text=True
            )
        except Exception as e:
            raise RuntimeError("Ollama n'est pas lancé. Lance 'ollama serve'.") from e

        # 2. Vérifier que le modèle existe
        models = subprocess.run(
            ["ollama", "list"],
            capture_output=True,
            text=True
        ).stdout

        if self.model_name not in models:
            raise RuntimeError(
                f"Le modèle {self.model_name} n'existe pas dans Ollama.\n"
                f"Crée-le avec : ollama create {self.model_name} -f Modelfile"
            )

        # 3. Réveiller le modèle (ta demande !)
        print(f"Lancement du modèle {self.model_name}...")
        subprocess.run(
            ["ollama", "run", self.model_name],
            capture_output=True,
            text=True
        )
        print("Modèle réveillé !")

        # 4. Initialiser le client Python
        self.client = Client(host="http://localhost:11434")

        print("Client Python prêt.")

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
        response = self.client.generate(
            model=self.model_name,
            prompt=prompt,
            options={
                "temperature": self.temperature,
                "num_predict": self.max_new_tokens,
            }
        )
        return response.get("response", "")


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
        except Exception as e:
            print(f"⚠️ Erreur de parsing JSON: {e}")
            print(f"Réponse brute du modèle: {raw_text[:200]}...")
            raw = {
                "response": "Peux-tu reformuler en précisant l'objet cible ?",
                "target_object": None,
                "obstacles": [],
                "status": "ambiguous",
                "confidence": 0.4,
            }
        data = self.formater.postprocess(raw, user_cmd)
        return Extraction(**data)

# --------------------------------------------------------------------------------------
#  Exécution directe du script
# --------------------------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyse de commandes NL -> labels connus (labelDict)")
    parser.add_argument("command", type=str, nargs="*", help="Commande en langage naturel")
    parser.add_argument("--model", dest="BIRA", default="BIRA", help="Nom du modèle Ollama à utiliser")
    args = parser.parse_args()

    #cmd = " ".join(args.command).strip()
    print("Ajouter votre commande: ")
    cmd = input()
    mgr = SLM_Manager(model_name=args.BIRA)
    mgr.load_model()

    if not cmd:
        print('Exemple : python SLM_Manager.py "BIRA, donne moi la pomme bleu"')
    else:
        extraction = mgr.analyze_command(cmd)
        print(json.dumps(extraction.to_payload(), ensure_ascii=False))
        print(f"status={extraction.status} confidence={extraction.confidence}")
