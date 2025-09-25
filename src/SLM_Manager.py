"""
SLM Manager par Marcelo Zevallos Gavidia et Ryan Ajakane
===========

Ce module encapsule l’appel à un Small Language Model (SLM) pour analyser une
commande en langage naturel et extraire trois informations clés :
- response        : feedback vocal à renvoyer à l’utilisateur,
- target_object   : label de l’objet cible (doit exister dans labelDict côté vision),
- obstacles       : liste de labels d’objets à éviter (idem).

Le flux est :
    user_cmd (str) --(Formatter.build_prompt)--> prompt
    prompt --(SLM generate)--> raw_text
    raw_text --(Formatter.parse)--> dict brut
    brut --(Formatter.postprocess)--> dict normalisé {response, target_object, obstacles, status, confidence}
    --> dataclass Extraction

Notes :
- Le `Formatter` est responsable du prompt, du parsing (JSON), de la normalisation
  via vos labels connus et de la gestion des cas incomplets/ambigus.
- Le SLM est chargé/déchargé dans cette classe et reste réutilisable entre appels.
"""

# TODO: Tester la classe SLM_Manager
from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Any, Optional
import json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import argparse

# TODO: compléter la classe Formatter
# from src.Formater import Formatter

# --------------------------------------------------------------------------------------
# Données structurées
# -------------------------------------------------------------------------------------
@dataclass
class Extraction:
    """
    Sortie structurée renvoyée par `SLM_Manager.analyze_command`.

    Attributes
    ----------
    response : str
        Feedback vocal prêt à énoncer à l’utilisateur.
    target_object : Optional[str]
        Label exact de l’objet cible (ex. 'car'), ou None si absent/indéterminé.
    obstacles : List[str]
        Liste des labels d’objets obstacles (peut être vide).
    status : str
        Statut de l’analyse : "ok" | "missing_target" | "ambiguous" | "empty".
    confidence : float, optional
        Score grossier (0..1) indiquant la confiance, par défaut 0.5.
    """
    response: str
    target_object: Optional[str]
    obstacle: Optional[str]
    status: str
    confidence: float = 0.5

    def to_payload(self) -> Dict[str, Any]:
        """
        Représentation simple (dict) attendue par les couches aval.
        """
        return {
            "response": self.response,
            "target_object": self.target_object,
            "obstacle": self.obstacle,
        }

# --------------------------------------------------------------------------------------
# Gestionnaire SLM
# --------------------------------------------------------------------------------------
class SLM_Manager:
    """
    Gère le modèle de langage (chargement, génération) et orchestre l’analyse.

    Paramètres
    ----------
    model_name : str
        Identifiant du modèle Hugging Face (ou chemin local).
    device : Optional[str]
        "cuda" si disponible sinon "cpu". Si None, auto-détection.
    formatter : Optional[Formatter]
        Instance de Formatter. Si None, une instance par défaut est créée.
    max_new_tokens : int
        Longueur max de génération du SLM.
    temperature : float
        Température d’échantillonnage (créativité).

    Usage
    -----
    >>> mgr = SLM_Manager()
    >>> mgr.load_model()
    >>> ext = mgr.analyze_command("Attrape la car et évite le bus.")
    >>> print(ext.to_payload())
    """

    def __init__(self,
                 model_name="meta-llama/Llama-3.2-1B",
                 device="cuda" if torch.cuda.is_available() else "cpu",
                 # formatter: Optional[Formatter] = None,
                 max_new_tokens: int = 192,
                 temperature: float = 0.4):
        """
        Initialize the SLM Manager with a Llama 3.2 model.

        Args:
            model_name (str): Hugging Face model ID or local path.
            device (str): Device to run the model on ('cuda' or 'cpu').
        """
        self.model_name = model_name
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        # self.formatter = Formatter()
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.model = None
        self.model = None
        self.tokenizer = None

    # -------------------------- Modèle --------------------------
    def load_model(self):
        """
        Charge le tokenizer et le modèle en mémoire (sur GPU si dispo).
        """
        print(f"Loading model: {self.model_name} on {self.device}")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
            low_cpu_mem_usage=True,
        ).to(self.device)
        print("Model loaded successfully.")

    # -------------------------- Génération --------------------------
    @torch.inference_mode()
    def generate_response(self, prompt: str) -> str:
        """
        Lance une génération de texte à partir d'un prompt.

        Parameters
        ----------
        prompt : str
            Prompt d'instruction construit par le Formatter.

        Returns
        -------
        str
            Texte complet renvoyé par le modèle (incluant potentiellement du bruit).
        """
        if not self.model or not self.tokenizer:
            raise ValueError("Model not loaded. Call load_model() first.")

        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        outputs = self.model.generate(
            **inputs,
            max_new_tokens=self.max_new_tokens,
            temperature=self.temperature,
            do_sample=True,
            eos_token_id=self.tokenizer.eos_token_id
        )
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return response

    # -------------------------- Chat debug --------------------------
    def chat(self):
        """
        Boucle interactive (debug). Tape 'quit' pour sortir.
        """
        print("Starting chat with Llama 3.2. Type 'quit' to exit.")
        while True:
            user_input = input("You: ")
            if user_input.lower() == "quit":
                break
            response = self.generate_response(user_input)
            print(f"Llama 3.2: {response}")

    # -------------------------- Chaîne complète --------------------------
    def analyze_command(self, user_cmd: str) -> Extraction:
        """
        Analyse une commande NL et renvoie une `Extraction`.

        Étapes :
          1) Construction du prompt via `Formatter.build_prompt(user_cmd)`.
          2) Appel modèle via `generate_response(prompt)`.
          3) Parsing robuste du premier JSON via `Formatter.parse`.
          4) Normalisation + statut via `Formatter.postprocess`.
          5) Conversion en dataclass `Extraction`.

        Parameters
        ----------
        user_cmd : str
            Commande en langage naturel (ex. "Attrape la car et évite le bus.")

        Returns
        -------
        Extraction
            Résultat normalisé et prêt à utiliser.
        """
        prompt = self.formatter.build_prompt(user_cmd)
        raw_text = self._generate(prompt)
        try:
            raw = self.formatter.parse(raw_text)
        except Exception:
            raw = {
                   "response": "Peux-tu reformuler en précisant l’objet cible ?",
                   "target_object": None,
                   "obstacles": []
                   }
        data = self.formatter.postprocess(raw, user_cmd)
        return Extraction(**data)

# --------------------------------------------------------------------------------------
# Main temporaire
# --------------------------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyse NL -> labels connus (labelDict)")
    parser.add_argument("command", type=str, nargs="*", help="Commande en langage naturel")
    parser.add_argument("--model", dest="model", default="meta-llama/Llama-3.2-1B")
    args = parser.parse_args()

    cmd = " ".join(args.command).strip()
    mgr = SLM_Manager(model_name=args.model)
    mgr.load_model()

    if not cmd:
        print("Exemple: python slm_manager.py 'Attrape la car et évite le bus.'")
    else:
        extraction = mgr.analyze_command(cmd)
        print(json.dumps(extraction.to_payload(), ensure_ascii=False))
        print(f"status={extraction.status} confidence={extraction.confidence}")
