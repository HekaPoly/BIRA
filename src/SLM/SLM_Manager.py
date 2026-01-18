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
- python -m SLM.SLM_Manager est utilisé pour ouvrir le slm.
"""
from __future__ import annotations
from typing import Optional
from pathlib import Path
import os
import json
import argparse
import subprocess

from ollama import Client

from SLM.Formater import Formater
from SLM.Extraction import Extraction

# Chargement optionnel de la configuration (.env) sans imposer la dépendance
try:
    from dotenv import load_dotenv

    load_dotenv()
except Exception:
    pass

# Import TensorRT si disponible (sinon fallback Ollama)
try:
    from SLM.tensorRT import TensorRTInferenceEngine
except Exception:
    TensorRTInferenceEngine = None

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
        max_new_tokens: int = 192,
        temperature: float = 0.4,
        prefer_tensorrt: Optional[bool] = None,
        tensorrt_engine_dir: Optional[str] = None,
        tensorrt_fallback_to_ollama: Optional[bool] = None,
    ):
        self.model_name = model_name
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.formater = formater or Formater()
        self.client = Client(host='http://localhost:11434')
        # Simple state tracking between calls
        self.previous_state: Optional[Extraction] = None
        self.current_state: Optional[Extraction] = None
        # Minimal chat history for free-chat mode (optional)
        self.chat_history = []
        # Switch to lock into structured mode once "bira" is invoked
        self.bira_called_once = False
        # TensorRT - activable via .env ou paramètres
        self.prefer_tensorrt = (
            os.getenv("USE_TENSORRT", "").lower() == "true"
            if prefer_tensorrt is None
            else prefer_tensorrt
        )
        self.tensorrt_fallback_to_ollama = (
            os.getenv("TENSORRT_FALLBACK_TO_OLLAMA", "true").lower() == "true"
            if tensorrt_fallback_to_ollama is None
            else tensorrt_fallback_to_ollama
        )
        default_engine_dir = Path(__file__).resolve().parent / "tensorRT" / "tensorrt_models" / "engines"
        self.tensorrt_engine_dir = Path(tensorrt_engine_dir or default_engine_dir)
        self.trt_engine: Optional[TensorRTInferenceEngine] = None
        self.trt_ready: bool = False

        if self.prefer_tensorrt and TensorRTInferenceEngine is None:
            print("TensorRTInferenceEngine introuvable, retour à Ollama.")
            self.prefer_tensorrt = False

        if self.prefer_tensorrt:
            self._init_tensorrt_engine()
    # ------------------------------------------------------------------
    def _init_tensorrt_engine(self) -> None:
        """
        Initialise le moteur TensorRT si l'engine est present, sinon bascule sur Ollama.
        """
        try:
            self.trt_engine = TensorRTInferenceEngine(engine_dir=str(self.tensorrt_engine_dir))
            self.trt_ready = bool(self.trt_engine.load_engine())
            if self.trt_ready:
                print(f"TensorRT pret (engine: {self.tensorrt_engine_dir})")
            elif not self.tensorrt_fallback_to_ollama:
                raise RuntimeError("Engine TensorRT introuvable")
        except Exception as exc:
            self.trt_ready = False
            msg = f"TensorRT indisponible ({exc}). "
            if self.tensorrt_fallback_to_ollama:
                msg += "Fallback Ollama active."
            print(msg)

    # ------------------------------------------------------------------
    def load_model(self):
        """
        Vérifie qu’Ollama tourne, puis lance le modèle avec subprocess
        pour le 'réveiller', puis crée le client Python.
        """

        if self.prefer_tensorrt and self.trt_ready and not self.tensorrt_fallback_to_ollama:
            print("TensorRT pret, lancement Ollama skip (fallback desactive).")
            return

        # 1. Vérifier que Ollama tourne
        try:
            subprocess.run(
                ["ollama", "list"],
                check=True,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
            )
        except Exception as e:
            raise RuntimeError("Ollama n'est pas lancé. Lance 'ollama serve'.") from e

        # 2. Vérifier que le modèle existe
        models = subprocess.run(
            ["ollama", "list"],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
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
            text=True,
            encoding="utf-8",
            errors="replace",
        )
        print("Modèle réveillé !")

        # 4. Initialiser le client Python
        self.client = Client(host="http://localhost:11434")

        print("Client Python prêt.")

    # ------------------------------------------------------------------
    def generate_response(self, prompt: str) -> str:
        """
        Envoie un prompt au modele Ollama et recupere la reponse textuelle complete.

        Parameters
        ----------
        prompt : str
            Instruction ou question a traiter.

        Returns
        -------
        str
            Sortie textuelle complete produite par le modele.
        """
        # TensorRT prioritaire si active et pret
        if self.prefer_tensorrt and self.trt_ready and self.trt_engine:
            try:
                resp = self.trt_engine.chat(
                    messages=[{"role": "user", "content": prompt}],
                    max_new_tokens=self.max_new_tokens,
                    temperature=self.temperature,
                )
                content = resp.get("message", {}).get("content", "")
                if content:
                    return content
            except Exception as exc:
                print(f"TensorRT a echoue ({exc}), fallback Ollama.")
                if not self.tensorrt_fallback_to_ollama:
                    raise

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
    def free_chat(self, user_text: str) -> str:
        """
        Mode libre : si l'entree ne commence pas par 'bira', on repond sans extraction JSON.
        """
        self.chat_history.append({"role": "user", "content": user_text})
        messages = [{"role": "system", "content": "Tu es BIRA, reponds librement au format texte."}] + self.chat_history

        resp = None
        if self.prefer_tensorrt and self.trt_ready and self.trt_engine:
            try:
                resp = self.trt_engine.chat(
                    messages=messages,
                    max_new_tokens=self.max_new_tokens,
                    temperature=self.temperature,
                )
            except Exception as exc:
                print(f"TensorRT a echoue ({exc}), fallback Ollama.")
                if not self.tensorrt_fallback_to_ollama:
                    raise

        if resp is None:
            resp = self.client.chat(
                model=self.model_name,
                messages=messages,
                options={
                    "temperature": self.temperature,
                    "num_predict": self.max_new_tokens,
                }
            )

        content = resp["message"]["content"] if isinstance(resp, dict) else resp.message.content
        # Conserver l'historique cote assistant
        self.chat_history.append({"role": "assistant", "content": content})
        return content

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
        extraction = Extraction(**data)

        # Update state tracking (previous -> current)
        self.previous_state = self.current_state
        self.current_state = extraction

        return extraction

    # ------------------------------------------------------------------
    def get_states(self) -> dict:
        """
        Retourne un snapshot simple de l'état courant et du précédent.
        """
        return {
            "current": self.current_state.to_payload() if self.current_state else None,
            "previous": self.previous_state.to_payload() if self.previous_state else None,
        }

# --------------------------------------------------------------------------------------
#  Exécution directe du script
# --------------------------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyse de commandes NL -> labels connus (labelDict)")
    parser.add_argument("command", type=str, nargs="*", help="Commande en langage naturel")
    parser.add_argument("--model", dest="model", default="BIRA", help="Nom du modèle Ollama à utiliser")
    args = parser.parse_args()

    # Créer et charger le modèle une seule fois
    mgr = SLM_Manager(model_name=args.model)
    mgr.load_model()

    small_talk = {"merci", "thanks", "salut", "hello", "allo", "bonjour", "bonsoir", "ca va", "ça va"}

    while True:
        try:
            print("Ajouter votre commande (ou 'quit' pour sortir) : ")
            cmd = input().strip()
            if not cmd:
                print('Exemple : BIRA, donne moi la pomme bleue')
                continue
            if cmd.lower() in {"quit", "exit"}:
                break

            cmd_lower = cmd.lower()
            starts_with_bira = cmd_lower.startswith("bira")
            is_chat_prefix = cmd_lower.startswith("chat:")

            # Routing simplifi?:
            # - small talk -> chat libre
            # - "chat:" -> chat libre
            # - commande qui commence par "bira" -> extraction structur?e
            # - sinon -> chat libre
            if any(tok in cmd_lower for tok in small_talk):
                answer = mgr.free_chat(cmd)
                print(answer)
            elif is_chat_prefix:
                chat_msg = cmd[len("chat:"):].strip() or cmd
                answer = mgr.free_chat(chat_msg)
                print(answer)
            elif starts_with_bira:
                extraction = mgr.analyze_command(cmd)
                print(json.dumps(extraction.to_payload(), ensure_ascii=False))
                print(f"status={extraction.status} confidence={extraction.confidence}")
            else:
                answer = mgr.free_chat(cmd)
                print(answer)

        except KeyboardInterrupt:
            print("\nArrêt demandé. Bye.")
            break
