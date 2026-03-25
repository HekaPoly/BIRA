
from __future__ import annotations
from typing import Optional
import json
import argparse
from cv_viewer import labels
from ollama import Client
import subprocess


def _parse_first_json(text: str) -> dict:
    """Parse the first complete JSON object from a string. Handles extra text or multiple JSON blobs."""
    text = text.strip()
    start = text.find("{")
    if start == -1:
        raise json.JSONDecodeError("No JSON object found", text, 0)
    depth = 0
    in_string = False
    escape = False
    quote = None
    for i in range(start, len(text)):
        c = text[i]
        if escape:
            escape = False
            continue
        if c == "\\" and in_string:
            escape = True
            continue
        if in_string:
            if c == quote:
                in_string = False
            continue
        if c in ('"', "'"):
            in_string = True
            quote = c
            continue
        if c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                return json.loads(text[start : i + 1])
    raise json.JSONDecodeError("Unbalanced braces", text, start)


# SYSTEM_BIRA = """
# Tu es BIRA, un assistant amical, enthousiaste et utile destiné à interpréter des commandes de préhension d’objets. 
# Tu dois toujours répondre EXCLUSIVEMENT en JSON valide.

# RÈGLES FONDAMENTALES :

# 3. response :
#    - Fournir une phrase de confirmation en première personne, avec un ton amical et enthousiaste.
#    - Reformuler l’action de manière active en incluant l’objet identifié.
#    - Ne jamais reprendre textuellement la commande de l’utilisateur.
#    - Exemples :
#        "D’accord, je saisis [objet]."
#        "Compris, j’attrape [objet] pour toi."

#    - Si la commande est trop vague (objet imprécis, terme flou), demander des clarifications avec enthousiasme.
#    - Si la demande concerne un groupe d’objets (ex. "les affaires", "les trucs"), demander des précisions.
#    - Si les objets mentionnés ne sont pas être détectés ou ne semblent pas présents, demander des clarifications.
#    - Exemples :
#        "Je suis ravi de t’aider ! Peux-tu préciser de quel objet il s’agit ?"
#        "Je suis ravi de t’aider, mais je ne parviens pas à repérer l’objet. Peux-tu me le décrire davantage ?"

       
# FORMAT DE SORTIE :
# Toujours renvoyer un JSON strictement valide, même si certains champs sont null ou vides.

# IL EST IMPERATIF DE SUIVRE STRICTEMENT LA STRUCTURE CI-DESSOUS, SOIT SUR QUE LE MODE EST PRESENT (GENRE IL FAUT VRAIMENT QUE LE MODE SOIT PRESENT SINON LE CODE PLANTE):
# Structure :
# {
#   "response": "...",
#   "mode": "confirmation" | "clarification" | "stop",
# }
# """

SYSTEM_BIRA = """
You are BIRA, a friendly, enthusiastic, and helpful assistant designed to interpret object-grasping commands.
The user will give instructions and your task is to find which action to do based on the detected objects. 
If the object is not detected, you must ask for clarification.
You must ALWAYS respond EXCLUSIVELY in valid JSON.

FUNDAMENTAL RULES:

Response:
   - Provide a confirmation sentence in the first person, with a friendly and enthusiastic tone.
   - Rephrase the action in an active way, including the identified object.
   - Examples:
       "Alright, I’m [ACTION] the [OBJECT]."
       "Got it, I’ll [ACTION] the [OBJECT] for you."

   - If the command is too vague (imprecise object, unclear term, unclear action), ask for clarification enthusiastically.
   - If the request refers to a group of objects (e.g., "the stuff", "the things"), ask for clarification.
   - If the mentioned objects cannot be detected or do not seem to be present, ask for clarification.
   - If the user whan to eat a specific [OBJECT], you must confirm by responding that you will bring the food [OBJECT]
   - Examples:
       "I’m happy to help, but I do not see [OBJECT]. Could you specify which one you mean?"
       "I’m happy to help, but I can’t seem to identify the [OBJECT]. Could you describe it a bit more?"
       "I’m happy to help, but I can’t seem to identify the object. Could you describe it a bit more?"

OUTPUT FORMAT:
Always return strictly valid JSON, even if some fields are null or empty.

IT IS IMPERATIVE TO STRICTLY FOLLOW THE STRUCTURE BELOW, MAKE SURE THAT THE MODE IS PRESENT
Structure:
[{
  "response": "...",
  "mode": "confirmation" | "clarification" | "stop"
},]
"""

class SLM_Manager:

    def __init__(
        self,
        model_name: str = "qwen3:1.7b",                       
        max_new_tokens: int = 500,
        temperature: float = 0.3,
        mode: str = "local",
        api_key: Optional[str] = None,
    ):
        self.model_name = model_name
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.mode = mode
        self.history = [{"role": "system", "content": SYSTEM_BIRA}]
        self.detections = None
        self.transcription = None
        self.prompt = None
        
        if mode == "cloud":
            if not api_key:
                raise ValueError("API key must be provided for cloud mode.")
            
            self.client = Client(
                host="https://ollama.com",
                headers={'Authorization': f'Bearer {api_key}'}
            )
            
        else:
            self.client = Client(host="http://localhost:11434")
    def reset_conversation(self) -> None:
        self.history = [{"role": "system", "content": SYSTEM_BIRA}]
        self.detections = None
        self.transcription = None
        self.prompt = None

    def set_transcription(self, transcription: str) -> None:
        self.transcription = transcription

    def set_detections(self, detection_labels: Optional[list[int]] = None, detected_objects: Optional[list] = None) -> None:
        resolved: list[str] = []

        if detection_labels:
            resolved.extend(labels.labelDict[lid] for lid in detection_labels if lid in labels.labelDict)

        if detected_objects:
            for obj in detected_objects:
                raw = getattr(obj, "raw_label", None)
                if raw is None:
                    continue
                key = int(raw)
                if key in labels.labelDict:
                    resolved.append(labels.labelDict[key])

        # Keep order while removing duplicates
        self.detections = list(dict.fromkeys(resolved)) if resolved else None

    def _build_prompt(self, transcription: str, detections: list[str]) -> str:
        return (
            "Analyze the following command and decide on the action to take: "
            f"'{transcription}'. The detected objects are: {detections}. "
            "Respond in JSON according to the rules."
        )

    def _parse_model_response(self, raw_response: str) -> dict:
        try:
            parsed = json.loads(raw_response)
        except json.JSONDecodeError:
            parsed = _parse_first_json(raw_response)

        if isinstance(parsed, list) and parsed:
            parsed = parsed[-1]

        if not isinstance(parsed, dict):
            raise ValueError("Model response is not a JSON object")

        mode = parsed.get("mode", "clarification")
        text = parsed.get("response", "I didn't understand. Could you repeat?")
        return {"response": str(text), "mode": str(mode)}

    def run_inference(self) -> dict:
        transcription = self.transcription
        detections = self.detections

        if not transcription:
            return {"response": "I didn't hear a command. Could you repeat?", "mode": "clarification"}

        if not detections:
            return {"response": "I don't see any relevant object. Could you clarify?", "mode": "clarification"}

        self.prompt = self._build_prompt(transcription, detections)
        raw_response = self.generate_response(self.prompt)

        try:
            response = self._parse_model_response(raw_response)
        except (json.JSONDecodeError, ValueError) as err:
            print("SLM JSON parse error:", err)
            response = {"response": "I didn't understand. Could you repeat?", "mode": "clarification"}

        if response["mode"] in {"stop"}:
            self.reset_conversation()

        return response

                

    # ------------------------------------------------------------------
    def load_model(self):
        """
        Vérifie qu’Ollama tourne, puis lance le modèle avec subprocess
        pour le 'réveiller', puis crée le client Python.
        """
        
        if self.mode == "cloud":
            print(f"Vérification du modèle '{self.model_name}' via Ollama Cloud...")
            try:
                _ = self.client.generate(
                    model=self.model_name,
                    prompt="ping",
                    options={"num_predict": 5}
                )
            except Exception as e:
                raise RuntimeError(
                    "Impossible de contacter Ollama Cloud. "
                    "Vérifie ta clé API et le nom du modèle."
                ) from e

            print("Modèle accessible via l'API Cloud.")
            print("Client Python prêt.")
            return
        

        # 1. Vérifier que Ollama tourne
        print("Vérification du serveur Ollama local...")
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

        # 4. Initialiser / revalider le client Python local
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
        self.history.append(
            {"role": "user", "content": prompt}
        )
        response = self.client.chat(
            model=self.model_name,
            messages=self.history,
                options={
                    "temperature": self.temperature,
                    "num_predict": self.max_new_tokens,
                    "format": "json",
                }
        )
                 
        return response["message"]["content"]




# --------------------------------------------------------------------------------------
#  Exécution directe du script
# --------------------------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test SLM_Manager inference locally")
    parser.add_argument("--model", dest="BIRA", default="qwen3:1.7b", help="Model name to use")
    parser.add_argument("--mode", default="local", choices=["local", "cloud"], help="Deployment mode")
    parser.add_argument("--api-key", default="", help="API key for cloud mode")
    
    args = parser.parse_args()

    # Initialize
    print(f"Initializing SLM_Manager (mode: {args.mode}, model: {args.BIRA})")
    mgr = SLM_Manager(model_name=args.BIRA, mode=args.mode, api_key=args.api_key if args.api_key else None)
    
    try:
        mgr.load_model()
    except Exception as e:
        print(f"Warning: Model load failed: {e}")
        print("Falling back to assuming the client is reachable...")
        
    print("\n--- Interactive SLM Test ---")
    print("Type 'q' or 'quit' to exit.")
    
    while True:
        try:
            transcription = input("\nEnter user command (transcription): ").strip()
            if transcription.lower() in ["q", "quit"]:
                break
                
            detections_input = input("Enter detected objects (comma-separated labels, e.g. '0, 41, 39' for person, cup, bottle): ").strip()
            if detections_input.lower() in ["q", "quit"]:
                break
                
            # Parse detections (dummy mapping of YOLO COCO classes)
            detection_labels = []
            if detections_input:
                try:
                    detection_labels = [int(x.strip()) for x in detections_input.split(",") if x.strip()]
                except ValueError:
                    print("Error: Detected objects must be integers.")
                    continue
            
            # Setup State
            mgr.set_transcription(transcription)
            mgr.set_detections(detection_labels=detection_labels)
            
            # Run Inference
            print("\n[Running inference...]")
            result = mgr.run_inference()
            
            # Display Result
            print("\n--- Result ---")
            print(f"Response: {result.get('response')}")
            print(f"Mode:     {result.get('mode')}")
            print("-------------")
            
        except KeyboardInterrupt:
            break
            
    print("\nExiting SLM test.")