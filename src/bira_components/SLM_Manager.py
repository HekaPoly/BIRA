
from __future__ import annotations
from typing import Optional
import json
import argparse
from cv_viewer import labels
from ollama import Client
import subprocess

from .bira_component import BiraComponent


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

class SLM_Manager(BiraComponent):

    def __init__(
        self,
        model_name: str = "gpt-oss:120b",                       
        max_new_tokens: int = 500,
        temperature: float = 0.3,
        mode: str = "cloud",
        api_key: str = "",
        mediator = None,
    ):
        super().__init__("SLM_Manager", mediator=mediator)
        self.model_name = model_name
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.mode = mode
        self.history = [{"role": "system", "content": SYSTEM_BIRA}]
        self.detections = None
        self.transcription = None
        self.prompt = None
        
        if mode == "cloud":
            if api_key is None:
                raise ValueError("API key must be provided for cloud mode.")
            
            self.client = Client(
                host="https://ollama.com",
                headers={'Authorization': f'Bearer {api_key}'}
            )
            
        else:
            self.client = Client(host="http://localhost:11434")
            
    def receive(self, message):
        print("SLM_Manager received message:", message)
        if message.keys().__contains__('start'):
            self.load_model()
            
        elif message.keys().__contains__('wake_up'):
            self.history = [
                {"role": "system", "content": SYSTEM_BIRA}
            ]
            self.detections = None
            self.transcription = None
            self.prompt = None
            
        elif message.keys().__contains__('detect_objects_ready'):
            print("SLM detect_objects_ready")
            # Prefer YOLO labels from computer_vision; fallback to ZED object_list (often empty after retrieve_objects)
            if self.detections is None:
                if message.get("detection_labels") is not None:
                    self.detections = [labels.labelDict[lid] for lid in message["detection_labels"] if lid in labels.labelDict]
                else:
                    self.detections = [labels.labelDict[int(obj.raw_label)] for obj in message['detect_objects_ready'].object_list]
            else:
                # Merge new detections with existing ones
                new_detections = []
                if message.get("detection_labels") is not None:
                    new_detections = [labels.labelDict[lid] for lid in message["detection_labels"] if lid in labels.labelDict]
                else:
                    new_detections = [labels.labelDict[int(obj.raw_label)] for obj in message['detect_objects_ready'].object_list]
                
                for det in new_detections:
                    if det not in self.detections:
                        self.detections.append(det)
            print("SLM Detections: ", self.detections)
             
            
        elif message.keys().__contains__('transcription_ready'):
            print("SLM transcription_ready")

            self.transcription = message['transcription_ready']
            
        elif message.keys().__contains__('generate_response'):
            print("SLM generate_response")
            print("Input :", self.transcription, self.detections)
            transcription = getattr(self, "transcription", None)
            detections = getattr(self, "detections", None)
            if transcription is None or detections is None:
                self.mediator.send(self, "generate_response")   
                return
            self.mediator.clear()
            # Create prompt
            print("transcription: ", transcription)
            print("detections: ", detections)
            #self.prompt = f"Analyse la commande suivante et décide de l'action à entreprendre : '{transcription}'. Les objets détectés sont : {detections}. Réponds en JSON selon les règles."
            self.prompt = f"Analyze the following command and decide on the action to take: '{transcription}'. The detected objects are: {detections}. Respond in JSON according to the rules."

            print( "SLM Prompt: ", self.prompt)
            response = self.generate_response(self.prompt)
            print("SLM: ", response)
            try:
                response = json.loads(response)[-1]
            except json.JSONDecodeError as e:
                print("SLM JSON parse error:", e)
                response = {"response": "I didn't understand. Could you repeat?", "mode": "clarification"}
            self.mediator.send(self, {"speak_request": response["response"]})
            print(response)

            if response["mode"] == "confirmation":
                self.mediator.send(self, {"eating": None})
                # Execute eating action
                # Verifier expression
                self.transcription = None
                self.detections = None
                self.images = None
                self.prompt = None
                self.history = [{"role": "system", "content": SYSTEM_BIRA}]
                self.mediator.send(self, {"sleep_mode": None})

            elif response["mode"] == "clarification": 
                self.transcription = None
                self.mediator.send(self, {"transcription_request": None})
                self.mediator.send(self, "detect_objects_request")
                self.mediator.send(self, "generate_response")
            elif response["mode"] == "stop":
                self.mediator.send(self, {"sleep_mode": None})
                self.transcription = None
                self.detections = None
                self.images = None
                self.prompt = None
                self.history = [{"role": "system", "content": SYSTEM_BIRA}]

                

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

        # 4. Initialiser le client Python
        self.client = Client(
            # USE LOCAL OLLAMA SERVER
            # host='http://localhost:11434'
            
            # USE API OLLAMA SERVER
            host='https://ollama.com',
            headers={'Authorization': 'Bearer ' + 'e96dd37a91844edab472f73e0b2b31a5.12O6_rnlOQaNsUCczRfqFsib'}
            )

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
            model="gpt-oss:120b",
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
    parser = argparse.ArgumentParser(description="Analyse de commandes NL -> labels connus (labelDict)")
    parser.add_argument("command", type=str, nargs="*", help="Commande en langage naturel")
    parser.add_argument("--model", dest="BIRA", default="BIRA", help="Nom du modèle Ollama à utiliser")
    args = parser.parse_args()

    #cmd = " ".join(args.command).strip()
    print("Ajouter votre commande: ")
    cmd = input()
    mgr = SLM_Manager(model_name=args.BIRA)
    mgr.load_model()