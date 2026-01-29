
from __future__ import annotations
from typing import Optional
import json
import argparse
from ollama import Client
import subprocess

from .bira_componant import BiraComponent


SYSTEM_BIRA = """
Tu es BIRA, un assistant amical, enthousiaste et utile destiné à interpréter des commandes de préhension d’objets. 
Tu dois toujours répondre EXCLUSIVEMENT en JSON valide.

RÈGLES FONDAMENTALES :

3. response :
   - Fournir une phrase de confirmation en première personne, avec un ton amical et enthousiaste.
   - Reformuler l’action de manière active en incluant l’objet identifié.
   - Ne jamais reprendre textuellement la commande de l’utilisateur.
   - Exemples :
       "D’accord, je saisis [objet]."
       "Compris, j’attrape [objet] pour toi."

   - Si la commande est trop vague (objet imprécis, terme flou), demander des clarifications avec enthousiasme.
   - Si la demande concerne un groupe d’objets (ex. "les affaires", "les trucs"), demander des précisions.
   - Si les objets mentionnés ne sont pas être détectés ou ne semblent pas présents, demander des clarifications.
   - Exemples :
       "Je suis ravi de t’aider ! Peux-tu préciser de quel objet il s’agit ?"
       "Je suis ravi de t’aider, mais je ne parviens pas à repérer l’objet. Peux-tu me le décrire davantage ?"

       
FORMAT DE SORTIE :
Toujours renvoyer un JSON strictement valide, même si certains champs sont null ou vides.

Structure :
{
  "response": "...",
  "mode": "confirmation" | "clarification" | "stop",
}
"""

class SLM_Manager(BiraComponent):

    def __init__(
        self,
        model_name: str = "BIRA",                       
        max_new_tokens: int = 128,
        temperature: float = 0.3,
        mode: str = "local",
        api_key: str = None,
        mediator = None,
    ):
        super().__init__("SLM_Manager", mediator=mediator)
        self.model_name = model_name
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.mode = mode
        
        if mode == "cloud":
            if api_key is None:
                raise ValueError("API key must be provided for cloud mode.")
            
            self.client = Client(
                host="https://ollama.com",
                headers={'Authorization': f'Bearer {api_key}'}
            )
            
        else:
            self.client = Client(host="http://localhost:11434")
            
    async def receive(self, message):
        if message.keys().__contains__('start'):
            self.load_model()
            
        elif message.keys().__contains__('wake_up'):
            self.history = [
                {"role": "system", "content": SYSTEM_BIRA}
            ]
            self.detections = None
            self.prompt = None
            
        elif message.keys().__contains__('detect_objects_ready'):
            self.detections = message['detect_objects_ready']
            self.mediator.send(self, "generate_response")
            
        elif message.keys().__contains__('transcription_ready'):
            self.transcription = message['transcription_ready']
            self.prompt = f"Analyse la commande suivante et décide de l'action à entreprendre : '{self.transcription}'. Les objets détectés sont : {json.dumps(self.detections)}. Réponds en JSON selon les règles."
            self.mediator.send(self, "generate_response")
            
        elif message.keys().__contains__('generate_response'):
            if self.prompt is None or self.detections is None:
                return
            
            response = self.generate_response(self.prompt)
            await self.mediator.send(self, {"response_ready": response})

            if response["mode"] == "confirmation":
                self.mediator.send(self, {"eating": None})
                # Execute eating action
                # Verifier expression 
                self.mediator.send(self, {"sleep": None})
                
            elif response["mode"] == "clarification": 
                self.mediator.send(self, {"transcription_request": None})
            elif response["mode"] == "stop":
                self.mediator.send(self, {"sleep": None})
                self.images = None
                self.prompt = None
                

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
            headers={'Authorization': 'Bearer ' + '747aadbe08f24aa5b2898948925dd80a.0hw2fdvQjib4R9VNLbxZVzHw'}
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
    parser = argparse.ArgumentParser(description="Analyse de commandes NL -> labels connus (labelDict)")
    parser.add_argument("command", type=str, nargs="*", help="Commande en langage naturel")
    parser.add_argument("--model", dest="BIRA", default="BIRA", help="Nom du modèle Ollama à utiliser")
    args = parser.parse_args()

    #cmd = " ".join(args.command).strip()
    print("Ajouter votre commande: ")
    cmd = input()
    mgr = SLM_Manager(model_name=args.BIRA)
    mgr.load_model()