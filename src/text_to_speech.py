import pyttsx3
import random

class TextToSpeech:
    def __init__(self, voice: str = None, rate: int = 130, volume: float = 0.8):
        self.engine = pyttsx3.init()
        self.rate = rate
        self.volume = volume
        self.engine.setProperty('rate', self.rate)
        self.engine.setProperty('volume', self.volume)
        self.engine.setProperty('voice', voice)
        

    def speak(self, text: str):
        MAX_LEN = 180

        for i in range(0, len(text), MAX_LEN):
            chunk = text[i:i+MAX_LEN]
            chunk = chunk.replace(",", " ").replace(";", " ")

            self.engine.setProperty('rate', self.rate + random.randint(-10, 10))
            self.engine.say(chunk)
            self.engine.runAndWait()
            self.engine.stop()

    def stop(self):
        self.engine.stop()

if __name__== "__main__":
    print("Testing TTS...", flush=True)
    text = "Voici les objets que j'ai détectés : person, ordinateur portable, ordinateur portable, clavier, ordinateur portable, ordinateur portable, ordinateur portable, voiture, oiseau, ordinateur portable, ordinateur portable, ordinateur portable, chat, télévision, télévision, télévision, ordinateur portable, person, person, person, person, person, ordinateur portable, télévision, télévision, télévision, person, person, télévision, person, person, télévision, person, person, télévision, télévision, person, ordinateur portable, ordinateur portable, person, télévision, télévision, ordinateur portable, ordinateur portable, ordinateur portable, person, person, ordinateur portable, person, person, person, person, person, person, ordinateur portable, person, ordinateur portable, person, ordinateur portable, person, person, person, person, person, person, person, person, ordinateur portable, ordinateur portable, person, person, person, person, person, person, person, person, person, person, person, person, person, person, person, person, person, person, person, person, person, chaise, chaise, ordinateur portable, valise, valise, ordinateur portable, valise, ordinateur portable, valise, chaise, ordinateur portable, chaise, ordinateur portable, valise, chaise, ordinateur portable, ordinateur portable, person, ordinateur portable, person, chaise, ordinateur portable, chaise, ordinateur portable, ordinateur portable, chien, ordinateur portable, chaise, chaise, télévision, télévision, télévision, chaise, ordinateur portable, chaise, télévision, télévision, ordinateur portable, télévision, chaise, ordinateur portable, télévision, télévision, télévision, télévision, télévision, télévision, télévision, télévision, télévision, chat, télévision, ordinateur portable, chaise, ordinateur portable, ordinateur portable, télévision, ordinateur portable, ordinateur portable, ordinateur portable, ordinateur portable, ordinateur portable, ordinateur portable, ordinateur portable, ordinateur portable, ordinateur portable, ordinateur portable, ordinateur portable, ordinateur portable, ordinateur portable, ordinateur portable, ordinateur portable, valise, ordinateur portable, ordinateur portable, ordinateur portable, ordinateur portable, ordinateur portable, ordinateur portable, télévision, ordinateur portable, person, ordinateur portable, chaise, ordinateur portable, ordinateur portable, ordinateur portable, ordinateur portable, chaise, ordinateur portable, ordinateur portable, person, ordinateur portable, person, souris, ordinateur portable, person, ordinateur portable, télévision, ordinateur portable, télévision, télévision, télévision, chien, télévision"
   
    tts = TextToSpeech(voice="fr+f3")
    print("Testing TTS...", flush=True)

    long_text = """Voici les objets que j'ai détectés : person, ordinateur portable, ... etc."""
    tts.speak(text)
