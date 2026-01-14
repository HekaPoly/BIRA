import pyttsx3
import random

from bira_componants.bira_componant import BiraComponent


class TextToSpeech(BiraComponent):
    def __init__(self, voice: str = None, rate: int = 130, volume: float = 0.8):
        self.engine = pyttsx3.init()
        self.rate = rate
        self.volume = volume
        self.engine.setProperty('rate', self.rate)
        self.engine.setProperty('volume', self.volume)
        self.engine.setProperty('voice', voice)
        

    def speak(self, text: str):
        """
        Convert the given text to speech and play it.
        Parameters:
            text (str): The text to be converted to speech.
        Returns:
            None
        """
        MAX_LEN = 180

        for i in range(0, len(text), MAX_LEN):
            chunk = text[i:i+MAX_LEN]
            chunk = chunk.replace(",", " ").replace(";", " ")

            self.engine.setProperty('rate', self.rate + random.randint(-10, 10))
            self.engine.say(chunk)
            self.engine.runAndWait()
            self.engine.stop()

    def stop(self):
        """
        Stop the speech engine.
        Parameters:
            None
        Returns:
            None
        """
        self.engine.stop()

if __name__== "__main__":
    tts = TextToSpeech(voice="fr+f3")
    print("Testing TTS...", flush=True)

    text = """Voici les objets que j'ai détectés : person, ordinateur portable, ... etc."""
    tts.speak(text)
