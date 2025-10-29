import pyttsx3 

class Speaker:
    def __init__(self, voice: str = None):
        self.engine = pyttsx3.init()

    def speak(self, text: str):
        self.engine.say(text)
        self.engine.runAndWait()

    def stop(self):
        self.engine.stop()

if __name__== "__main__":
    tts = Speaker(voice="Zira")
    print("Testing TTS...", flush=True)
    tts.speak("Hello, this is a test.")
