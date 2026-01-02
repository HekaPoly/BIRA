import whisper # Need to explore forks that may offer better speed or accuracy
import pyaudio
import wave
import sys
import tempfile
from ctypes import *

import micro

# Load the Whisper model once
print("Loading Whisper model...")
model = whisper.load_model("small", device="cuda")
print("Model loaded")

# Records audio directly from the microphone until the user presses Enter 
# and then transcribes it to text using Whisper, returning that transcription.
class SpeechToText:
    def __init__ (self, local_model, extern_API, language):
        self.local_model = local_model
        self.extern_API = extern_API
        self.language = language

    def transcribe(self, audio_data='recording.wav'):
        result = model.transcribe(audio_data, language=self.language)
        return str(result["text"].strip())
    
if __name__ == "__main__":
    print("Speech to Text module")
    stt = SpeechToText(local_model=True, extern_API=False, language="french")
    transcription = stt.transcribe('recording.wav')
    print("Transcription:", transcription)
    
    