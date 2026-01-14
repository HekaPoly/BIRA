import whisper

from bira_componants.bira_componant import BiraComponent

# Load the Whisper model once
print("Loading Whisper model...")
model = whisper.load_model("small", device="cuda")
print("Model loaded")


class SpeechToText(BiraComponent):
    def __init__ (self, language="french"):
        self.language = language

    def transcribe(self, audio_data='recording.wav'):
        """
        Transcribe audio data to text using Whisper model.
        
        Parameters: 
            audio_data (str): Path to the audio file to transcribe.
            
        Returns:
            str: Transcribed text.
        """
        result = model.transcribe(audio_data, language=self.language)
        return str(result["text"].strip())
    
if __name__ == "__main__":
    print("Speech to Text module")
    stt = SpeechToText(language="french")
    transcription = stt.transcribe('recording.wav')
    print("Transcription:", transcription)
    
    