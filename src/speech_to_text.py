import numpy as np
from faster_whisper import WhisperModel
from micro import Micro

class SpeechToText:
    def __init__(self, local_model="small", device="cuda", language="french"):
        """
        Initialize an optimized Whisper model using faster-whisper.

        Parameters:
            local_model (str): The Whisper model size ("small").
            device (str): The device to run on ("cuda").
            language (str): The target transcription language ("french").
        """
        try:
            self.model = WhisperModel(local_model, device=device, compute_type="float16")

        except Exception as e:
            print(f"Error loading Whisper model: {e}")
            self.model = None

        self.language = language

    def transcribe(self, audio_data: np.ndarray) -> str:
        """
        Transcribe raw audio data (NumPy array) into text.

        Parameters:
            audio_data (np.ndarray): Audio samples, typically recorded from a microphone.

        Returns:
            str: The transcribed text.
        """
        if self.model is None:
            print("Whisper model is not loaded")
            return ""
        
        if audio_data is None or len(audio_data) == 0:
            print("No audio data provided")
            return ""
        
        if not isinstance(audio_data, np.ndarray):
            print("Audio data must be a NumPy array")
            return ""
        
        try:
            if audio_data.ndim > 1:
                audio_data = audio_data.mean(axis=1)

            if audio_data.dtype == np.int16:
                audio_data = audio_data.astype(np.float32) / 32768.0

            print("Transcribing audio")
            segments, info = self.model.transcribe(audio_data, language=self.language)

            text = " ".join([segment.text.strip() for segment in segments])
            print(f"Transcription complete in {info.duration:.1f}s")
            return text.strip()
        
        except Exception as e:
            print(f"Error during transcription: {e}")
            return ""

    def transcribe_from_micro(self, micro) -> str:
        """
        Record from an existing Micro object and transcribe the captured audio.

        Parameters:
            micro (Micro): An instance of the Micro class that records audio.

        Returns:
            str: The transcribed text.
        """
        try:
            if not isinstance(micro, Micro):
                raise ValueError("Provided object is not an instance of Micro")
            
            if micro.audio_data is None or len(micro.audio_data) == 0:
                print("No audio data found")
                print("Starting recording")
                micro.record(duration=3)
                print("Transcribing captured audio")
            else:
                return self.transcribe(micro.audio_data)
            
        except Exception as e:
            print(f"Error in transcribe_from_micro: {e}")
            return ""

if __name__ == "__main__":
    micro = Micro(frequency=16000, max_duration=10)
    stt = SpeechToText(local_model="small", device="cuda", language="french")

    micro.record(duration=3)
    transcribed_text = stt.transcribe_from_micro(micro)
    print("Transcribed text: ", transcribed_text)
