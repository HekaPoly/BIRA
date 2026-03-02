import os

from openai import OpenAI

from bira_components.bira_component import BiraComponent

# Toggle: True = OpenAI Whisper API, False = local whisper model
USE_OPENAI_API = True

# API key (only used when USE_OPENAI_API is True)
OPENAI_API_KEY = None  # or set directly: OPENAI_API_KEY = "sk-..."

# Local whisper model (loaded only when USE_OPENAI_API is False)
_local_model = None


def _get_local_model():
    global _local_model
    if _local_model is None:
        import whisper
        print("Loading Whisper model (local)...")
        _local_model = whisper.load_model("small", device="cuda")
        print("Model loaded")
    return _local_model


class SpeechToText(BiraComponent):
    def __init__(self, language="en", mediator=None, use_openai_api=None):
        super().__init__("speech_to_text", mediator)
        self.language = language
        self.use_openai_api = use_openai_api if use_openai_api is not None else USE_OPENAI_API
        self._client = OpenAI(api_key=OPENAI_API_KEY) if (self.use_openai_api and OPENAI_API_KEY) else None

    def receive(self, message):
        print("STT", message.keys())
        if "transcribe_1" in message:
            print("Received transcribe_1 request")
            result = self.transcribe(message["transcribe_1"])
            print("heres the result:", result)
            self.mediator.send(self, {"transcription_ready": result})

    def transcribe_async(self, audio_path):
        print(f"Transcribing audio: {audio_path} in language: {self.language}")
        if self.use_openai_api:
            if not self._client:
                raise ValueError("OPENAI_API_KEY is not set. Set the global or OPENAI_API_KEY env var.")
            with open(audio_path, "rb") as f:
                response = self._client.audio.transcriptions.create(
                    model="whisper-1",
                    file=f,
                    language=self.language,
                )
            text = response.text.strip() if response.text else ""
        else:
            model = _get_local_model()
            res = model.transcribe(audio_path, language=self.language)
            text = (res.get("text") or "").strip()
        print("Stopped Transcription")
        return text

    def transcribe(self, audio_data="recording.wav"):
        print("Starting transcription...")
        try:
            result = self.transcribe_async(audio_data)
            print(result)
            return result
        except Exception as e:
            print("Error during transcription:", e)
            return "Je veux manger des fraises"


if __name__ == "__main__":
    print("Speech to Text module (OpenAI API)" if USE_OPENAI_API else "Speech to Text module (local Whisper)")
    stt = SpeechToText(language="fr")
    transcription = stt.transcribe("recording.wav")
    print("Transcription:", transcription)
