# from openai import OpenAI

# Toggle: True = OpenAI Whisper API, False = local whisper model
USE_OPENAI_API = False

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


class SpeechToText():
    def __init__(self, language="en", mediator=None, use_openai_api=None):

        self.language = language
        self.use_openai_api = use_openai_api if use_openai_api is not None else USE_OPENAI_API
        self._client = OpenAI(api_key=OPENAI_API_KEY) if (self.use_openai_api and OPENAI_API_KEY) else None

    def transcribe(self, audio_data="recording.wav"):
        """
        Transcribe an audio file to text using either the OpenAI Whisper API or a local Whisper model.

        Args:
            audio_data (str): Path to the audio file to transcribe. Defaults to "recording.wav".

        Returns:
            str: The transcribed text. If an error occurs during transcription, a fallback message
                is returned instead.
        """
        print("Starting transcription...")
        try:
            result = self.transcribe_async(audio_data)
            print(result)
            return result
        except Exception as e:
            print("Error during transcription:", e)
            return "I want to eat some strawberries"


    def transcribe_async(self, audio_path):
        """
        Perform the actual transcription of an audio file, using either the OpenAI Whisper API
        or a local Whisper model depending on the instance configuration.

        This method is called internally by :meth:`transcribe` and handles the low-level
        transcription logic. When using the OpenAI API, it opens the audio file and sends it
        to the ``whisper-1`` endpoint. When using the local model, it loads the model on first
        use (lazy initialization) and runs inference on the given file.

        Args:
            audio_path (str): Path to the audio file to transcribe (e.g. ``"recording.wav"``).

        Returns:
            str: The transcribed text, stripped of leading/trailing whitespace.

        Raises:
            ValueError: If ``use_openai_api`` is ``True`` but no API key has been configured
                (neither via :data:`OPENAI_API_KEY` nor the ``OPENAI_API_KEY`` environment variable).
            FileNotFoundError: If the file at ``audio_path`` does not exist.
        """

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

    # #TODO: remove used method
    # def receive(self, message):
    #     print("STT", message.keys())
    #     if "transcribe_1" in message:
    #         print("Received transcribe_1 request")
    #         result = self.transcribe(message["transcribe_1"])
    #         print("Here's the result:", result)
    #         self.mediator.send(self, {"transcription_ready": result})


if __name__ == "__main__":
    print("Speech to Text module (OpenAI API)" if USE_OPENAI_API else "Speech to Text module (local Whisper)")
    stt = SpeechToText(language="fr")
    transcription = stt.transcribe("recording.wav")
    print("Transcription:", transcription)
