from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

class STTAPI(BiraComponent, language="french", mediator=None):
    def __init__(self, language="french", mediator=None):
        super().__init__("stt_api", mediator)
        self.language = language
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self._lock = asyncio.Lock()
    
    def receive(self, message):
        if message.keys().__contains__('transcribe_1'):
            result = self.transcribe(message['transcribe_1'])
            self.mediator.send(self, {"transcription_request": result})

    def transcribe_async(self, audio_file_path):
        return asyncio.to_thread(
            self.transcribe,
            audio_file_path
        )
    
    def transcribe_sync(self, audio_file_path):
        with self._lock:
            result = self.transcribe_async(audio_file_path)
            return result

    def transcribe(self, audio_file_path):
        with open(audio_file_path, "rb") as audio_file:
            transcription = self.client.audio.transcriptions.create(
                file=audio_file,
                model="whisper-1",
                response_format="text"
            )
            return transcription

