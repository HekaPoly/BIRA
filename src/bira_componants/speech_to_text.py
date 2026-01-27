import asyncio
import whisper

from bira_componants.bira_componant import BiraComponent

# Load the Whisper model once
print("Loading Whisper model...")
model = whisper.load_model("small", device="cuda")
print("Model loaded")


class SpeechToText(BiraComponent):
    def __init__ (self, language="french", mediator=None):
        super().__init__("speech_to_text", mediator)
        self.language = language
        self.model = model
        self._lock = asyncio.Lock()
        
    async def receive(self, message):
        if message.keys().__contains__('transcribe_1'):
            result = await self.transcribe(message['transcribe_1'])
            self.mediator.notify(self, {"transcription_ready": result})

    async def transcribe_async(self, audio_path):
        return await asyncio.to_thread(
            self.model.transcribe,
            audio_path,
            language=self.language
        )

    async def transcribe(self, audio_data='recording.wav'):
        async with self._lock:
            result = await self.transcribe_async(audio_data)
        return result["text"].strip()
    
if __name__ == "__main__":
    print("Speech to Text module")
    stt = SpeechToText(language="french")
    transcription = stt.transcribe('recording.wav')
    print("Transcription:", transcription)
    
    