import whisper
import pyaudio
import wave
import sys
import tempfile
from ctypes import *

class SpeechToText:
    def __init__(self, model_size="small", device="cuda", language="french"):
        try:
            self.model = whisper.load_model(model_size, device=device)
        except Exception as e:
            raise RuntimeError(f"Impossible de charger le modèle Whisper: {e}")
        self.language = language
        
    def _create_wav_file(self, filename, channels=1, sample_rate=16000, bits_per_sample=16):
            wav_file = wave.open(filename, 'wb')
            wav_file.setnchannels(channels)
            wav_file.setsampwidth(bits_per_sample // 8)
            wav_file.setframerate(sample_rate)
            return wav_file
    
    def _suppress_alsa_warnings(self):
            ERROR_HANDLER_FUNC = CFUNCTYPE(None, c_char_p, c_int, c_char_p, c_int, c_char_p)
            def py_error_handler(filename, line, function, err, fmt):
                return

            c_error_handler = ERROR_HANDLER_FUNC(py_error_handler)
            asound = cdll.LoadLibrary('libasound.so')
            asound.snd_lib_error_set_handler(c_error_handler)

    def _start_recording(self, wav_file, audio_format=pyaudio.paInt16, channels=1, sample_rate=16000, chunk_size=1024):
            def callback(in_data, frame_count, time_info, status):
                wav_file.writeframes(in_data)
                return in_data, pyaudio.paContinue
            # Initialize PyAudio
            audio = pyaudio.PyAudio()

            # Start recording audio
            stream = audio.open(format=audio_format,
                                channels=channels,
                                rate=sample_rate,
                                input=True,
                                frames_per_buffer=chunk_size,
                                stream_callback=callback)
            return stream, audio
    

    # Records audio directly from the microphone until the user presses Enter 
    # and then transcribes it to text using Whisper, returning that transcription.
    def transcribe_directly(self):
        
        # Create a temporary file to store the recorded audio (this will be deleted once we've finished transcription)
        temp_file = tempfile.NamedTemporaryFile(suffix=".wav")

        sample_rate = 16000
        bits_per_sample = 16
        chunk_size = 1024
        audio_format = pyaudio.paInt16
        channels = 1

        wav_file = self._create_wav_file(temp_file.name, channels, sample_rate, bits_per_sample)

        self._suppress_alsa_warnings()
        stream, audio = self._start_recording(wav_file, audio_format, channels, sample_rate, chunk_size)

        input("Press Enter to stop recording...")
        # Stop and close the audio stream
        stream.stop_stream()
        stream.close()
        audio.terminate()

        # Close the wave file
        wav_file.close()

        # And transcribe the audio to text (suppressing warnings about running on a CPU)
        result = self.model.transcribe(temp_file.name, language=self.language)  

        return str(result["text"].strip())

    def transcribe_for(self, seconds=5):
        import time
        # Create a temporary file to store the recorded audio (this will be deleted once we've finished transcription)
        temp_file = tempfile.NamedTemporaryFile(suffix=".wav")

        sample_rate = 16000
        bits_per_sample = 16
        chunk_size = 1024
        audio_format = pyaudio.paInt16
        channels = 1

        # Open the wave file for writing
        wav_file = self._create_wav_file(temp_file.name, channels, sample_rate, bits_per_sample)
        
        # Suppress ALSA warnings (https://stackoverflow.com/a/13453192)
        self._suppress_alsa_warnings()
        
        stream, audio = self._start_recording(wav_file, audio_format, channels, sample_rate, chunk_size)
        

        # Stop and close the audio stream
        print(f"Recording for {seconds} seconds...")
        time.sleep(seconds)

        stream.stop_stream()
        stream.close()
        audio.terminate()

        # Close the wave file
        wav_file.close()

        # And transcribe the audio to text (suppressing warnings about running on a CPU)
        result = self.model.transcribe(temp_file.name, language=self.language)  

        return str(result["text"].strip())

if __name__ == "__main__":
    stt = SpeechToText()  
    print(stt.transcribe_directly())  
