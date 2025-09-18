import sounddevice as sd
import numpy as np
import wave

class Micro:
    def __init__ (self, sampling_frequency, maximum_duration, peripheral_device):
        self.sampling_frequency = sampling_frequency
        self.maximum_duration = maximum_duration
        self.peripheral_device = peripheral_device
        self.stream = None

    def start_recording(self):
        self.stream = sd.InputStream(
            samplerate = self.sample_rate,
            channels = self.channels,
            dtype = 'int16',
            blocksize = block_size,
            device = self.device,
            callback = callback
        )
        self.stream.start()
        print("Enregistrement démarré")
    
    def stop_recording(self):
        if self.stream != None:
            self.stream.stop()
            self.stream.close()
            self.stream = None
        print("Enregistrement arrêté")
    
    def save_recording(self, filename):
        if self.recording is None:
            raise ValueError("Aucun enregistrement à sauvegarder")

        with wave.open(filename, 'wb') as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(self.sample_rate)
            wf.writeframes(self.recording.tobytes())
        print(f"Fichier sauvegardé sous {filename}")
    
    def get_stream(self):
        if self.stream is None:
            self.stream = sd.InputStream(
                samplerate = self.sample_rate,
                channels = 1,
                dtype = 'int16',
                device = self.device
            )
        return self.stream

# Trouver quoi écrire
# if __name__ == "__main__":
#     print()
