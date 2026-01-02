import sounddevice as sd
import numpy as np
import wave

class Micro:
    def __init__ (self, sampling_frequency, maximum_duration):
        self.sampling_frequency = sampling_frequency
        self.maximum_duration = maximum_duration
        self.recording = None

    def start_recording(self):
        self.recording = sd.InputStream(
            samplerate = self.sample_rate,
            channels = self.channels,
            dtype = "int16",
            blocksize = block_size,
            device = self.device,
            callback = callback
        )
        self.recording.start()
        print("Recording started")
    
    def stop_recording(self):
        if self.recording != None:
            self.recording.stop()
            self.recording.close()
            self.recording = None
        print("Recording stopped")
    
    def save_recording(self, filename):
        if self.recording is None:
            raise ValueError("No recording to save")

        with wave.open(filename, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(self.sample_rate)
            wf.writeframes(self.recording.tobytes())
        print(f"Filename saved as {filename}")
    
    def get_stream(self):
        if self.recording is None:
            self.recording = sd.InputStream(
                samplerate = self.sample_rate,
                channels = 1,
                dtype = "int16",
                device = self.device
            )
        return self.recording

    def get_volume(self):
        """
        Estimate the average volume (in decibels).

        Returns
        -------
        float or None
            The average RMS volume level in decibels (dB), or None if no data is available.
        """
        if self.audio_data is None:
            print("No audio data available!")
            return None

        audio_float = self.audio_data.astype(np.float32) / 32768.0
        rms = np.sqrt(np.mean(np.square(audio_float)))
        db = 20 * np.log10(rms + 1e-6)

        print(f"Average volume: {db:.2f} dB")
        return db
    
    
if __name__ == "__main__":
    micro = Micro(44100, 5)
    micro.start_recording()
    micro.save_recording("test.wav")
