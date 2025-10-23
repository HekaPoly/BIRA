import sounddevice as sd
import numpy as np
import wave

class Micro:
    """
    A class to manage microphone recording, playback, and basic audio analysis.

    """

    def __init__(self, frequency=44100, max_duration=10, device=None):
        """
        Initialize the Micro class with the given recording parameters.

        Parameters
        ----------
        frequency : int, 
            Sampling frequency in Hz.
        max_duration : int
            Maximum recording time in seconds.
        device : int or None
            Audio input device index.
        """
        self.frequency = frequency
        self.max_duration = max_duration
        self.device = device

        self.is_recording = False
        self.audio_data = None
        self.stream = None

    def start(self):
        """
        Start recording audio from the selected input device.
        If recording is already in progress, the method does nothing.
        The audio stream runs asynchronously using a callback.
        """
        if self.is_recording:
            print("Recording is already in progress.")
            return

        self.audio_data = []

        self.stream = sd.InputStream(
            samplerate=self.frequency,
            channels=1,
            dtype="int16",
            device=self.device,
            callback=self._capture_audio
        )

        self.stream.start()
        self.is_recording = True
        print("Recording started.")

    def _capture_audio(self, data, frames, time, status):
        """
        Internal callback function to capture audio chunks from the stream.

        Parameters
        ----------
        data : numpy.ndarray
            The audio data buffer received from the microphone.
        frames : int
            The number of frames in the current buffer.
        time : CData
            Timing information from the audio driver.
        status : sounddevice.CallbackFlags
            Status flags.
        """
        if status:
            print(f"Status: {status}")
        self.audio_data.append(data.copy())

    def stop(self):
        """
        Stop the audio recording and finalize the data buffer.

        Notes
        -----
        - If no recording is in progress, the method prints a message.
        - After stopping, the recorded audios are concatenated into a single array.
        """
        if not self.is_recording:
            print("No recording is currently in progress!")
            return

        if self.stream:
            self.stream.stop()
            self.stream.close()
            self.stream = None

        if self.audio_data:
            self.audio_data = np.concatenate(self.audio_data)

        self.is_recording = False
        print("Recording stopped.")

    def record(self, duration=5):
        """
        Record audio for a fixed duration.

        Parameters
        ----------
        duration : int or float
            Recording time in seconds (default is 5).
        """
        print(f"Recording for {duration} seconds...")
        self.start()
        sd.sleep(int(duration * 1000))
        self.stop()

    def save(self, filename="recording.wav"):
        """
        Save the recorded audio data to a WAV file.

        Parameters
        ----------
        filename : str, optional
            The name of the output file (default is "recording.wav").
        """
        if self.audio_data is None:
            print("No recording to save!")
            return

        with wave.open(filename, "wb") as file:
            file.setnchannels(1)
            file.setsampwidth(2)  # 16-bit PCM
            file.setframerate(self.frequency)
            file.writeframes(self.audio_data.tobytes())

        print(f"File saved: {filename}")

    def duration(self):
        """
        Get the duration of the recorded audio.

        Returns
        -------
        float
            The duration of the recording in seconds, or 0 if no data is available.
        """
        if self.audio_data is None:
            return 0
        return len(self.audio_data) / self.frequency

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
    my_mic = Micro(frequency=16000, max_duration=10)

    my_mic.record(duration=3)
    my_mic.save("my_audio.wav")

    print(f"Duration: {my_mic.duration():.2f} seconds")
    print(f"Samples: {len(my_mic.audio_data)}")
    my_mic.get_volume()
