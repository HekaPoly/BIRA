import sounddevice as sd
import numpy as np
import wave

class Micro:
    def __init__(self, frequency=44100, device=None):
        """
        Initialize the Micro object.

        Parameters:
            frequency (int): Sampling rate in Hz (default: 44100).
            device (str or None): Name or ID of the audio input device.

        Returns:
            None
        """
        self.frequency = frequency
        self.device = device
        self.is_recording = False
        self.audio_data = None
        self.stream = None

    def get_volume(self):
        """
        Calculate and return the average volume of the recorded audio.
        Returns:
            None
        """
        if self.audio_data is None or len(self.audio_data) == 0:
            return 0.0

        # CASE 1 : streaming (list of chunks)
        if isinstance(self.audio_data, list):
            audio = np.concatenate(self.audio_data)
        else:
            audio = self.audio_data

        audio = audio.astype(np.float32) / 32768.0
        return float(np.sqrt(np.mean(audio ** 2)))

    def wait_for_volume(self, threshold=0.2):
        """
        Loops until a certain volume is reached
        Parameters:
            threshold (float): Volume threshold to trigger the break (default: 0.2).
        Returns:
            None
        """
        self._start_recording()

        while True:
            sd.sleep(100)

            if self.audio_data and len(self.audio_data) > 0:
                last_chunk = self.audio_data[-1]
                audio = last_chunk.astype(np.float32) / 32768.0
                volume = np.sqrt(np.mean(audio ** 2))
                print(f"Current volume: {volume:.3f}")
                if volume >= threshold:
                    print(f"Command detected with volume: {volume:.3f}")
                    self._stop_recording()
                    break

    def record(self):
        print('start transcription_request')
        self._record(duration=5)
        self._save_recording('recording.wav')

    def _start_recording(self):
        """
        Start recording audio from the selected device.
        Does nothing if a recording is already in progress.

        Returns:
            None
        """
        if self.is_recording:
            print("Recording is already in progress")
            return

        self.audio_data = []

        self.stream = sd.InputStream(
            samplerate=self.frequency,
            channels=1,
            dtype="int16",
            device=self.device,
            callback=self._get_stream
        )

        self.stream.start()
        self.is_recording = True
        print("Recording started")

    def _get_stream(self, data, _frames, _time, status):
        """
        Internal callback function used by sounddevice to collect audio data.

        Parameters:
            data (numpy.ndarray): Audio buffer received from the device.
            _frames (int): Number of frames in the buffer.
            _time (CData): Timing information from sounddevice.
            status (CallbackFlags): Stream status (e.g., underflow/overflow).

        Returns:
            None
        """
        if status:
            print(f"Status: {status}")
        self.audio_data.append(data.copy())

    def _stop_recording(self):
        """
        Stop the current recording and finalize the audio data.
        Does nothing if no recording is active.

        Returns:
            None
        """
        if not self.is_recording:
            print("No recording in progress")
            return

        if self.stream:
            self.stream.stop()
            self.stream.close()
            self.stream = None

        if self.audio_data:
            self.audio_data = np.concatenate(self.audio_data)
        self.is_recording = False
        print("Recording stopped")

    def _record(self, duration=5):
        """
        Record audio automatically for a given duration.

        Parameters:
            duration (int): Duration of the recording in seconds (default: 5).
        """
        print(f"Recording for {duration} seconds")
        self._start_recording()
        sd.sleep(duration * 1000)
        self._stop_recording()
        # TODO: change to sd.rec(...)

    def _save_recording(self, filename="recording.wav"):
        """
        Save the recorded audio as a WAV file.

        Parameters:
            filename (str): Output file name (default: "recording.wav").

        Returns:
            None
        """
        if self.audio_data is None:
            print("No recording to save")
            return

        with wave.open(filename, "wb") as file:
            file.setnchannels(1)
            file.setsampwidth(2)
            file.setframerate(self.frequency)
            file.writeframes(self.audio_data.tobytes())
        print(f"File saved: {filename}")


if __name__ == "__main__":
    my_micro = Micro(frequency=16000)
    my_micro.record()
    print(f"Average volume: {my_micro.get_volume():.3f}")
