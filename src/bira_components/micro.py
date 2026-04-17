import os
from typing import Optional

import sounddevice as sd
import numpy as np
import wave

from bira_components import history as bira_history

RECORDING_FILE_PATH = 'recording.wav'
DEFAULT_RECORD_SECONDS = 5.0

class Micro:
    def __init__(
        self,
        frequency=44100,
        device=None,
        default_record_seconds: float = DEFAULT_RECORD_SECONDS,
        max_live_buffer_seconds: float = DEFAULT_RECORD_SECONDS,
    ):
        """
        Initialize the Micro object.

        Parameters:
            frequency (int): Sampling rate in Hz (default: 44100).
            device (str or None): Name or ID of the audio input device.
            default_record_seconds (float): Default recording duration in seconds.
            max_live_buffer_seconds (float): Rolling window used during wake-word style capture.

        Returns:
            None
        """
        self.frequency = frequency
        self.device = device
        self.default_record_seconds = float(default_record_seconds)
        self.max_live_buffer_seconds = float(max_live_buffer_seconds)
        self.is_recording = False
        self.audio_data = None
        self.stream = None
        self._active_max_buffer_frames: Optional[int] = None
        self._buffered_frames = 0

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
        self._start_recording(max_buffer_seconds=self.max_live_buffer_seconds)
        bira_history.log_event(
            "recording_wait_started",
            component="micro",
            threshold=threshold,
            max_live_buffer_seconds=self.max_live_buffer_seconds,
        )

        while True:
            sd.sleep(100)

            if self.audio_data and len(self.audio_data) > 0:
                last_chunk = self.audio_data[-1]
                audio = last_chunk.astype(np.float32) / 32768.0
                volume = np.sqrt(np.mean(audio ** 2))
                print(f"Current volume: {volume:.3f}")
                if volume >= threshold:
                    print(f"Command detected with volume: {volume:.3f}")
                    bira_history.log_event(
                        "recording_threshold_reached",
                        component="micro",
                        threshold=threshold,
                        volume=round(float(volume), 3),
                    )
                    self._stop_recording()
                    break

    def record(self, duration_seconds: Optional[float] = None):
        duration = float(duration_seconds) if duration_seconds is not None else self.default_record_seconds
        print('start transcription_request')
        bira_history.log_event(
            "recording_requested",
            component="micro",
            duration_seconds=duration,
            output_file=RECORDING_FILE_PATH,
        )
        self._record(duration=duration)
        self._save_recording(RECORDING_FILE_PATH)

    def _start_recording(self, max_buffer_seconds: Optional[float] = None):
        """
        Start recording audio from the selected device.
        Does nothing if a recording is already in progress.

        Parameters:
            max_buffer_seconds (float or None): Maximum duration kept in memory while recording.
                If None, keep all collected samples until stop.

        Returns:
            None
        """
        if self.is_recording:
            print("Recording is already in progress")
            bira_history.log_event("recording_start_ignored", component="micro", reason="already_recording")
            return

        self.audio_data = []
        self._buffered_frames = 0
        if max_buffer_seconds is None:
            self._active_max_buffer_frames = None
        else:
            self._active_max_buffer_frames = max(1, int(float(max_buffer_seconds) * self.frequency))

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
        bira_history.log_event(
            "recording_started",
            component="micro",
            frequency=self.frequency,
            device=self.device,
            max_buffer_seconds=max_buffer_seconds,
        )

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
            bira_history.log_event("recording_stream_status", component="micro", status=str(status))
        chunk = data.copy()
        self.audio_data.append(chunk)
        self._buffered_frames += int(len(chunk))

        max_frames = self._active_max_buffer_frames
        if max_frames is not None:
            while self.audio_data and self._buffered_frames > max_frames:
                removed = self.audio_data.pop(0)
                self._buffered_frames -= int(len(removed))

    def _stop_recording(self):
        """
        Stop the current recording and finalize the audio data.
        Does nothing if no recording is active.

        Returns:
            None
        """
        if not self.is_recording:
            print("No recording in progress")
            bira_history.log_event("recording_stop_ignored", component="micro", reason="not_recording")
            return

        if self.stream:
            self.stream.stop()
            self.stream.close()
            self.stream = None

        if self.audio_data:
            self.audio_data = np.concatenate(self.audio_data)
        else:
            self.audio_data = np.array([], dtype=np.int16)
        self._active_max_buffer_frames = None
        self._buffered_frames = int(len(self.audio_data))
        self.is_recording = False
        print("Recording stopped")
        sample_count = int(len(self.audio_data)) if self.audio_data is not None else 0
        duration_seconds = round(sample_count / self.frequency, 3) if sample_count else 0.0
        bira_history.log_event(
            "recording_stopped",
            component="micro",
            sample_count=sample_count,
            duration_seconds=duration_seconds,
        )

    def _record(self, duration=DEFAULT_RECORD_SECONDS):
        """
        Record audio automatically for a given duration.

        Parameters:
            duration (int): Duration of the recording in seconds (default: 5).
        """
        print(f"Recording for {duration} seconds")
        self._start_recording(max_buffer_seconds=duration)
        sd.sleep(max(1, int(float(duration) * 1000)))
        self._stop_recording()
        # TODO: change to sd.rec(...)

    def _save_recording(self, filename=RECORDING_FILE_PATH):
        """
        Save the recorded audio as a WAV file.

        Parameters:
            filename (str): Output file name (default: "recording.wav").

        Returns:
            None
        """
        if self.audio_data is None:
            print("No recording to save")
            bira_history.log_event(
                "recording_save_skipped",
                component="micro",
                filename=filename,
                reason="no_audio_data",
            )
            return

        with wave.open(filename, "wb") as file:
            file.setnchannels(1)
            file.setsampwidth(2)
            file.setframerate(self.frequency)
            file.writeframes(self.audio_data.tobytes())
        print(f"File saved: {filename}")
        sample_count = int(len(self.audio_data))
        duration_seconds = round(sample_count / self.frequency, 3) if sample_count else 0.0
        bira_history.log_event(
            "recording_saved",
            component="micro",
            filename=filename,
            sample_count=sample_count,
            duration_seconds=duration_seconds,
        )

    @staticmethod
    def clear_recording():
        if os.path.exists(RECORDING_FILE_PATH):
            os.remove(RECORDING_FILE_PATH)
            print(f"{RECORDING_FILE_PATH} has been deleted.")
            bira_history.log_event(
                "recording_file_deleted",
                component="micro",
                filename=RECORDING_FILE_PATH,
            )
        else:
            print("The file does not exist.")
            bira_history.log_event(
                "recording_file_missing",
                component="micro",
                filename=RECORDING_FILE_PATH,
            )


if __name__ == "__main__":
    my_micro = Micro(frequency=16000)
    my_micro.record()
    print(f"Average volume: {my_micro.get_volume():.3f}")
