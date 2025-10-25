import sounddevice as sd
import numpy as np
import wave

class Micro:
    """
    Classe pour l'enregistrement audio via microphone.
    
    Attributes:
        frequency (int): Fréquence d'échantillonnage en Hz
        max_duration (int): Durée maximale d'enregistrement en secondes
        device (int): Numéro du périphérique audio à utiliser
        is_recording (bool): État de l'enregistrement
        audio_data (numpy.ndarray): Données audio enregistrées
        flow (sd.InputStream): Flux d'entrée audio
    """

    def __init__(self, frequency=44100, max_duration=10, device=None):
        """
        Initialize the microphone recorder.
        
        Args:
            frequency (int, optional): Sampling frequency in Hz. Defaults to 44100.
            max_duration (int, optional): Maximum recording duration in seconds. Defaults to 10.
            device (int, optional): Audio device ID. Defaults to None (default device).
        """
        self.frequency = frequency      
        self.max_duration = max_duration      
        self.device = device 
        
        self.is_recording = False
        self.audio_data = None
        self.flow = None
    
    def start(self) :
        """
        Start audio recording.
        
        If already recording, prints a message and returns without action.
        """
        if self.is_recording:
            print("L'enregistrement est en cours")
            return

        self.audio_data = []
        
        self.flow = sd.InputStream(
            samplerate=self.frequency,
            channels=1,          
            dtype="int16",        
            device=self.device,
            callback=self._retrieve_audio
        )
        
        self.flow.start()
        self.is_recording = True
        print("Enregistrement démarré")
    
    def _retrieve_audio(self, data, frames, time, status):
        """
        Callback function to collect audio data from the stream.
        
        Args:
            data (numpy.ndarray): Audio data frames
            frames (int): Number of frames
            time (CData): Timestamp information
            status (sd.CallbackFlags): Status flags
        """
        if status:
            print(f"Statut : {status}")
        self.audio_data.append(data.copy())
    
    def stop(self):
        """
        Stop audio recording and process collected data.
        
        If no recording is in progress, prints a message and returns.
        Concatenates all audio chunks into a single numpy array.
        """
        if not self.is_recording:
            print("Aucun enregistrement en cours")
            return
        
        if self.flow:
            self.flow.stop()
            self.flow.close()
            self.flow = None
        
        if self.audio_data:
            self.audio_data = np.concatenate(self.audio_data, axis=0)
        
        self.is_recording = False
        print("Enregistrement arrêté")
    
    def record(self, duration):
        """
        Record audio for a specified duration.
        
        Args:
            duration (int): Recording duration in seconds
        """
        print(f"Enregistrement de {duration} secondes")
        self.start()
        sd.sleep(duration * 1000)
        self.stop()

    # def record(self):
    #     print("Enregistrement démarré")
    #     self.start()
        
    #     input("Appuyez sur le bouton du microphone pour arrêter l'enregistrement\n")
    #     self.stop()
    
    # def record_with_timeout(self, timeout=None)
    #     print("Enregistrement démarré")
    #     self.start()
        
    #     if timeout:
    #         print(f"Timeout dans {timeout} secondes")
    #         import threading
    #         def stop_after_delay():
    #             sd.sleep(timeout * 1000)
    #             if self.is_recording:
    #                 self.stop()
    #         threading.Thread(target=stop_after_delay, daemon=True).start()
        
    #     input("Appuyez sur le bouton du microphone pour arrêter l'enregistrement\n")
    #     if self.is_recording:
    #         self.stop()
    
    def save(self, file_name="recording.wav"):
        """
        Save recorded audio to a WAV file.
        
        Args:
            file_name (str, optional): Output filename. Defaults to "recording.wav".
        """
        if self.audio_data is None or len(self.audio_data) == 0:
            print("Aucun enregistrement à sauvegarder")
            return
        
        with wave.open(file_name, "wb") as fichier:
            fichier.setnchannels(1)
            fichier.setsampwidth(2)
            fichier.setframerate(self.frequency)
            fichier.writeframes(self.audio_data.tobytes())
        
        print(f"Fichier sauvegardé : {file_name}")
    
    def duration_recording(self):
        """
        Get the duration of the current recording.
        
        Returns:
            float: Recording duration in seconds, 0 if no data
        """
        if self.audio_data is None or len(self.audio_data) == 0:
            return 0
        return len(self.audio_data) / self.frequency
    
    def get_volume(self):
        """
        Calculate the average volume of the recorded audio.
        
        Returns:
            float: Volume level between 0 and 1, or 0 if no data
        """
        if self.audio_data is None or len(self.audio_data) == 0:
            return 0.0
        
        audio_float = self.audio_data.astype(np.float32) / 32768.0
        volume = np.sqrt(np.mean(audio_float**2))
        
        return volume

if __name__ == "__main__":
    # print("Périphériques audio disponibles : ")
    # print(sd.query_devices())
    # print("\n" + "="*50 + "\n")
    
    microphone = Micro(frequency=16000, max_duration=10)
    
    print("Début de l'enregistrement")
    microphone.record(duration=3)
    
    microphone.save("audio.wav")
    
    print(f"Durée : {microphone.duration_recording():.2f} secondes")
    if microphone.audio_data is not None:
        print(f"Échantillons : {len(microphone.audio_data)}")
    else:
        print("Aucun échantillon enregistré")
    
    volume_level = microphone.get_volume()
    print(f"Niveau sonore : {volume_level:.4f}")
