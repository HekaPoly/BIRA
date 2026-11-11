from src.bira_components.micro import Micro
from src.bira_components.speech_to_text import SpeechToText
import torch
import torchaudio
import time
import numpy as np
from enum import Enum, auto
import sounddevice as sd

#--------------------------------
# Constants for the HeyBira model
#--------------------------------
SAMPLE_RATE = 16000          # À AJUSTER : doit être celui de l'entraînement du modèle
WINDOW_SECONDS = 1.0         # À AJUSTER : durée d'audio que le modèle attend
WAKE_THRESHOLD = 0.9         # probabilité minimale pour considérer le mot entendu
CONFIRMATIONS = 2            # nombre de fenêtres consécutives au-dessus du seuil
STRIDE_SECONDS = 0.25        # on ré-analyse l'audio toutes les 0,25 s
SILENCE_VOLUME = 0.01        # en dessous, on considère que personne n'a parlé
N_MELS = 40                  # À AJUSTER : nombre de bandes du spectrogramme de Mel
MODEL_PATH = "..."

class EtatBira(Enum):
    """ Enumeration des deux etats possible de Bira"""
    VEILLE = auto()
    EVEIL = auto()

class MicroEcoute(Micro):
    """
    Micro avec un tampon circulaire : il ne garde que les dernières secondes
    d'audio. Contrairement à Micro._start_recording (qui accumule tout et
    finirait par saturer la mémoire), on peut écouter indéfiniment.
    """
 
    def start_listening(self, buffer_seconds=2.0):
        if self.stream is not None:
            return
        taille = int(buffer_seconds * self.frequency)
        self._buffer = np.zeros(taille, dtype=np.int16)
        self.stream = sd.InputStream(
            samplerate=self.frequency,
            channels=1,
            dtype="int16",
            device=self.device,
            callback=self._fill_buffer,
        )
        self.stream.start()

    def _fill_buffer(self, data, _frames, _time, status):
        """Appelée automatiquement par sounddevice, dans un autre thread."""
        if status:
            print(f"Status: {status}")
        # On colle le nouveau bloc à la fin et on garde les N derniers échantillons
        self._buffer = np.concatenate((self._buffer, data[:, 0]))[-len(self._buffer):]
 
    def last_seconds(self, seconds):
        """Retourne une copie des `seconds` dernières secondes d'audio."""
        n = int(seconds * self.frequency)
        return self._buffer[-n:].copy()
 
    def stop_listening(self):
        """À appeler avant Micro.record(), car un seul flux peut utiliser le micro."""
        if self.stream is not None:
            self.stream.stop()
            self.stream.close()
            self.stream = None


class HeyBira():
    """
        Objet qui contient les fonctionnalités du mandat HeyBira
    """

    def __init__(self):
        self.etat = EtatBira.VEILLE
        raise NotImplementedError

    def convert_audio():
        """
            Fonction qui traite et convertit les audio en dataset
        """
        raise NotImplementedError

    def detect_keyword(self):
        """
            Fonction qui appelle le model AI pour detecter le mot cle "HeyBira"  
            Analyse les dernieres secondes d'audio et renvoie True si le mot cle est detecté
        """
        audio = self.micro.last_seconds(WINDOW_SECONDS)
        proba = self.model.evaluate_audio(audio)
        if proba >= WAKE_THRESHOLD:
            self._hits += 1
        else:
            self._hits = 0
        return self._hits >= CONFIRMATIONS
        raise NotImplementedError

    def Bira_sleep_mode(self):
        """
            Fonction de mise en veille du bras robotique, ecoute continue avec aucune transcription
        """
        self.etat = EtatBira.VEILLE
        self._hits = 0
        self.micro.start_listening()
        print("[BIRA] Veille")

        raise NotImplementedError

    def response_delay(self):
        """
            Fonction qui fixe les delai de mise en veille du bras ex: tout les 30s sans réponse
            True si le delai est dépassé, False sinon
        """
        return time.monotonic() - self._last_activity > self.awake_timeout
        raise NotImplementedError

    def wake_up(self):
        """Passe en éveil : on libère le micro pour l'enregistrement des commandes."""
        self.micro.stop_listening()
        self.etat = EtatBira.EVEIL
        self._last_activity = time.monotonic()
        print("[BIRA] Éveil")

    def listen_for_command(self):
        """Enregistre 5 s, ignore le silence, transcrit. Retourne le texte ou None."""
        self.micro.record()  # écrit recording.wav (voir Micro.record)
        if self.micro.get_volume() < SILENCE_VOLUME:
            return None      # évite que Whisper "invente" du texte sur du silence
        try:
            # On appelle transcribe_async directement : transcribe() renvoie une
            # phrase bidon en cas d'erreur, ce qui ferait agir le robot pour rien.
            texte = self.stt.transcribe_async("recording.wav")
        except Exception as e:
            print("Erreur de transcription :", e)
            return None
        return texte or None

    def run(self, on_command):
        """
        Boucle principale. `on_command(texte)` est appelée pour chaque commande.
        Ctrl+C pour arrêter proprement.
        """
        self.bira_sleep_mode()
        try:
            while True:
                if self.etat == EtatBira.VEILLE:
                    if self.detect_keyword():
                        self.wake_up()
                    else:
                        time.sleep(STRIDE_SECONDS)
                else:  # EVEIL
                    texte = self.listen_for_command()
                    if texte:
                        self._last_activity = time.monotonic()
                        on_command(texte)
                    elif self.response_delay_expired():
                        self.bira_sleep_mode()
        except KeyboardInterrupt:
            print("Arrêt demandé")
        finally:
            self.micro.stop_listening()

class HeyBiraModel():
    """
        Charge le modele entrainé pour detecter le mot cle "HeyBira"
    """
    def __init__(self, model_path, device):
        self.model = None
        self.model_path = model_path # Chemin vers le modèle pré-entraîné
        self.device = device #or ("cuda" if torch.cuda.is_available() else "cpu")

        # Doit reproduire EXACTEMENT le prétraitement utilisé à l'entraînement
        self.melspec = torchaudio.transforms.MelSpectrogram(
            sample_rate=SAMPLE_RATE, n_fft=400, hop_length=160, n_mels=N_MELS
        ).to(self.device)

        raise NotImplementedError
    
    def charger_model(self):
        """
            Fonction qui charge le model AI
        """
        self.model = torch.load(self.model_path, map_location=self.device, weights_only=True)
        self.model.eval()  # configurer le modele en mode "prédiction"/evaluation (désactive dropout, etc.)
        raise NotImplementedError

    def train_model():
        """
            Fonction qui entraine le model AI
        """
        raise NotImplementedError

    def evaluate_audio(self, audio_data):
        """
            Fonction qui evalue la correspondance de l'audio au mot cle "HeyBira"
            Args:
                audio_data (numpy array): Données audio à évaluer.
            Returns:
                #bool: True si le mot cle est détecté, False sinon. (non)
                une probabilité entre 0 et 1 indiquant la confiance de la détection.
        """
        # int16 (-32768..32767) -> float (-1..1), comme attendu par torchaudio
        audio = torch.from_numpy(audio_data.astype(np.float32) / 32768.0)

        # Avoir la longueur exacte attendue par le modèle
        n = int(SAMPLE_RATE * WINDOW_SECONDS)
        if len(audio) < n:
            audio = torch.nn.functional.pad(audio, (n - len(audio), 0)) # Ajout de 0
        audio = audio[-n:] #tronquer à la longueur attendue

        with torch.no_grad():  # désactive le calcul du gradient : plus rapide, moins de mémoire
            spectro = self.melspec(audio.to(self.device))   # bascule le tenseur sur le processeur configuré
                                                            # transforme l'audio en spectrogramme de Mel
            spectro = torch.log(spectro + 1e-6) # logarithme pour stabiliser les valeurs
            entree = spectro.unsqueeze(0).unsqueeze(0) # ajoute les dimensions batch et channel
            sortie = self.model(entree) # passage dans le modèle et renvoie un score brut
            return torch.sigmoid(sortie).item() # contraindre le score brut entre 0 et 1 
        raise NotImplementedError



def main():
    """
        Fonction qui simule le fonctionnement complet
    """
    # Initialisation du micro/pour les tests
       
    micro = MicroEcoute(frequency=SAMPLE_RATE)
    stt = SpeechToText(language="fr")
    stt.preload()                      # charge Whisper une fois, au démarrage
    model = HeyBiraModel()
    model.load()

    bira = HeyBira(micro, stt, model)
    bira.run(on_command=lambda texte: print("Commande reçue :", texte))


if __name__ == "__main__":
    main()

