import sounddevice as sd
import numpy as np
import wave

class Micro:
    def __init__(self, frequence=44100, duree_max=10, peripherique=None):
        # Paramètres de base
        self.frequence = frequence      # Qualité audio (Hz)
        self.duree_max = duree_max      # Durée maximum d'enregistrement
        self.peripherique = peripherique # Microphone à utiliser
        
        # État de l'enregistrement
        self.est_en_train = False
        self.donnees_audio = None
        self.flux = None
    
    def demarrer(self):
        """Démarre l'enregistrement"""
        if self.est_en_train:
            print("⚠️ L'enregistrement est déjà en cours!")
            return
        
        # Préparer la liste pour stocker l'audio
        self.donnees_audio = []
        
        # Créer le flux audio
        self.flux = sd.InputStream(
            samplerate=self.frequence,
            channels=1,           # 1 = mono, 2 = stéréo
            dtype="int16",        # Format des données
            device=self.peripherique,
            callback=self._recuperer_audio
        )
        
        self.flux.start()
        self.est_en_train = True
        print("🎤 Enregistrement démarré!")
    
    def _recuperer_audio(self, donnees, frames, temps, status):
        """Fonction appelée automatiquement quand il y a du son"""
        if status:
            print(f"Status: {status}")
        self.donnees_audio.append(donnees.copy())
    
    def arreter(self):
        """Arrête l'enregistrement"""
        if not self.est_en_train:
            print("⚠️ Aucun enregistrement en cours!")
            return
        
        if self.flux:
            self.flux.stop()
            self.flux.close()
            self.flux = None
        
        # Convertir toutes les données en un seul tableau
        if self.donnees_audio:
            self.donnees_audio = np.concatenate(self.donnees_audio)
        
        self.est_en_train = False
        print("⏹️ Enregistrement arrêté!")
    
    def enregistrer(self, duree=5):
        """Enregistre automatiquement pendant X secondes"""
        print(f"⏱️ Enregistrement de {duree} secondes...")
        self.demarrer()
        sd.sleep(duree * 1000)  # Attendre X secondes
        self.arreter()
    
    def sauvegarder(self, nom_fichier="enregistrement.wav"):
        """Sauvegarde l'audio dans un fichier WAV"""
        if self.donnees_audio is None:
            print("❌ Aucun enregistrement à sauvegarder!")
            return
        
        with wave.open(nom_fichier, "wb") as fichier:
            fichier.setnchannels(1)      # Mono
            fichier.setsampwidth(2)      # 16-bit
            fichier.setframerate(self.frequence)
            fichier.writeframes(self.donnees_audio.tobytes())
        
        print(f"💾 Fichier sauvegardé: {nom_fichier}")
    
    def duree(self):
        """Donne la durée de l'enregistrement"""
        if self.donnees_audio is None:
            return 0
        return len(self.donnees_audio) / self.frequence

# 🎯 EXEMPLE D'UTILISATION SIMPLE
if __name__ == "__main__":
    # 1. Voir les micros disponibles
    print("📱 Périphériques audio disponibles:")
    print(sd.query_devices())
    print("\n" + "="*50 + "\n")
    
    # 2. Créer un micro
    mon_micro = Micro(frequence=16000, duree_max=10)
    
    # 3. Enregistrer 3 secondes
    mon_micro.enregistrer(duree=3)
    
    # 4. Sauvegarder
    mon_micro.sauvegarder("mon_audio.wav")
    
    # 5. Afficher les infos
    print(f"🕒 Durée: {mon_micro.duree():.2f} secondes")
    print(f"📊 Échantillons: {len(mon_micro.donnees_audio)}")