from src.bira_components.micro import Micro
from src.bira_components.speech_to_text import SpeechToText

class HeyBira():
    """
        Objet qui contient les fonctionnalités du mandat HeyBira
    """

    def __init__(self):
        raise NotImplementedError

    def convert_audio():
        """
            Fonction qui traite et convertit les audio en dataset
        """
        raise NotImplementedError

    def detect_keyword():
        """
            Fonction qui appelle le model AI pour detecter le mot cle "HeyBira"  
        """
        raise NotImplementedError

    def Bira_sleep_mode():
        """
            Fonction de mise en veille du bras robotique
        """
        raise NotImplementedError

    def response_delay():
        """
            Fonction qui fixe les delai de mise en veille du bras ex: tout les 30s sans réponse
        """
        raise NotImplementedError



class HeyBiraModel():
    """
        Objet qui configure, entraine le model et predit ex:
    """
    def __init__(self):

        raise NotImplementedError
    
    def configure_model():
        """
            Fonction qui configure le model AI
        """
        raise NotImplementedError

    def train_model():
        """
            Fonction qui entraine le model AI
        """
        raise NotImplementedError

    def evaluate_audio():
        """
            Fonction qui evalue la correspondance 
            de l'audio au mot cle "HeyBira"
        """
        raise NotImplementedError



def main():
    """
        Fonction qui simule le fonctionnement complet
    """
    # Initialisation du micro/pour les tests
    my_micro = None

    while True :
        if my_micro is None:
            frequency = int(input("Enter sampling frequency: "))  # Fréquence d'échantillonnage
            my_micro = Micro(frequency=frequency)

        duration = int(input("Enter recording duration (seconds): "))  # Durée de l'enregistrement

        my_micro.record(duration=duration)

        enregistrement = input("Do you want to save the recording? (y/n): ").strip().lower()

        if enregistrement == "y":
            my_micro.save_recording("recording.wav")

        print(f"Average volume: {my_micro.get_volume():.3f}")

main()

