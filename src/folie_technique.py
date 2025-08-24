import enum
import multiprocessing
from detector_viewer import DetectorViewer
import sounddevice as sd
from time import sleep
import detector
import speech_to_text
import threading
import time
import numpy as np
import pyaudio
import random

from text_viewer import TextViewer
from uart import send_data_through_UART
import random

VOLUME_THRESHOLD = 0.1

class FolieTechniqueAction(enum.Enum):
    HI = "Salut! :D",
    LEFT = "Tourner a gauche",
    RIGHT = "Tourner a droite",
    UP = "Pointer vers le haut",
    DANCE = "Alors on danse!",
    FIND_PERSONS = "Trouver des personnes",
    POINT_PERSON = "Pointer une personne",
    RESET = "Reset BIRA"

class FolieTechnique:
    def hello_dance(self,):
        send_data_through_UART(300, 0)
        sleep(1)
        self.hello(2)
        send_data_through_UART(60, 0)
        sleep(1)
        self.hello(2)
    
    def base_dance(self, sleep_time=3):
        send_data_through_UART(90, 0)
        sleep(sleep_time)
        send_data_through_UART(90, 1)
        sleep(sleep_time)
        send_data_through_UART(0, 0)
        sleep(sleep_time)
        send_data_through_UART(0, 1)
        sleep(sleep_time)
    
    def base_dance_2(self, sleep_time=5):
        send_data_through_UART(90, 0)
        sleep(sleep_time)
        send_data_through_UART(90, 1)
        sleep(sleep_time)
        send_data_through_UART(180, 0)
        sleep(sleep_time)
        send_data_through_UART(180, 1)
        sleep(sleep_time)
        send_data_through_UART(90, 0)
        sleep(sleep_time)
        send_data_through_UART(90, 1)
        sleep(sleep_time)
        send_data_through_UART(0, 0)
        sleep(sleep_time)
        send_data_through_UART(0, 1)
        sleep(sleep_time)

    def hello(self, repetitions=3, timeout=4, is_3ddl=False):
        self.text_queue.put("Je tourne la premiere membrure.")
        send_data_through_UART(150, 1)
        sleep(3)
        if is_3ddl:
            self.text_queue.put("Je tourne la deuxieme membrure.")
            send_data_through_UART(70, 2, 10)
            sleep(2)
            for _ in range(repetitions):
                self.text_queue.put("Salut! :D")
                send_data_through_UART(75, 2, 5)
                sleep(1)
                send_data_through_UART(55, 2, 5)
                sleep(1)
        else:
            for _ in range(repetitions):
                send_data_through_UART(165, 1)
                sleep(timeout)
                send_data_through_UART(135, 1)
                sleep(timeout)
        
    def handle_action(self, action: FolieTechniqueAction, opt):
        if action == FolieTechniqueAction.HI:
            self.hello(is_3ddl=True)
        elif action == FolieTechniqueAction.LEFT:
            self.text_queue.put("Je tourne la base.")
            send_data_through_UART(0, 0)
        elif action == FolieTechniqueAction.RIGHT:
            self.text_queue.put("Je tourne la base.")
            send_data_through_UART(170, 0)
        elif action == FolieTechniqueAction.UP:
            self.text_queue.put("Je tourne la premiere membrure.")
            send_data_through_UART(150, 1)
            sleep(5)
            self.text_queue.put("Je tourne la deuxieme membrure.")
            send_data_through_UART(70, 2, 5)
            sleep(1)
        elif action == FolieTechniqueAction.DANCE:
            i = random.randint(0, 2)
            if i == 0:
                self.base_dance()
            elif i == 1:
                self.base_dance_2()
            elif i == 2:
                self.hello_dance()

        elif action == FolieTechniqueAction.FIND_PERSONS:
            detector.object_detection(0, 60, opt)

        #elif action == FolieTechniqueAction.POINT_PERSON:
            # detector.object_detection(0, 60, opt)
            # if need to show specific element to cv
            # vision_queue.put({"label": label_value, "coordinate_dict": coordinate_dict})
            # PERSON_ANGLE_HORIZONTAL = 80
            # PERSON_ANGLE_VERTICAL = 90
            # send_data_through_UART(PERSON_ANGLE_VERTICAL, 1)
            # sleep(0.5)
            # send_data_through_UART(PERSON_ANGLE_HORIZONTAL, 0)
            # sleep(0.5)
        else:
            print("No action matched.")
        
        sleep(5)

        
    def get_command_from_text(self, raw_text):
        lower_text = raw_text.lower()
        
        if any(word in lower_text for word in ["bonjour", "hi", "hello", "salu", "allo"]):
            return FolieTechniqueAction.HI
        elif any(word in lower_text for word in ["gauche", "left"]):
            return FolieTechniqueAction.LEFT
        elif any(word in lower_text for word in ["droit", "right"]):
            return FolieTechniqueAction.RIGHT
        elif any(word in lower_text for word in ["haut", "up", "ciel"]):
            return FolieTechniqueAction.UP
        elif any(word in lower_text for word in ["danse", "dance"]):
            return FolieTechniqueAction.DANCE
        elif sum(1 for word in ["trouve", "find", "person", "people"] if word in lower_text) >= 2:
            return FolieTechniqueAction.FIND_PERSONS
        #elif sum(1 for word in ["montre", "point", "person", "people"] if word in lower_text) >= 2:
        #    return FolieTechniqueAction.POINT_PERSON
        else:
            return None
    
    def run_reset(self):
        self.text_queue.put({"subtitle": "Je me remets en position de base (base)."})
        send_data_through_UART(80, 0)
        sleep(3)
        self.text_queue.put({"subtitle": "Je me remets en position de base (2e membrure)."})
        send_data_through_UART(0, 2, 5)
        sleep(3)
        self.text_queue.put({"subtitle": "Je me remets en position de base (1er membrure)."})
        send_data_through_UART(0, 1)
        sleep(3)
    
    def run_sound_detection(self):
        has_detect = False
        sample_rate = 16000
        bits_per_sample = 16
        chunk_size = 1024
        audio_format = pyaudio.paInt16
        channels = 1

        audio = pyaudio.PyAudio()
        stream = audio.open(format=audio_format,
                        channels=channels,
                        rate=sample_rate,
                        input=True,
                        frames_per_buffer=chunk_size)

        while not has_detect:
            data = stream.read(chunk_size, exception_on_overflow=False)
            audio_data = np.frombuffer(data, dtype=np.float32)
            volume = np.linalg.norm(audio_data)
            print(volume)
            if volume > VOLUME_THRESHOLD:
                print("Audio detected, starting Folie Technique...")       
                has_detect = True
            sleep(0.1)

        stream.stop_stream()
        stream.close()

    def run_text_window(self, queue: multiprocessing.Queue):
        app = TextViewer(queue)
        app.open()
        

    def run_folie_app(self, opt):
        print("Starting Folie Technique...")
        self.text_queue = multiprocessing.Queue()
        text_process = multiprocessing.Process(target=self.run_text_window, args=(self.text_queue,))
        text_process.start()

        self.text_queue.put("Je suis BIRA, le bras robotique de HEKA.")
        self.run_reset()

        self.text_queue.put("Dit BIRA!")
        self.run_sound_detection()
        
        
        self.text_queue.put({"text": "Prépare toi!", "countdown": 3})
        self.text_queue.put({"text": "Dit ta commande!", "countdown": 10})
        text = speech_to_text.transcribe_for(13)
        command = self.get_command_from_text(text)
        
        
        while(command is None):
            self.text_queue.put({"text": "Jai mal compris,\npeux-tu repeter ta commande. :)", "countdown": 10})
            text = speech_to_text.transcribe_for(10)
            command = self.get_command_from_text(text)


        self.text_queue.put({"text": command.value, "countdown": 3})
        sleep(3)
        self.text_queue.put("Jai compris ta commande, je vais faire l'action.")
        if command == FolieTechniqueAction.FIND_PERSONS:
            text_process.terminate()
            self.text_queue.close()


        self.handle_action(command, opt)

        print("Done action, resetting...")

        if command == FolieTechniqueAction.FIND_PERSONS:
            self.text_queue = multiprocessing.Queue()
            text_process = multiprocessing.Process(target=self.run_text_window, args=(self.text_queue,))
            text_process.start()

        self.text_queue.put("Jai fini de faire l'action. \n\nA la prochaine! :D")
        self.run_reset()
        sleep(3)
        print("End")
        self.text_queue.close()
        text_process.terminate()
        


                        
                
