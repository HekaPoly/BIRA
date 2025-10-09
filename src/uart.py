# Helper functions to send data to microcontroller through UART
# Author: Fabrice Renard
# Date: 30/09/2023

from platform import platform
from serial import Serial
from serial.tools import list_ports
from time import sleep

class UARTTransmitter():
    def __init__(self, baud_rate=115200, uart_init_delay=2):
        """
        Initialise la classe avec les paramètres principaux.
        """
        self.baud_rate = baud_rate
        self.uart_init_delay = uart_init_delay
        self.VELOCITY_SHIFT = 3
        self.ANGLE_SHIFT = 16
        self.NUMBER_OF_BYTES = 4

    def get_serial_ports_list(self) -> list:
        """ 
        This function returns a list of active serial ports;

        Parameters:
            None

        Returns:
            list_com_ports (list): Active serial ports.
        """
        com_ports = list_ports.comports()
        list_com_ports = []

        if (len(com_ports) != 0):
            for port in com_ports:
                if "Windows" in platform():
                    list_com_ports.append(port.name)
                elif "Linux" in platform():
                    list_com_ports.append("/dev/" + port.name)

        return list_com_ports

    def send_data_through_UART(self, angle: int, motorId: int = 0) -> bool:
        """
        This function takes angle as input to send it to a microcontroller through UART;

        Parameters:
            angle (int): The angle to send to the microcontroller. Must be in between 0 and 360.

        Returns:
            dataSuccessfullySent (bool): Result of data transmission (Successful or Unsuccessful).
        """
        angle =  int((2.15*int(angle)+360) % 360 )
        assert(angle >= 0 and angle <= 360)
        serial_ports = self.get_serial_ports_list()
        if len(serial_ports) != 1:
            raise Exception("Erreur: il doit y avoir seulement un port serial connecte")
        
        serial_port = serial_ports[0]

        VELOCITY = 75
        VELOCITY <<= self.VELOCITY_SHIFT

        angle <<= self.ANGLE_SHIFT
        data_successfully_sent = False
        data = 0x00000000
        data += motorId
        data += angle
        data += VELOCITY

        ser = Serial(
                        port            = serial_port,
                        baudrate        = self.baud_rate,
                        timeout         = None,
                        write_timeout   = 0,
                        xonxoff         = False,
                        rtscts          = False,
                        dsrdtr          = False)
        try:
            ser.isOpen()
        except Exception as e:
            print(e)
            print("Unable to open serial communication port. Try selecting a different port.")

        byte_data = data.to_bytes(self.NUMBER_OF_BYTES, byteorder='little')

        try:
            sleep(self.uart_init_delay)
            
            ser.write(byte_data)
            data_successfully_sent = True
        except Exception as e:
            print(e)

        if data_successfully_sent:
            print('Data sent: 0x' + byte_data.hex())
        
        return data_successfully_sent