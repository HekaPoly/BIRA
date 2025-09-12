# Helper functions to send data to microcontroller through UART
# Author: Fabrice Renard
# Date: 30/09/2023

from platform import platform
from serial import Serial
from serial.tools import list_ports
from time import sleep

class UARTTransmitter:
    def __init__(self, DEFAULT_BAUD_RATE, VELOCITY_SHIFT, ANGLE_SHIFT, NUMBER_OF_BYTES, UART_INIT_DELAY):
        self.DEFAULT_BAUD_RATE = 115200
        self.VELOCITY_SHIFT = 3
        self.ANGLE_SHIFT = 16 
        self.NUMBER_OF_BYTES = 4
        self.UART_INIT_DELAY = 2

    def get_serial_ports_list() -> list:
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

    def send_data_through_UART(angle: int, motor_id: int = 0) -> bool:
        """
        This function takes angle as input to send it to a microcontroller through UART;

        Parameters:
            angle (int): The angle to send to the microcontroller. Must be in between 0 and 360.

        Returns:
            dataSuccessfullySent (bool): Result of data transmission (Successful or Unsuccessful).
        """
        angle =  int((2.15*int(angle)+360) % 360)
        assert(angle >= 0 and angle <= 360)
        serial_ports = UARTTransmitter.get_serial_ports_list()
        if len(serial_ports) != 1:
            raise Exception("Erreur: il ne doit y avoir qu'un seul port serie connecte")
        
        serial_port = serial_ports[0]

        VELOCITY = 75
        velocity <<= UARTTransmitter.VELOCITY_SHIFT

        angle <<= UARTTransmitter.ANGLE_SHIFT
        data_successfully_sent = False
        data = 0x00000000
        data += angle
        data += motor_id
        data += VELOCITY

        ser = Serial(
                        port            = serial_port,
                        baudrate        = UARTTransmitter.DEFAULT_BAUD_RATE,
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

        byte_data = data.to_bytes(UARTTransmitter.NUMBER_OF_BYTES, byteorder='little')

        try:
            sleep(UARTTransmitter.UART_INIT_DELAY)
            
            ser.write(byte_data)
            data_successfully_sent = True
        except Exception as e:
            print(e)

        if data_successfully_sent:
            print('Data sent: 0x' + byte_data.hex())
        
        return data_successfully_sent
    