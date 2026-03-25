# Helper functions to send data to microcontroller through UART
# Author: Fabrice Renard
# Date: 30/09/2023

from platform import platform
from serial import Serial
from serial.tools import list_ports
from time import sleep

from .bira_component import BiraComponent


class UARTTransmitter():
    def __init__(self, 
                 default_baud_rate = 115200, 
                 velocity_shift = 3, 
                 angle_shift = 16, 
                 bytes = 4, 
                 uart_init_delay = 2):
        self.default_baud_rate = default_baud_rate
        self.velocity_shift = velocity_shift
        self.angle_shift = angle_shift
        self.bytes = bytes
        self.uart_init_delay = uart_init_delay

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

    def send_data_through_uart(self, angle: int, motor_id: int = 0, velocity: int = 75):
        """
        This function takes angle as input to send it to a microcontroller through UART;

        Parameters:
            angle (int): The angle to send to the microcontroller. Must be in between 0 and 360.
            motor_id (int): The ID of the motor to control.
            velocity (int): The velocity of the motor.

        Returns:
            None.
        """
        angle =  int(angle) % 360
        if angle < 0 or angle > 360:
            raise Exception("Error: The angle must be between 0 and 360 degrees.")
        
        serial_ports = self.get_serial_ports_list()
        if len(serial_ports) != 1:
            raise Exception("Error: There must be only one serial port connected.")
        
        serial_port = serial_ports[0]

        velocity <<= self.velocity_shift

        angle <<= self.angle_shift

        data = 0x00000000
        data += angle
        data += motor_id
        data += velocity

        ser = Serial(
                        port            = serial_port,
                        baudrate        = self.default_baud_rate,
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

        byte_data = data.to_bytes(self.bytes, byteorder='little')

        try:
            sleep(self.uart_init_delay)
            ser.write(byte_data)
            print(f"Data sent successfully: 0x{byte_data.hex()}")
        except Exception as e:
            print(e)
        finally:
            ser.close()
    