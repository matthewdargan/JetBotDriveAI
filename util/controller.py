"""Interface to get controller inputs to send to JetBot."""
from pickle import dumps
from socket import socket, AF_INET, SOCK_DGRAM
from typing import Dict

import pygame

TICKS_PER_SECOND: int = 30
JOYSTICK_DZ_THRESH: float = 0.15
PRECISION: int = 3

UDP_IP: str = '192.168.0.14'
UDP_PORT: int = 5005


class Controller:
    def __init__(self):
        """Initializes controller state."""
        pygame.init()
        pygame.joystick.init()
        self.status: Dict[str, float] = {'motor_right': 0, 'motor_left': 0, 'right_trigger': 0}
        self.clock: pygame.time.Clock = pygame.time.Clock()
        self._joystick: pygame.joystick.JoystickType = pygame.joystick.Joystick(0)
        self._joystick.init()

        # Set up a socket UDP connection
        self.sock: socket = socket(AF_INET, SOCK_DGRAM)
        print(f'UDP socket open on IP {UDP_IP}:{UDP_PORT} for controller data')

    def monitor_status(self):
        """Monitors the status of controller inputs and sets deadzones."""
        while True:
            # Event object must be acquired before data from axes can be read
            for event in pygame.event.get():
                if event.type == pygame.JOYAXISMOTION:
                    # Handle left joystick axes positions: right and up are positive (after inverting y axis)
                    ls_x = round(self._joystick.get_axis(0), PRECISION)
                    ls_y = -round(self._joystick.get_axis(1), PRECISION)

                    # Compensate for controller deadzone
                    if abs(ls_x) < JOYSTICK_DZ_THRESH:
                        ls_x = 0.0
                    if abs(ls_y) < JOYSTICK_DZ_THRESH:
                        ls_y = 0.0

                    # Calculate intermediaries for tank control conversion
                    v = (1 - abs(ls_x)) * (ls_y / 1) + ls_y
                    w = (1 - abs(ls_y)) * (ls_x / 1) + ls_x

                    # Handle right trigger position, normalize value
                    self.status['right_trigger'] = round((self._joystick.get_axis(4) + 1.0) / 2.0, PRECISION)

                    # Translate intermediaries and apply right trigger speed multiplier to compute tank control values
                    self.status['motor_right'] = (v + w) / 2 * self.status['right_trigger']
                    self.status['motor_left'] = (v - w) / 2 * self.status['right_trigger']

            print(self.status)

            # Send the controller status over the socket
            self.sock.sendto(dumps(self.status), (UDP_IP, UDP_PORT))

            # Slow loop down to a maximum amount of ticks per second
            self.clock.tick(TICKS_PER_SECOND)


if __name__ == '__main__':
    controller = Controller()
    controller.monitor_status()
