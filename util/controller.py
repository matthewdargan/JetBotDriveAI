"""Interface to get controller inputs to send to JetBot."""
from pickle import dumps
from socket import socket, AF_INET, SOCK_DGRAM

import pygame

JOYSTICK_DZ_THRESH = 0.15
PRECISION = 3
TICKS_PER_SECOND = 30

UDP_IP = '192.168.0.14'
UDP_PORT = 5005


class Controller:
    def __init__(self, ticks_per_second):
        """
        Initializes controller state.
        :param ticks_per_second: constant for controls per second
        """
        pygame.init()
        pygame.joystick.init()
        self.status = {}
        self.clock = pygame.time.Clock()
        self._joystick: pygame.joystick.JoystickType = pygame.joystick.Joystick(0)
        self._joystick.init()
        self.ticks_per_second: int = ticks_per_second

        # Set up a socket UDP connection
        self.sock = socket(AF_INET, SOCK_DGRAM)
        print(f'UDP socket open on IP {UDP_IP}:{UDP_PORT} for controller data')

    def monitor_status(self):
        """Monitors the status of controller inputs and sets deadzones."""
        while True:
            # Event object must be acquired before data from axes can be read
            for event in pygame.event.get():
                if event.type == pygame.JOYAXISMOTION:
                    # Handle left joystick axes positions: right and down are positive
                    self.status['ls_x'] = round(self._joystick.get_axis(0), PRECISION)
                    self.status['ls_y'] = round(self._joystick.get_axis(1), PRECISION)

                    # Handle right trigger position, normalize value
                    self.status['right_trigger'] = round((self._joystick.get_axis(5) + 1.0) / 2.0, PRECISION)

            # Compensate for controller deadzone
            if abs(self.status['ls_x']) < JOYSTICK_DZ_THRESH:
                self.status['ls_x'] = 0.0
            if abs(self.status['ls_y']) < JOYSTICK_DZ_THRESH:
                self.status['ls_y'] = 0.0

            print(self.status)
            # Send the controller status over the socket
            self.sock.send(dumps(self.status))

            # Slow loop down to a maximum amount of ticks per second
            self.clock.tick(self.ticks_per_second)


if __name__ == '__main__':
    controller = Controller(ticks_per_second=30)
    controller.monitor_status()
