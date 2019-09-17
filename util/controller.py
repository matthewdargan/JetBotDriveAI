import pygame
from time import sleep

JOYSTICK_DZ_THRESH = 0.15
PRECISION = 3
TICKS_PER_SECOND = 30


class Controller:
    def __init__(self):
        pygame.init()
        pygame.joystick.init()
        self.status = {}
        self.clock = pygame.time.Clock()
        self._joystick: pygame.joystick.JoystickType = pygame.joystick.Joystick(0)
        self._joystick.init()

    def monitor_status(self):
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
            # Slow loop down to a maximum amount of ticks per second
            self.clock.tick(TICKS_PER_SECOND)


if __name__ == '__main__':
    controller = Controller()
    controller.monitor_status()
