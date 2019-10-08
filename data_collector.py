"""Data collector for getting images and pairing them with human-input controls."""
import os
import pickle
import time
from argparse import ArgumentParser
from socket import socket, AF_INET, SOCK_DGRAM
from typing import Dict, Tuple, Union

import traitlets
from jetbot import Camera, Robot

FRAMES_DIRECTORY: str = 'frames'
CONTROLS_PICKLE_FILENAME: str = 'controlsPickle'

TICKS_PER_SECOND: int = 10

UDP_IP: str = '192.168.0.20'
UDP_PORT: int = 5005


class Collector:
    def __init__(self, frames_directory, ticks_per_second, controls_pickle_filename):
        """
        Initializes data collection state.
        :param frames_directory: directory to save frames into
        :param ticks_per_second: constant for data collection fps
        :param controls_pickle_filename: filename for the pickle file for storing human-input controls
        """
        self.robot: Robot = Robot()
        self.controls: Dict[int, Dict[str, float]] = {}
        self.current_controls: Tuple[float, float, float] = (0, 0, 0)
        self.controls_pickle_filename: str = controls_pickle_filename
        self.camera: Camera = Camera.instance()

        # Reset camera width/height to 25% of the capture width/height
        self.camera.width = 820
        self.camera.height = 616

        self.frames_directory: str = frames_directory
        self.frame_index: int = 0
        self.ticks_per_second: int = ticks_per_second
        self.sock: socket = socket(AF_INET, SOCK_DGRAM)
        print(f'UDP socket open on IP {UDP_IP}:{UDP_PORT} for controller data')

    def run_data_collection(self, last_frame: Union[int, None] = None):
        """
        Runs data collection for controller inputs and camera frames until pause.
        :param last_frame: last frame that data was good for, delete all frames after this frame
        """
        print('Starting data collection')
        self.resume(last_frame)

        while True:
            try:
                # Save image and controls
                # TODO: Fix
                self.robot.right_motor._write_value(float(self.current_controls[0]))
                self.robot.left_motor._write_value(float(self.current_controls[1]))
                self.save_frame()
                self.save_controls()
                self.frame_index += 1

                # Sleep tick
                time.sleep(1. / self.ticks_per_second)
            except KeyboardInterrupt:
                self.pause()
                exit(0)

    def save_frame(self):
        """Saves a frame into a file in the frames_directory."""
        file_path = f"{self.frames_directory}/{self.frame_index}.png"
        with open(file_path, 'wb') as f:
            f.write(self.camera.value)

    def save_controls(self):
        """Saves controls from the human-input controller as a left-motor vector and a right-motor vector."""
        data, _ = self.sock.recvfrom(1024)
        self.controls[self.frame_index] = pickle.loads(data)
        self.current_controls = (
            self.controls[self.frame_index]['motor_right'], self.controls[self.frame_index]['motor_left'])
        print(data)

    def pause(self):
        """Pauses data collection and pickles controls."""
        with open(self.controls_pickle_filename, 'wb') as f:
            pickle.dump(self.controls, f)

        print('Pausing data collection')

    def resume(self, last_frame: Union[int, None]):
        """
        Unpickles controls and resumes data collection.
        :param last_frame: last frame that data was good for, delete all frames after this frame
        """
        if os.path.isfile(self.controls_pickle_filename):
            with open(self.controls_pickle_filename, 'rb') as f:
                try:
                    self.controls = pickle.load(f)

                    # Delete all entries from the last frame onwards
                    for key in self.controls.keys():
                        if key > last_frame:
                            del self.controls[key]

                    # Delete all images from the last frame onwards
                    last_frame_name: str = f"{last_frame}.png"
                    for file in os.listdir(FRAMES_DIRECTORY):
                        if file > last_frame_name:
                            os.remove(file)

                    self.frame_index = len(self.controls)
                    print('Resuming data collection')

                    # traitlets.dlink((self.current_controls[0], 'value'), (self.robot.right_motor, 'value'),
                    #                 transform=lambda x: -x)
                    # traitlets.dlink((self.current_controls[1], 'value'), (self.robot.left_motor, 'value'),
                    #                 transform=lambda x: -x)
                except FileNotFoundError:
                    print(f"{self.controls_pickle_filename} not found")


if __name__ == "__main__":
    # Parse arguments for the mode jetBot should start in (either data collection or model running)
    parser = ArgumentParser('jetbot_drive_ai')
    parser.add_argument('-m', type=str, help='Mode to start the jetbot in', dest='mode', required=True)
    parser.add_argument(
        '-f', type=int, help='Frame to delete onwards in case of a mistake', dest='frame', required=False
    )
    args = parser.parse_args()

    collector = Collector(FRAMES_DIRECTORY, TICKS_PER_SECOND, CONTROLS_PICKLE_FILENAME)

    # New dataset
    if 'n' in args.mode:
        print('Starting new dataset')
        os.makedirs(FRAMES_DIRECTORY, exist_ok=True)
        collector.run_data_collection()
    # Resume on a current dataset in case bad inputs were made
    elif 'r' in args.mode and args.frame:
        print(f'Resuming from frame {args.frame}')
        os.makedirs(FRAMES_DIRECTORY, exist_ok=True)
        collector.run_data_collection(args.frame)
    else:
        print('Invalid mode')
