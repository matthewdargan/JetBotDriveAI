"""Data collector for getting images and pairing them with human-input controls."""
import os
import pickle
import time
from typing import Dict, Tuple

from jetbot import Camera, Robot
from util.controller import Controller


class Collector:
    def __init__(self, frames_directory, ticks_per_second, controls_pickle_filename):
        """
        Initializes data collection state.
        :param frames_directory: directory to save frames into
        :param ticks_per_second: constant for data collection fps and controls per second
        :param controls_pickle_filename: filename for the pickle file for storing human-input controls
        """
        self.robot = Robot()

        # TODO: Find way to network controller inputs over to JetBot during data collection in real-time
        self.controller: Controller = Controller()

        self.controls: Dict[int, Tuple[float, float]] = {}
        self.controls_pickle_filename: str = controls_pickle_filename

        # TODO: Figure out way to set capture width and height here
        self.camera = Camera.instance()
        self.frames_directory: str = frames_directory
        self.frame_index: int = 0
        self.ticks_per_second: int = ticks_per_second

    def run_data_collection(self):
        """Runs data collection for controller inputs and camera frames until pause."""
        print('Starting data collection')
        self.resume()

        while True:
            try:
                # Save image and controls
                self.save_frame()
                self.save_controls()
                self.frame_index += 1

                # Sleep tick
                time.sleep(1. / self.ticks_per_second)
            except KeyboardInterrupt:
                self.pause()

    def save_frame(self):
        """Saves a frame into a file in the frames_directory."""
        file_path = f"{self.frames_directory}/{self.frame_index}.png"
        with open(file_path, 'wb') as f:
            f.write(self.camera.value)

    def save_controls(self):
        """Saves controls from the human-input controller as a left-motor vector and a right-motor vector."""
        # TODO: Receive controller inputs from the network here
        self.controls[self.frame_index] = get_controls(self.controller)

    def pause(self):
        """Pauses data collection and pickles controls."""
        with open(self.controls_pickle_filename, 'wb') as f:
            pickle.dump(self.controls, f)

        print('Pausing data collection')

    def resume(self):
        """Unpickles controls and resumes data collection."""
        if os.path.isfile(self.controls_pickle_filename):
            with open(self.controls_pickle_filename, 'rb') as f:
                try:
                    self.controls = pickle.load(f)
                    self.frame_index = self.controls.keys().sort()[-1] + 1
                    print('Resuming data collection')
                except FileNotFoundError:
                    print(f"{self.controls_pickle_filename} not found")
