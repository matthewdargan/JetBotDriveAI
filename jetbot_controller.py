"""JetBot controller for either running data collection or running the self-driving model."""
import os
from argparse import ArgumentParser

from data_collection import Collector

# Directory for saving image data in
FRAMES_DIRECTORY: str = 'frames'

# Constant for data collection fps and controls per second
TICKS_PER_SECOND: int = 30

# Filename for the pickle file for storing human-input controls
CONTROLS_PICKLE_FILENAME: str = 'controlsPickle'

if __name__ == "__main__":
    # Parse arguments for the mode jetBot should start in (either data collection or model running)
    parser = ArgumentParser('jetbot_drive_ai')
    parser.add_argument('-m', type=str, help='Mode to start the jetbot in', dest='mode', required=True)
    args = parser.parse_args()

    if 'collect' in args.mode:
        os.makedirs(FRAMES_DIRECTORY, exist_ok=True)
        collector = Collector(FRAMES_DIRECTORY, TICKS_PER_SECOND, CONTROLS_PICKLE_FILENAME)
        collector.run_data_collection()
    elif 'train' in args.mode:
        pass
    else:
        print('Invalid mode')
