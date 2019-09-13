"""JetBot controller for either running data collection or running the self-driving model."""
import os
from argparse import ArgumentParser

from data_collection import Collector

# Directory for saving image data in
frames_directory: str = 'frames'

# Constant for data collection fps and controls per second
ticks_per_second = 10

# Filename for the pickle file for storing human-input controls
controls_pickle_filename = 'controlsPickle'

if __name__ == "__main__":
    # Parse arguments for the mode jetBot should start in (either data collection or model running)
    parser = ArgumentParser('jetbot_drive_ai')
    parser.add_argument('-m', type=str, help='Mode to start the jetbot in', dest='mode', required=True)
    args = parser.parse_args()

    if 'collect' in args.mode:
        os.makedirs(frames_directory, exist_ok=True)
        collector = Collector(frames_directory, ticks_per_second, controls_pickle_filename)
        collector.run_data_collection()
    else:
        pass
