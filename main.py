from configparser import ConfigParser
from core import ActiveLearning
import argparse

# Read in the configurations from the params.ini file
config = ConfigParser()
config.read('params.ini')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--d', type=str, help='Dataset name')
    args = parser.parse_args()
    ActiveLearning(dataset_name=args.d)