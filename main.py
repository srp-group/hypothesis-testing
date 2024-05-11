from configparser import ConfigParser
from core import ActiveLearning
import argparse

# Read in the configurations from the params.ini file
config = ConfigParser()
config.read('params.ini')

if __name__ == '__main__':
    # get the arguments from the command line
    parser = argparse.ArgumentParser()
    parser.add_argument('--d', type=str, help='Dataset name')
    parser.add_argument('--r', type=int, help='random seed')
    args = parser.parse_args()
    # start Active Learning! :)
    ActiveLearning(dataset_name=args.d, random_seed=args.r).run()