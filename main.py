import pandas as pd
from configparser import ConfigParser
from core import ActiveLearning

# Read in the configurations from the params.ini file
config = ConfigParser()
config.read('params.ini')

if __name__ == '__main__':
    ActiveLearning()