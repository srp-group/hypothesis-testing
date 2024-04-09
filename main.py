import pandas as pd
from configparser import ConfigParser

# Read in the configurations from the params.ini file
config = ConfigParser()
config.read('params.ini')

def read_file(filename, dataset):

    # Read the file
    with open(filename, 'r') as file:
        lines = file.readlines()

    data = []
    for line in lines:
        parts = line.strip().split(' ')
        parts = list(filter(lambda x: x.strip(), parts)) # Filter the values if there are empty spaces present
        label = int(parts[0])
        features = [0] * int(config[dataset]['n_features']) # Initialise the features list according to the number of n_features
        for part in parts[1:]:
            index, value = part.split(':')
            features[int(index) - 1] = float(value)  # Subtracting 1 to adjust 1-based to 0-based indexing

        data.append([label] + features) # Add the label and feature in a line

    # Create column names for DataFrame
    columns = ['label'] + [f'feature_{i}' for i in range(1, int(config[dataset]['n_features'])+1)]

    # Create a DataFrame from the list of lists and return it
    return pd.DataFrame(data, columns=columns)


# Read each file and assign the returned df
dna_train = read_file(config['DNA']['train'], 'DNA')
dna_test = read_file(config['DNA']['test'], 'DNA')
dna_val = read_file(config['DNA']['val'], 'DNA')

splice_train = read_file(config['SPLICE']['train'], 'SPLICE')
splice_test = read_file(config['SPLICE']['test'], 'SPLICE')

protein_train = read_file(config['PROTEIN']['train'], 'PROTEIN')
protein_test = read_file(config['PROTEIN']['test'], 'PROTEIN')
protein_val = read_file(config['PROTEIN']['val'], 'PROTEIN')