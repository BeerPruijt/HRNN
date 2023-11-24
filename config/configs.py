import os

# Relative path to the data directory
DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'data')
RESULTS_DIR = os.path.join(os.path.dirname(__file__), '..', 'results')

# Hyperparameters GRU
gru_params = {
}

# This is the estimation window so it it's set to 3 we have for example ([4, 5, 6], [7]) as a sequence
sequence_length = 5
