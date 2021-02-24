import numpy as np
import sounddevice as sd

from cfg_pyroom import fs
from DA_pyroom import DAwithPyroom

"""
CAUTION: Simulated dataset is in int16 dtype
"""
NPY_NAME = r'my_dataset'

BASE_PATH = r'./'

# Set systems paths
INPUT_PATH = BASE_PATH + NPY_NAME + '.npy'
OUTPUT_PATH = BASE_PATH + NPY_NAME + '_DA'+'.npy'

# Init class DA with pyroom
my_sim = DAwithPyroom(INPUT_PATH)

# Call method sim_dataset to create simulated dataset
my_dataset_simulated = my_sim.sim_dataset()

# Save GT
np.save(OUTPUT_PATH, my_dataset_simulated)

# Listen to a sample from the simulation
sd.play(my_dataset_simulated[1,:], fs)



