import numpy as np
import sounddevice as sd
import os
import os.path

from cfg_pyroom import fs
from DA_pyroom import DAwithPyroom


BASE_PATH = r'/home/luis/Downloads/ctc_SC/NPY_files/pyroom_input/'

for item in sorted(os.listdir(BASE_PATH)):
    NPY_NAME = item.split('.')[0]
    print(NPY_NAME)

    # Set systems paths
    INPUT_PATH = BASE_PATH + NPY_NAME + '.npy'
    OUTPUT_PATH = BASE_PATH + NPY_NAME + '_DA'+'.npy'

    # Init class DA with pyroom
    my_sim = DAwithPyroom(INPUT_PATH, float_flag=True)

    # Call method sim_dataset to create simulated dataset
    my_dataset_simulated = my_sim.sim_dataset(position=0)

    # Save GT
    np.save(OUTPUT_PATH, my_dataset_simulated)

    # Listen to a sample from the simulation
    # sd.play(my_dataset_simulated[1,:], fs)





