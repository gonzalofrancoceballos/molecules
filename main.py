import numpy as np
from pathlib import Path
import processing
import modeling

from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession

config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)

def main():
    data_path = Path("/home/gonzalo_franco/workspaces/python/molecules/data")
    resolution = 0.7
    min_xyz = -5
    max_xyz = 5
    possible_elements = ['N', 'C', 'H', 'O', 'F']
    possible_elements_dict = {}
    for i, e in enumerate(possible_elements):
        possible_elements_dict[e] = i

    range_of_values = np.arange(2 * min_xyz, 2 * max_xyz + resolution, resolution)
    range_of_values = np.round(range_of_values, 3)

    print("Loading data")
    molecule_structure_dict, train_data = processing.load_data(data_path, min_xyz, max_xyz, resolution)

    print("Training model")
    modeling.train_model(train_data,
                         molecule_structure_dict, 
                         possible_elements_dict, 
                         range_of_values,
                         batch_size=32,
                         tensorboard_dir="tensorboard/test_2")


if __name__ == "__main__":
    main()


