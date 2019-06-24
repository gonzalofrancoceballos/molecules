import numpy as np
from pathlib import Path
import processing
import modeling


def main():
    data_path = Path("./data")
    resolution = 0.5
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
    modeling.train_model(train_data, molecule_structure_dict, possible_elements_dict, range_of_values)


if __name__ == "__main__":
    main()


