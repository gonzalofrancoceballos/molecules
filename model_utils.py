import processing

import numpy as np
from tensorflow import keras


class DataGenerator(keras.utils.Sequence):
    """
    Generator class
    """

    def __init__(self,
                 train_data,
                 molecule_structure_dict,
                 possible_elements_dict,
                 range_of_values):

        self.train_data = train_data
        self.molecule_structure_dict = molecule_structure_dict
        self.possible_elements_dict = possible_elements_dict
        self.range_of_values = range_of_values

        self.on_epoch_end()

    def __len__(self):
        """
        Denotes the number of batches per epoch
        """
        return self.train_data.shape[0]

    def __getitem__(self, index):
        element = self.train_data.iloc[index]
        x = processing.get_canvas(element["molecule_name"],
                                  element["atom_index_0"],
                                  element["atom_index_1"],
                                  self.molecule_structure_dict,
                                  self.possible_elements_dict,
                                  self.range_of_values)
        y = np.array([element["scalar_coupling_constant"]])
        return x, y

    def on_epoch_end(self):
        """
        Updates indexes after each epoch
        """
        self.train_data = self.train_data.sample(frac=1).reset_index(drop=True)