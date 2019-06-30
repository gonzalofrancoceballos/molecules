import processing

import numpy as np
import keras


class DataGenerator(keras.utils.Sequence):
    """
    Generator class
    """

    def __init__(self,
                 data,
                 molecule_structure_dict,
                 possible_elements_dict,
                 range_of_values,
                 batch_size,
                 shuffle_on_reset=True):

        self._data = data.sample(frac=1).reset_index(drop=True)
        self.data_size = data.shape[0]
        self.index_range = np.arange(0, self.data_size, dtype=int)
        self._molecule_structure_dict = molecule_structure_dict
        self._possible_elements_dict = possible_elements_dict
        self._range_of_values = range_of_values
        self.batch_size = batch_size
        self.shuffle_on_reset = shuffle_on_reset
        self.on_epoch_end()

    def __len__(self):
        """
        Denotes the number of batches per epoch
        """
        return int(np.floor(self._data.shape[0] / self.batch_size))

    def __getitem__(self, index):
        i_select = self.index_range[(index * self.batch_size):((index + 1) * self.batch_size)]

        x = []
        y = []
        for i in i_select:
            element = self._data.iloc[i]
            x_i = processing.get_canvas(element["molecule_name"],
                                        element["atom_index_0"],
                                        element["atom_index_1"],
                                        self._molecule_structure_dict,
                                        self._possible_elements_dict,
                                        self._range_of_values)
            y_i = np.array([element["scalar_coupling_constant"]])
            x.append(x_i)
            y.append(y_i)

        x = np.vstack(x)
        y = np.vstack(y)

        return x, y

    def on_epoch_end(self):
        """
        Updates indexes after each epoch
        """
        if self.shuffle_on_reset:
            np.random.shuffle(self.index_range)

