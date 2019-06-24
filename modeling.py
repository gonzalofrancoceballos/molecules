import model_utils
import models


def train_model(train_data, molecule_structure_dict, possible_elements_dict, range_of_values):
    """

    :param train_data:
    :param molecule_structure_dict:
    :param possible_elements_dict:
    :param range_of_values:
    :return:
    """

    n = len(range_of_values)

    data_generator = model_utils.DataGenerator(train_data,
                                               molecule_structure_dict,
                                               possible_elements_dict,
                                               range_of_values)

    model = models.get_model(n)
    model.compile(optimizer='adam',
                  loss='mae',
                  metrics=['mae'])

    model.fit_generator(generator=data_generator,
                        use_multiprocessing=True, epochs=100,
                        workers=2)
