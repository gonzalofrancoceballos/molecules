import model_utils
import models
import losses
from tensorflow.keras import callbacks


def train_model(train_data, molecule_structure_dict, possible_elements_dict, range_of_values,
                batch_size=4, tensorboard_dir="tensorboard/test_1"):
    """

    :param train_data:
    :param molecule_structure_dict:
    :param possible_elements_dict:
    :param range_of_values:
    :param batch_size:
    :param tensorboard_dir:
    :return:
    """

    n = len(range_of_values)

    data_generator = model_utils.DataGenerator(train_data,
                                               molecule_structure_dict,
                                               possible_elements_dict,
                                               range_of_values,
                                               batch_size=batch_size)

    model = models.get_model(n)
    model.compile(optimizer='adam',
                  loss=losses.log_mae,
                  metrics=['mae', losses.log_mae])

    tensorboard = callbacks.TensorBoard(log_dir=f"./../{tensorboard_dir}",
                                        histogram_freq=1,
                                        batch_size=16,
                                        write_grads=True,
                                        update_freq="batch")

    model.fit_generator(generator=data_generator,
                        use_multiprocessing=True, epochs=100,
                        workers=2,
                        callbacks=[tensorboard])
