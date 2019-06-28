from tensorflow.keras import layers, models


def get_model(n):
    """
    Builds and returns model
    
    :param n: size of data (type: int)
    :return: model object
    """
    model = models.Sequential()

    model.add(layers.Conv3D(32, (5, 5, 5), activation='relu', input_shape=(n + 1, n + 1, n + 1, 6)))
    model.add(layers.MaxPooling3D((2, 2, 2)))

    model.add(layers.Conv3D(16, (3, 3, 3), activation='relu'))
    model.add(layers.MaxPooling3D((2, 2, 2)))

    model.add(layers.Conv3D(64, (3, 3, 3), activation='relu'))
    model.add(layers.MaxPooling3D((2, 2, 2)))

    model.add(layers.Flatten())
    model.add(layers.Dense(32, activation='relu'))
    model.add(layers.Dense(10, activation='relu'))
    model.add(layers.Dense(1, activation='linear'))

    model.summary()