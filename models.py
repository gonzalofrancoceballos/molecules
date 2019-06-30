from keras import layers, models
import numpy as np


def get_model(n):
    """
    Builds and returns model
    
    :param n: size of data (type: int)
    :return: model object
    """
    model = models.Sequential()
    
 
    model.add(layers.Conv3D(128, (n-1, n-1, n-1), activation='relu', input_shape=(n+1, n+1, n+1, 6)))
    model.add(layers.MaxPooling3D((2,2,2)))
    
#     n1 = int(np.floor(n/2))
#     n2 = int(np.floor((n-n1)/2))
#     model.add(layers.Conv3D(128, (n1, n1, n1), activation='relu', input_shape=(n+1, n+1, n+1, 6)))
#     model.add(layers.MaxPooling3D((2,2,2)))

#     model.add(layers.Conv3D(128, (n2, n2, n2), activation='relu'))
#     model.add(layers.MaxPooling3D((2,2,2)))
    
#     model.add(layers.Conv3D(128, (n, n, n), activation='relu', input_shape=(n + 1, n + 1, n + 1, 6)))
#     model.add(layers.MaxPooling3D((2, 2, 2)))

#     model.add(layers.Conv3D(64, (3, 3, 3), activation='relu'))
#     model.add(layers.MaxPooling3D((2, 2, 2)))
    
#     model.add(layers.Conv3D(16, (3, 3, 3), activation='relu'))
#     model.add(layers.MaxPooling3D((2, 2, 2)))
    
#     model.add(layers.Conv3D(16, (2, 2, 2), activation='relu'))
#     model.add(layers.MaxPooling3D((2, 2, 2)))

    model.add(layers.Flatten())
    model.add(layers.Dense(128, activation='relu'))
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dense(32, activation='relu'))
    model.add(layers.Dense(1, activation='linear'))

    model.summary()
    
    return model