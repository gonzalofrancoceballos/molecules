import numpy as np


def apply_resolution(x, resolution):
    """
    Rounds numbers to a certain resolution

    :param x: input to be rounded (type: floar or np.array)
    :param resolution: desired resolution
    """

    return resolution * np.round(x / resolution)