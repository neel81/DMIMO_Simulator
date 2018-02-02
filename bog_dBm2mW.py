import numpy as np


def bog_dBm2mW(dBm):
    return np.power(10, np.divide(dBm, 10))
