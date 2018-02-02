import numpy as np


def bog_mW2dBm(mW):
    return 10*np.log10(mW)
