import numpy as np
import random


class Antenna():
    def __init__(self, device, offset):
        self.device = device
        self.offset = offset

    def coordinates(self):
        return self.device.coordinates + self.offset

    def __repr__(self):
        return 'Antenna<{0},{1},{2}>'.format(
            self.offset[0], self.offset[1], self.offset[2])


def create_ULA(device, antenna_count, antenna_spacing, rotation_rad=None):
    if rotation_rad is None:
        rotation_rad = random.uniform(0, 1) * np.pi
    array_offsets = np.zeros((antenna_count, 3))
    array_offsets[:, 0] = np.linspace(-0.5, 0.5, antenna_count)*(
        (antenna_count-1)*antenna_spacing)
    if rotation_rad != 0:
        R = [[np.cos(rotation_rad), -np.sin(rotation_rad), 0],
             [np.sin(rotation_rad), np.cos(rotation_rad), 0],
             [0, 0, 1]]
        array_offsets = (R@array_offsets.T).T
    return [Antenna(device, offset) for offset in array_offsets]
