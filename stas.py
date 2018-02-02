import bog_params as params
import numpy as np
from antenna import create_ULA
from bog_dBm2mW import bog_dBm2mW
from get_noise_power import get_noise_power


class STA():

    def __init__(self, ident, x, y, z, antenna_count):
        self.ident = ident
        self.x = x
        self.y = y
        self.z = z
        if antenna_count > 0:
            self.antennas = create_ULA(self, antenna_count, 0.5*params.Pathloss.wavelength)
        else:
            self.antennas = []
        self.tx_power_mW = bog_dBm2mW(params.STA.TotalTxPower_dBm)
        self.wau = None
        self.noise_figure_dB = params.STA.NoiseFigure_dB
        self.n_mW = bog_dBm2mW(get_noise_power(params.Frame.DATA_Bandwidth_MHz,
                                               self.noise_figure_dB))

    @property
    def coordinates(self):
        return np.array([self.x, self.y, self.z])

    @property
    def antenna_coordinates(self):
        antenna_offsets = np.array([a.offset for a in self.antennas])
        return self.coordinates + antenna_offsets

    @property
    def antenna_count(self):
        return len(self.antennas)

    def __repr__(self):
        return "STA_{}".format(self.ident)
