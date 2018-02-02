from enum import Enum
import numpy as np
from antenna import create_ULA
import bog_params as params


class WAU():

    class PHYState(Enum):
        LISTEN = 0
        TX = 1

    class CCAState(Enum):
        CH_BUSY = 0
        CH_IDLE = 1

    def __init__(self, ident, x, y, z, antenna_count):
        self.ident = ident
        self.x = x
        self.y = y
        self.z = z
        self.pu = None
        self.stas = []
        if antenna_count > 0:
            self.antennas = create_ULA(self, antenna_count, 0.5*params.Pathloss.wavelength)
        else:
            self.antennas = []
        # statistics variables (move these to bog_waus?)
        self.statistics_sta_sinr = []
        self.sinr_intervals = []
        self.sinr_interval_prev_t = None
        self.sinr_interval_sinr_t = None
        self.cwmin = params.Backoff.aCWmin
        self.cca_state = WAU.CCAState.CH_IDLE
        self.phy_state = WAU.PHYState.LISTEN
        self.statistics_wau_mcsindex = []
        self.statistics_tx_events = []
        self.statistics_ntotalbytes = 0
        self.tx_changelinkdr_us = -1
        self.tx_nbytes = -1
        self.tx_power_mW = 0

    @property
    def is_listening(self):
        return self.phy_state is WAU.PHYState.LISTEN

    @property
    def is_txing(self):
        return self.phy_state is WAU.PHYState.TX

    @property
    def is_ch_idle(self):
        return self.cca_state is WAU.CCAState.CH_IDLE

    @property
    def is_ch_busy(self):
        return self.cca_state is WAU.CCAState.CH_BUSY

    def set_ch_busy(self):
        self.cca_state = WAU.CCAState.CH_BUSY

    def set_ch_idle(self):
        self.cca_state = WAU.CCAState.CH_IDLE

    def start_tx(self, p_mW, current_time):
        assert self.is_listening
        assert self.tx_power_mW == 0
        self.phy_state = WAU.PHYState.TX
        self.tx_power_mW = p_mW
        self.initialize_sinr_stats(current_time)

    def stop_tx(self):
        assert self.is_txing
        assert self.tx_power_mW > 0
        self.phy_state = WAU.PHYState.LISTEN
        self.tx_power_mW = 0

    @property
    def coordinates(self):
        return np.array([self.x, self.y, self.z])

    @property
    def antenna_count(self):
        return len(self.antennas)

    @property
    def antenna_coordinates(self):
        antenna_offsets = np.array([a.offset for a in self.antennas])
        return self.coordinates + antenna_offsets

    def collect_statistics(self, current_time):
        assert self.pu
        self.statistics_tx_events.append((self.pu.frame.tx_start_time_us,
                                          current_time))
        self.statistics_ntotalbytes += self.tx_nbytes
        self.store_average_sinr(current_time)
        self.tx_nbytes = -1

    def initialize_sinr_stats(self, current_time):
        self.sinr_interval_prev_t = 0
        self.sinr_interval_prev_sinr = None
        self.sinr_intervals = []

    def update_sinr_stats(self, current_time, p_mW, i_mW, n_mW):
        sinr = np.zeros(len(p_mW))
        for k in range(len(p_mW)):
            sinr[k] = p_mW[k]/(i_mW[k] + n_mW)
        delta_t = current_time - self.sinr_interval_prev_t
        if self.sinr_interval_prev_sinr is not None and delta_t > 0:
            self.sinr_intervals.append((delta_t, self.sinr_interval_prev_sinr))
        self.sinr_interval_prev_t = current_time
        self.sinr_interval_prev_sinr = sinr

    def store_average_sinr(self, current_time):
        assert current_time > self.sinr_interval_prev_t
        delta_t = current_time - self.sinr_interval_prev_t
        self.sinr_intervals.append((delta_t, self.sinr_interval_prev_sinr))
        t_total = 0
        sum_sinr = 0
        for (delta_t, sinr) in self.sinr_intervals:
            t_total += delta_t
            sum_sinr += delta_t*sinr
        avg_sinr = sum_sinr / t_total
        self.statistics_sta_sinr.append(avg_sinr)

    def __repr__(self):
        return "WAU_{}".format(self.ident)
