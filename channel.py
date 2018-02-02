import numpy as np
from rice_channel import rice_channel


class Channel():

    def __init__(self, receivers, transmitters, breakpoint_distance,
                 carrier_freq, k_factor_dB, nlos_fading_stdev):
        # Store list that maps each row of channel array to appropriate
        # receiver node ident.
        rx_map = []
        for r in receivers:
            rx_map.extend([r.ident]*r.antenna_count)
        self.rx_ident_map = np.array(rx_map)
        # Store list that maps each column of channel array to appropriate
        # transmitter node ident.
        tx_map = []
        for t in transmitters:
            tx_map.extend([t.ident]*t.antenna_count)
        self.tx_ident_map = np.array(tx_map)
        self._H = rice_channel(receivers, transmitters, breakpoint_distance,
                               carrier_freq, k_factor_dB, nlos_fading_stdev)

    def full(self):
        return self._H

    def subchannel(self, receivers, transmitters):
        # Slice channel array to get channel between receiver(s) and
        # transmitter(s).

        try:
            rx_iterator = iter(receivers)
        except TypeError:
            # put single receiver object into a list
            rx_iterator = iter([receivers])
        row_inds = []
        for r in rx_iterator:
            row_inds.extend(np.where(self.rx_ident_map == r.ident)[0])

        try:
            tx_iterator = iter(transmitters)
        except TypeError:
            # put single transmitter object into a list
            tx_iterator = iter([transmitters])
        col_inds = []
        for t in tx_iterator:
            col_inds.extend(np.where(self.tx_ident_map == t.ident)[0])

        return self._H[row_inds][:, col_inds]
