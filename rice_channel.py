import numpy as np
from numpy import matlib as mat
import math


def los_channel(rx_ant_coordinates, tx_ant_coordinates, wavelength):
    shape = rx_ant_coordinates.shape
    if (len(shape) == 1):
        nr = 1
    else:
        nr = len(rx_ant_coordinates)
    shape = tx_ant_coordinates.shape
    if (len(shape) == 1):
        nt = 1
    else:
        nt = len(tx_ant_coordinates)
    rr = mat.repmat(rx_ant_coordinates, nt, 1)
    tt = np.kron(tx_ant_coordinates, np.ones((nr, 1)))
    diff = np.power((rr - tt), 2)
    diff = np.sum(diff, axis=1)
    dist = np.power(diff, 0.5)
    dist = dist.reshape(nr, nt)
    dist = np.array(dist)
    los = np.zeros((nr, nt), dtype=complex)
    for ii in range(nr):
        for jj in range(nt):
            los[ii][jj] = np.exp(1j * 2 * np.pi * dist[ii][jj] / wavelength)
    return los


def rice_channel(rx_nodes, tx_nodes, breakpoint_distance, carrier_freq,
                 k_factor_dB, nlos_fading_stdev):
    los_fading_stdev = 3
    wavelength = 3.0 * (10 ** 8) / carrier_freq

    total_tx_ant = sum(txn.antenna_count for txn in tx_nodes)
    total_rx_ant = sum(rxn.antenna_count for rxn in rx_nodes)

    n_rx_nodes = len(rx_nodes)
    n_tx_nodes = len(tx_nodes)

    path_loss = np.zeros((n_rx_nodes, n_tx_nodes))
    shadow_fading = np.zeros((n_rx_nodes, n_tx_nodes))

    is_los = np.ones((n_rx_nodes, n_tx_nodes), dtype=bool)
    material_losses_dB = np.zeros((n_rx_nodes, n_tx_nodes))

    H = np.zeros((total_rx_ant, total_tx_ant), dtype=complex)

    row_id2 = 0

    for rr in range(n_rx_nodes):
        rxn = rx_nodes[rr]
        n_rx_ant = rxn.antenna_count

        row_id1 = row_id2
        row_id2 += n_rx_ant

        H_los = np.zeros((n_rx_ant, total_tx_ant), dtype=complex)
        H_nlos = np.zeros((n_rx_ant, total_tx_ant), dtype=complex)

        col_idx2 = 0

        for tt in range(n_tx_nodes):
            txn = tx_nodes[tt]
            n_tx_ant = txn.antenna_count

            col_idx1 = col_idx2
            col_idx2 = col_idx2 + n_tx_ant

            # d = math.sqrt(sum([(a-b)**2 for (a, b) in zip(rxn.coordinates, txn.coordinates)]))
            d = np.linalg.norm(rxn.coordinates - txn.coordinates)

            if (d < breakpoint_distance):
                # To compute PL between same rx/tx nodes (d = 0);
                # diagonal entries of tx_tx and rx_rx matrices (these entries
                # will never be used).
                pl = 20 * math.log10(4*np.pi*np.max([0.001, d]) / wavelength)
            else:
                pl = 20 * math.log10(
                    4 * np.pi * float(breakpoint_distance) / wavelength)
                pl += 35 * math.log10(float(d)/breakpoint_distance)
                is_los[rr][tt] = False
            path_loss[rr][tt] = max(0.0, pl)

            if (material_losses_dB[rr][tt] > 0):
                is_los[rr][tt] = False

            if (is_los[rr][tt]):
                k = 10 ** (k_factor_dB / 10.0)
                shadow_fading[rr][tt] = los_fading_stdev * np.random.normal()
            else:
                k = 0
                shadow_fading[rr][tt] = nlos_fading_stdev * np.random.normal()

            # compute total attenuation
            attenuation_dB = (path_loss[rr][tt] + shadow_fading[rr][tt]
                              + material_losses_dB[rr][tt])
            attenuation = 10 ** (-attenuation_dB / 20.0)

            # compute channel matrix component which is LOS
            if (is_los[rr][tt]):
                h_los = attenuation * math.sqrt(k/(k + 1)) * los_channel(
                    rxn.antenna_coordinates, txn.antenna_coordinates, wavelength)
            else:
                h_los = 0.0

            # compute channel matrix component which is NLOS
            h_iid = math.sqrt(0.5)*(np.random.randn(n_rx_ant, n_tx_ant)
                                    + 1j*np.random.randn(n_rx_ant, n_tx_ant))
            h_nlos = attenuation * math.sqrt(1 / (k + 1)) * h_iid
            H_los[:, list(range(col_idx1, col_idx2))] = h_los
            H_nlos[:, list(range(col_idx1, col_idx2))] = h_nlos

        H_i = H_los + H_nlos
        H[list(range(row_id1, row_id2)), :] = H_i

    return H
