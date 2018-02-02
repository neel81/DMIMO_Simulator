import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from latexify import latexify
from latexify import format_axes


def compute_T_CBF(M, n_rx_ant, res_mode):
    no_of_bits = 12
    if res_mode == "high_res":
        no_of_bits = 16
    if n_rx_ant == 2:
        cbf_bits = 13 * no_of_bits
    elif n_rx_ant == 1:
        cbf_bits = 7 * no_of_bits
    no_of_subcarriers_cbf = 122
    no_of_subcarriers_snr = 62
    preamble = 40 * (10 ** -6)
    T_CBF = ((cbf_bits * no_of_subcarriers_cbf) + (4 * no_of_subcarriers_snr * n_rx_ant)) / (
            29.3 * (10 ** 6)) + preamble
    return T_CBF


def compute_overhead(M, K, S, TD):
    # K here denotes number of distinct STAs
    # S denotes number of STAs with one rx antenna chosen
    T_SIFS = 16 * 10 ** -6
    TS = (7.4 + 36 + 4 * M) * (10 ** -6)
    overhead = TS

    for i in range(K - S):  # account for overhead only once from STAs which use both antennas
        overhead += compute_T_CBF(M, 2, "high_res")
    for i in range(S):  # account for overhead from STAs which use only one rx antenna
        overhead += compute_T_CBF(M, 1, "high_res")

    overhead += (2 * K * T_SIFS) + (K - 1) * (40 * 10 ** -6)

    # TD = 1 * (10 ** -3)  # duration of TXOP = 10 ms
    percentage = (overhead / (overhead + TD))*100
    return percentage


M = 8
overhead = []
TD_range = range(1, 11, 2)

for TD in TD_range:
    i = 0
    overhead_ = []
    for k in range(4, 9, 1):
        overhead_.append(compute_overhead(M, k, i, TD*10**-3))
        i += 2
    overhead.append(overhead_)

params = latexify(columns=2)
matplotlib.rcParams.update(params)

colors = iter(cm.nipy_spectral(np.linspace(0, 1, len(TD_range))))
for overhead_ in overhead:
    plt.plot(overhead_, color=next(colors), marker='*', linewidth=2.5, label='TxOP time = {} ms'.format(TD_range[overhead.index(overhead_)]))
plt.xlabel('Number of users chosen')
plt.xticks(range(0, 5, 1), [4, 5, 6, 7, 8])
plt.ylabel('Channel sounding overhead (in percentage)')
plt.legend()
plt.title('Overhead percentage computed as per 802.11ac standards')
plt.tight_layout()
plt.savefig('channel_sounding_overhead.pdf')
plt.show()
