import numpy as np
import random
from channel import Channel
import bog_params as params
from waus import WAU
from stas import STA
from antenna import create_ULA
import cvxpy as cvx
import matplotlib.pyplot as plt
import pylab as pl
from operator import itemgetter
import matplotlib.cm as cm
import matplotlib
from latexify import latexify
from latexify import format_axes

def set_wau_positions():
    wau_ind = 0
    WAUs = []
    no_of_floors = list(range(params.World.nRoomsZ))
    no_of_walls_Y = list(range(params.World.nRoomsY))
    no_of_walls_X = list(range(params.World.nRoomsX))

    for roomZ in no_of_floors:
        for roomY in no_of_walls_Y:
            for roomX in no_of_walls_X:
                wau_ind = wau_ind + 1
                x = params.World.roomX_m * (0.5 + roomX)
                y = params.World.roomY_m * (0.5 + roomY)
                z = 1.5 + (roomZ + 1) * params.World.roomZ_m
                WAUs.append(
                    WAU(wau_ind, x, y, z, 0))
    return WAUs


def set_sta_positions(WAUs, n_STAs):
    STAs = []
    for sta_ident in range(n_STAs):
        #        x1 = params.World.roomX_m + (-2)*params.World.roomX_m
        #        x2 = params.World.roomX_m + 2*params.World.roomX_m
        #        y1 = params.World.roomY_m + (-1)*params.World.roomY_m
        #        y2 = params.World.roomY_m + 1*params.World.roomY_m

        x1 = 0
        x2 = 2 * params.World.roomX_m
        y1 = 0
        y2 = 2 * params.World.roomY_m

        x = x1 + random.uniform(0, 1) * (x2 - x1)
        y = y1 + random.uniform(0, 1) * (y2 - y1)
        z = WAUs[0].z
        sta = STA(sta_ident, x, y, z, 0)
        STAs.append(sta)
    return STAs

n_STAs = 40
N_O = 1.38*(10**-23)*290*80*(10**6)

WAUs = set_wau_positions()
STAs = set_sta_positions(WAUs, n_STAs)

n_WAUs = len(WAUs)

for sta in STAs:
    sta.antennas = create_ULA(sta, params.General.nAntennas_STA,
                              0.5 * params.Pathloss.wavelength)

for wau in WAUs:
    wau.antennas = create_ULA(wau, params.General.nAntennas_WAU,
                              0.5 * params.Pathloss.wavelength)

chan_wau2sta = Channel(STAs, WAUs, params.Pathloss.breakpoint_distance,
                       params.Pathloss.fc_GHz * 1e9,
                       params.Pathloss.k_factor_dB,
                       params.Pathloss.nlos_fading_stdev)

chan_wau2sta = chan_wau2sta._H / np.sqrt(N_O)  # Noise normalized channel gain matrix
# chan_wau2sta = chan_wau2sta._H

print('Shape of channel matrix is : {}'.format(chan_wau2sta.shape))

list_of_corr_matrices = []

for i in range(0, 4):
    corr_matrix = np.zeros((n_STAs, n_STAs))
    for j in range(0, n_STAs):
        row_j = range(2*j, 2*j+2)
        column = range(2*i, 2*i+2)
        ch_j = chan_wau2sta[np.ix_(row_j, column)]
        u_j, s_j, v_j = np.linalg.svd(ch_j)
        for k in range(j, n_STAs):
            row_k = range(2*k, 2*k+2)
            ch_k = chan_wau2sta[np.ix_(row_k, column)]
            u_k, s_k, v_k = np.linalg.svd(ch_k)
            corr_jk = np.linalg.norm(ch_j @ ch_k.conj().T)/(np.linalg.norm(ch_j)*np.linalg.norm(ch_k))
            corr_matrix[k, j] = corr_jk
    list_of_corr_matrices.append(corr_matrix)

params = latexify(columns=2)
matplotlib.rcParams.update(params)

for i in range(len(list_of_corr_matrices)):
    cmap = cm.get_cmap('jet')
    cmap.set_bad('w')
    corr_matrix = list_of_corr_matrices[i]
    mask = np.tri(corr_matrix.shape[0], k=0)
    mask = mask.T
    corr_matrix = np.ma.array(corr_matrix, mask=mask)
    plt.matshow(corr_matrix, cmap=cmap, vmin=0, vmax=1)
    cbar = plt.colorbar(format='%.2f')
    cbar.set_label('Channel correlation between users')
    plt.title('Channel correlation matrix for WAU {}'.format(i+1), y=1.1)
    plt.gca().xaxis.tick_bottom()
    plt.tight_layout()
    plt.xlabel('User index')
    plt.ylabel('User index')
    plt.xticks(fontname='Computer Modern Sans Serif')
    plt.yticks(fontname='Computer Modern Sans Serif')
    plt.savefig('corr_WAU_'+str(i)+'.pdf')

# Averaging correlations using Fischer's Z transformation

cross_correlations = np.zeros((n_STAs, n_STAs))

for i in range(0, 40):
    cross_corrs = []
    for k in range(4):
        cross_corrs.append(0.5*np.log((1+list_of_corr_matrices[k][:, i])/(1-list_of_corr_matrices[k][:, i])))
    cross_corrs = np.array(cross_corrs)
    corr_i = np.mean(cross_corrs, axis=0)
    corr_i = (np.exp(2*corr_i) - 1)/(np.exp(2*corr_i) + 1)
    cross_correlations[:, i] = corr_i

# cmap = cm.get_cmap('jet')
# cmap.set_bad('w')
# mask = np.tri(cross_correlations.shape[0], k=0)
# mask = mask.T
# cross_correlations = np.ma.array(cross_correlations, mask=mask)
# plt.matshow(cross_correlations, cmap=cmap, vmin=0, vmax=1)
# cbar = plt.colorbar(format='%.2f')
# cbar.set_label('Channel correlation between users')
# plt.title('Averaged channel correlation \n Fischer Z Transform')
# plt.show()

# Correlation over all four WAUs

big_corr_matrix = np.zeros((n_STAs, n_STAs))
for j in range(0, n_STAs):
    row_j = range(2*j, 2*j+2)
    ch_j = chan_wau2sta[row_j, :]
    u_j, s_j, v_j = np.linalg.svd(ch_j)
    for k in range(j, n_STAs):
        row_k = range(2*k, 2*k+2)
        ch_k = chan_wau2sta[row_k, :]
        u_k, s_k, v_k = np.linalg.svd(ch_k)
        corr_jk = np.linalg.norm(ch_j @ ch_k.conj().T)/(np.linalg.norm(ch_j)*np.linalg.norm(ch_k))
        big_corr_matrix[k, j] = corr_jk

map = cm.get_cmap('jet')
cmap.set_bad('w')
mask = np.tri(big_corr_matrix.shape[0], k=0)
mask = mask.T
big_corr_matrix = np.ma.array(big_corr_matrix, mask=mask)
plt.matshow(big_corr_matrix, cmap=cmap, vmin=0, vmax=1)
cbar = plt.colorbar(format='%.2f')
cbar.set_label('Channel correlation between users')
plt.gca().xaxis.tick_bottom()
plt.title('Channel correlation across four WAUs', y=1.1)
plt.xticks(fontname='Computer Modern Sans Serif')
plt.yticks(fontname='Computer Modern Sans Serif')
plt.xlabel('User index')
plt.ylabel('User index')
plt.tight_layout()
plt.savefig('corr_4waus.pdf')
plt.show()



