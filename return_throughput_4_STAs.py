#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 16 18:46:36 2017

@author: neel45
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 27 14:47:36 2017

@author: neel45
"""

import numpy as np
import random
from channel import Channel
import bog_params as params
from waus import WAU
from stas import STA
from antenna import create_ULA
import cvxpy as cvx
import matplotlib.pyplot as plt 
from operator import itemgetter 

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
                x = params.World.roomX_m*(0.5+roomX)
                y = params.World.roomY_m*(0.5+roomY)
                z = 1.5 + (roomZ+1)*params.World.roomZ_m
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
        x2 = 2*params.World.roomX_m 
        y1 = 0 
        y2 = 2*params.World.roomY_m 
        
        x = x1 + random.uniform(0, 1)*(x2-x1)
        y = y1 + random.uniform(0, 1)*(y2-y1)
        z = WAUs[0].z
        sta = STA(sta_ident, x, y, z, 0)
        STAs.append(sta)
    return STAs 

N_T = N_C = params.General.nAntennas 

N_O = 1.38*(10**-23)*290*80*(10**6) 
p_c = 0.0316

n_STAs = 4

WAUs = []
STAs = [] 

WAUs = set_wau_positions() 
STAs = set_sta_positions(WAUs, n_STAs) 

plt.figure() 
for wau in WAUs: 
    plt.scatter(wau.x, wau.y, marker='^', color='black', s=70) 
for sta in STAs: 
    plt.scatter(sta.x, sta.y, marker='*', color='red') 

plt.xlabel('x (in meters)') 
plt.xlim([-1,21]) 
plt.ylabel('y (in meters)')
plt.ylim([-1,21])
plt.show() 

n_WAUs = len(WAUs)

for sta in STAs:
        sta.antennas = create_ULA(sta, params.General.nAntennas,
                                          0.5*params.Pathloss.wavelength)
    
for wau in WAUs:
        wau.antennas = create_ULA(wau, params.General.nAntennas,
                                          0.5*params.Pathloss.wavelength)
    
chan_wau2sta = Channel(STAs, WAUs, params.Pathloss.breakpoint_distance,
                                        params.Pathloss.fc_GHz * 1e9,
                                        params.Pathloss.k_factor_dB,
                                        params.Pathloss.nlos_fading_stdev) 

chan_wau2sta = chan_wau2sta._H/np.sqrt(N_O) # Noise normalized channel gain matrix 

def cvx_power_alloc_equal_power(W, Pa, nt, nc, nac):
    Pc = Pa*np.ones((nt, 1))
    Q = np.kron(np.eye(nc), np.ones((nac, 1)))
    d = cvx.Variable(nc)
    objective = cvx.Maximize(cvx.sum_entries(cvx.log1p(d)/np.log(2))) 
    ## Constraining SNR to be between min required for MCS = 0 ( = 3.9 dB) and for MCS = 12 ( = 36 dB)
    constraints = [W @ Q @ d <= Pc, d >= 0, d <= 10**3.6]
    prob = cvx.Problem(objective, constraints)
    # prob.solve(solver='SCS') 
    prob.solve(max_iters=50000) 
    # print('Problem status: {}'.format(prob.status))
    return np.asarray(np.abs(d.value)).flatten(), prob.status  

def compute_snr(M, K, snr): 
    snr_list = [] 
    for x in snr: 
        snr_list.append(10*np.log10(((M - K + 1)/K)*(10**(x/10))))
    return snr_list 

def lookup_rate(snr): 
    if snr < 3.9: 
        rate = 0.0
    elif snr >= 3.9 and snr < 6.9: 
        rate = 36.0  
    elif snr >= 6.9 and snr < 9.3: 
        rate = 72.1 
    elif snr >= 9.3 and snr < 12.0: 
        rate = 108.1
    elif snr >= 12.0 and snr < 15.8: 
        rate = 144.1
    elif snr >= 15.8 and snr < 19.3: 
        rate = 216.2
    elif snr >= 19.3 and snr < 21.2: 
        rate = 288.2
    elif snr >= 21.2 and snr < 23.0: 
        rate = 324.3
    elif snr >= 23.0 and snr < 28.5: 
        rate = 360.3
    elif snr >= 28.5 and snr < 29.5: 
        rate = 432.4 
    elif snr >= 29.5 and snr < 32.5: 
        rate = 480.4 
    elif snr >= 32.5 and snr < 34.0: 
        rate = 540.4
    elif snr >= 34.0 :
        rate = 600.4 
    return rate 

def compute_T_CBF(M, n_rx_ant, res_mode): 
    no_of_bits = 12 
    if res_mode == "high_res": 
        no_of_bits = 16
    if n_rx_ant == 2 :
        cbf_bits = 13*no_of_bits 
    elif n_rx_ant == 1 :
        cbf_bits = 7*no_of_bits 
    no_of_subcarriers_cbf = 122 
    no_of_subcarriers_snr = 62 
    preamble = 40*(10**-6) 
    T_CBF = ((cbf_bits*no_of_subcarriers_cbf) + (4*no_of_subcarriers_snr*n_rx_ant))/(29.3*(10**6)) + preamble 
    return T_CBF 

def compute_throughput(M, K, S, b, snr_list): 
    ## K here denotes number of distinct STAs 
    ## S denotes number of STAs with one rx antenna chosen 
    T_SIFS = 16*10**-6 
    TS = (7.4 + 36 + 4*M)*(10**-6) 
    overhead = TS 
    
    for i in range(K - S):   ## account for overhead only once from STAs which use both antennas 
        overhead += compute_T_CBF(M, 2, "high_res") 
    for i in range(S):   ## account for overhead from STAs which use only one rx antenna 
        overhead += compute_T_CBF(M, 1, "high_res") 
    
    overhead += (2*K*T_SIFS) + (K-1)*(40*10**-6) 
    
    R = 0 
    TD = [] 
    for snr in snr_list: 
        r = lookup_rate(snr)*(10**6)
        TD.append(b*1500*8/r) 
    max_TD = max(TD) 
    # print('Ratio of overhead for number of rx antennas = {} is {}%'.format(K,overhead/(max_TD + overhead)*100)) 
    R = len(snr_list)*b*1500*8/(max_TD + overhead) 
    return R 

def range_(m, k): 
    if m == k: 
        return [m] 
    else: 
        return list(range(m, k+1)) 

b = 64  # Maximum frame aggregation rate 

T = 4
C = 4 

def max_val(l, i=-1):
    return max(enumerate(map(itemgetter(i), l)),key=itemgetter(1)) 

max_norm = list(np.linalg.norm(chan_wau2sta, axis=-1))   
T = 4 
C = n_STAs 

N_T = N_C = params.General.nAntennas 

W = np.linalg.pinv(chan_wau2sta) 

d_values = cvx_power_alloc_equal_power(W, p_c, T*N_T, C, N_C)[0] 
d_values = [10*np.log10(val) for val in d_values for _ in (0, 1)] 
rates_random = [lookup_rate(snr) for snr in d_values] 
R = compute_throughput(T*N_T, 4, 0, b, d_values)/(10**6) 

print('Throughput = {} Mbps'.format(R))
    