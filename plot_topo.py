#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 27 14:19:53 2017

@author: neel45
"""

import numpy as np
import random
from channel import Channel
import bog_params as params
from waus import WAU
from stas import STA
from antenna import create_ULA
import matplotlib.pyplot as plt 
import pylab as pl 

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

N_T = params.General.nAntennas_WAU 
N_C = params.General.nAntennas_STA 

N_O = 1.38*(10**-23)*290*80*(10**6) 
p_c = 0.0316

n_STAs = 120

plot = False ## bool to plot cases in which SNR drops below 15 dB or throughput less than 800 Mbps (set to True if individual plots are desired)

WAUs = []
STAs = [] 

WAUs = set_wau_positions() 
STAs = set_sta_positions(WAUs, n_STAs) 

plt.figure() 
for wau in WAUs: 
    plt.scatter(wau.x, wau.y, marker='^', color='black', s=70) 
for sta in STAs: 
    plt.scatter(sta.x, sta.y, marker='*', color='red') 

plt.xlabel(r'x (in meters)') 
plt.ylabel(r'y (in meters)') 
plt.title(r'Distribution of STAs in a sample D-MIMOO group')
plt.show() 

n_WAUs = len(WAUs)

for sta in STAs:
        sta.antennas = create_ULA(sta, params.General.nAntennas_STA,
                                          0.5*params.Pathloss.wavelength)
    
for wau in WAUs:
        wau.antennas = create_ULA(wau, params.General.nAntennas_WAU,
                                          0.5*params.Pathloss.wavelength)
    
chan_wau2sta = Channel(STAs, WAUs, params.Pathloss.breakpoint_distance,
                                        params.Pathloss.fc_GHz * 1e9,
                                        params.Pathloss.k_factor_dB,
                                        params.Pathloss.nlos_fading_stdev) 

chan_wau2sta = chan_wau2sta._H/np.sqrt(N_O) # Noise normalized channel gain matrix 

indices = [] 
main_indices = [] 
i = 1 
flag = True

chan_matrix = np.absolute(chan_wau2sta) 
chan_matrix.sort() 

max_vector = list(chan_matrix[:,-1])
decreasing_order = []   ## contains STAs (both rx antennas) arranged in decreasing order of RSS 

k = 1 
while flag: 
    max_elem = max(max_vector)
    ind = max_vector.index(max_elem)
    if ind%2 ==  0: 
        colocated_index = ind+1 
        decreasing_order.append(ind) 
        decreasing_order.append(colocated_index) 
    else: 
        colocated_index = ind-1 
        decreasing_order.append(colocated_index) 
        decreasing_order.append(ind) 
    max_vector[ind] = 0
    max_vector[colocated_index] = 0 
    if len(decreasing_order) == n_STAs*2: 
        flag = False 

max_norm = list(np.linalg.norm(chan_wau2sta, axis=-1)) 
dec_order_norm = []   ## contains STAs (both rx antennas) arranged in decreasing order of norm 

k = 1
flag = True 
while flag: 
    max_elem = max(max_norm)
    ind = max_norm.index(max_elem) 
    if ind%2 ==  0: 
        colocated_index = ind+1 
        dec_order_norm.append(ind) 
        dec_order_norm.append(colocated_index) 
    else: 
        colocated_index = ind-1 
        dec_order_norm.append(colocated_index) 
        dec_order_norm.append(ind) 
    max_norm[ind] = 0
    max_norm[colocated_index] = 0 
    if len(dec_order_norm) == n_STAs*2: 
        flag = False 

main_indices = []
for j in range(0, len(decreasing_order), 8): 
    main_indices.append(decreasing_order[j:j+8]) 

main_indices_norm = []
for j in range(0, len(dec_order_norm), 8): 
    main_indices_norm.append(dec_order_norm[j:j+8]) 

plt.figure() 
for wau in WAUs: 
    plt.scatter(wau.x, wau.y, marker='^', color='black', s=70) 
import matplotlib.cm as cm 
colors = iter(cm.nipy_spectral(np.linspace(0, 1, len(main_indices)))) 
for sub_group in main_indices: 
    clr = next(colors)
    z = 0 
    while z < (len(sub_group)): 
        sta = STAs[sub_group[z]//2] 
        pl.scatter(sta.x, sta.y, color = clr, marker = '*', s = 0.1)
        pl.text(sta.x, sta.y, str(main_indices.index(sub_group)), color=clr, fontsize=10.5)
        z += 2 
# pl.margins(0.1)
plt.title(r'Grouping with RSS-based selection') 
plt.xlabel(r'x (in meters)')
plt.ylabel(r'y (in meters)')
plt.show()

plt.figure()  
import matplotlib.cm as cm 
colors = iter(cm.nipy_spectral(np.linspace(0, 1, len(main_indices_norm)))) 
for sub_group in main_indices_norm: 
    clr = next(colors)
    z = 0 
    while z < (len(sub_group)): 
        sta = STAs[sub_group[z]//2] 
        plt.scatter(sta.x, sta.y, color = clr, marker = '*', s = 0.1) 
        pl.text(sta.x, sta.y, str(main_indices_norm.index(sub_group)), color=clr, fontsize=10.5)
        z += 2 
for wau in WAUs: 
    plt.scatter(wau.x, wau.y, marker='^', color='black', s=70) 
plt.title(r'Grouping with Norm-based selection') 
plt.xlabel(r'x (in meters)')
plt.ylabel(r'y (in meters)')
plt.show()