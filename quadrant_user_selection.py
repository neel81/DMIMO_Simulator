#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 13 11:26:16 2017

@author: Neelakantan
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
import pylab as pl
from operator import itemgetter 
import matplotlib.cm as cm
import matplotlib
from latexify import latexify
from latexify import format_axes
from statistics import median
import itertools
import pickle


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

def cvx_power_alloc_equal_power(W, Pa, nt, nc, nac):
    #Pc = Pa*np.ones((nt, 1))  ## per-antenna power constraint
    Q = np.kron(np.eye(nc), np.ones((nac, 1)))
    d = cvx.Variable(nc)
    objective = cvx.Maximize(cvx.sum_entries(cvx.log1p(d)/np.log(2)))
    if nc > 1:
        constraints = [cvx.sum_entries(W @ Q @ d) <= Pa, d >= 0, d <= 10**3.6] 
    if nc == 1: 
        constraints = [cvx.sum_entries(W @ Q) * d <= Pa, d >= 0, d <= 10**3.6] 
    prob = cvx.Problem(objective, constraints)
    # prob.solve(solver='SCS') 
    prob.solve(max_iters=50000) 
    # print('Problem status: {}'.format(prob.status))
    return np.asarray(np.abs(d.value)).flatten(), prob.status  

def cvx_power_alloc_unequal_power(W, Pa, nt, nc, nac): 
    #Pc = Pa*np.ones((nt,1))  ## per-antenna power constraint
    d = cvx.Variable(nc*nac) 
    objective = cvx.Maximize(cvx.sum_entries(cvx.log1p(d)/np.log(2))) 
    constraints = [cvx.sum_entries(W @ d) <= Pa, d >= 0, d <= 10**3.6] 
    prob = cvx.Problem(objective, constraints) 
    prob.solve(max_iters=50000) 
    # print('Problem status: {}'.format(prob.status))
    return np.asarray(np.abs(d.value)).flatten() 

def waterfilling_solver(b, A, tol=0.01, max_iters=4000):
    """
    Inputs:
        b: levels (e.g., channel gains)
        A: power constraint
        tol: tolerance relative to power constraint
    """
    if np.min(b) < 1:
        xi = np.array([0, len(b)/np.min(b)*A], dtype=np.float_)
    else:
        xi = np.array([0, len(b)*A], dtype=np.float_)
    for i in range(max_iters):
        m = (xi[0] + xi[1])/2
        levels = np.maximum(m - 1/b, 0)
        d = np.sum(levels) - A
        if d < -tol:
            xi[0] = m
        elif d > tol:
            xi[1] = m
        else:
            return levels
    raise ValueError("Could not find solution", b, A, xi, d)


def palloc_wf_equal_power(W, Pa, nt, nc, nac):
    Q = np.kron(np.eye(nc), np.ones((nac, 1)))
    b = 1/(np.ones((1, nt))@W@Q)
    x = waterfilling_solver(b[0], Pa)
    return x*b 


def palloc_wf_unequal_power(W, Pa, nt, nc, nac):
    b = 1/(np.ones((1, nt))@W)
    x = waterfilling_solver(b[0], Pa)
    return x*b


def compute_snr(M, K, snr): 
    snr_list = [] 
    for x in snr: 
        snr_list.append(10*np.log10(((M - K + 1)/K)*(10**(x/10))))
    return snr_list 


def lookup_rate(snr): 
    if snr < 3.9: 
        rate = 0.0
    elif 3.9 <= snr < 6.9:
        rate = 36.0  
    elif 6.9 <= snr < 9.3:
        rate = 72.1 
    elif 9.3 <= snr < 12.0:
        rate = 108.1
    elif 12.0 <= snr < 15.8:
        rate = 144.1
    elif 15.8 <= snr < 19.3:
        rate = 216.2
    elif 19.3 <= snr < 21.2:
        rate = 288.2
    elif 21.2 <= snr < 23.0:
        rate = 324.3
    elif 23.0 <= snr < 28.5:
        rate = 360.3
    elif 28.5 <= snr < 29.5:
        rate = 432.4 
    elif 29.5 < snr < 32.5:
        rate = 480.4 
    elif 32.5 <= snr < 34.0:
        rate = 540.4
    elif snr >= 34.0 :
        rate = 600.4 
    return rate 


def lookup_mcs(snr): 
    if snr < 3.9: 
        mcs = -1
    elif 3.9 <= snr < 6.9:
        mcs = 0
    elif 6.9 <= snr < 9.3:
        mcs = 1
    elif 9.3 <= snr < 12.0:
        mcs = 2
    elif 12.0 <= snr < 15.8:
        mcs = 3
    elif 15.8 <= snr < 19.3:
        mcs = 4
    elif 19.3 <= snr < 21.2:
        mcs = 5
    elif 21.2 <= snr < 23.0:
        mcs = 6
    elif 23.0 <= snr < 28.5:
        mcs = 7
    elif 28.5 <= snr < 29.5:
        mcs = 8
    elif 29.5 <= snr < 32.5:
        mcs = 9
    elif 32.5 <= snr < 34.0:
        mcs = 10
    elif snr >= 34.0 :
        mcs = 11
    return mcs


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


def compute_throughput(M, K, S, snr_list): 
    # K here denotes number of distinct STAs
    # S denotes number of STAs with one rx antenna chosen
    T_SIFS = 16*10**-6 
    TS = (7.4 + 36 + 4*M)*(10**-6) 
    overhead = TS 
    
    for i in range(K - S):   # account for overhead only once from STAs which use both antennas
        overhead += compute_T_CBF(M, 2, "high_res") 
    for i in range(S):   # account for overhead from STAs which use only one rx antenna
        overhead += compute_T_CBF(M, 1, "high_res") 
    
    overhead += (2*K*T_SIFS) + (K-1)*(40*10**-6) 
    
    R = 0 
    bits_sent = 0 
    TD = 10*(10**-3)  # duration of TXOP = 10 ms
    for snr_ in snr_list:
        r = lookup_rate(snr_)
        bits_sent += (r*TD)
    # print('Ratio of overhead for number of rx antennas = {} is {}%'.format(K,overhead/(max_TD + overhead)*100)) 
    R = bits_sent/(TD + overhead)
    return R 


def range_(m, k): 
    if m == k: 
        return [m] 
    else: 
        return list(range(m, k+1)) 


def max_val(l, i=-1):
    return max(enumerate(map(itemgetter(i), l)),key=itemgetter(1)) 


def check_snr(snr_list): 
    poor_STA_index = [] 
    if any(z<3.9 for z in snr_list): 
        poor_STA_index = [y for y, snr in enumerate(snr_list) if snr < 3.9]
    return poor_STA_index 


N_T = params.General.nAntennas_WAU 
N_C = params.General.nAntennas_STA 

T = 4 
C = 4 

N_O = 1.38*(10**-23)*290*80*(10**6) 
p_c = 0.01

n_STAs = 40

tput_random = []  # Random STA selection
tput_random_streams = [] 
tput_RR_norm = [] # Round Robin selection based on norm of each row (all streams per STA)
tput_norm_1_sel = [] # Round Robin selection based on norm (fewer than max streams per STA) --- computed SNR
tput_norm_1_sel_all8 = []
tput_corr_users = []
tput_oracle_users = []
tput_oracle_streams = []

per_stream_tput_random_users = []
per_stream_tput_random_streams = []
per_stream_tput_norm_users = []
per_stream_tput_norm_streams = []

no_of_single_STAs_norm = [] 
no_of_single_STAs_random_streams = [] 

per_stream_SNR_random = [] 
per_stream_SNR_random_streams = [] 
per_stream_SNR_norm = [] 
per_stream_SNR_norm1 = []
per_stream_SNR_corr_users = []

per_stream_mcs_random = [] 
per_stream_mcs_random_streams = [] 
per_stream_mcs_norm = [] 
per_stream_mcs_norm1 = []
per_stream_mcs_corr_users = []

infeasible_random = 0 
infeasible_norm = 0
infeasible_corr_users = 0
infeasible_random_streams = 0
infeasible_norm_streams = 0
infeasible_oracle = 0

cond_random_users = []
cond_random_streams = []
cond_norm_users = []
cond_norm_streams = []

correlations_random = []
correlations_random_streams = []
correlations_norm = []
correlations_norm_streams = []

no_of_streams_random_users = []
no_of_streams_random_streams = []
no_of_streams_norm_users = []
no_of_streams_norm_streams = []

no_of_times_user_chosen = [0]*(n_STAs) 

for big_iter in range(54): 
    print('Drop = {}'.format(big_iter))
    WAUs = []
    STAs = [] 
    
    WAUs = set_wau_positions() 
    STAs = set_sta_positions(WAUs, n_STAs) 
    
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

    # Find correlation between users across all four WAUs.

    big_corr_matrix = np.zeros((n_STAs, n_STAs))
    for j in range(0, n_STAs):
        row_j = range(2 * j, 2 * j + 2)
        ch_j = chan_wau2sta[row_j, :]
        u_j, s_j, v_j = np.linalg.svd(ch_j)
        for k in range(j, n_STAs):
            row_k = range(2 * k, 2 * k + 2)
            ch_k = chan_wau2sta[row_k, :]
            u_k, s_k, v_k = np.linalg.svd(ch_k)
            corr_jk = np.linalg.norm(ch_j @ ch_k.conj().T) / (np.linalg.norm(ch_j) * np.linalg.norm(ch_k))
            big_corr_matrix[k, j] = corr_jk
            big_corr_matrix[j, k] = corr_jk


    #  Old code to order streams in decreasing order of norm -- no round robin selection of WAUs

#    max_norm = list(np.linalg.norm(chan_wau2sta, axis=-1))
#    flag = True
#    dec_order_1 = []   ## contains rx antennas (not STAs) in decreasing order of norm
#    k = 1
#    while flag: 
#        max_elem = max(max_norm)
#        ind = max_norm.index(max_elem)
#        dec_order_1.append(ind) 
#        max_norm[ind] = 0
#        if len(dec_order_1) == n_STAs*2: 
#            flag = False 
#    
#    main_ind_norm = [] 
#    for j in range(0, len(dec_order_1), 8): 
#        main_ind_norm.append(dec_order_1[j:j+8]) 
    
    ordering_waus_2streams = [] 
    
    for r in range(0, chan_wau2sta.shape[1], 2):
        norms = [] 
        for c in range(0, chan_wau2sta.shape[0], 2):
            chan_vector = chan_wau2sta[np.ix_(range(c, c+2), range(r, r+2))]
            norms.append(np.linalg.norm(chan_vector)) 
        k = 1
        flag = True 
        dec_order_norm = [] 
        while flag: 
            max_elem = max(norms)
            ind = norms.index(max_elem)
            dec_order_norm.append(ind)
            norms[ind] = 0
            if len(dec_order_norm) == n_STAs: 
                flag = False 
        ordering_waus_2streams.append(dec_order_norm)
    
    ordering_waus_1stream = [] 
    
    for r in range(0, chan_wau2sta.shape[1], 2):
        norms = [] 
        for c in range(0,chan_wau2sta.shape[0]): 
            chan_vector = chan_wau2sta[c, range(r, r+2)]
            norms.append(np.linalg.norm(chan_vector)) 
        k = 1
        flag = True 
        dec_order_norm = [] 
        while flag: 
            max_elem = max(norms)
            ind = norms.index(max_elem)
            dec_order_norm.append(ind)
            norms[ind] = 0
            if len(dec_order_norm) == 2*n_STAs: 
                flag = False 
        ordering_waus_1stream.append(dec_order_norm)
    
    no_of_trials = 500 
    max_norm = list(np.linalg.norm(chan_wau2sta, axis=-1)) 
    
    chosen_stas = [] 
    chosen_streams = [] 
    index_list = [0]*4 
    index_list_streams = [0]*4

    selected_users = []
    
    for i in range(no_of_trials):  # Each 'i' represents one TXOP
        if i % 100 == 0:
            print('Iteration = {}'.format(i)) 
        
        STA_random = [] 
        j = 0 
        while j<4: 
            id = random.randint(0, 2*n_STAs-1) 
            if id not in STA_random and id+1 not in STA_random and id-1 not in STA_random: 
                if id%2 == 0: 
                    STA_random.append(id) 
                    STA_random.append(id+1) 
                else: 
                    STA_random.append(id-1) 
                    STA_random.append(id)
                j = j+1  
        
        streams_random = random.sample(range(0, 2*n_STAs-1), 8)
        
        potential_stas = [] 
        rr = random.sample([0,1,2,3],4)   # Chose WAUs randomly for selecting users
        for z in rr:
            STA_list = ordering_waus_2streams[z] 
            m = index_list[z] 
            sta = STA_list[m] 
            while sta in chosen_stas: 
                m = (m+1) % len(STA_list)
                sta = STA_list[m] 
            index_list[z] = m 
            chosen_stas.append(sta) 
            if len(chosen_stas) == n_STAs:
                chosen_stas = [] 
                index_list = [0]*4 
            potential_stas.append(sta) 
         
        potential_streams = [] 
        for z in rr:
            stream_list = ordering_waus_1stream[z] 
            m = index_list_streams[z] 
            stream_1 = stream_list[m] 
            while stream_1 in chosen_streams: 
                m = (m+1) % len(stream_list)
                stream_1 = stream_list[m] 
            chosen_streams.append(stream_1) 
            if len(chosen_streams) == 2*n_STAs:
                chosen_streams = [] 
                index_list_streams = [0]*4 
            stream_2 = stream_list[m] 
            while stream_2 in chosen_streams: 
                m = (m+1)%len(stream_list) 
                stream_2 = stream_list[m] 
            index_list_streams[z] = m 
            chosen_streams.append(stream_2) 
            if len(chosen_streams) == 2*n_STAs:
                chosen_streams = [] 
                index_list_streams = [0]*4
            potential_streams.append(stream_1) 
            potential_streams.append(stream_2)
        
        
#        plt.figure() 
#        for wau in WAUs: 
#            plt.scatter(wau.x, wau.y, marker='^', color='black', s=70) 
#        for sta in potential_stas: 
#            plt.scatter(STAs[sta].x, STAs[sta].y, marker='*', color='red') 
#            pl.text(STAs[sta].x, STAs[sta].y, str(sta), color='blue', fontsize=10.5)
#        plt.xlim([0,21]) 
#        plt.ylim([0,21])
#        plt.show() 
        
        streams_norm = [] 
        
        for z in range(len(potential_stas)): 
            streams_norm.append(2*potential_stas[z]) 
            streams_norm.append(2*potential_stas[z]+1)

        # Selection of users based on cross_correlation
        # For each TXOP, start by picking a random previously unselected user

        chosen_users_TXOP = []
        u1 = np.random.randint(0, n_STAs)
        while u1 in selected_users:
            u1 = np.random.randint(0, n_STAs)
        selected_users.append(u1)
        chosen_users_TXOP.append(u1)
        corr_vector_u1 = big_corr_matrix[:, u1].copy()
        flag = True
        while flag:
            u2 = np.argmin(corr_vector_u1)
            while u2 in selected_users:
                u2 = np.argmin(corr_vector_u1)
                corr_vector_u1[u2] = 5  # some high number
            corr_vector_u1[u2] = 5  # some high number
            selected_users.append(u2)
            chosen_users_TXOP.append(u2)
            if len(selected_users) == n_STAs:
                selected_users = []
            if len(chosen_users_TXOP) == 4:
                flag = False

        chosen_streams_TXOP = []
        for user in chosen_users_TXOP:
            chosen_streams_TXOP.append(2*user)
            chosen_streams_TXOP.append(2*user + 1)
        
        R_list_new = [] 
        #streams_norm_1 = main_ind_norm[i%len(main_ind_norm)].copy() 
        streams_norm_1 = potential_streams 
    
        no_rx_antennas = range_(1, min(T*N_T, len(streams_norm_1)))
        for k in no_rx_antennas: 
            considered_group = random.sample(streams_norm_1, k) 
            channel_norms = [max_norm[stream] for stream in considered_group] 
            snr = 10*np.log10([p_c*(n**2) for n in channel_norms])
            computed_snr = compute_snr(T*N_T, k, snr) 
            distinct_STAs = len(considered_group) 
            single_rx_STAs = 0 
            for rx_ant in considered_group: 
                if (rx_ant+1 in considered_group) or (rx_ant-1 in considered_group): 
                    distinct_STAs -= 0.5 
                else: 
                    single_rx_STAs += 1 
            distinct_STAs = int(distinct_STAs)
            R = compute_throughput(T*N_T, distinct_STAs, single_rx_STAs, computed_snr)
            potential_list = [] 
            potential_list.append(considered_group) 
            potential_list.append(R)
            R_list_new.append(potential_list) 
        
        index = max_val(R_list_new)[0]
        streams_norm_1_sel = R_list_new[index][0]   # Norm based selection of rx antennas ---based on computed SNR 
        # no_of_streams_norm.append(len(streams_norm_1_sel))
        
        for c in range(len(no_of_times_user_chosen)): 
            if (2*c in streams_norm_1_sel) or (2*c+1 in streams_norm_1_sel): 
                no_of_times_user_chosen[c] += 1 
            
        H_random = chan_wau2sta[STA_random, :].copy()
        cond_random_users.append(10*np.log10(np.linalg.cond(H_random)))
        correlations = []
        for m in range(0, H_random.shape[0], 2):
            ch_m = H_random[list(range(m, m+2)), :].copy()
            for n in range(m+2, H_random.shape[0], 2):
                ch_n = H_random[list(range(n, n+2)), :].copy()
                correlation = np.linalg.norm(ch_m @ ch_n.conj().T) / (np.linalg.norm(ch_m) * np.linalg.norm(ch_n))
                correlations.append(correlation)
        correlations_random.append(median(correlations))

        nss = 1  # Number of streams per STA = 1

        H_random_streams = []
        for stream in streams_random:
            ant_list = []
            if stream % 2 == 0:
                ant_list.append(stream)
                ant_list.append(stream + 1)
            else:
                ant_list.append(stream - 1)
                ant_list.append(stream)
            H_STA = chan_wau2sta[ant_list, :].copy()
            [U, S, V] = np.linalg.svd(H_STA)
            He = np.diag(S[:nss]) @ V[:nss, :]
            H_random_streams.append(He)

        H_random_streams = np.vstack(H_random_streams)
        H_random_streams_try = chan_wau2sta[streams_random, :].copy()
        cond_random_streams.append(10*np.log10(np.linalg.cond(H_random_streams_try)))
        correlations = []
        for m in range(0, H_random_streams.shape[0], 1):
            ch_m = H_random_streams[m, :].copy()
            for n in range(m+1, H_random_streams.shape[0], 1):
                if (streams_random[n] != streams_random[m] + 1) and (
                        streams_random[n] != streams_random[m] - 1):
                    ch_n = H_random_streams[n, :].copy()
                    correlation = np.linalg.norm(ch_m @ ch_n.conj().T) / (np.linalg.norm(ch_m) * np.linalg.norm(ch_n))
                    correlations.append(correlation)
        correlations_random_streams.append(median(correlations))

        H_RR_norm = chan_wau2sta[streams_norm, :].copy()
        cond_norm_users.append(10*np.log10(np.linalg.cond(H_RR_norm)))
        correlations = []
        for m in range(0, H_RR_norm.shape[0], 2):
            ch_m = H_RR_norm[list(range(m, m+2)), :].copy()
            for n in range(m+2, H_RR_norm.shape[0], 2):
                ch_n = H_RR_norm[list(range(n, n+2)), :].copy()
                correlation = np.linalg.norm(ch_m @ ch_n.conj().T) / (np.linalg.norm(ch_m) * np.linalg.norm(ch_n))
                correlations.append(correlation)
        correlations_norm.append(median(correlations))

        H_corr_users = chan_wau2sta[chosen_streams_TXOP, :].copy()
        
        H_Norm_single_stream = [] 
        for stream in streams_norm_1_sel:
            ant_list = [] 
            if stream % 2 == 0:
                ant_list.append(stream) 
                ant_list.append(stream + 1) 
            else: 
                ant_list.append(stream - 1) 
                ant_list.append(stream)
            H_STA = chan_wau2sta[ant_list,:]
            [U, S, V] = np.linalg.svd(H_STA)
            He = np.diag(S[:nss])@V[:nss, :] 
            H_Norm_single_stream.append(He) 
            
        H_Norm_single_stream_all8 = [] 
        for stream in streams_norm_1: 
            ant_list = [] 
            if stream % 2 == 0:
                ant_list.append(stream) 
                ant_list.append(stream + 1) 
            else: 
                ant_list.append(stream - 1) 
                ant_list.append(stream)
            H_STA = chan_wau2sta[ant_list, :].copy()
            [U, S, V] = np.linalg.svd(H_STA)
            He = np.diag(S[:nss])@V[:nss, :] 
            H_Norm_single_stream_all8.append(He) 
            
        H_Norm_single_stream = np.vstack(H_Norm_single_stream)

        H_Norm_single_stream_all8 = np.vstack(H_Norm_single_stream_all8)

        H_try = chan_wau2sta[streams_norm_1, :].copy()
        cond_norm_streams.append(10 * np.log10(np.linalg.cond(H_try)))
        correlations = []
        for m in range(0, H_try.shape[0], 1):
            ch_m = H_try[m, :].copy()
            for n in range(m+1, H_try.shape[0], 1):
                if (streams_norm_1[n] != streams_norm_1[m] + 1) and (
                        streams_norm_1[n] != streams_norm_1[m] - 1):
                    ch_n = H_try[n, :].copy()
                    correlation = np.linalg.norm(ch_m @ ch_n.conj().T) / (np.linalg.norm(ch_m) * np.linalg.norm(ch_n))
                    correlations.append(correlation)
        correlations_norm_streams.append(median(correlations))

        W_random = np.linalg.pinv(H_random) 
        W_random_streams = np.linalg.pinv(H_random_streams)
        W_RR_norm = np.linalg.pinv(H_RR_norm)
        W_norm_1_sel = np.linalg.pinv(H_Norm_single_stream) 
        W_norm_1_sel_all8 = np.linalg.pinv(H_Norm_single_stream_all8)
        W_corr_users = np.linalg.pinv(H_corr_users)
    
        W_sqr_random = np.square(np.absolute(W_random)) 
        W_sqr_random_streams = np.square(np.absolute(W_random_streams))
        W_sqr_norm = np.square(np.absolute(W_RR_norm))
        W_sqr_norm_1_sel = np.square(np.absolute(W_norm_1_sel)) 
        W_sqr_norm_1_sel_all8 = np.square(np.absolute(W_norm_1_sel_all8))
        W_sqr_corr_users = np.square(np.absolute(W_corr_users))
        
        d_values_random = [-1]*4 
        d_values_norm = [-1]*4
        d_values_corr_users = [-1]*4
        
        poor_stream_random = 0 
        poor_stream_norm = 0
        poor_stream_corr_users = 0
        poor_stream_norm_streams = 0
        poor_stream_random_streams = 0
        
        while any(z < 3.9 for z in d_values_random): 
            # d_values_random = cvx_power_alloc_equal_power(W_sqr_random, p_c, T*N_T, len(STA_random)//2, N_C)[0]
            d_values_random = palloc_wf_equal_power(W_sqr_random, p_c, T*N_T, len(STA_random)//2, N_C)[0]
            d_values_random = [10*np.log10(val) for val in d_values_random for _ in (0, 1)] 
            list_of_poor_STA_indices = check_snr(d_values_random) 
            if len(list_of_poor_STA_indices) > 0: 
                for index in list_of_poor_STA_indices: 
                    STA_random[index] = -1
                    print('Found a poor STA --- Random') 
                    poor_stream_random += 1
                while any(m == -1 for m in STA_random):
                    STA_random.remove(-1)
                if poor_stream_random == 8:
                    infeasible_random += 1 
                    print('Infeasible --- Random')
                    break
                H_random = chan_wau2sta[STA_random,:].copy() 
                W_random = np.linalg.pinv(H_random) 
                W_sqr_random = np.square(np.absolute(W_random))
        no_of_streams_random_users.append(len(STA_random))
        if STA_random:
            R = compute_throughput(T*N_T, len(d_values_random)//2, 0, d_values_random)
            temp = []
            for snr in d_values_random:
                per_stream_SNR_random.append(snr)
                mcs_id = lookup_mcs(snr)
                per_stream_mcs_random.append(max(0, mcs_id))
                temp.append(lookup_rate(snr))
            variance = max(temp) - min(temp)
            per_stream_tput_random_users.append(variance)
            tput_random.append(R)
        
        while any(z < 3.9 for z in d_values_norm):
            # d_values_norm = cvx_power_alloc_equal_power(W_sqr_norm, p_c, T*N_T, len(streams_norm)//2, N_C)[0]
            d_values_norm = palloc_wf_equal_power(W_sqr_norm, p_c, T*N_T, len(streams_norm)//2, N_C)[0]
            d_values_norm = [10*np.log10(val) for val in d_values_norm for _ in (0, 1)]
            list_of_poor_STA_indices = check_snr(d_values_norm)
            if len(list_of_poor_STA_indices) > 0: 
                for index in list_of_poor_STA_indices: 
                    streams_norm[index] = -1
                    print('Found a poor STA --- Norm')
                    poor_stream_norm += 1
                while any(m == -1 for m in streams_norm):
                    streams_norm.remove(-1)
                if poor_stream_norm == 8:
                    infeasible_norm += 1 
                    print('Infeasible --- Norm') 
                    break
                H_norm = chan_wau2sta[streams_norm,:].copy() 
                W_norm = np.linalg.pinv(H_norm) 
                W_sqr_norm = np.square(np.absolute(W_norm))
        no_of_streams_norm_users.append(len(streams_norm))
        if streams_norm:
            R = compute_throughput(T*N_T, len(d_values_norm)//2, 0, d_values_norm)
            temp = []
            for snr in d_values_norm:
                per_stream_SNR_norm.append(snr)
                mcs_id = lookup_mcs(snr)
                per_stream_mcs_norm.append(max(0, mcs_id))
                temp.append(lookup_rate(snr))
            variance = max(temp) - min(temp)
            per_stream_tput_norm_users.append(variance)
            tput_RR_norm.append(R)

        while any(z<3.9 for z in d_values_corr_users):
            # d_values_norm = cvx_power_alloc_equal_power(W_sqr_norm, p_c, T*N_T, len(streams_norm)//2, N_C)[0]
            d_values_corr_users = palloc_wf_equal_power(W_sqr_corr_users, p_c, T*N_T, len(chosen_streams_TXOP)//2, N_C)[0]
            d_values_corr_users = [10*np.log10(val) for val in d_values_corr_users for _ in (0, 1)]
            list_of_poor_STA_indices = check_snr(d_values_corr_users)
            if len(list_of_poor_STA_indices) > 0:
                for index in list_of_poor_STA_indices:
                    chosen_streams_TXOP[index] = -1
                    print('Found a poor STA --- Correlation')
                    poor_stream_corr_users += 1
                if poor_stream_corr_users == 8:
                    infeasible_corr_users += 1
                    print('Infeasible --- Correlation')
                    break
                while any(m == -1 for m in chosen_streams_TXOP):
                    chosen_streams_TXOP.remove(-1)
                H_corr_users = chan_wau2sta[chosen_streams_TXOP, :].copy()
                W_corr_users = np.linalg.pinv(H_corr_users)
                W_sqr_corr_users = np.square(np.absolute(W_corr_users))
        R = compute_throughput(T*N_T, 4, 0, d_values_corr_users)
        for snr in d_values_corr_users:
            per_stream_SNR_corr_users.append(snr)
            mcs_id = lookup_mcs(snr)
            per_stream_mcs_corr_users.append(max(0, mcs_id))
        tput_corr_users.append(R)


        # d_values_norm_1_sel = cvx_power_alloc_unequal_power(W_sqr_norm_1_sel, p_c, T*N_T, len(streams_norm_1_sel), 1)
        d_values_norm_1_sel = palloc_wf_unequal_power(W_sqr_norm_1_sel, p_c, T*N_T, len(streams_norm_1_sel), 1)[0]
        d_STAs = len(streams_norm_1_sel) 
        single_STAs = 0 
        for rx_ant in streams_norm_1_sel: 
            if (rx_ant+1 in streams_norm_1_sel) or (rx_ant-1 in streams_norm_1_sel): 
                d_STAs -= 0.5 
            else: 
                single_STAs += 1 
        d_STAs = int(d_STAs)
        R = compute_throughput(T*N_T, d_STAs, single_STAs, 10*np.log10(d_values_norm_1_sel))
        snr_norm1 = 10*np.log10(d_values_norm_1_sel)
        for snr in snr_norm1: 
            per_stream_SNR_norm1.append(snr)
            mcs_id = lookup_mcs(snr) 
            per_stream_mcs_norm1.append(max(0, mcs_id))
        tput_norm_1_sel.append(R)

        snr_norm1 = [-1]*4
        while any(z < 3.9 for z in snr_norm1):
            d_values_norm_1_sel_all8 = palloc_wf_unequal_power(W_sqr_norm_1_sel_all8, p_c, T*N_T, len(streams_norm_1), 1)[0]
            snr_norm1 = 10*np.log10(d_values_norm_1_sel_all8)
            list_of_poor_STA_indices = check_snr(snr_norm1)
            if len(list_of_poor_STA_indices) > 0:
                for index in list_of_poor_STA_indices:
                    streams_norm_1[index] = -1
                    print('Found a poor stream --- Norm streams')
                    poor_stream_norm_streams += 1
                while any(m == -1 for m in streams_norm_1):
                    streams_norm_1.remove(-1)
                if poor_stream_norm_streams == 8:
                    infeasible_norm_streams += 1
                    print('Infeasible --- Norm streams')
                    break
                H_Norm_single_stream_all8 = []
                for stream in streams_norm_1:
                    ant_list = []
                    if stream % 2 == 0:
                        ant_list.append(stream)
                        ant_list.append(stream + 1)
                    else:
                        ant_list.append(stream - 1)
                        ant_list.append(stream)
                    H_STA = chan_wau2sta[ant_list, :].copy()
                    [U, S, V] = np.linalg.svd(H_STA)
                    He = np.diag(S[:nss]) @ V[:nss, :]
                    H_Norm_single_stream_all8.append(He)
                H_Norm_single_stream_all8 = np.vstack(H_Norm_single_stream_all8)
                W_norm_1_sel_all8 = np.linalg.pinv(H_Norm_single_stream_all8)
                W_sqr_norm_1_sel_all8 = np.square(np.absolute(W_norm_1_sel_all8))
        temp = []
        for snr in snr_norm1:
            per_stream_SNR_norm1.append(snr)
            mcs_id = lookup_mcs(snr)
            per_stream_mcs_norm1.append(mcs_id)
            temp.append(lookup_rate(snr))
        variance = max(temp) - min(temp)
        per_stream_tput_norm_streams.append(variance)
        d_STAs = len(streams_norm_1)
        single_STAs = 0
        for rx_ant in streams_norm_1:
            if (rx_ant + 1 in streams_norm_1) or (rx_ant - 1 in streams_norm_1):
                d_STAs -= 0.5
            else:
                single_STAs += 1
        d_STAs = int(d_STAs)
        R = compute_throughput(T * N_T, d_STAs, single_STAs, snr_norm1)
        tput_norm_1_sel_all8.append(R)
        no_of_streams_norm_streams.append(len(snr_norm1))

        snr_random_streams = [-1]*4
        while any(z < 3.9 for z in snr_random_streams):
            d_values_random_streams = palloc_wf_unequal_power(W_sqr_random_streams, p_c, T*N_T, len(streams_random), 1)[0]
            snr_random_streams = 10*np.log10(d_values_random_streams)
            list_of_poor_STA_indices = check_snr(snr_random_streams)
            if len(list_of_poor_STA_indices) > 0:
                for index in list_of_poor_STA_indices:
                    streams_random[index] = -1
                    print('Found a poor stream --- Random streams')
                    poor_stream_random_streams += 1
                while any(m == -1 for m in streams_random):
                    streams_random.remove(-1)
                if poor_stream_random_streams == 8:
                    infeasible_random_streams += 1
                    print('Infeasible --- Random streams')
                    break
                H_random_streams = []
                for stream in streams_random:
                    ant_list = []
                    if stream % 2 == 0:
                        ant_list.append(stream)
                        ant_list.append(stream + 1)
                    else:
                        ant_list.append(stream - 1)
                        ant_list.append(stream)
                    H_STA = chan_wau2sta[ant_list, :].copy()
                    [U, S, V] = np.linalg.svd(H_STA)
                    He = np.diag(S[:nss]) @ V[:nss, :]
                    H_random_streams.append(He)
                H_random_streams = np.vstack(H_random_streams)
                W_random_streams = np.linalg.pinv(H_random_streams)
                W_sqr_random_streams = np.square(np.absolute(W_random_streams))
        d_STAs = len(streams_random)
        single_STAs = 0
        for rx_ant in streams_random:
            if (rx_ant + 1 in streams_random) or (rx_ant - 1 in streams_random):
                d_STAs -= 0.5
            else:
                single_STAs += 1
        d_STAs = int(d_STAs)
        R = compute_throughput(T * N_T, d_STAs, single_STAs, 10 * np.log10(d_values_random_streams))
        temp = []
        for snr in snr_random_streams: 
            per_stream_SNR_random_streams.append(snr)
            mcs_id = lookup_mcs(snr)
            per_stream_mcs_random_streams.append(mcs_id)
            temp.append(lookup_rate(snr))
        variance = max(temp) - min(temp)
        per_stream_tput_random_streams.append(variance)
        tput_random_streams.append(R)
        no_of_streams_random_streams.append(len(d_values_random_streams))

print('Infeasible random = {}'.format(infeasible_random)) 
print('Infeasible norm = {}'.format(infeasible_norm))
print('Infeasible correlation based = {}'.format(infeasible_corr_users))

params = latexify(columns=2)
matplotlib.rcParams.update(params)

plt.figure() 

tput_random = np.sort(tput_random) 
F2 = np.array(range(len(tput_random)))/float(len(tput_random)) 
plt.plot(tput_random, F2, color='red', linestyle=':', linewidth=3, label='Random user selection \n(Two streams per user)')

tput_random_streams = np.sort(tput_random_streams) 
F2 = np.array(range(len(tput_random_streams)))/float(len(tput_random_streams)) 
plt.plot(tput_random_streams, F2, color='green', linestyle='-.', linewidth=3, label='Random user selection \n(Exactly one stream per user)')

tput_RR_norm = np.sort(tput_RR_norm) 
F2 = np.array(range(len(tput_RR_norm)))/float(len(tput_RR_norm)) 
plt.plot(tput_RR_norm, F2, color='blue', linestyle='--', linewidth=3, label='Norm-based \n(Two streams per user)')

tput_norm_1_sel_all8 = np.sort(tput_norm_1_sel_all8)
F2 = np.array(range(len(tput_norm_1_sel_all8)))/float(len(tput_norm_1_sel_all8))
plt.plot(tput_norm_1_sel_all8, F2, color='orange', linestyle='-', linewidth=3, label='Norm-based \n(Some users with single stream)')

# tput_corr_users = np.sort(tput_corr_users)
# F2 = np.array(range(len(tput_corr_users)))/float(len(tput_corr_users))
# plt.plot(tput_corr_users, F2, color='black', label='Correlation based user selection')

plt.legend() 
plt.xlabel(r'Sum Throughput $t$ (in Mbps)')
plt.ylabel(r'Pr[T$<= t$]')
plt.title('Distribution of sum downlink throughput T (in Mbps) \n Group dimension = 20m x 20m, Number of users = {}'.format(n_STAs))
plt.tight_layout()
plt.savefig('group_throughput.pdf')
plt.show()

params = latexify(fig_width=12.5, fig_height=8, columns=2)
matplotlib.rcParams.update(params)

plt.figure()
# no_of_streams_random_users = np.sort(no_of_streams_random_users)
# F2 = np.array(range(len(no_of_streams_random_users)))/float(len(no_of_streams_random_users))
plt.plot(no_of_streams_random_users, color='red', marker='x', linestyle=':', label='Random user selection \n(Two streams per user)')

# no_of_streams_random_streams = np.sort(no_of_streams_random_streams)
# F2 = np.array(range(len(no_of_streams_random_streams)))/float(len(no_of_streams_random_streams))
no_of_streams_random_streams = [x + 10 for x in no_of_streams_random_streams]
plt.plot(no_of_streams_random_streams, color='green', marker='o', linestyle='-.', label='Random user selection \n(Exactly one stream per user)')

# no_of_streams_norm_users = np.sort(no_of_streams_norm_users)
# F2 = np.array(range(len(no_of_streams_norm_users)))/float(len(no_of_streams_norm_users))
no_of_streams_norm_users = [x + 20 for x in no_of_streams_norm_users]
plt.plot(no_of_streams_norm_users, color='blue', marker='D', linestyle='--', label='Norm-based \n(Two streams per user)')

# no_of_streams_norm_streams = np.sort(no_of_streams_norm_streams)
# F2 = np.array(range(len(no_of_streams_norm_streams)))/float(len(no_of_streams_norm_streams))
no_of_streams_norm_streams = [x + 30 for x in no_of_streams_norm_streams]
plt.plot(no_of_streams_norm_streams, color='orange', marker='^', linestyle='-', label='Norm-based \n(Some users with single stream)')

plt.legend().draggable()
plt.xlabel(r'TxOP index')
plt.ylabel(r'Number of streams served S')
plt.title('Distribution of number of streams served S \n Group dimension = 20m x 20m, Number of users = {}'.format(n_STAs))
plt.yticks([0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32, 34, 36, 38], [0, 2, 4, 6, 8, 0, 2, 4, 6, 8, 0, 2, 4, 6, 8, 0, 2, 4, 6, 8])
plt.tight_layout()
plt.savefig('no_of_streams_served.pdf')
plt.show()

plt.figure()

per_stream_tput_random_users = np.sort(per_stream_tput_random_users)
F2 = np.array(range(len(per_stream_tput_random_users)))/float(len(per_stream_tput_random_users))
plt.plot(per_stream_tput_random_users, F2, color='red', linestyle=':', linewidth=3, label='Random user selection \n(Two streams per user)')

per_stream_tput_random_streams = np.sort(per_stream_tput_random_streams)
F2 = np.array(range(len(per_stream_tput_random_streams)))/float(len(per_stream_tput_random_streams))
plt.plot(per_stream_tput_random_streams, F2, color='green', linestyle='-.', linewidth=3, label='Random user selection \n(Exactly one stream per user)')

per_stream_tput_norm_users = np.sort(per_stream_tput_norm_users)
F2 = np.array(range(len(per_stream_tput_norm_users)))/float(len(per_stream_tput_norm_users))
plt.plot(per_stream_tput_norm_users, F2, color='blue', linestyle='--', linewidth=3, label='Norm-based \n(Two streams per user)')

per_stream_tput_norm_streams = np.sort(per_stream_tput_norm_streams)
F2 = np.array(range(len(per_stream_tput_norm_streams)))/float(len(per_stream_tput_norm_streams))
plt.plot(per_stream_tput_norm_streams, F2, color='orange', linestyle='-', linewidth=3, label='Norm-based \n(Some users with single stream)')

with open('var_oracle_users_1.txt', 'rb') as f:
    var_oracle_users = pickle.load(f)
var_oracle_users = np.sort(var_oracle_users)
F2 = np.array(range(len(var_oracle_users)))/float(len(var_oracle_users))
line,= plt.plot(var_oracle_users, F2, color='black', linewidth=3, label='Oracle user selection')
line.set_dashes([8, 4, 4, 8])

# tput_corr_users = np.sort(tput_corr_users)
# F2 = np.array(range(len(tput_corr_users)))/float(len(tput_corr_users))
# plt.plot(tput_corr_users, F2, color='black', label='Correlation based user selection')

plt.legend()
plt.xlabel(r'Variance of per stream throughput $v$ (in Mbps)')
plt.ylabel(r'Pr[V$<= v$]')
plt.title('Distribution of variance of per stream \n downlink throughput V (in Mbps) \n Group dimension = 20m x 20m, Number of users = {}'.format(n_STAs))
plt.tight_layout()
plt.savefig('per_stream_throughput.pdf')
plt.show()

plt.figure()

cond_random_users = np.sort(cond_random_users)
F2 = np.array(range(len(cond_random_users)))/float(len(cond_random_users))
plt.plot(cond_random_users, F2, color='red', linestyle=':', linewidth=3, label='Random user selection \n(Two streams per user)')

cond_random_streams = np.sort(cond_random_streams)
F2 = np.array(range(len(cond_random_streams)))/float(len(cond_random_streams))
plt.plot(cond_random_streams, F2, color='blue', linestyle='--', linewidth=3, label='Norm-based \n(Two streams per user)')

cond_norm_users = np.sort(cond_norm_users)
F2 = np.array(range(len(cond_norm_users)))/float(len(cond_norm_users))
plt.plot(cond_norm_users, F2, color='green', linestyle='-.', linewidth=3, label='Random user selection \n(Exactly one stream per user)')

cond_norm_streams = np.sort(cond_norm_streams)
F2 = np.array(range(len(cond_norm_streams)))/float(len(cond_norm_streams))
plt.plot(cond_norm_streams, F2, color='orange', linestyle='-', linewidth=3, label='Norm-based \n(Some users with single stream)')

plt.legend()
plt.xlabel(r'Condition Number $x$ (in dB)')
plt.ylabel(r'Pr[X$<= x$]')
plt.title('Distribution of condition number X of channel matrices \n Group dimension = 20m x 20m, Number of users = {}'.format(n_STAs))
plt.tight_layout()
plt.savefig('condition_number.pdf')
plt.show()

plt.figure()

correlations_random = np.sort(correlations_random)
F2 = np.array(range(len(correlations_random)))/float(len(correlations_random))
plt.plot(correlations_random, F2, color='red', linestyle=':', linewidth=3, label='Random user selection \n(Two streams per user)')

correlations_random_streams = np.sort(correlations_random_streams)
F2 = np.array(range(len(correlations_random_streams)))/float(len(correlations_random_streams))
plt.plot(correlations_random_streams, F2, color='green', linestyle='-.', linewidth=3, label='Random user selection \n(Exactly one stream per user)')

correlations_norm = np.sort(correlations_norm)
F2 = np.array(range(len(correlations_norm)))/float(len(correlations_norm))
plt.plot(correlations_norm, F2, color='blue', linestyle='--', linewidth=3, label='Norm-based \n(Two streams per user)')

correlations_norm_streams = np.sort(correlations_norm_streams)
F2 = np.array(range(len(correlations_norm_streams)))/float(len(correlations_norm_streams))
plt.plot(correlations_norm_streams, F2, color='orange', linestyle='-', linewidth=3, label='Norm-based \n(Some users with single stream)')

plt.legend()
plt.xlabel(r'Median pair-wise correlation coefficient $c$ (in dB)')
plt.ylabel(r'Pr[C $<= c$]')
plt.title('Distribution of median pair-wise correlation coefficient C \n of users chosen as part of transmission group \n Group dimension = 20m x 20m, Number of users = {}'.format(n_STAs))
plt.tight_layout()
plt.savefig('correlation_coefficients.pdf')
plt.show()


#plt.figure() 
#
#no_of_single_STAs_norm = np.sort(no_of_single_STAs_norm) 
#F2 = np.array(range(len(no_of_single_STAs_norm)))/float(len(no_of_single_STAs_norm))
#plt.plot(no_of_single_STAs_norm, F2, color='orange') 
#
#plt.xlabel('Number of STAs with single stream n')
#plt.ylabel('Pr[N <= n]')
#plt.xlim([1,9])
#plt.title('Distribution of number of STAs chosen with single stream N \n Room size = $20$m x $20$m, Number of STAs = {}'.format(n_STAs))
#plt.show()  

plt.figure()
per_stream_SNR_random = np.sort(per_stream_SNR_random) 
F2 = np.array(range(len(per_stream_SNR_random)))/float(len(per_stream_SNR_random)) 
plt.plot(per_stream_SNR_random, F2, color='red', label = 'Random user selection \n(Two streams per user)') 

per_stream_SNR_random_streams = np.sort(per_stream_SNR_random_streams) 
F2 = np.array(range(len(per_stream_SNR_random_streams)))/float(len(per_stream_SNR_random_streams)) 
plt.plot(per_stream_SNR_random_streams, F2, color='green', label = 'Random user selection \n(Exactly one stream per user)') 

per_stream_SNR_norm = np.sort(per_stream_SNR_norm) 
F2 = np.array(range(len(per_stream_SNR_norm)))/float(len(per_stream_SNR_norm)) 
plt.plot(per_stream_SNR_norm, F2, color='blue', label = 'Norm-based \n(Two streams per user)') 

per_stream_SNR_norm1 = np.sort(per_stream_SNR_norm1) 
F2 = np.array(range(len(per_stream_SNR_norm1)))/float(len(per_stream_SNR_norm1)) 
plt.plot(per_stream_SNR_norm1, F2, color='orange', label = 'Norm-based \n(Some users with single stream)') 

plt.legend() 
plt.xlabel('Downlink SNR x (in dB)') 
plt.ylabel('Pr[SNR <= x]') 
plt.title('Distribution of downlink SNR (in dB) per stream \n Group dimension = $20$m x $20$m, Number of users = {}'.format(n_STAs)) 
plt.show() 

plt.figure() 
per_stream_mcs_random = np.sort(per_stream_mcs_random) 
F2 = np.array(range(len(per_stream_mcs_random)))/float(len(per_stream_mcs_random)) 
plt.plot(per_stream_mcs_random, F2, color='red', linestyle=':', linewidth=3, label='Random user selection \n(Two streams per user)')

per_stream_mcs_random_streams = np.sort(per_stream_mcs_random_streams) 
F2 = np.array(range(len(per_stream_mcs_random_streams)))/float(len(per_stream_mcs_random_streams)) 
plt.plot(per_stream_mcs_random_streams, F2, color='green', linestyle='-.', linewidth=3, label='Random user selection \n(Exactly one stream per user)')

per_stream_mcs_norm = np.sort(per_stream_mcs_norm) 
F2 = np.array(range(len(per_stream_mcs_norm)))/float(len(per_stream_mcs_norm)) 
plt.plot(per_stream_mcs_norm, F2, color='blue', linestyle='--', linewidth=3, label='Norm-based \n(Two streams per user)')

per_stream_mcs_norm1 = np.sort(per_stream_mcs_norm1) 
F2 = np.array(range(len(per_stream_mcs_norm1)))/float(len(per_stream_mcs_norm1)) 
plt.plot(per_stream_mcs_norm1, F2, color='orange', linestyle='-', linewidth=3, label='Norm-based \n(Some users with single stream)')

# per_stream_mcs_corr_users = np.sort(per_stream_mcs_corr_users)
# F2 = np.array(range(len(per_stream_mcs_corr_users)))/float(len(per_stream_mcs_corr_users))
# plt.plot(per_stream_mcs_corr_users, F2, color='black', label='Correlation based user selection')

plt.legend() 
plt.xlabel('MCS index $n$')
plt.ylabel(r'Pr[N$<= n$]')
plt.title('Distribution of downlink MCS index N per stream \n Group dimension = 20m x 20m, Number of users = {}'.format(n_STAs))
plt.tight_layout()
plt.savefig('group_mcs.pdf')
plt.show() 

# plt.figure()
# no_of_streams_norm = np.sort(no_of_streams_norm)
# F2 = np.array(range(len(no_of_streams_norm)))/float(len(no_of_streams_norm))
# plt.plot(no_of_streams_norm, F2, color='orange')
# plt.title('Distribution of number of streams N chosen \n by Algorithm 1; Group dimension = $20$m x $20$m')
# plt.xlabel('Number of streams n')
# plt.ylabel('Pr[N <= n]')
# plt.show()
#
# fig,ax = plt.subplots(figsize=(15,8))
# width = 0.35
# ind = np.arange(len(no_of_times_user_chosen))
# sum_number = sum(no_of_times_user_chosen)
# no_of_times_user_chosen = [x/sum_number for x in no_of_times_user_chosen]
# rects = ax.bar(ind, no_of_times_user_chosen, width, color='red')
#
# ax.set_ylabel('Relative frequency of choosing the user i')
# ax.set_xlabel('User index i')
# ax.set_title('Distribution of number of times users are chosen')
# ax.set_xticks(ind + width / 2)
# labels = []
# for v in ind:
#     labels.append(str(v+1))
# ax.set_xticklabels(labels)
# plt.show()