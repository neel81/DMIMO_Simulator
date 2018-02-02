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
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt 
import pylab as pl
from operator import itemgetter 
import matplotlib.cm as cm
from statistics import median
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
    prob.solve(max_iters=50000)
    return np.asarray(np.abs(d.value)).flatten(), prob.status  

def cvx_power_alloc_unequal_power(W, Pa, nt, nc, nac): 
    #Pc = Pa*np.ones((nt,1))  ## per-antenna power constraint
    d = cvx.Variable(nc*nac) 
    objective = cvx.Maximize(cvx.sum_entries(cvx.log1p(d)/np.log(2))) 
    constraints = [cvx.sum_entries(W @ d) <= Pa, d >= 0, d <= 10**3.6] 
    prob = cvx.Problem(objective, constraints) 
    prob.solve(max_iters=50000)
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
    elif snr >= 34.0:
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
    elif snr >= 34.0:
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
tput_oracle_users = []

no_of_single_STAs_norm = [] 
no_of_single_STAs_random_streams = [] 

per_stream_SNR_random = [] 
per_stream_SNR_random_streams = [] 
per_stream_SNR_norm = [] 
per_stream_SNR_norm1 = []

per_stream_mcs_random = [] 
per_stream_mcs_random_streams = [] 
per_stream_mcs_norm = [] 
per_stream_mcs_norm1 = []

no_of_streams_norm = []

infeasible_random = 0 
infeasible_norm = 0
infeasible_norm_all8 = 0
infeasible_random_streams = 0
infeasible_oracle = 0

variance_tput_random_users = []
variance_tput_norm_users = []
variance_tput_norm_streams = []
variance_tput_random_streams = []
variance_tput_oracle_users = []

no_of_times_user_chosen = [0]*n_STAs

big_schedule = []

for big_iter in range(54): 
    print('Drop = {}'.format(big_iter))
    
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

    # Oracle based scheduling -- 2 streams per user
    # First find throughput of all user combinations (40 users and 8 groups), arrange them in decreasing order
    # Choose a schedule that maximizes group user throughput

    list_of_streams_and_tput_oracle = []

    for x1 in range(0, n_STAs - 3, 1):
        for x2 in range(x1 + 1, n_STAs - 2, 1):
            for x3 in range(x2 + 1, n_STAs - 1, 1):
                for x4 in range(x3 + 1, n_STAs, 1):
                    list_of_streams = []
                    list_of_x = [x1, x2, x3, x4]
                    for x in list_of_x:
                        list_of_streams.append(2 * x)
                        list_of_streams.append(2 * x + 1)
                    H_oracle = chan_wau2sta[list_of_streams, :].copy()
                    W_oracle = np.linalg.pinv(H_oracle)
                    W_sqr_oracle = np.square(np.absolute(W_oracle))
                    d_values_oracle = [-1]*4
                    poor_stream_oracle = 0
                    while any(z < 3.9 for z in d_values_oracle):
                        d_values_oracle = palloc_wf_equal_power(W_sqr_oracle, p_c, T * N_T, len(list_of_streams) // 2, N_C)[
                            0]
                        d_values_oracle = [10 * np.log10(val) for val in d_values_oracle for _ in (0, 1)]
                        list_of_poor_STA_indices = check_snr(d_values_oracle)
                        if len(list_of_poor_STA_indices) > 0:
                            for index in list_of_poor_STA_indices:
                                list_of_streams[index] = -1
                                poor_stream_oracle += 1
                            if poor_stream_oracle == 8:
                                infeasible_oracle += 1
                                print('Infeasible -- Oracle')
                                break
                            while any(m == -1 for m in list_of_streams):
                                list_of_streams.remove(-1)
                            H_oracle = chan_wau2sta[list_of_streams, :].copy()
                            W_oracle = np.linalg.pinv(H_oracle)
                            W_sqr_oracle = np.square(np.absolute(W_oracle))
                    R = compute_throughput(T * N_T, len(d_values_oracle)//2, 0, d_values_oracle)
                    temp_rate = []
                    for snr in d_values_oracle:
                        per_stream_SNR_norm.append(snr)
                        mcs_id = lookup_mcs(snr)
                        per_stream_mcs_norm.append(max(0, mcs_id))
                        temp_rate.append(lookup_rate(snr))
                    jain_fairness = 0
                    if all(rate > 0 for rate in temp_rate):
                        jain_fairness = (sum(temp_rate)**2)/(len(d_values_oracle)*sum([x**2 for x in temp_rate]))
                    temp = []
                    temp.append(list_of_streams)
                    temp.append(R)
                    temp.append(jain_fairness)
                    list_of_streams_and_tput_oracle.append(temp)

    print('Completed oracle for drop = {}'.format(big_iter))
    list_of_streams_and_tput_oracle.sort(key=lambda z: z[1], reverse=True)

    scheduled_streams = []
    chosen_streams = []

    for l in range(len(list_of_streams_and_tput_oracle)):
        list_of_streams = list_of_streams_and_tput_oracle[l][0]
        tput = list_of_streams_and_tput_oracle[l][1]
        var = list_of_streams_and_tput_oracle[l][2]

        stream_occurence = 0
        for stream in list_of_streams:
            if stream in scheduled_streams:
                stream_occurence += 1

        if stream_occurence == 0 and len(scheduled_streams) < 2*n_STAs:
            for stream in list_of_streams:
                scheduled_streams.append(stream)
            chosen_streams.append(list_of_streams)
            tput_oracle_users.append(tput)
            variance_tput_oracle_users.append(var)

    # assert len(scheduled_streams) == len(set(scheduled_streams)), "Scheduled streams does not have unique streams"

    colors = iter(cm.nipy_spectral(np.linspace(0, 1, len(chosen_streams))))
    for schedule in chosen_streams:
        clr = next(colors)
        for user in schedule:
                plt.scatter(STAs[user//2].x, STAs[user//2].y, color=clr, marker='*', s=0.15)
                pl.text(STAs[user//2].x, STAs[user//2].y, str(chosen_streams.index(schedule) + 1), color=clr, fontsize=10.5)

    for wau in WAUs:
        plt.scatter(wau.x, wau.y, marker='^', color='black', s=70)

    plt.title('Scheduled users by Oracle for iteration = {}'.format(big_iter))
    plt.xlabel('x (in meters)')
    plt.ylabel('y (in meters)')
    plt.tight_layout()
    plt.savefig('scheduled_users_'+str(big_iter)+'.png')

    # Oracle based scheduling -- users may have single streams
    # First find throughput of all user combinations (40 users and 8 groups), arrange them in decreasing order
    # Choose a schedule that maximizes group user throughput

    # list_of_streams_and_tput_oracle = []
    #
    # for x1 in range(1, 2*n_STAs - 7, 1):
    #     for x2 in range(x1, 2*n_STAs - 6, 1):
    #         for x3 in range(x2, 2*n_STAs - 5, 1):
    #             for x4 in range(x3, 2*n_STAs - 4, 1):
    #                 for x5 in range(x4, 2*n_STAs - 3, 1):
    #                     for x6 in range(x5, 2*n_STAs - 2, 1):
    #                         for x7 in range(x6, 2*n_STAs - 1, 1):
    #                             for x8 in range(x7, 2*n_STAs, 1):
    #                                 list_of_streams = [x1, x2, x3, x4, x5, x6, x7, x8]
    #                                 H_oracle_individual = []
    #                                 for stream in list_of_streams:
    #                                     ant_list = []
    #                                     if stream % 2 == 0:
    #                                         ant_list.append(stream)
    #                                         if stream + 1 not in list_of_streams:
    #                                             ant_list.append(stream + 1)
    #                                     else:
    #                                         if stream-1 not in list_of_streams:
    #                                             ant_list.append(stream - 1)
    #                                         ant_list.append(stream)
    #                                     H_STA = chan_wau2sta[ant_list, :]
    #                                     [U, S, V] = np.linalg.svd(H_STA)
    #                                     He = np.diag(S[:nss]) @ V[:nss, :]
    #                                     H_oracle_individual.append(He)
    #                                 H_oracle = np.vstack(H_oracle_individual)
    #                                 W_oracle = np.linalg.pinv(H_oracle)
    #                                 W_sqr_oracle = np.square(np.absolute(W_oracle))
    #                                 d_values_oracle = \
    #                                     palloc_wf_equal_power(W_sqr_oracle, p_c, T * N_T, len(list_of_streams) // 2, N_C)[
    #                                         0]
    #                                 d_values_oracle = [10 * np.log10(val) for val in d_values_oracle for _ in (0, 1)]
    #                                 R = compute_throughput(T * N_T, len(d_values_oracle), 0, d_values_oracle)
    #                                 temp = []
    #                                 temp.append(list_of_streams)
    #                                 temp.append(R)
    #                                 list_of_streams_and_tput_oracle.append(temp)
    #
    # print('Completed oracle streams for drop = {}'.format(big_iter))
    # list_of_streams_and_tput_oracle.sort(key=lambda z: z[1], reverse=True)
    #
    # scheduled_streams = []
    #
    # for l in range(len(list_of_streams_and_tput_oracle)):
    #     list_of_streams = list_of_streams_and_tput_oracle[l][0]
    #     tput = list_of_streams_and_tput_oracle[l][1]
    #
    #     stream_occurence = 0
    #     for stream in list_of_streams:
    #         if stream in scheduled_streams:
    #             stream_occurence += 1
    #
    #     if stream_occurence == 0:
    #         for stream in list_of_streams:
    #             scheduled_streams.append(stream)
    #             tput_oracle_streams.append(tput)
    
    ordering_waus_2streams = [] 
    
    for r in range(0, chan_wau2sta.shape[1], 2):
        norms = [] 
        for c in range(0, chan_wau2sta.shape[0], 2):
            chan_vector = chan_wau2sta[np.ix_(range(c, c+2), range(r, r+2))]
            norms.append(np.linalg.norm(chan_vector))
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
        while j < 4:
            id = random.randint(0, 2*n_STAs-1) 
            if id not in STA_random and id+1 not in STA_random and id-1 not in STA_random: 
                if id % 2 == 0:
                    STA_random.append(id) 
                    STA_random.append(id+1) 
                else: 
                    STA_random.append(id-1) 
                    STA_random.append(id)
                j = j+1  
        
        streams_random = random.sample(range(0, 2*n_STAs-1), 8)
        
        potential_stas = [] 
        rr = random.sample([0, 1, 2, 3], 4)   # Chose WAUs randomly for selecting users
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
        
        streams_norm = [] 
        
        for z in range(len(potential_stas)): 
            streams_norm.append(2*potential_stas[z]) 
            streams_norm.append(2*potential_stas[z]+1)
            
        H_random = chan_wau2sta[STA_random, :].copy()

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

        H_RR_norm = chan_wau2sta[streams_norm, :].copy()
        
        nss = 1  # Number of streams per STA = 1

        streams_norm_1 = potential_streams

        H_Norm_single_stream_all8 = [] 
        for stream in streams_norm_1: 
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
            H_Norm_single_stream_all8.append(He)

        H_Norm_single_stream_all8 = np.vstack(H_Norm_single_stream_all8)

        W_random = np.linalg.pinv(H_random) 
        W_random_streams = np.linalg.pinv(H_random_streams)
        W_RR_norm = np.linalg.pinv(H_RR_norm)
        W_norm_1_sel_all8 = np.linalg.pinv(H_Norm_single_stream_all8)
    
        W_sqr_random = np.square(np.absolute(W_random)) 
        W_sqr_random_streams = np.square(np.absolute(W_random_streams))
        W_sqr_norm = np.square(np.absolute(W_RR_norm))
        W_sqr_norm_1_sel_all8 = np.square(np.absolute(W_norm_1_sel_all8))
        
        d_values_random = [-1]*4 
        d_values_norm = [-1]*4
        d_values_random_streams = [-1]*8
        d_values_norm_1_sel_all8 = [-1]*8
        
        poor_stream_random = 0 
        poor_stream_norm = 0
        poor_stream_random_streams = 0
        poor_stream_norm_all8 = 0
        
        while any(z < 3.9 for z in d_values_random):
            d_values_random = palloc_wf_equal_power(W_sqr_random, p_c, T*N_T, len(STA_random)//2, N_C)[0]
            d_values_random = [10*np.log10(val) for val in d_values_random for _ in (0, 1)] 
            list_of_poor_STA_indices = check_snr(d_values_random) 
            if len(list_of_poor_STA_indices) > 0: 
                for index in list_of_poor_STA_indices: 
                    STA_random[index] = -1
                    print('Found a poor STA -- Random')
                    poor_stream_random += 1 
                if poor_stream_random == 8:
                    infeasible_random += 1 
                    print('Infeasible -- Random')
                    break 
                while any(m == -1 for m in STA_random): 
                    STA_random.remove(-1) 
                H_random = chan_wau2sta[STA_random,:].copy() 
                W_random = np.linalg.pinv(H_random) 
                W_sqr_random = np.square(np.absolute(W_random)) 
        R = compute_throughput(T*N_T, len(d_values_random)//2, 0, d_values_random)
        temp = []
        for snr in d_values_random: 
            per_stream_SNR_random.append(snr)
            mcs_id = lookup_mcs(snr) 
            per_stream_mcs_random.append(max(0, mcs_id))
            temp.append(lookup_rate(snr))
        if all(rate > 0 for rate in temp):
            jain_fairness = (sum(temp) ** 2) / (len(d_values_random) * sum([x ** 2 for x in temp]))
            variance_tput_random_users.append(jain_fairness)
        tput_random.append(R) 
        
        while any(z < 3.9 for z in d_values_norm):
            d_values_norm = palloc_wf_equal_power(W_sqr_norm, p_c, T*N_T, len(streams_norm)//2, N_C)[0]
            d_values_norm = [10*np.log10(val) for val in d_values_norm for _ in (0, 1)]
            list_of_poor_STA_indices = check_snr(d_values_norm)
            if len(list_of_poor_STA_indices) > 0: 
                for index in list_of_poor_STA_indices: 
                    streams_norm[index] = -1
                    print('Found a poor STA -- Norm')
                    poor_stream_norm += 1 
                if poor_stream_norm == 8:
                    infeasible_norm += 1 
                    print('Infeasible -- Norm')
                    break 
                while any(m == -1 for m in streams_norm): 
                    streams_norm.remove(-1) 
                H_norm = chan_wau2sta[streams_norm,:].copy() 
                W_norm = np.linalg.pinv(H_norm) 
                W_sqr_norm = np.square(np.absolute(W_norm)) 
        R = compute_throughput(T*N_T, len(d_values_norm)//2, 0, d_values_norm)
        temp = []
        for snr in d_values_norm: 
            per_stream_SNR_norm.append(snr)
            mcs_id = lookup_mcs(snr) 
            per_stream_mcs_norm.append(max(0, mcs_id))
            temp.append(lookup_rate(snr))
        if all(rate > 0 for rate in temp):
            jain_fairness = (sum(temp) ** 2) / (len(d_values_norm) * sum([x ** 2 for x in temp]))
            variance_tput_norm_users.append(jain_fairness)
        tput_RR_norm.append(R)

        while any(z < 3.9 for z in d_values_norm_1_sel_all8):
            d_values_norm_1_sel_all8 = palloc_wf_unequal_power(W_sqr_norm_1_sel_all8, p_c, T*N_T, len(streams_norm_1), 1)[0]
            d_values_norm_1_sel_all8 = 10*np.log10(d_values_norm_1_sel_all8)
            list_of_poor_STA_indices = check_snr(d_values_norm_1_sel_all8)
            if len(list_of_poor_STA_indices) > 0:
                for index in list_of_poor_STA_indices:
                    streams_norm_1[index] = -1
                    print('Found a poor STA -- Norm streams')
                    poor_stream_norm_all8 += 1
                if poor_stream_norm_all8 == 8:
                    infeasible_norm_all8 += 1
                    print('Infeasible -- Norm streams')
                    break
                while any(m == -1 for m in streams_norm_1):
                    streams_norm_1.remove(-1)
                H_Norm_single_stream_all8 = []
                for stream in streams_norm_1:
                    ant_list = []
                    if stream % 2 == 0:
                        ant_list.append(stream)
                        ant_list.append(stream + 1)
                    else:
                        ant_list.append(stream - 1)
                        ant_list.append(stream)
                    H_STA = chan_wau2sta[ant_list, :]
                    [U, S, V] = np.linalg.svd(H_STA)
                    He = np.diag(S[:nss]) @ V[:nss, :]
                    H_Norm_single_stream_all8.append(He)
                H_Norm_single_stream_all8 = np.vstack(H_Norm_single_stream_all8)
                W_norm_1_sel_all8 = np.linalg.pinv(H_Norm_single_stream_all8)
                W_sqr_norm_1_sel_all8 = np.square(np.absolute(W_norm_1_sel_all8))
        d_STAs = len(streams_norm_1)
        single_STAs = 0
        for rx_ant in streams_norm_1:
            if (rx_ant+1 in streams_norm_1) or (rx_ant-1 in streams_norm_1):
                d_STAs -= 0.5
            else:
                single_STAs += 1
        d_STAs = int(d_STAs)
        R = compute_throughput(T*N_T, d_STAs, single_STAs, d_values_norm_1_sel_all8)
        temp = []
        for snr in d_values_norm_1_sel_all8:
            per_stream_SNR_norm1.append(snr)
            mcs_id = lookup_mcs(snr)
            per_stream_mcs_norm1.append(max(0, mcs_id))
            temp.append(lookup_rate(snr))
        if all(rate > 0 for rate in temp):
            jain_fairness = (sum(temp) ** 2) / (len(d_values_norm_1_sel_all8) * sum([x ** 2 for x in temp]))
            variance_tput_norm_streams.append(jain_fairness)
        tput_norm_1_sel_all8.append(R)

        while any(z < 3.9 for z in d_values_random_streams):
            d_values_random_streams = \
            palloc_wf_unequal_power(W_sqr_random_streams, p_c, T * N_T, len(streams_random), 1)[0]
            d_values_random_streams = 10*np.log10(d_values_random_streams)
            list_of_poor_STA_indices = check_snr(d_values_random_streams)
            if len(list_of_poor_STA_indices) > 0:
                for index in list_of_poor_STA_indices:
                    streams_random[index] = -1
                    print('Found a poor STA -- Random streams')
                    poor_stream_random_streams += 1
                if poor_stream_random_streams == 8:
                    infeasible_random_streams += 1
                    print('Infeasible -- Random streams')
                    break
                while any(m == -1 for m in streams_random):
                    streams_random.remove(-1)
                H_random_streams = []
                for stream in streams_random:
                    ant_list = []
                    if stream % 2 == 0:
                        ant_list.append(stream)
                        ant_list.append(stream + 1)
                    else:
                        ant_list.append(stream - 1)
                        ant_list.append(stream)
                    H_STA = chan_wau2sta[ant_list, :]
                    [U, S, V] = np.linalg.svd(H_STA)
                    He = np.diag(S[:nss]) @ V[:nss, :]
                    H_random_streams.append(He)
                H_random_streams = np.vstack(H_random_streams)
                W_random_streams = np.linalg.pinv(H_random_streams)
                W_sqr_random_streams = np.square(np.absolute(W_random_streams))
        d_STAs = len(streams_random)
        single_STAs = 0
        for rx_ant in streams_random:
            if (rx_ant+1 in streams_random) or (rx_ant-1 in streams_random):
                d_STAs -= 0.5
            else:
                single_STAs += 1
        d_STAs = int(d_STAs)
        R = compute_throughput(T*N_T, d_STAs, single_STAs, d_values_random_streams)
        temp = []
        for snr in d_values_random_streams:
            per_stream_SNR_random_streams.append(snr)
            mcs_id = lookup_mcs(snr)
            per_stream_mcs_random_streams.append(max(0, mcs_id))
            temp.append(lookup_rate(snr))
        if all(rate > 0 for rate in temp):
            jain_fairness = (sum(temp) ** 2) / (len(d_values_random_streams) * sum([x ** 2 for x in temp]))
            variance_tput_random_streams.append(jain_fairness)
        tput_random_streams.append(R)

print('Simulation completed!')

with open('tput_random_users_2.txt', 'wb') as myfile:
    pickle.dump(tput_random, myfile)

with open('tput_random_streams_2.txt', 'wb') as myfile:
    pickle.dump(tput_random_streams, myfile)

with open('tput_norm_users_2.txt', 'wb') as myfile:
    pickle.dump(tput_RR_norm, myfile)

with open('tput_norm_streams_2.txt', 'wb') as myfile:
    pickle.dump(tput_norm_1_sel_all8, myfile)

with open('tput_oracle_users_2.txt', 'wb') as myfile:
    pickle.dump(tput_oracle_users, myfile)

with open('mcs_random_users_2.txt', 'wb') as myfile:
    pickle.dump(per_stream_mcs_random, myfile)

with open('mcs_random_streams_2.txt', 'wb') as myfile:
    pickle.dump(per_stream_mcs_random_streams, myfile)

with open('mcs_norm_users_2.txt', 'wb') as myfile:
    pickle.dump(per_stream_mcs_norm, myfile)

with open('mcs_norm_streams_2.txt', 'wb') as myfile:
    pickle.dump(per_stream_mcs_norm1, myfile)

with open('var_random_users_2.txt', 'wb') as myfile:
    pickle.dump(variance_tput_random_users, myfile)

with open('var_random_streams_2.txt', 'wb') as myfile:
    pickle.dump(variance_tput_random_streams, myfile)

with open('var_norm_users_2.txt', 'wb') as myfile:
    pickle.dump(variance_tput_norm_users, myfile)

with open('var_norm_streams_2.txt', 'wb') as myfile:
    pickle.dump(variance_tput_norm_streams, myfile)

with open('var_oracle_users_2.txt', 'wb') as myfile:
    pickle.dump(variance_tput_oracle_users, myfile)

