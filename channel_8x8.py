import numpy as np
import random
from channel import Channel
import bog_params as params
from waus import WAU
from stas import STA
from antenna import create_ULA
import cvxpy as cvx
import matplotlib.pyplot as plt 
from bog_mW2dBm import bog_mW2dBm 


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

def set_sta_positions(WAUs):
    STAs = []
    for sta_ident, wau in enumerate(WAUs):
        x1 = wau.x + (-2)*params.World.roomX_m
        x2 = wau.x + (5)*params.World.roomX_m
        y1 = wau.y + (-2)*params.World.roomY_m
        y2 = wau.y + 5*params.World.roomY_m

        x = x1 + random.uniform(0, 1)*(x2-x1)
        y = y1 + random.uniform(0, 1)*(y2-y1)
        z = wau.z
        sta = STA(sta_ident, x, y, z, 0)
        STAs.append(sta)
        wau.stas.append(sta)
        wau.stas[0].wau = wau
    return STAs 

def cvx_power_alloc_equal_power(W, Pa, nt, nc, nac):
    Pc = Pa*np.ones((nt, 1))
    Q = np.kron(np.eye(nc), np.ones((nac, 1)))
    d = cvx.Variable(nc)
    objective = cvx.Maximize(cvx.sum_entries(cvx.log1p(d)/np.log(2)))
    constraints = [W @ Q @ d <= Pc, d >= 0]
    prob = cvx.Problem(objective, constraints)
    prob.solve() 
    print('Problem status: {}'.format(prob.status))
    return np.asarray(np.abs(d.value)).flatten(), prob.status 

def cvx_power_alloc_unequal_power(W, Pa, nt, nc, nac): 
    Pc = Pa*np.ones((nt,1))
    d = cvx.Variable(nc*nac) 
    objective = cvx.Maximize(cvx.sum_entries(cvx.log1p(d)/np.log(2))) 
    constraints = [W @ d <= Pc, d >= 0] 
    prob = cvx.Problem(objective, constraints) 
    prob.solve(solver='SCS') 
    print('Problem status: {}'.format(prob.status))
    return np.asarray(np.abs(d.value)).flatten()

N_T = N_C = params.General.nAntennas 

N_O = 1.38*(10**-23)*290*80*(10**6) 
p_c = 0.1

sensitivity_dBm = np.array([-92, -89, -85, -83, -79, -76, -74, -69, -65, -63])
sensitivity_dBm = sensitivity_dBm - 10*np.log10(N_O/(10**-3))
rates = [32.5, 65, 97.5, 130, 195, 260, 292.5, 325, 390, 433.3]

tput_opt_eq = [] 
tput_opt_rate_mapped = [] 
tput_opt_uneq = [] 
tput_naive = [] 
tput_naive_rate_mapped = [] 

for i in range(200): 
    
    print('Iteration = {}'.format(i))

    WAUs = []
    STAs = []
    chan_wau2sta = None 
    
    WAUs = set_wau_positions()
    STAs = set_sta_positions(WAUs) 
    
    T = len(WAUs) # Number of tx WAUs 
    C = len(STAs) # Number of clients 
    
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
    
    W = np.linalg.pinv(chan_wau2sta) # Precoding matrix 
    W_new = np.square(np.absolute(W)) 
    
    # Optimization formulation 
#    d = cvx.Variable(C) 
#    Q = np.kron(np.eye(C), np.ones((N_C, 1))) 
#    obj = cvx.sum_entries(cvx.mul_elemwise(1/(N_O*np.log(2)), cvx.log1p(d)))  
#    objective = cvx.Maximize(obj) 
#    constraints = [W_new @ Q @ d <= power_constraint] 
#    prob = cvx.Problem(objective, constraints) 
#    result = prob.solve() 
    
    d_values = cvx_power_alloc_equal_power(W_new, p_c, T*N_T, C, N_C)[0] 
    sum_tput_opt = 80*(10**6)*sum([N_C*np.log2(1 + y) for y in d_values]) 
    rx_power_opt_dBm = [10*np.log10(x/(10**-3)) for x in d_values]
    data_rate = 0.0 
    for r in rx_power_opt_dBm: 
        mcs = max(0, sum(r > sensitivity_dBm) - 1) 
        data_rate += N_C*rates[mcs] 
    
    tput_opt_rate_mapped.append(data_rate)  
    tput_opt_eq.append(sum_tput_opt) 
    
#    d_values = cvx_power_alloc_unequal_power(W_new, p_c, T*N_T, C, N_C) 
#    sum_tput_opt = 80*(10**6)*sum([np.log2(1 + y) for y in d_values]) 
#    tput_opt_uneq.append(sum_tput_opt) 
    
    # Naive power scaling 
    sum_W = np.sum(W_new, axis = 1) 
    
    rx_powers = (p_c/max(sum_W))*np.ones(C*N_C) 
    rx_power_opt_dBm = [10*np.log10(x/(10**-3)) for x in rx_powers]

    data_rate = 0.0 
    for r in rx_power_opt_dBm: 
        mcs = max(0, sum(r > sensitivity_dBm) - 1) 
        data_rate += rates[mcs] 
    
    tput_naive_rate_mapped.append(data_rate)
    sum_tput_naive = 80*(10**6)*sum([np.log2(1 + y) for y in rx_powers]) 
    tput_naive.append(sum_tput_naive) 

tput_opt_eq = [t/(10**6) for t in tput_opt_eq] 
tput_opt_uneq = [t/(10**6) for t in tput_opt_uneq]
tput_naive = [t/(10**6) for t in tput_naive] 
#
#plt.figure(1) 
#plt.plot(tput_opt, color='blue', label='Optimization result') 
#plt.plot(tput_naive, color='red', label='Naive power scaling')
#plt.xlabel('Iteration') 
#plt.ylabel('Sum throughput [in Mbps]') 
#plt.legend() 
#plt.grid() 

plt.figure() 

bins_tput = np.arange(min(tput_opt_eq), max(tput_opt_eq), 10)
counts_tput, bin_edges_tput = np.histogram(tput_opt_eq, bins=bins_tput, normed=True)
cdf_tput = np.cumsum(counts_tput)
cdf_tput = [x/sum(counts_tput) for x in cdf_tput]
plt.plot(bin_edges_tput[1:], cdf_tput, color='red', label='Optimization result') 

#bins_tput = np.arange(min(tput_opt_uneq), max(tput_opt_uneq), 10)
#counts_tput, bin_edges_tput = np.histogram(tput_opt_uneq, bins=bins_tput, normed=True)
#cdf_tput = np.cumsum(counts_tput)
#cdf_tput = [x/sum(counts_tput) for x in cdf_tput]
#plt.plot(bin_edges_tput[1:], cdf_tput, color='black', label='Optimization result --- Case II') 

#bins_tput = np.arange(min(tput_opt_rate_mapped), max(tput_opt_rate_mapped), 10)
#counts_tput, bin_edges_tput = np.histogram(tput_opt_rate_mapped, bins=bins_tput, normed=True)
#cdf_tput = np.cumsum(counts_tput)
#cdf_tput = [x/sum(counts_tput) for x in cdf_tput]
#plt.plot(bin_edges_tput[1:], cdf_tput, color='black', label='Optimization result (rate mapped)') 

bins_tput = np.arange(min(tput_naive), max(tput_naive), 10)
counts_tput, bin_edges_tput = np.histogram(tput_naive, bins=bins_tput, normed=True)
cdf_tput = np.cumsum(counts_tput)
cdf_tput = [x/sum(counts_tput) for x in cdf_tput]
plt.plot(bin_edges_tput[1:], cdf_tput, color='blue', label='Naive result') 

#bins_tput = np.arange(min(tput_naive_rate_mapped), max(tput_naive_rate_mapped), 10)
#counts_tput, bin_edges_tput = np.histogram(tput_naive_rate_mapped, bins=bins_tput, normed=True)
#cdf_tput = np.cumsum(counts_tput)
#cdf_tput = [x/sum(counts_tput) for x in cdf_tput]
#plt.plot(bin_edges_tput[1:], cdf_tput, color='orange', label='Naive result (rate mapped)') 

plt.legend() 
plt.xlabel('Sum throughput [in Mbps]')
plt.ylabel('CDF of sum throughput')
plt.grid() 
plt.savefig('./Power_Allocation.png',bbox_inches='tight') 
    
    
    
    
    
    