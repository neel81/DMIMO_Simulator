import bog_enums as enums
import numpy as np


class Backoff():
    Algorithm = enums.Backoff.Algorithm.bo_counter_per_bog
    SameRandomSlot = False
    RandomSlot = enums.Backoff.RandomSlot.r_max
    ResetPolicy = enums.Backoff.ResetPolicy.txg_and_winner_bogs
    IDLEProportion = 1  # 0.0-1.0
    aSlotTime_us = 9  # OFDM back slot length
    aCWmin = 15  # AC = best effort


class TXG():
    AllowExtension = True
    CombineWinners = True
    MinSize = 1


class CCA():
    Threshold_dBm = -82


class General():
    nAntennas_WAU = 2 
    nAntennas_STA = 2 
    WAU_grouping = enums.WAU_grouping.adjacent
    sim_time_us = 100000
    plot_topo = False
    plot_stats = True
    print_sim_time = True


class Frame():
    TXOP_Duration_us = 500
    DATA_Bandwidth_MHz = 80
    DATA_GI_ns = 400
    ACK_Length_B = 14
    ACK_MCS = 0
    ACK_Bandwidth_MHz = 20
    ACK_GI_ns = 400
    SIFS_us = 16


class PU():
    TotalTxPower_dBm = 20


class WAU():
    NoiseFigure_dB = 8


class STA():
    TotalTxPower_dBm = 20
    NoiseFigure_dB = 8


class Pathloss():
    fc_GHz = 5  # carrier frequency
    mu_dB = 0  # shadowing mean
    sigma_dB = 0  # shadowing deviation
    wallloss_dB = 0  # wall attenuation
    floorloss_dB = 0  # floor attenuation
    wavelength = 3e8 / (fc_GHz * 1e9)
    k_factor_dB = 5
    nlos_fading_stdev = 5
    breakpoint_distance = 10


class World():
    roomX_m = 10  # Room width
    roomY_m = 10  # Room length
    roomZ_m = 3  # Room height
    nRoomsX = 2  # Rooms in a row
    nRoomsY = 2  # Rooms in a column
    nRoomsZ = 1  # Number of floors
    wallsX = roomX_m * np.array(list(range(nRoomsX)))
    wallsY = roomY_m * np.array(list(range(nRoomsY)))
    wallsZ = roomZ_m * np.array(list(range(nRoomsZ)))
