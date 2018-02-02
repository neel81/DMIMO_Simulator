class Backoff():

    class Algorithm():
        bo_counter_per_bog = 0
        bo_counter_per_wau = 1
        bo_counter_per_pu = 2

    class RandomSlot():
        r_min = 0
        r_max = 1
        r_avg = 2

    class ResetPolicy():
        winner_bogs = 0
        txg_and_winner_bogs = 1
        all_bogs = 2


class WAU_grouping():
    independent = 0
    adjacent = 1
    far = 2
