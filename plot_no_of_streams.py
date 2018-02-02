import matplotlib
import matplotlib.pyplot as plt
from latexify import latexify
from latexify import format_axes
import numpy as np

n_STAs = 40

random_users = [8]*100
random_users = np.array(random_users)
random_users[0:27] = 2
random_users[27:39] = 4
random_users[39:48] = 6

random_streams = [8]*100
norm_users = [8]*100
norm_users = np.array(norm_users)
norm_users[0:15] = 4
norm_users[15:25] = 6

norm_streams = [8]*100

params = latexify(columns=2)
matplotlib.rcParams.update(params)

plt.figure()

random_users = np.sort(random_users)
F2 = np.array(range(len(random_users)))/float(len(random_users))
plt.plot(random_users, F2, color='red', linestyle=':', linewidth=3, label='Random user selection \n(Two streams per user)')

random_streams = np.sort(random_streams)
F2 = np.array(range(len(random_streams)))/float(len(random_streams))
plt.plot(random_streams, F2, color='green', linestyle='-.', linewidth=3, label='Random user selection \n(Exactly one stream per user)')

norm_users = np.sort(norm_users)
F2 = np.array(range(len(norm_users)))/float(len(norm_users))
plt.plot(norm_users, F2, color='blue', linestyle='--', linewidth=3, label='Norm-based \n(Two streams per user)')

norm_streams = np.sort(norm_streams)
F2 = np.array(range(len(norm_streams)))/float(len(norm_streams))
plt.plot(norm_streams, F2, color='orange', linestyle='-', linewidth=3, label='Norm-based \n(Some users with single stream)')

plt.legend()
plt.xlabel(r'Number of streams served $s$')
plt.ylabel(r'Pr[S$<= s$]')
plt.title('Distribution of number of streams served S \n Group dimension = 20m x 20m, Number of users = {}'.format(n_STAs))
plt.tight_layout()
plt.savefig('no_of_streams_served.pdf')
plt.show()