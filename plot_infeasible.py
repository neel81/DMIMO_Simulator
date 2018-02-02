#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 19 14:55:57 2017

@author: neel45
"""

import matplotlib
import matplotlib.pyplot as plt 
import numpy as np
from latexify import latexify
from latexify import format_axes

SMALL_SIZE = 9 
MEDIUM_SIZE = 11 
BIGGER_SIZE = 13

y = [12.7,3.6,1.2,0]

params = latexify(fig_width=9.5, fig_height=7, columns=2)
matplotlib.rcParams.update(params)
fig,ax = plt.subplots() 
width = 0.1
ind = np.arange(len(y))

rects = ax.bar(ind, y, width, color='red', align='center')

ax.set_ylabel('Percentage of TxOPs')
ax.set_xlabel('User grouping strategy')
ax.set_title('Percentage of TxOPs for which user grouping was infeasible')
ax.set_xticks(ind) 
labels = ['Random \n 2 streams per user','Norm-based \n 2 streams per user', 'Random \n 1 stream per user', 'Norm-based \n Some users with \n fewer than 2 streams']

rects[0].set_color('r') 
rects[1].set_color('b')
rects[2].set_color('g')

ax.set_xticklabels(labels)
plt.tight_layout()
plt.savefig('plot_infeasible.pdf')
plt.show() 