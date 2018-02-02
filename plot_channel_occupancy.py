import csv
import matplotlib
import matplotlib.pyplot as plt
from latexify import latexify
from latexify import format_axes

with open('./chocc_DMIMO/Chocc_data_1.csv') as f:
    reader = csv.reader(f)
    chocc_list = list(reader)