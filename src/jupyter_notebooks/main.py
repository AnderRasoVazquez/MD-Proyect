import csv
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

input_file = 'files/verbal_autopsies_clean.csv'

df = pd.read_csv(input_file, header = 0)

print(df['sex'])

plt.figure()
df['sex'].plot()


# with open('files/verbal_autopsies_clean.csv', newline='') as csvfile:
#     spamreader = csv.reader(csvfile, delimiter=',')
#     count = 0
#     for row in spamreader:
#         count += 1
#         print(row)
#         #print(', '.join(row))
#         if count == 10:
#             break