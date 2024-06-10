
import matplotlib.pyplot as plt
import csv
import numpy as np
import matplotlib.pyplot as plt 
import csv 
import torch
from scipy import interpolate
x1 = [] 
y1 = [] 

i = 0
with open('/home/yanqiao/Downloads/csv_range.csv','r') as csvfile: 
    plots = csv.reader(csvfile, delimiter = ',') 
    for row in plots: 
        i += 1
        if i == 1: continue
        x1.append(float(row[1])) 
        y1.append(float(row[2])) 
print('x len: ', len(x1))
print('y len: ', len(y1))

f = interpolate.interp1d(x1, y1)

# xnew = np.arange(1, len(x1), 0.1)

ynew = f(x1)   # use interpolation function returned by `interp1d`

plt.plot(y1, '.', ynew ,'-', color='b', label='range+color')



x2 = [] 
y2 = [] 
i = 0
with open('/home/yanqiao/Downloads/csv_range_only.csv','r') as csvfile: 
    plots = csv.reader(csvfile, delimiter = ',') 
    for row in plots: 
            i += 1
            if i == 1: continue
            x2.append(float(row[1])) 
            y2.append(float(row[2])) 

print('x len: ', len(x2))
print('y len: ', len(y2))
f = interpolate.interp1d(x2, y2)
ynew = f(x2) 
plt.plot(y2, '.', ynew ,'-', color='r', label='range only')
plt.legend()
plt.show()
