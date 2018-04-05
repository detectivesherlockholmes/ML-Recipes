'''
Created on 01-Apr-2018

@author: sherlock
'''
import numpy as np
import matplotlib.pyplot as plt

grehounds = 500
labs = 500
grey_height = 28 + 4*np.random.randn(grehounds)
lab_height = 24 + 4*np.random.randn(labs)

plt.hist([grey_height, lab_height], stacked= True, color= ['r','b'])
plt.show()
