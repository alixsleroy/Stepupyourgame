#### SET UP IN SETTING AND POTENTIAL THE DT LIST AND PARA OF THE POTENTIAL 
#### SET UP THE PATH TO THE DATA OF INTEREST 
## hard problem generated on lap top 
import os
os.chdir("/home/s2133976/OneDrive/ExtendedProject/Code/Stepupyourgame/Stepupyourgame/data/eddie")



import matplotlib.pyplot as plt
import re
import sys
import numpy as np
from scipy.stats import norm
import scipy.integrate as integrate

import pandas as pd
font_size=25
lw=4

myblue = (0,119/235,187/235)
myred=(187/235,85/235,102/235)
myyellow=(221/235,170/235,51/235)
mygrey=(187/235,187/235,187/235)
mygreen="#66BB55"
mymagenta="#7733DD"

import matplotlib.ticker as mtick
plt.rc('xtick', labelsize=font_size) 
plt.rc('ytick', labelsize=font_size) 

from settings_and_potential_eddie import *
print(dtlist)

### EDDIE RESULTS 

dta_noada = pd.DataFrame()
dta_tr = pd.DataFrame()
dta_re = pd.DataFrame()

# for i in range(len(dtlist)):
i=0
dti=dtlist[i]
file_i = "data_a275/vec_noadai="+str(i)+".txt"
x_noada=np.hstack(openCfile(file_i))
file_i = "data_a275/vec_tri="+str(i)+".txt"
x_tr=np.hstack(openCfile(file_i))
# file_i = "data_hard_a3/vec_rei="+str(i)+".txt"
# x_re=np.hstack(openCfile(file_i))

dta_noada["x"+str(i)] = x_noada
dta_tr["x"+str(i)] = x_tr
# dta_re["x"+str(i)] = x_re