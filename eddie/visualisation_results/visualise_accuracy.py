import matplotlib.pyplot as plt
import re
import os
import sys
import numpy as np
from scipy.stats import norm
import scipy.integrate as integrate
from settings_and_potential_eddie import *
import pandas as pd


font_size=35
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


## hard problem generated on lap top 
os.chdir("/home/s2133976/OneDrive/ExtendedProject/Code/Stepupyourgame/Stepupyourgame/data/eddie")


dta_noada = pd.DataFrame()
dta_tr = pd.DataFrame()
dta_re = pd.DataFrame()

for i in range(len(dtlist)):
    dti=dtlist[i]
    file_i = "data_hard/vec_noadai="+str(i)+".txt"
    x_noada=np.hstack(openCfile(file_i))
    file_i = "data_hard/vec_tri="+str(i)+".txt"
    x_tr=np.hstack(openCfile(file_i))
    # file_i = "data_easy500000/vec_rei="+str(i)+".txt"
    # x_re=np.hstack(openCfile(file_i))

    dta_noada["x"+str(i)] = x_noada
    dta_tr["x"+str(i)] = x_tr
    # dta_re["x"+str(i)] = x_re


weak_list_noada=[]
weak_list_tr=[]
# weak_list_re=[]



## When no access to the true moment
a=-10
b=10
norm=np.round(integrate.quad(lambda q: np.exp(-U(q)/tau), a,b)[0],16)
true_mom_1 = np.round(integrate.quad(lambda q: np.exp(-U(q)/tau)*q, a,b)[0],16)/norm

# print("true moment 1:\n")
# print(true_mom_1)
for j in range(len(dtlist)):
    x = dta_noada["x"+str(j)]
    weak_list_noada.append(np.abs((np.sum(x)/len(x)-true_mom_1)/true_mom_1))
    x = dta_tr["x"+str(j)]
    weak_list_tr.append(np.abs((np.sum(x)/len(x)-true_mom_1)/true_mom_1))
    # x = dta_re["x"+str(j)]  
    # weak_list_re.append(np.abs((np.sum(x)/len(x)-true_mom_1)/true_mom_1))


    # ###########################################
fig, (ax1)= plt.subplots(1,1,figsize=(7,7))# plt.figure(figsize=(4,4))
ax1.set_title("First moment")
ax1.plot(np.log(dtlist),np.log(weak_list_noada),"x",label="Overdamped",color=myred)
x,y_x,a_round = get_slope(weak_list_noada,dtlist)
ax1.plot(x,y_x,"--",label="slope: "+str(a_round),color=myred)
ax1.plot(np.log(dtlist),np.log(weak_list_tr),"x",label="Transformed",color=myblue)
x,y_x,a_round = get_slope(weak_list_tr,dtlist)
ax1.plot(x,y_x,"--",label="slope: "+str(a_round),color=myblue)
# ax1.plot(np.log(dtlist),np.log(weak_list_re),"x",label="Naive rescaling",color=mygreen)
# x,y_x,a_round = get_slope(weak_list_re,dtlist)
# ax1.plot(x,y_x,"--",label="slope: "+str(a_round),color=mygreen)
ax1.legend()

os.chdir("/home/s2133976/OneDrive/ExtendedProject/Code/Stepupyourgame/Stepupyourgame/eddie/visualisation_results")
fig.savefig('figures/accuracy_hard.png')

plt.show()