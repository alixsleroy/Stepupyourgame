import matplotlib.pyplot as plt
import re
import os
import sys
import numpy as np
from scipy.stats import norm
import scipy.integrate as integrate
from settings_and_potential import *
import pandas as pd

# nrank = sys.argv[0]
os.chdir("/home/s2133976/OneDrive/ExtendedProject/Code/Stepupyourgame/Stepupyourgame/data/C/data_overdamped")



dta_noada = pd.DataFrame()
dta_tr = pd.DataFrame()
# dta_re = pd.DataFrame()

for i in range(len(dtlist)):
    dti=dtlist[i]
    file_i = "vec_noadai="+str(i)+".txt"
    x_noada=np.hstack(openCfile(file_i))
    file_i = "vec_tri="+str(i)+".txt"
    x_tr=np.hstack(openCfile(file_i))
    # file_i = "data_hard/vec_rei="+str(i)+".txt"
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
ax1.plot(np.log(dtlist),np.log(weak_list_noada),"x",label="ada",color="red")
x,y_x,a_round = get_slope(weak_list_noada,dtlist)
ax1.plot(x,y_x,"--",label="slope: "+str(a_round),color="red")
ax1.plot(np.log(dtlist),np.log(weak_list_tr),"x",label="tr",color="green")
x,y_x,a_round = get_slope(weak_list_tr,dtlist)
ax1.plot(x,y_x,"--",label="slope: "+str(a_round),color="green")
# ax1.plot(np.log(dtlist),np.log(weak_list_re),"x",label="re",color="blue")
# x,y_x,a_round = get_slope(weak_list_re,dtlist)
# ax1.plot(x,y_x,"--",label="slope: "+str(a_round),color="blue")
ax1.legend()
os.chdir("/home/s2133976/OneDrive/ExtendedProject/Code/Stepupyourgame/Stepupyourgame/C++/overdamped/visualisation")
fig.savefig('figures/accuracy_a3.png')

plt.show()