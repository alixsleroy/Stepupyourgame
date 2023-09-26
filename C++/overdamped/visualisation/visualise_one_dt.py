import matplotlib.pyplot as plt
import re
import os
import sys
import numpy as np
from scipy.stats import norm
from settings_and_potential import *

myblue = (0,119/235,187/235)
myred=(187/235,85/235,102/235)
myyellow=(221/235,170/235,51/235)
mygrey=(187/235,187/235,187/235)
mygreen="#66BB55"
mymagenta="#7733DD"


# nrank = sys.argv[0]
os.chdir("/home/s2133976/OneDrive/ExtendedProject/Code/Stepupyourgame/Stepupyourgame/data/C/data_overdamped_onedt")


# global T
# global gamma
# global tau
# global h
# global n_samples
# global Nt

range_bins=[-2,3]





#######################################################
##### what do you want to run
#######################################################
## run the sample to check the look of the distributions 
check_sample=0

x_noada=np.hstack(openCfile("vec_noada.txt"))
x_tr=np.hstack(openCfile("vec_tr.txt"))
x_re=np.hstack(openCfile("vec_re.txt"))

fig, ((axs))= plt.subplots(1,1,figsize=(20,5))# plt.figure(figsize=(4,4))
fig.subplots_adjust(left=0.1,bottom=0.1, right=0.9, top=0.9, wspace=0.4, hspace=0.4)
nbins=300

# SDE
histogram_sde,bins = np.histogram(x_noada,bins=nbins,range=range_bins, density=True)
midx_sde = (bins[0:-1]+bins[1:])/2


# transformed
histogram_tr,bins = np.histogram(x_tr,bins=nbins,range=range_bins, density=True)
midx_re = (bins[0:-1]+bins[1:])/2

# rescaled
histogram_re,bins = np.histogram(x_re,bins=nbins,range=range_bins, density=True)
midx_tr = (bins[0:-1]+bins[1:])/2

# Invariant distribution
rho = np.exp(- U(midx_sde)/tau)
rho = rho / (np.sum(rho)* (midx_sde[1]-midx_sde[0]) ) # Normalize rho by dividing by its approx. integral

axs.plot(midx_sde,rho,linewidth=2.5,label='Invariant distribution $\\rho(x,\\infty)$',color="orange")
axs.plot(midx_sde,histogram_sde,"-",linewidth=2.5,label='Overdamped SDE',color=myred)
axs.plot(midx_re,histogram_re,"-.",linewidth=2.5,label='Overdamped SDE\nwith naive time rescaling',color=myblue)
axs.plot(midx_tr,histogram_tr,"--",linewidth=2.5,label='Transformed\noverdamped SDE',color=mygreen)
axs.legend()
os.chdir("/home/s2133976/OneDrive/ExtendedProject/Code/Stepupyourgame/Stepupyourgame/C++/overdamped/visualisation")
fig.savefig("figures/onedt_doublewell.png")


