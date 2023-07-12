import matplotlib.pyplot as plt
import re
import os
import sys
import numpy as np
from scipy.stats import norm
from settings_and_potential import *
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



# nrank = sys.argv[0]
os.chdir("/home/s2133976/OneDrive/ExtendedProject/Code/Weak SDE approximation/C++/underdamped")


# global T
# global gamma
# global tau
# global h
# global n_samples
# global Nt

range_bins=[-3,3]

tau=0.1

s=10

# #/////////////////////////////////
# #// Spring potential definition //
# #/////////////////////////////////
# a=  20.0
# b=  0.1
# x0= 0.1
# c=  0.1
# def U(x):
#     res = (a**1.5*b**0.5*x0*np.arctan((a/b)**0.5*(x-x0))+(a*b*(a*x0*(x-x0)-b))/(a*(x-x0)**2+b)+c*(x-x0)**2+2*c*(x-x0)*x0)*0.5
#     return res

#/////////////////////////////////////////
#// Anisotropique potential definition //
#////////////////////////////////////////
def U(x):
    res = np.log(s*(x*x-1)*(x*x-1))
    return res
#define DIVTERM          //define to use
m=0.001      #     // minimum step scale factor
M=1.5       #       // maximum step scale factor
dt=0.0005     #      // artificial time stepsize
gamma=0.1         #   // friction coefficient
tau=0.1          #  // 'temperature'
numruns=500000         # // total number of trajectories
numsam=10000  
r=0.01
def getg(x):
    f=4*(x*x-1)
    fabs=np.abs(f)
    xi=fabs*fabs*fabs*r+m*m
    den=1/M+1/np.sqrt(xi)
    g=1/den
    return(g)


#######################################################
##### what do you want to run
#######################################################
## run the sample to check the look of the distributions 

q_noada,p_noada,g=openCfile_qp("data_one_dt/vec_noada.txt")
q_tr,p_tr,g=openCfile_qp("data_one_dt/vec_tr.txt")
g1,g2,g3=openCfile_qp("data_one_dt/val_g_ada.txt")


fig, ((axs))= plt.subplots(2,2,figsize=(25,25))# plt.figure(figsize=(4,4))
# fig.subplots_adjust(left=0.1,bottom=0.1, right=0.9, top=0.9, wspace=0.4, hspace=0.4)
nbins=105

############
## Q vals ##
############
# SDE
histogram_noada,bins = np.histogram(q_noada,bins=nbins,range=range_bins, density=True)
midx_sde = (bins[0:-1]+bins[1:])/2

#transformed
histogram_tr,bins = np.histogram(q_tr,bins=nbins,range=range_bins, density=True)
midx_tr = (bins[0:-1]+bins[1:])/2

# # rescaled
# histogram_re,bins = np.histogram(x_re,bins=nbins,range=range_bins, density=True)
# midx_re = (bins[0:-1]+bins[1:])/2

# Invariant distribution for q
rho = np.exp(- U(midx_sde)/tau)
# # normal dis
# rho = np.exp(- 0.5*midx_tr*midx_tr/tau)
rho = rho / (np.sum(rho)* (midx_sde[1]-midx_sde[0]) ) # Normalize rho by dividing by its approx. integral

# Log Q 
#######
axs[0,0].semilogy(midx_sde,(rho),"--",linewidth=lw*2,label='Invariant distribution $\\rho(x,\\infty)$',color=mygrey)
axs[0,0].semilogy(midx_sde,(histogram_noada),"-",linewidth=lw,label='Underdamped SDE',color=myred)
# axs.plot(midx_re,histogram_re,"--",linewidth=2.5,label='Underdamped SDE\nwith naive time rescaling',color=myblue)
axs[0,0].semilogy(midx_tr,(histogram_tr),"--",linewidth=lw,label='Transformed\nunderdamped SDE',color=myblue)
axs[0,0].set_ylim(np.exp(-2),np.exp(3))
axs[0,0].set_xlabel("q", fontsize=font_size)
def ticks(y, pos):
    return r'$e^{:.0f}$'.format(np.log(y))

# axs[1].xaxis.set_major_formatter(mtick.FuncFormatter(ticks))
axs[0,0].yaxis.set_major_formatter(mtick.FuncFormatter(ticks))


axs[0,0].legend(loc='upper center',
          ncol=1,fontsize=font_size) #, bbox_to_anchor=(0.5, 1.33),


# Q
###
axs[0,1].plot(midx_sde,rho,"--",linewidth=lw*2,label='Invariant distribution $\\rho(x,\\infty)$',color=mygrey)
axs[0,1].plot(midx_sde,histogram_noada,"-",linewidth=lw,label='Underdamped SDE',color=myred)
# axs.plot(midx_re,histogram_re,"--",linewidth=2.5,label='Underdamped SDE\nwith naive time rescaling',color=myblue)
axs[0,1].plot(midx_tr,histogram_tr,"--",linewidth=lw,label='Transformed\nunderdamped SDE',color=myblue)
axs[0,1].set_ylim(0,np.max(rho)+0.3)
axs[0,1].set_xlabel("q", fontsize=font_size)
axs[0,1].legend(fontsize=font_size)


############
## P vals ## 
############
range_bins=[-5.,5.]

# no ada
histogram_noada,bins = np.histogram(p_noada,bins=nbins,range=range_bins, density=True)
midx_noada = (bins[0:-1]+bins[1:])/2

# transformed
histogram_tr,bins = np.histogram(p_tr,bins=nbins,range=range_bins, density=True)
midx_tr = (bins[0:-1]+bins[1:])/2

# Invariant distribution for p
### momentum p invariant
rho = np.exp(-(midx_noada**2)/(2*tau))
rho = rho / (np.sum(rho)* (midx_noada[1]-midx_noada[0]) ) # Normalize rho by dividing by its approx. integral

axs[1,0].plot(midx_noada,rho,linewidth=lw*2,label='Invariant distribution $\\rho(x,\\infty)$',color=mygrey)
axs[1,0].plot(midx_noada,histogram_noada,"-",linewidth=lw,label='Underdamped SDE',color=myred)
# axs.plot(midx_re,histogram_re,"--",linewidth=2.5,label='Overdamped SDE\nwith naive time rescaling',color=myblue)
axs[1,0].plot(midx_tr,histogram_tr,"--",linewidth=lw,label='Transformed\nunderdamped SDE',color=myblue)
axs[1,0].set_xlabel("p", fontsize=font_size)
# axs[1,0].legend(fontsize=font_size)
# plt.show()

############
## Histograms g vals ## 
############
range_bins=[0,5]
histogram_g,bins = np.histogram(g1,bins=nbins,range=range_bins, density=True)
midx_noada = (bins[0:-1]+bins[1:])/2
mean_g=round(np.mean(g1),2)

axs[1,1].plot(midx_noada,histogram_g,linewidth=lw*2,label='Average step='+str(mean_g),color=mygrey)
axs[1,1].set_xlabel("g", fontsize=font_size)
axs[1,1].legend(fontsize=font_size, loc="upper center")

fig.savefig('figures/one_dt.png')
