##### Build a main to compute accuracy plot with different schemes and different examples problem
## How to use those files 
# 1) Set up in settings.py the values of M and m and choose the potential to use "bond", "square", "power4", "spring"
 
#######################################################
##### Import packages and set up paths
#######################################################

import matplotlib.pyplot as plt
import sys

## Path for data
path="/home/s2133976/OneDrive/ExtendedProject/Code/Stepupyourgame/Stepupyourgame/data/python/accuracy_spring_potential_overdamped/saved_pickles_over/"
## path for figures 
path_figures="/home/s2133976/OneDrive/ExtendedProject/Code/Stepupyourgame/Stepupyourgame/Python/accuracy computation/accuracy_spring_potential/visualisation/figures/"

## Set global variables 
global T
global gamma
global tau
global h
global n_samples
global Nt

## Import the package and the values of the global var to run the samples 
sys.path.append("Python/accuracy computation/accuracy_spring_potential")
from settings import *
from useful_tools import *
from accuracy_over_func import *

#######################################################
##### Plot presentations (colors)
#######################################################
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

#######################################################
##### what do you want to run
#######################################################
## run the sample to check the look of the distributions 
check_sample=0
plot_sample=0
run_for_h_var=0
run_errors_moments=1
test=0
#######################################################
##### Set up important settings
#######################################################
    
## SQUARE POTENTIAL RESULTS
# ## for precise results on accuracy curves 
# dt_list= [2 ** (R-10) for R in range(7)]
# T=1.0
# tau=1
# n_samples=1000000
# x=0
# range_bins=[-10,10] # range to plot samples on histograms
# nbins=50
# range_int=[-10,10] # integration range 

##############################
## SPRING POTENTIAL RESULTS ##
##############################
## easier problem 
#####################
# with parameters
m=0.001
M=1.5
pot="spring"
a=1
b=1
x0=0.5
c=0.1
dt_list= [0.009,0.01,0.02,0.03,0.04,0.05,0.06,0.07,0.08,0.09,0.1,0.2,0.3,0.4,0.5,0.6]
T=100
tau=0.1
n_samples=5000123
x=0
range_bins=[-10,10] # range to plot samples on histograms
nbins=50
range_int=[-10,10] # integration range

## very hard problem 
#####################
# ## with parameters
# m=0.001
# M=1.5
# pot="spring"
# a=10
# b=0.1
# x0=0.5
# c=0.1
# dt_list= [0.009,0.01,0.02,0.03,0.04,0.05,0.06,0.07,0.08,0.09,0.1,0.2,0.3,0.4,0.5,0.6]
# #dt_list= [0.0007,0.0009,0.001,0.003,0.005,0.007,0.009,0.01,0.03,0.05,0.07,0.09,0.1,0.3,0.5]
# T=100 
# tau=0.1
# n_samples=10000000
# x=0
# range_bins=[-1,1] # range to plot samples on histograms
# nbins=50
# range_int=[-10,10] # integration range 

#####################################################
## Verify that this is the distributions of squar pot
#####################################################
if check_sample==1:
    check_methods(Nt,tau,n_samples,h,x,range_bins,T)

if plot_sample==1:
    list_param = str(pot)+'-tau='+str(tau)+'-M='+str(M)+'m='+str(m)+"-Nt="+str(Nt)+"-ns="+str(n_samples)+"-h="+str(h)
    dta_noada=pd.read_pickle(path+"plot_hist/dta_noada_"+list_param)
    dta_tr=pd.read_pickle(path+"plot_hist/dta_tr_"+list_param)
    dta_re=pd.read_pickle(path+"plot_hist/dta_re_"+list_param)
    plot_one_distr_over(dta_noada,dta_tr,dta_re,tau,range_bins,h,T)


###########################################
## Obtain the samples for different dt
###########################################
if run_for_h_var==1:
    sample_listdt(T,tau,n_samples,x,dt_list,pot)


# ###########################################
# ## Open the samples previously saved
# ###########################################
if run_errors_moments==1:

    # List of parameters 
    list_param = str(pot)+'-tau='+str(tau)+'-M='+str(M)+'m='+str(m)+"-T="+str(T)+"-ns="+str(n_samples)

    dta_noada=pd.read_pickle(path+"dta_noada_"+list_param)
    dta_tr=pd.read_pickle(path+"dta_tr_"+list_param)


    # ###########################################
    # ## Interpolate the slope
    # ###########################################
    x = np.log(dt_list)
    A = np.vstack([x, np.ones(len(x))]).T

    # ###########################################
    # ## Compute the log moments
    # ###########################################
    fig, (ax1,ax2,ax3,ax4)= plt.subplots(1,4,figsize=(20,7))# plt.figure(figsize=(4,4))
    fig.subplots_adjust(left=0.1,bottom=0.1, right=0.9, top=0.9, wspace=0.4, hspace=0.4)

    ax1.set_title("First moment",fontsize=font_size)
    ax2.set_title("Second moment",fontsize=font_size)
    ax3.set_title("Third moment",fontsize=font_size)
    ax4.set_title("Fourth moment",fontsize=font_size)

    ## no adaptive 
    ###############
    lab="Overdamped"
    mom1_list,mom2_list,mom3_list,mom4_list,mom_1_plussd,mom_2_plussd,mom_3_plussd,mom_4_plussd=moment_list(dt_list,tau,dta_noada,range_int)
    ax1.plot(x,np.log(mom1_list),"x",color=myred)
    ax2.plot(x,np.log(mom2_list),"x",color=myred)
    ax3.plot(x,np.log(mom3_list),"x",color=myred)
    ax4.plot(x,np.log(mom4_list),"x",color=myred)

    ## interpolation linear  
    #first moment
    m, c = np.linalg.lstsq(A, np.log(mom1_list), rcond=None)[0]
    ax1.plot(x, m*x + c,"--",label="slope: "+str(round(m,2)),color=myred)
    #second moment
    m, c = np.linalg.lstsq(A, np.log(mom2_list), rcond=None)[0]
    ax2.plot(x, m*x + c,"--",label="slope: "+str(round(m,2)),color=myred)
    #third moment
    m, c = np.linalg.lstsq(A, np.log(mom3_list), rcond=None)[0]
    ax3.plot(x, m*x + c,"--",label="slope: "+str(round(m,2)),color=myred)
    #fourth moment
    m, c = np.linalg.lstsq(A, np.log(mom4_list), rcond=None)[0]
    ax4.plot(x, m*x + c,"--",label="slope: "+str(round(m,2)),color=myred)


    ## Transformed 
    ###############
    # lab="Transformed overdamped"
    mom1_list,mom2_list,mom3_list,mom4_list,mom_1_plussd,mom_2_plussd,mom_3_plussd,mom_4_plussd=moment_list(dt_list,tau,dta_tr,range_int)
    ax1.plot(x,np.log(mom1_list),"x",color=myblue)
    ax2.plot(x,np.log(mom2_list),"x",color=myblue)
    ax3.plot(x,np.log(mom3_list),"x",color=myblue)
    ax4.plot(x,np.log(mom4_list),"x",color=myblue)


    ## interpolation linear  
    #first moment
    m, c = np.linalg.lstsq(A, np.log(mom1_list), rcond=None)[0]
    ax1.plot(x, m*x + c,"--",label="slope: "+str(round(m,2)),color=myblue)
    #second moment
    m, c = np.linalg.lstsq(A, np.log(mom2_list), rcond=None)[0]
    ax2.plot(x, m*x + c,"--",label="slope: "+str(round(m,2)),color=myblue)
    #third moment
    m, c = np.linalg.lstsq(A, np.log(mom3_list), rcond=None)[0]
    ax3.plot(x, m*x + c,"--",label="slope: "+str(round(m,2)),color=myblue)
    #fourth moment
    m, c = np.linalg.lstsq(A, np.log(mom4_list), rcond=None)[0]
    ax4.plot(x, m*x + c,"--",label="slope: "+str(round(m,2)),color=myblue)


    ax1.legend(fontsize=font_size*0.8,ncol=1,loc="upper left")#,bbox_to_anchor=(1.5, 1.3))
    ax2.legend(fontsize=font_size*0.8,ncol=1,loc="upper left")#,bbox_to_anchor=(1.5, 1.3))
    ax3.legend(fontsize=font_size*0.8,ncol=1,loc="upper left")#,bbox_to_anchor=(1.5, 1.3))
    ax4.legend(fontsize=font_size*0.8,ncol=1,loc="upper left")#,bbox_to_anchor=(1.5, 1.3))


    fig.savefig(path_figures+'moments_logplot_'+str(pot)+'.png')


    # ###########################################
    # ## Compute the accuracy curves
    # ###########################################


    fig, (ax1)= plt.subplots(1,1,figsize=(10,7))# plt.figure(figsize=(4,4))
    fig.subplots_adjust(left=0.1,bottom=0.1, right=0.9, top=0.9, wspace=0.4, hspace=0.4)

    ## No ada 
    accuracy_list =error_list(nbins,dt_list,range_bins,tau,dta_noada)
    x,y_x,a_round=get_slope(accuracy_list,dt_list)
    ax1.plot(np.log(dt_list),np.log(accuracy_list),label="No ada")
    ax1.plot(x,y_x,"--",label="slope="+str(a_round))

    ## Tr
    accuracy_list =error_list(nbins,dt_list,range_bins,tau,dta_tr)
    x,y_x,a_round=get_slope(accuracy_list,dt_list)
    ax1.plot(np.log(dt_list),np.log(accuracy_list),label="Transfo")
    ax1.plot(x,y_x,"--",label="slope="+str(a_round))

    # ## Re
    # accuracy_list =error_list(nbins,dt_list,range_bins,tau,dta_rescale)
    # x,y_x,a_round=get_slope(accuracy_list,dt_list)
    # ax1.plot(np.log(dt_list),np.log(accuracy_list),label="Rescale")
    # ax1.plot(x,y_x,"--",label="slope="+str(a_round))

    #
    ax1.set_ylabel("$\log(\epsilon)$")
    ax1.set_xlabel("$\log(h)$")
    ax1.set_title("Accuracy on EM with potential "+pot)
    ax1.legend()
    fig.savefig(path+'accuracy_plot_'+str(pot)+'.png')


if test==1:
    # save the values of the function g
    list_param = str(pot)+'-tau='+str(tau)+'-M='+str(M)+'m='+str(m)+"-T="+str(T)+"-ns="+str(n_samples)
    dta_g_transfo=pd.read_pickle(path+"dta_g_tr_"+list_param)
    dta_g_rescale=pd.read_pickle(path+"dta_g_re_"+list_param)

    ############################################
    ### Mean step size for each size of step h
    ############################################

    fig, (axs)= plt.subplots(1,len(dt_list),figsize=(10,7))# plt.figure(figsize=(4,4))
    fig.subplots_adjust(left=0.1,bottom=0.1, right=0.9, top=0.9, wspace=0.4, hspace=0.4)

    # each mean value of steps 
    list_mean_g_transfo=[]
    list_mean_g_rescale=[]

    for j in range(len(dt_list)):
        ## Transformed step size
        ########################
        var = dta_g_transfo["g"+str(j)]
        ## plot histograms 
        mean_tr=np.mean(var)
        lowerbound=np.min(var)
        upperbound=np.max(var)
        nbins=100
        histogram,bins = np.histogram(var,bins=nbins,range=[lowerbound,upperbound], density=True)
        midx = (bins[0:-1]+bins[1:])/2
        axs[j].bar(midx,histogram,width=(upperbound-lowerbound)/nbins,color=myblue,label="Transformed",alpha=0.5)

        ## Rescale step size 
        #####################
        var = dta_g_rescale["g"+str(j)]
        ## plot histograms 
        mean_re=np.mean(var)
        lowerbound=np.min(var)
        upperbound=np.max(var)
        nbins=100
        histogram,bins = np.histogram(var,bins=nbins,range=[lowerbound,upperbound], density=True)
        midx = (bins[0:-1]+bins[1:])/2
        axs[j].bar(midx,histogram,width=(upperbound-lowerbound)/nbins,color=mygreen,label="Rescale",alpha=0.5)

        ## labels histograms
        axs[j].set_xlabel("$g(x)$")
        axs[j].legend()
        txt = "Tr="+str(round(mean_tr,3))+" \nRe="+str(round(mean_re,3))
        axs[j].set_title(txt)

    fig.savefig(path_figures+'mean_g_plot_'+list_param+'.png')
