##### Build a main to compute accuracy plot with different schemes and different examples problem

import matplotlib.pyplot as plt
import sys
import matplotlib.colors as colors

sys.path.append("Python/accuracy/accuracy_moredims")

# Import the package to run the samples 
from pot_definition import *
from settings import *
from useful_tools import *
from sample_over_noada import *
from sample_over_transformed import *
from sample_over_rescaled import *

global T
global gamma
global tau
global h
global n_samples
global Nt

# #######################################################
# ##### what do you want to run
# #######################################################
# ## run the sample to check the look of the distributions 
run_samples=1
run_plots=1 #color plot 
run_plots2=1 #scatter plot
histx=1 #histogram in the x axis

# #######################################################
# ##### Set up important settings
# #######################################################
    
## set up others values 
T=10
tau=1
n_samples=10000
h=0.02
Nt=int(T*1/h)+1
xinit=np.ones(2)*0
xinit[0]=1
xinit[1]=0

if run_samples:
    # ###################
    # ## No adaptivity
    # ###################
    ## no ada
    # print(xinit)
    example = sample_noada(xinit,n_samples,Nt,h,tau)
    dta = dta_format_over(example)
    dta.to_pickle("Python/accuracy/accuracy_moredims/saved_pickles/dta_noada"+str(h))
    # ###################
    # ## Transformed
    # ###################
    example = sample_tr(xinit,n_samples,Nt,h,tau)
    dta = dta_format_over(example)
    dta.to_pickle("Python/accuracy/accuracy_moredims/saved_pickles/dta_tr"+str(h))

    # ###################
    # ## Rescaled
    # ###################
    example = sample_re(xinit,n_samples,Nt,h,tau)
    dta = dta_format_over(example)
    dta.to_pickle("Python/accuracy/accuracy_moredims/saved_pickles/dta_re"+str(h))



############################
## Plot a color gradient map
############################
if run_plots==1:
    fig, (ax1,ax2,ax3) = plt.subplots(ncols=3,figsize=(16,5))# plt.figure(figsize=(4,4))
    fig.subplots_adjust(left=0.1,
                                bottom=0.1, 
                                right=0.9, 
                                top=0.9, 
                                wspace=0.4, 
                                hspace=0.4)

    ## No adaptivity 
    #################
    dta_noada=pd.read_pickle("Python/accuracy/accuracy_moredims/saved_pickles/dta_noada"+str(h))
    xplot,yplot = dta_noada["x"], dta_noada["y"]
    H, xedges, yedges = np.histogram2d(xplot,yplot,200)
    ax1.pcolormesh(xedges,yedges, H, norm=colors.LogNorm(vmin=0.001, vmax=H.max()), cmap=plt.cm.jet) 
    ax1.set_xlim(xplot.min(), xplot.max())
    ax1.set_ylim(yplot.min(), yplot.max())
    ax1.set_xlabel('x')
    ax1.set_ylabel('y')
    ax1.set_title('no adaptivity')
    ax1.grid()

    ## Transformed 
    ##############
    dta_tr=pd.read_pickle("Python/accuracy/accuracy_moredims/saved_pickles/dta_tr"+str(h))
    xplot,yplot  = dta_tr["x"], dta_tr["y"]
    H, xedges, yedges = np.histogram2d(xplot,yplot,200)
    ax2.pcolormesh(xedges,yedges, H, norm=colors.LogNorm(vmin=0.001, vmax=H.max()), cmap=plt.cm.jet) 
    ax2.set_xlim(xplot.min(), xplot.max())
    ax2.set_ylim(yplot.min(), yplot.max())
    ax2.set_xlabel('x')
    ax2.set_ylabel('y')
    ax2.set_title('transformed')
    ax2.grid()

    ## Rescaled 
    ############
    dta_re=pd.read_pickle("Python/accuracy/accuracy_moredims/saved_pickles/dta_re"+str(h))
    xplot,yplot = dta_re["x"], dta_re["y"]

    # ax3.plot(xplot,yplot,"o")
    H, xedges, yedges = np.histogram2d(xplot,yplot,200)
    ax3.pcolormesh(xedges,yedges, H, norm=colors.LogNorm(vmin=0.001, vmax=H.max()), cmap=plt.cm.jet) 
    ax3.set_xlim(xplot.min(), xplot.max())
    ax3.set_ylim(yplot.min(), yplot.max())
    ax3.set_xlabel('x')
    ax3.set_ylabel('y')
    ax3.set_title('rescaled')
    ax3.grid()

    xstring="x"+str(xinit[0])+"y"+str(xinit[1])
    plt.savefig("Python/accuracy/accuracy_moredims/saved_plots/scatter/plot_distr_T="+str(T)+"tau="+str(tau)+"nsample="+str(n_samples)+"h"+str(h)+"M="+str(M)+"m="+str(m)+xstring+".png", bbox_inches='tight')
    plt.show()


############################
## Plot a scatter plot
############################
if run_plots2==1:
    fig, (ax1,ax2,ax3) = plt.subplots(ncols=3,figsize=(16,5))# plt.figure(figsize=(4,4))
    fig.subplots_adjust(left=0.1,
                                bottom=0.1, 
                                right=0.9, 
                                top=0.9, 
                                wspace=0.4, 
                                hspace=0.4)
    
    xstring=", xinit="+str(xinit[0])+" and yinit="+str(xinit[1])
    fig.suptitle("T="+str(T)+", tau="+str(tau)+", nsample="+str(n_samples)+", h="+str(h)+", M="+str(M)+", m="+str(m)+xstring)
    
    xmin=-5
    xmax=5
    ymin=-5
    ymax=5


    ## No adaptivity 
    #################
    dta_noada=pd.read_pickle("Python/accuracy/accuracy_moredims/saved_pickles/dta_noada"+str(h))
    xplot,yplot = dta_noada["x"], dta_noada["y"]
    ax1.plot(xplot,yplot,"o",markersize=4)
    ax1.plot(xinit[0],xinit[1],"o",markersize=4,color="red")
    # H, xedges, yedges = np.histogram2d(xplot,yplot,200)
    # ax1.pcolormesh(xedges, yedges, H, norm=colors.LogNorm(vmin=0.001, vmax=H.max()), cmap=plt.cm.jet) 
    ax1.set_xlim(xplot.min(), xplot.max())
    ax1.set_ylim(yplot.min(), yplot.max())
    ax1.set_xlabel('x')
    ax1.set_ylabel('y')
    ax1.set_title('no adaptivity')
    ax1.grid()
 

    ## Transformed 
    ##############
    dta_tr=pd.read_pickle("Python/accuracy/accuracy_moredims/saved_pickles/dta_tr"+str(h))
    xplot,yplot  = dta_tr["x"], dta_tr["y"]
    ax2.plot(xplot,yplot,"o",markersize=4)
    ax2.plot(xinit[0],xinit[1],"o",markersize=4,color="red")
    ax2.set_xlabel('x')
    ax2.set_ylabel('y')
    ax2.set_xlim(xplot.min(), xplot.max())
    ax2.set_ylim(yplot.min(), yplot.max())
    ax2.set_title('transformed')
    ax2.grid()

    ## Rescaled 
    ############
    dta_re=pd.read_pickle("Python/accuracy/accuracy_moredims/saved_pickles/dta_re"+str(h))
    xplot,yplot = dta_re["x"], dta_re["y"]
    ax3.plot(xplot,yplot,"o",markersize=4)
    ax3.plot(xinit[0],xinit[1],"o",markersize=4,color="red")
    ax3.set_xlabel('x')
    ax3.set_ylabel('y')
    ax3.set_xlim(xplot.min(), xplot.max())
    ax3.set_ylim(yplot.min(), yplot.max())
    ax3.set_title('rescaled')
    ax3.grid()

    plt.savefig("Python/accuracy/accuracy_moredims/saved_plots/scatter/plot_points_T="+str(T)+"tau="+str(tau)+"nsample="+str(n_samples)+"h"+str(h)+"M="+str(M)+"m="+str(m)+xstring+".png", bbox_inches='tight')
    plt.show()




############################
## Plot histograms
############################
if histx==1:
    nbins=100
    xmin=-10
    xmax=10
    range_bins = [xmin,xmax]
    ymax=2
    ymin=0
    fig, (ax1,ax2,ax3) = plt.subplots(ncols=3,figsize=(16,5))# plt.figure(figsize=(4,4))
    fig.subplots_adjust(left=0.1,
                                bottom=0.1, 
                                right=0.9, 
                                top=0.9, 
                                wspace=0.4, 
                                hspace=0.4)
    
    xstring=", xinit="+str(xinit[0])+" and yinit="+str(xinit[1])
    fig.suptitle("T="+str(T)+", tau="+str(tau)+", nsample="+str(n_samples)+", h="+str(h)+", M="+str(M)+", m="+str(m)+xstring)
    

    ## No adaptivity 
    #################
    dta_noada=pd.read_pickle("Python/accuracy/accuracy_moredims/saved_pickles/dta_noada"+str(h))
    xplot,yplot = dta_noada["x"], dta_noada["y"]

    # do the histogram for x 
    histogram,bins = np.histogram(xplot,bins=nbins,range=range_bins, density=True)
    midx = (bins[0:-1]+bins[1:])/2
    ax1.plot(midx,histogram) #,label='SDE',color="red")

    # add the true distribution 
    rho = rhox(midx,tau)
    rho = rho / ( np.sum(rho)* (midx[1]-midx[0]) ) # Normalize rho by dividing by its approx. integral
    ax1.plot(midx,rho,'--',label='Invariant',color="orange")
    ax1.legend(loc='lower center',bbox_to_anchor=(1.3, 0.5),ncol=1, fancybox=True, shadow=True)

    #set up labels 
    ax1.set_xlim(xmin,xmax)
    ax1.set_ylim([ymin, ymax])
    ax1.set_xlabel('x')
    ax1.set_ylabel('$\\rho(x)$')
    ax1.set_title('no adaptivity')
    ax1.grid()
 

    ## Transformed 
    ##############
    dta_tr=pd.read_pickle("Python/accuracy/accuracy_moredims/saved_pickles/dta_tr"+str(h))
    xplot,yplot  = dta_tr["x"], dta_tr["y"]

    # do the histogram for x 
    histogram,bins = np.histogram(xplot,bins=nbins,range=range_bins, density=True)
    midx = (bins[0:-1]+bins[1:])/2
    ax2.plot(midx,histogram) #,label='SDE',color="red")

    # add the true distribution 
    rho = rhox(midx,tau)
    rho = rho / ( np.sum(rho)* (midx[1]-midx[0]) ) # Normalize rho by dividing by its approx. integral
    ax2.plot(midx,rho,'--',label='Invariant',color="orange")
    ax2.legend(loc='lower center',bbox_to_anchor=(1.3, 0.5),ncol=1, fancybox=True, shadow=True)

    #set up labels 
    ax2.set_xlim(xmin,xmax)
    ax2.set_ylim(ymin, ymax)
    ax2.set_xlabel('x')
    ax2.set_ylabel('$\\rho(x)$')
    ax2.set_title('transformed')
    ax2.grid()

    ## Rescaled 
    ############
    dta_re=pd.read_pickle("Python/accuracy/accuracy_moredims/saved_pickles/dta_re"+str(h))
    xplot,yplot = dta_re["x"], dta_re["y"]

    # do the histogram for x 
    histogram,bins = np.histogram(xplot,bins=nbins,range=range_bins, density=True)
    midx = (bins[0:-1]+bins[1:])/2
    ax3.plot(midx,histogram) #,label='SDE',color="red")

    # add the true distribution 
    rho = rhox(midx,tau)
    rho = rho / ( np.sum(rho)* (midx[1]-midx[0]) ) # Normalize rho by dividing by its approx. integral
    ax3.plot(midx,rho,'--',label='Invariant',color="orange")
    ax3.set_ylim([0, 4])
    ax3.legend(loc='lower center',bbox_to_anchor=(1.3, 0.5),ncol=1, fancybox=True, shadow=True)

    # set up lables 
    ax3.set_xlim(xmin,xmax)
    ax3.set_ylim(ymin,ymax)
    ax3.set_xlabel('x')
    ax3.set_ylabel('$\\rho(x)$')
    ax3.set_title('rescaled')
    ax3.grid()

    plt.savefig("Python/accuracy/accuracy_moredims/saved_plots/histx/plot_points_T="+str(T)+"tau="+str(tau)+"nsample="+str(n_samples)+"h"+str(h)+"M="+str(M)+"m="+str(m)+xstring+".png", bbox_inches='tight')
    plt.show()




