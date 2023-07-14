## Import the package to run the samples 

from useful_tools import *
import os
import sys
# nrank = sys.argv[0]
os.chdir("/home/s2133976/OneDrive/ExtendedProject/Code/Weak SDE approximation/Python/accuracy/")

# Parameters of the simulations 
T=50
tau=0.1
n_samples=1000 #0000
n_samples2_rank = int(n_samples/12)
x=0
m=0.001
M=1.5
dt_list= [0.0005,0.001,0.01,0.05]
pot="spring"
test=0
run_errors_moments=1
range_bins=[-4,2]
nbins=50
range_int=[-10,10] #range of the intergration born

# ###########################################
# ## Open the samples previously saved
# ###########################################
if run_errors_moments==1:

    # List of parameters 
    list_param = 'spring-tau='+str(tau)+'-M='+str(M)+'m='+str(m)+"-T="+str(T)+"-ns="+str(n_samples)
    list_param = 'spring'
    dta_noada=pd.read_pickle("accuracy_openmp/data/dta_noada_"+list_param)
    dta_transfo=pd.read_pickle("accuracy_openmp/data/dta_tr_"+list_param)
    dta_rescale=pd.read_pickle("accuracy_openmp/data/dta_re_"+list_param)


    # ###########################################
    # ## Compute the moments
    # ###########################################
    fig, (ax1,ax2,ax3,ax4)= plt.subplots(1,4,figsize=(20,7))# plt.figure(figsize=(4,4))
    ax1.set_title("First moment")
    ax2.set_title("Second moment")
    ax3.set_title("Third moment")
    ax4.set_title("Fourth moment")

    lab="no ada"
    mom1_list,mom2_list,mom3_list,mom4_list,mom_1_plussd,mom_2_plussd,mom_3_plussd,mom_4_plussd=moment_list(dt_list,tau,dta_noada,range_int)

    ax1.plot(dt_list,mom1_list,label=lab,color="red")
    ax2.plot(dt_list,mom2_list,label=lab,color="red")
    ax3.plot(dt_list,mom3_list,label=lab,color="red")
    ax4.plot(dt_list,mom4_list,label=lab,color="red")

    # add CI (95%)
    lab="95% CI"
    ax1.plot(dt_list,mom_1_plussd,"--",label=lab,color="red")
    ax2.plot(dt_list,mom_2_plussd,"--",label=lab,color="red")
    ax3.plot(dt_list,mom_3_plussd,"--",label=lab,color="red")
    ax4.plot(dt_list,mom_4_plussd,"--",label=lab,color="red")

    lab="transformed"
    mom1_list,mom2_list,mom3_list,mom4_list,mom_1_plussd,mom_2_plussd,mom_3_plussd,mom_4_plussd=moment_list(dt_list,tau,dta_transfo,range_int)
    ax1.plot(dt_list,mom1_list,label=lab,color="green")
    ax2.plot(dt_list,mom2_list,label=lab,color="green")
    ax3.plot(dt_list,mom3_list,label=lab,color="green")
    ax4.plot(dt_list,mom4_list,label=lab,color="green")

    # add CI (95%)
    lab="95% CI"
    ax1.plot(dt_list,mom_1_plussd,"--",label=lab,color="green")
    ax2.plot(dt_list,mom_2_plussd,"--",label=lab,color="green")
    ax3.plot(dt_list,mom_3_plussd,"--",label=lab,color="green")
    ax4.plot(dt_list,mom_4_plussd,"--",label=lab,color="green")

    ## Do not consider rescaled 
    # lab="rescaled"
    # mom1_list,mom2_list,mom3_list,mom4_list,mom_1_plussd,mom_2_plussd,mom_3_plussd,mom_4_plussd=moment_list(dt_list,tau,dta_rescale,range_int)
    # ax1.plot(dt_list,mom1_list,label=lab,color="blue")
    # ax2.plot(dt_list,mom2_list,label=lab,color="blue")
    # ax3.plot(dt_list,mom3_list,label=lab,color="blue")
    # ax4.plot(dt_list,mom4_list,label=lab,color="blue")

    # # add CI (95%)
    # lab="95% CI"
    # ax1.plot(dt_list,mom_1_plussd,"--",label=lab,color="blue")
    # ax2.plot(dt_list,mom_2_plussd,"--",label=lab,color="blue")
    # ax3.plot(dt_list,mom_3_plussd,"--",label=lab,color="blue")
    # ax4.plot(dt_list,mom_4_plussd,"--",label=lab,color="blue")

    ax1.legend()
    fig.savefig('accuracy_openmp/saved_figures/'+pot+'/moments_plot_'+str(pot)+'.png')


    # ###########################################
    # ## Compute the log moments
    # ###########################################
    fig, (ax1,ax2,ax3,ax4)= plt.subplots(1,4,figsize=(20,7))# plt.figure(figsize=(4,4))
    ax1.set_title("First moment")
    ax2.set_title("Second moment")
    ax3.set_title("Third moment")
    ax4.set_title("Fourth moment")

    lab="no ada"
    mom1_list,mom2_list,mom3_list,mom4_list,mom_1_plussd,mom_2_plussd,mom_3_plussd,mom_4_plussd=moment_list(dt_list,tau,dta_noada,range_int)
    ax1.loglog(dt_list,mom1_list,label=lab,color="red")
    ax2.loglog(dt_list,mom2_list,label=lab,color="red")
    ax3.loglog(dt_list,mom3_list,label=lab,color="red")
    ax4.loglog(dt_list,mom4_list,label=lab,color="red")

    # Add CI
    lab="95% CI"
    ax1.loglog(dt_list,mom_1_plussd,"--",label=lab,color="red")
    ax2.loglog(dt_list,mom_2_plussd,"--",label=lab,color="red")
    ax3.loglog(dt_list,mom_3_plussd,"--",label=lab,color="red")
    ax4.loglog(dt_list,mom_4_plussd,"--",label=lab,color="red")

    lab="transformed"
    mom1_list,mom2_list,mom3_list,mom4_list,mom_1_plussd,mom_2_plussd,mom_3_plussd,mom_4_plussd=moment_list(dt_list,tau,dta_transfo,range_int)
    ax1.loglog(dt_list,mom1_list,label=lab,color="green")
    ax2.loglog(dt_list,mom2_list,label=lab,color="green")
    ax3.loglog(dt_list,mom3_list,label=lab,color="green")
    ax4.loglog(dt_list,mom4_list,label=lab,color="green")

    # Add CI
    lab="95% CI"
    ax1.loglog(dt_list,mom_1_plussd,"--",label=lab,color="green")
    ax2.loglog(dt_list,mom_2_plussd,"--",label=lab,color="green")
    ax3.loglog(dt_list,mom_3_plussd,"--",label=lab,color="green")
    ax4.loglog(dt_list,mom_4_plussd,"--",label=lab,color="green")

    ## Do not consider rescaled
    # lab="rescaled"
    # mom1_list,mom2_list,mom3_list,mom4_list,mom_1_plussd,mom_2_plussd,mom_3_plussd,mom_4_plussd=moment_list(dt_list,tau,dta_rescale,range_int)
    # ax1.loglog(dt_list,mom1_list,label=lab,color="blue")
    # ax2.loglog(dt_list,mom2_list,label=lab,color="blue")
    # ax3.loglog(dt_list,mom3_list,label=lab,color="blue")
    # ax4.loglog(dt_list,mom4_list,label=lab,color="blue")

    # # Add CI
    # lab="95% CI"
    # ax1.loglog(dt_list,mom_1_plussd,"--",label=lab,color="blue")
    # ax2.loglog(dt_list,mom_2_plussd,"--",label=lab,color="blue")
    # ax3.loglog(dt_list,mom_3_plussd,"--",label=lab,color="blue")
    # ax4.loglog(dt_list,mom_4_plussd,"--",label=lab,color="blue")

    ax1.legend()
    fig.savefig('accuracy_openmp/saved_figures/'+pot+'/moments_logplot_'+str(pot)+'.png')


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
    accuracy_list =error_list(nbins,dt_list,range_bins,tau,dta_transfo)
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
    fig.savefig('accuracy_openmp/saved_figures/'+pot+'/accuracy_plot_'+str(pot)+'.png')


if test==1:
    # save the values of the function g
    list_param = str(pot)+'-tau='+str(tau)+'-M='+str(M)+'m='+str(m)+"-T="+str(T)+"-ns="+str(n_samples)
    dta_g_transfo=pd.read_pickle("accuracy_openmp/data/dta_g_tr_"+list_param)
    dta_g_rescale=pd.read_pickle("accuracy_openmp/data/dta_g_re_"+list_param)

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
        axs[j].bar(midx,histogram,width=(upperbound-lowerbound)/nbins,color="green",label="Transformed",alpha=0.5)

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
        axs[j].bar(midx,histogram,width=(upperbound-lowerbound)/nbins,color="blue",label="Rescale",alpha=0.5)

        ## labels histograms
        axs[j].set_xlabel("$g(x)$")
        axs[j].legend()
        txt = "Tr="+str(round(mean_tr,3))+" \nRe="+str(round(mean_re,3))
        axs[j].set_title(txt)

    fig.savefig('accuracy_openmp/saved_figures/'+pot+'/mean_g_plot_'+list_param+'.png')
