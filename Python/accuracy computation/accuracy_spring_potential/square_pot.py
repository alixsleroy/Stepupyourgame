##### Build a main to compute accuracy plot with different schemes and different examples problem
## How to use those files 
# 1) Set up in settings.py the values of M and m and choose the potential to use "bond", "square", "power4", "spring"
 
import matplotlib.pyplot as plt
import sys
sys.path.append("Python/accuracy/accuracy_1dim")

## Import the package to run the samples 
from settings import *
from useful_tools import *
from accuracy_over_func import *

global T
global gamma
global tau
global h
global n_samples
global Nt

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
############################################### ########
    

T=5
tau=0.1
n_samples=20000
h=0.05
Nt= 10000 #int(T*1/h)+1
x=0
range_bins=[-1,1] # range to plot samples on histograms
nbins=20
range_int=[-10,10] # integration range 

# # ## for precise results on accuracy curves 
# dt_list= [0.01,0.02,0.03,0.04,0.05,0.06,0.07,0.08,0.09,0.1,0.2,0.3,0.4,0.5,0.6]
# ## set up others values
# T=100
# tau=0.1
# n_samples=1000
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
    dta_noada=pd.read_pickle("Python/accuracy/accuracy_1dim/saved_pickles_over/plot_hist/dta_noada_"+list_param)



###########################################
## Obtain the samples for different dt
###########################################
if run_for_h_var==1:
        ####################################################
    ## Obtain order of a list of sample for different dt
    ####################################################

    # List of datasets to save the information on the sample values
    dta_noada = pd.DataFrame()
    for j in range(len(dt_list)):
        h=dt_list[j]
        Nt=int(T*1/h)+1 #set up the number of steps we will take

        ## compute the non adaptive value
        try1 =sample_noada(x,n_samples,Nt,h,tau) 
        dt = dta_format_over(try1)
        dta_noada["x"+str(j)] = dt["x"]
        


    # save the values of x
    list_param = str(pot)+'-tau='+str(tau)+'-M='+str(M)+'m='+str(m)+"-T="+str(T)+"-ns="+str(n_samples)
    dta_noada.to_pickle("Python/accuracy/accuracy_1dim/saved_pickles_over/dta_noada_"+list_param)


# ###########################################
# ## Open the samples previously saved
# ###########################################
if run_errors_moments==1:

    # List of parameters 
    list_param = str(pot)+'-tau='+str(tau)+'-M='+str(M)+'m='+str(m)+"-T="+str(T)+"-ns="+str(n_samples)

    dta_noada=pd.read_pickle("Python/accuracy/accuracy_1dim/saved_pickles_over/dta_noada_"+list_param)
    # dta_transfo=pd.read_pickle("Python/accuracy/accuracy_1dim/saved_pickles_over/dta_tr_"+list_param)
    # does not consider the rescale for the accuracy computation
    # dta_rescale=pd.read_pickle("Python/accuracy/accuracy_1dim/saved_pickles_over/dta_re_"+list_param)


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

    # lab="transformed"
    # mom1_list,mom2_list,mom3_list,mom4_list,mom_1_plussd,mom_2_plussd,mom_3_plussd,mom_4_plussd=moment_list(dt_list,tau,dta_transfo,range_int)
    # ax1.plot(dt_list,mom1_list,label=lab,color="green")
    # ax2.plot(dt_list,mom2_list,label=lab,color="green")
    # ax3.plot(dt_list,mom3_list,label=lab,color="green")
    # ax4.plot(dt_list,mom4_list,label=lab,color="green")

    # # add CI (95%)
    # lab="95% CI"
    # ax1.plot(dt_list,mom_1_plussd,"--",label=lab,color="green")
    # ax2.plot(dt_list,mom_2_plussd,"--",label=lab,color="green")
    # ax3.plot(dt_list,mom_3_plussd,"--",label=lab,color="green")
    # ax4.plot(dt_list,mom_4_plussd,"--",label=lab,color="green")

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
    fig.savefig('Python/accuracy/accuracy_1dim/saved_figures/'+pot+'/moments_plot_'+str(pot)+'.png')


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
    
    ## No ada 
    # accuracy_list =error_list(nbins,dt_list,range_bins,tau,dta_noada)
    x,y_x,a_round=get_slope(mom1_list,dt_list)
    ax1.plot(np.log(dt_list),np.log(mom1_list),label="No ada")
    ax1.plot(x,y_x,"--",label="slope="+str(a_round))


    # Add CI
    lab="95% CI"
    ax1.loglog(dt_list,mom_1_plussd,"--",label=lab,color="red")
    ax2.loglog(dt_list,mom_2_plussd,"--",label=lab,color="red")
    ax3.loglog(dt_list,mom_3_plussd,"--",label=lab,color="red")
    ax4.loglog(dt_list,mom_4_plussd,"--",label=lab,color="red")

    # lab="transformed"
    # mom1_list,mom2_list,mom3_list,mom4_list,mom_1_plussd,mom_2_plussd,mom_3_plussd,mom_4_plussd=moment_list(dt_list,tau,dta_transfo,range_int)
    # ax1.loglog(dt_list,mom1_list,label=lab,color="green")
    # ax2.loglog(dt_list,mom2_list,label=lab,color="green")
    # ax3.loglog(dt_list,mom3_list,label=lab,color="green")
    # ax4.loglog(dt_list,mom4_list,label=lab,color="green")

    # # Add CI
    # lab="95% CI"
    # ax1.loglog(dt_list,mom_1_plussd,"--",label=lab,color="green")
    # ax2.loglog(dt_list,mom_2_plussd,"--",label=lab,color="green")
    # ax3.loglog(dt_list,mom_3_plussd,"--",label=lab,color="green")
    # ax4.loglog(dt_list,mom_4_plussd,"--",label=lab,color="green")

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
    fig.savefig('Python/accuracy/accuracy_1dim/saved_figures/'+pot+'/moments_logplot_'+str(pot)+'.png')


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

    # ## Tr
    # accuracy_list =error_list(nbins,dt_list,range_bins,tau,dta_transfo)
    # x,y_x,a_round=get_slope(accuracy_list,dt_list)
    # ax1.plot(np.log(dt_list),np.log(accuracy_list),label="Transfo")
    # ax1.plot(x,y_x,"--",label="slope="+str(a_round))

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
    fig.savefig('Python/accuracy/accuracy_1dim/saved_figures/'+pot+'/accuracy_plot_'+str(pot)+'.png')


if test==1:
    # save the values of the function g
    list_param = str(pot)+'-tau='+str(tau)+'-M='+str(M)+'m='+str(m)+"-T="+str(T)+"-ns="+str(n_samples)
    dta_g_transfo=pd.read_pickle("Python/accuracy/accuracy_1dim/saved_pickles_over/dta_g_tr_"+list_param)
    dta_g_rescale=pd.read_pickle("Python/accuracy/accuracy_1dim/saved_pickles_over/dta_g_re_"+list_param)

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

    fig.savefig('Python/accuracy/accuracy_1dim/saved_figures/'+pot+'/mean_g_plot_'+list_param+'.png')
