##### Build a main to compute accuracy plot with different schemes and different examples problem
## How to use those files 
# 1) Set up in settings.py the values of M and m and choose the potential to use "bond", "square", "power4", "spring"
 
import matplotlib.pyplot as plt
import sys
sys.path.append("Python/accuracy/accuracy_burnin")

## Import the package to run the samples 
from settings import *
from useful_tools import *
from sample_over_noada import sample_noada

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
run_for_h_var=0
run_errors_moments=1
#######################################################
##### Set up important settings
############################################### ########
    
## SQUARE POTENTIAL RESULTS
# ## for precise results on accuracy curves 
dt_list= [2 ** (R-10) for R in range(7)]
T=20
tau=1
n_samples=50000
x=5
range_bins=[-10,10] # range to plot samples on histograms
nbins=50
range_int=[-10,10] # integration range 


##########################################
## Obtain the samples for different dt
###########################################
if run_for_h_var==1:

    ####################################################
    ## Obtain order of a list of sample for different dt
    ####################################################

    # List of datasets to save the information on the sample values
    # dta_noadaX1  = pd.DataFrame()
    # dta_noadaX2 = pd.DataFrame()
    # dta_noadaX3 = pd.DataFrame()
    # dta_noadaX4 = pd.DataFrame()

    error1=[]
    error2=[]
    error3=[]
    error4=[]

    for j in range(len(dt_list)):
        h=dt_list[j]
        Nt=int(T*1/h)+1 #set up the number of steps we will take

        ## compute the non adaptive value
        resx =sample_noada(x,n_samples,Nt,h,tau) 
        error1.append(np.sum(resx[:,1])/n_samples)
        error2.append(np.sum(resx[:,2])/n_samples)
        error3.append(np.sum(resx[:,3])/n_samples)
        error4.append(np.sum(resx[:,4])/n_samples)

    print(error1)
    print(error2)
    print(error3)
    print(error4)

    ## When no access to the true moment
    a=range_int[0]
    b=range_int[1]
    norm=np.round(integrate.quad(lambda q: np.exp(-U(q)/tau), a,b)[0],16)
    true_mom_1 = np.round(integrate.quad(lambda q: np.exp(-U(q)/tau)*(q), a,b)[0],16)/norm
    true_mom_2 = np.round(integrate.quad(lambda q: np.exp(-U(q)/tau)*q*q, a,b)[0],16)/norm
    true_mom_3 = np.round(integrate.quad(lambda q: np.exp(-U(q)/tau)*(q)*q*q, a,b)[0],16)/norm
    true_mom_4 = np.round(integrate.quad(lambda q: np.exp(-U(q)/tau)*q*q*q*q, a,b)[0],16)/norm

    error1=np.array(error1)
    error2=np.array(error2)
    error3=np.array(error3)
    error4=np.array(error4)

    error1=np.abs((error1-true_mom_1))
    error2=np.abs((error2-true_mom_2)/true_mom_2)
    error3=np.abs((error3-true_mom_3))
    error4=np.abs((error4-true_mom_4)/true_mom_4)

    print(error1)
    print(error2)
    print(error3)
    print(error4)

    np.savetxt("Python/accuracy/accuracy_burnin/data/error1.out",error1)
    np.savetxt("Python/accuracy/accuracy_burnin/data/error2.out",error2)
    np.savetxt("Python/accuracy/accuracy_burnin/data/error3.out",error3)
    np.savetxt("Python/accuracy/accuracy_burnin/data/error4.out",error4)
    

        # dta_noadaX1["x"+str(j)] = sumXs[0]
        # dta_noadaX2["x"+str(j)] = sumXs[1]   
        # dta_noadaX3["x"+str(j)] = sumXs[2]   
        # dta_noadaX4["x"+str(j)] = sumXs[3]

    # # save the values of x
    # list_param = str(pot)+'X1-tau='+str(tau)+'-M='+str(M)+'m='+str(m)+"-T="+str(T)+"-ns="+str(n_samples)
    # dta_noadaX1.to_pickle("Python/accuracy/accuracy_burnin/data/dta_noada_"+list_param)
    # list_param = str(pot)+'X2-tau='+str(tau)+'-M='+str(M)+'m='+str(m)+"-T="+str(T)+"-ns="+str(n_samples)
    # dta_noadaX2.to_pickle("Python/accuracy/accuracy_burnin/data/dta_noada_"+list_param)
    # list_param = str(pot)+'X3-tau='+str(tau)+'-M='+str(M)+'m='+str(m)+"-T="+str(T)+"-ns="+str(n_samples)
    # dta_noadaX3.to_pickle("Python/accuracy/accuracy_burnin/data/dta_noada_"+list_param)
    # list_param = str(pot)+'X4-tau='+str(tau)+'-M='+str(M)+'m='+str(m)+"-T="+str(T)+"-ns="+str(n_samples)
    # dta_noadaX4.to_pickle("Python/accuracy/accuracy_burnin/data/dta_noada_"+list_param)
    

# print(error1)

# ###########################################
# ## Open the samples previously saved
# ###########################################
if run_errors_moments==1:


    # ###########################################
    # ## Compute the moments
    # ###########################################
    fig, (ax1,ax2,ax3,ax4)= plt.subplots(1,4,figsize=(20,7))# plt.figure(figsize=(4,4))
    ax1.set_title("First moment")
    ax2.set_title("Second moment")
    ax3.set_title("Third moment")
    ax4.set_title("Fourth moment")

    lab="no ada"
    ax1.plot(dt_list,error1,label=lab,color="red")
    ax2.plot(dt_list,error2,label=lab,color="red")
    ax3.plot(dt_list,error3,label=lab,color="red")
    ax4.plot(dt_list,error4,label=lab,color="red")

    ax1.legend()
    fig.savefig('Python/accuracy/accuracy_burnin/fig/'+pot+'/moments_plot_'+str(pot)+'.png')


    # ###########################################
    # ## Compute the log moments
    # ###########################################
    fig, (ax1,ax2,ax3,ax4)= plt.subplots(1,4,figsize=(20,7))# plt.figure(figsize=(4,4))
    ax1.set_title("First moment")
    ax2.set_title("Second moment")
    ax3.set_title("Third moment")
    ax4.set_title("Fourth moment")

    lab="no ada"
    ax1.loglog(dt_list,error1,label=lab,color="red")
    ax2.loglog(dt_list,error2,label=lab,color="red")
    ax3.loglog(dt_list,error3,label=lab,color="red")
    ax4.loglog(dt_list,error4,label=lab,color="red")
    

    ## No ada 
    # accuracy_list =error_list(nbins,dt_list,range_bins,tau,dta_noada)
    x,y_x,a_round=get_slope(error1,dt_list)
    ax1.loglog(x,y_x,"x",label="slope="+str(a_round))
    x,y_x,a_round=get_slope(error2,dt_list)
    ax2.loglog(x,y_x,"x",label="slope="+str(a_round))
    x,y_x,a_round=get_slope(error3,dt_list)
    ax3.loglog(x,y_x,"x",label="slope="+str(a_round))
    x,y_x,a_round=get_slope(error4,dt_list)
    ax4.loglog(x,y_x,"x",label="slope="+str(a_round))

    ax1.legend()
    ax2.legend()
    ax3.legend()
    ax4.legend()

    fig.savefig('Python/accuracy/accuracy_burnin/fig/'+str(pot)+'/moments_logplot_'+str(pot)+'.png')


    # # ###########################################
    # # ## Compute the accuracy curves
    # # ###########################################


    # fig, (ax1)= plt.subplots(1,1,figsize=(10,7))# plt.figure(figsize=(4,4))
    # fig.subplots_adjust(left=0.1,bottom=0.1, right=0.9, top=0.9, wspace=0.4, hspace=0.4)

    # ## No ada 
    # accuracy_list =error_list(nbins,dt_list,range_bins,tau,dta_noada)
    # x,y_x,a_round=get_slope(accuracy_list,dt_list)
    # ax1.plot(np.log(dt_list),np.log(accuracy_list),label="No ada")
    # ax1.plot(x,y_x,"--",label="slope="+str(a_round))

    # ## Tr
    # accuracy_list =error_list(nbins,dt_list,range_bins,tau,dta_transfo)
    # x,y_x,a_round=get_slope(accuracy_list,dt_list)
    # ax1.plot(np.log(dt_list),np.log(accuracy_list),label="Transfo")
    # ax1.plot(x,y_x,"--",label="slope="+str(a_round))

    # # ## Re
    # # accuracy_list =error_list(nbins,dt_list,range_bins,tau,dta_rescale)
    # # x,y_x,a_round=get_slope(accuracy_list,dt_list)
    # # ax1.plot(np.log(dt_list),np.log(accuracy_list),label="Rescale")
    # # ax1.plot(x,y_x,"--",label="slope="+str(a_round))

    # #
    # ax1.set_ylabel("$\log(\epsilon)$")
    # ax1.set_xlabel("$\log(h)$")
    # ax1.set_title("Accuracy on EM with potential "+pot)
    # ax1.legend()
    # fig.savefig('Python/accuracy/accuracy_1dim/saved_figures/'+pot+'/accuracy_plot_'+str(pot)+'.png')

