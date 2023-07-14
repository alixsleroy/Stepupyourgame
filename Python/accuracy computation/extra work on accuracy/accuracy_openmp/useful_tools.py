import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
plt.rc('xtick', labelsize=13) 
plt.rc('ytick', labelsize=13) 
from settings import *
from pot_defintion_openmp import *
import scipy.integrate as integrate

def dta_format_under(mat):
    dta = pd.DataFrame(mat,columns=["sim","t","q","p","g"])
    return dta

def dta_format_over(mat):
    dta = pd.DataFrame(mat,columns=["sim","count","t","x","g","gp"])
    return dta


def plot_one_distr_over(df_noada,df_tr,df_re,tau,range_bins,h,T):

    ## Set up the plots 
    Nt=int(T*1/h)+1

    fig, ((axs))= plt.subplots(1,2,figsize=(20,5))# plt.figure(figsize=(4,4))
    fig.subplots_adjust(left=0.1,bottom=0.1, right=0.9, top=0.9, wspace=0.4, hspace=0.4)
    txt = 'M={:.2f}, m={:.2f}, a={:d}, b={:.2f}, c={:.2f}, $x_0$={:.2f}, $N_t$={:.2f}, $\\tau$={:.2f}, h={:.3f}'.format(round(M,2),round(m,2),a,b,round(c,2),round(x0,2),Nt, tau, h)
    fig.suptitle(txt, fontsize=15)

    # fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    ## Loop through the values 
    
    # List of time
    counti = df_noada['count'].max()
    # * Dataframe using hessian
    df_noada_i=df_noada[df_noada['count']==counti]
    # * Dataframe using first derivative 
    df_tr_i=df_tr[df_tr['count']==counti]
    # * Dataframe using no adaptivity 
    df_re_i=df_re[df_re['count']==counti]

    nbins=300

    # SDE
    histogram_sde,bins = np.histogram(df_noada_i["x"],bins=nbins,range=range_bins, density=True)
    midx_sde = (bins[0:-1]+bins[1:])/2

    # Invariant distribution
    rho = np.exp(- U(midx_sde)/tau)
    rho = rho / (np.sum(rho)* (midx_sde[1]-midx_sde[0]) ) # Normalize rho by dividing by its approx. integral

    # Transformed
    histogram_tr,bins = np.histogram(df_tr_i["x"],bins=nbins,range=range_bins, density=True)
    midx_tr = (bins[0:-1]+bins[1:])/2

    # Rescaled
    histogram_re,bins = np.histogram(df_re_i["x"],bins=nbins,range=range_bins, density=True)
    midx_re = (bins[0:-1]+bins[1:])/2
    
    # Plots 
    axs[0].plot(midx_sde,rho,linewidth=2.5,label='Invariant distribution $\\rho(x,\\infty)$',color="orange")
    axs[0].plot(midx_sde,histogram_sde,"--",linewidth=2.5,label='Overdamped SDE',color="red")
    axs[0].plot(midx_re,histogram_re,"--",linewidth=2.5,label='Overdamped SDE\nwith naive time rescaling',color="blue")
    axs[0].plot(midx_tr,histogram_tr,"--",linewidth=2.5,label='Transformed\noverdamped SDE',color="green")

    # Legend 
    axs[0].legend(loc='upper left',fancybox=True,shadow=True,fontsize=12)
    axs[0].set_xlabel("x",fontsize=15)


    ### plot the step size 

    ## Transformed 
    var =df_tr['g']
    mean_tr=np.mean(var)
    lowerbound=np.min(var)
    upperbound=np.max(var)
    nbins=100
    histogram,bins = np.histogram(var,bins=nbins,range=[lowerbound,upperbound], density=True)
    midx = (bins[0:-1]+bins[1:])/2
    axs[1].bar(midx,histogram,width=(upperbound-lowerbound)/nbins,color="green",label="Transformed",alpha=0.5)

    ## Rescaled 
    var =df_re['g']
    mean_re=np.mean(var)
    lowerbound=np.min(var)
    upperbound=np.max(var)
    nbins=100
    histogram,bins = np.histogram(var,bins=nbins,range=[lowerbound,upperbound], density=True)
    midx = (bins[0:-1]+bins[1:])/2
    axs[1].bar(midx,histogram,width=(upperbound-lowerbound)/nbins,color="blue",label="Rescaled",alpha=0.5)
    
    # 
    axs[1].set_xlabel("g(x)", fontsize=15)
    # axs[1].legend()
    txt = str(round(mean_tr,3)) #+" Rescaled mean g(x): "+str(round(mean_re,3))
    axs[1].set_title(txt)

    fig.savefig('Python/accuracy/accuracy_1dim/saved_figures/'+pot+'/hist_stepsize'+str(pot)+'tau'+str(tau)+'h'+str(h)+'M='+str(M)+'m'+str(m)+'T'+str(T)+'Nt'+str(Nt)+'.png')

    #plt.show()



def plot_one_distr_under(df_noada,df_transfo,df_rescale,tau):
    plt.close()
    fig, (ax1,ax2)= plt.subplots(1, 2,figsize=(16,6))# plt.figure(figsize=(4,4))
    fig.subplots_adjust(left=0.1,
                            bottom=0.1, 
                            right=0.9, 
                            top=0.9, 
                            wspace=0.4, 
                            hspace=0.4)

    # fig.suptitle("$\\beta$="+str(tau)+", $\\gamma=$"+str(gamma))

    # No ada 
    qf_list,pf_list,gf_list,tf_list = df_noada["q"], df_noada["p"], df_noada["g"], df_noada["t"]
    ## p
    histogram,bins = np.histogram(qf_list,bins=100,range=[0,1.5], density=True)
    midx_q = (bins[0:-1]+bins[1:])/2
    ax1.plot(midx_q,histogram,label='sde',color="red")
    ## q 
    histogram,bins = np.histogram(pf_list,bins=100,range=[-1.5,1.5], density=True)
    midx_p = (bins[0:-1]+bins[1:])/2
    ax2.plot(midx_p,histogram,label='sde',color="red")

    # # Ada transfo 
    # qf_list,pf_list,gf_list,tf_list = df_transfo["q"], df_transfo["p"], df_transfo["g"], df_transfo["t"]
    # ## p
    # histogram,bins = np.histogram(qf_list,bins=100,range=[0,1.5], density=True)
    # midx_q = (bins[0:-1]+bins[1:])/2
    # ax1.plot(midx_q,histogram,label='transformed sde',color="green")
    # ## q 
    # histogram,bins = np.histogram(pf_list,bins=100,range=[-1.5,1.5], density=True)
    # midx_p = (bins[0:-1]+bins[1:])/2
    # ax2.plot(midx_p,histogram,label='transformed sde',color="green")

    # Rescale 
    qf_list,pf_list,gf_list,tf_list = df_rescale["q"], df_rescale["p"], df_rescale["g"], df_rescale["t"]
    ## p
    histogram,bins = np.histogram(qf_list,bins=100,range=[0,1.5], density=True)
    midx_q = (bins[0:-1]+bins[1:])/2
    ax1.plot(midx_q,histogram,label='rescaled',color="blue")
    ## q 
    histogram,bins = np.histogram(pf_list,bins=100,range=[-1.5,1.5], density=True)
    midx_p = (bins[0:-1]+bins[1:])/2
    ax2.plot(midx_p,histogram,label='rescaled',color="blue")

    ### position q invariant
    rho = np.exp(- U(midx_q)/tau)
    rho = rho / ( np.sum(rho)* (midx_q[1]-midx_q[0]) ) # Normalize rho by dividing by its approx. integral
    ax1.plot(midx_q,rho,'--',label='invariant',color="orange")
    #ax1.legend() 

    ### momentum p invariant
    rho = np.exp(-(midx_p**2)/(2*tau))
    rho = rho / ( np.sum(rho)* (midx_p[1]-midx_p[0]) ) # Normalize rho by dividing by its approx. integral
    ax2.plot(midx_p,rho,'--',label='invariant',color="orange")

    ax2.legend(loc='lower center',bbox_to_anchor=(-0.4, -.25),
            ncol=4, fancybox=True, shadow=True)

    fig.show()

    plt.show()


def error_list(nbins,dt_list,range_bins,tau,dta_noada):
    ####################################################
    ## Obtain the error on each bins of the sample
    ####################################################

    error_list=[]

    #compute the integral to normalise
    a=range_bins[0]
    b=range_bins[1]
    Z1 = integrate.quad(lambda q: np.exp(-U(q)/tau), a+0.001,b)[0]


    for j in range(len(dt_list)):
        ## -- no ada
        x = dta_noada["x"+str(j)]
        histogram,bins = np.histogram(x,bins=nbins,range=range_bins, density=True)
        histogram = histogram/np.sum(histogram)

        ## -- 
        error=0
        for i in range(0,len(bins)-1):
            a = bins[i]
            b = bins[i+1]
            Z_bin = integrate.quad(lambda q: np.exp(-U(q)/tau), a,b)[0]/Z1
            error = error+np.abs(histogram[i]-Z_bin)
        error_list.append(error)
    return(error_list)

def moment_list(dt_list,tau,dta_noada,range_int):
    ####################################################
    ## Obtain the error on the moments
    ####################################################

    mom1_list=[]
    mom2_list=[]
    mom3_list=[]
    mom4_list=[]

    mom_1_plussd=[]
    mom_2_plussd=[]
    mom_3_plussd=[]
    mom_4_plussd=[]


    ## When no access to the true moment
    a=range_int[0]
    b=range_int[1]
    norm=np.round(integrate.quad(lambda q: np.exp(-U(q)/tau), a,b)[0],16)
    true_mom_1 = np.round(integrate.quad(lambda q: np.exp(-U(q)/tau)*np.abs(q), a,b)[0],16)/norm
    true_mom_2 = np.round(integrate.quad(lambda q: np.exp(-U(q)/tau)*q*q, a,b)[0],16)/norm
    true_mom_3 = np.round(integrate.quad(lambda q: np.exp(-U(q)/tau)*np.abs(q)**3, a,b)[0],16)/norm
    true_mom_4 = np.round(integrate.quad(lambda q: np.exp(-U(q)/tau)*q*q*q*q, a,b)[0],16)/norm

    for j in range(len(dt_list)):

        x = dta_noada["x"+str(j)]

        # compute first moment
        mom_1 =np.sum(np.abs(x))/len(x)
        mom_1=np.abs(mom_1-true_mom_1)/mom_1
        mom_sd_1=np.std(np.abs(x))/np.sqrt(len(x))*1.96 #(Z for alpha 0.05)
        mom1_list.append(mom_1)
        mom_1_plussd.append(mom_1+mom_sd_1)

        #compute second moment
        mom_2 = np.sum(np.power(np.abs(x),2))/len(x)
        mom_2 = np.abs(mom_2-true_mom_2)/mom_2
        mom_sd_2=np.std(np.power(np.abs(x),2))/np.sqrt(len(x))*1.96 #(Z for alpha 0.05)
        mom2_list.append(mom_2)
        mom_2_plussd.append(mom_2+mom_sd_2)

        # compute third moment
        mom_3 = np.sum(np.power(np.abs(x),3))/len(x)
        mom_3=np.abs(mom_3-true_mom_3)/mom_3
        mom_sd_3=np.std(np.power(np.abs(x),3))/np.sqrt(len(x))*1.96 #(Z for alpha 0.05)
        mom3_list.append(mom_3)
        mom_3_plussd.append(mom_3+mom_sd_3)

        # compute fourth moment 
        mom_4 = np.sum(np.power(np.abs(x),4))/len(x)
        mom_4=np.abs(mom_4-true_mom_4)/mom_4
        mom_sd_4=np.std(np.power(np.abs(x),4))/np.sqrt(len(x))*1.96 #(Z for alpha 0.05)
        mom4_list.append(mom_4)
        mom_4_plussd.append(mom_4+mom_sd_4)

    return(mom1_list,mom2_list,mom3_list,mom4_list,mom_1_plussd,mom_2_plussd,mom_3_plussd,mom_4_plussd)


def get_slope(accuracy_list,dt_list):
    #######################################
    ## Obtain the slope of the error decay
    #######################################
    logx1=np.log(dt_list[0])
    logx2=np.log(dt_list[-1])
    logy1=np.log(accuracy_list[0])
    logy2=np.log(accuracy_list[-1])
    a=(logy1-logy2)/(logx1-logx2)
    b=logy1-a*logx1
    x=np.linspace(logx1,logx2,1000)
    y_x=a*x+b
    a_round=np.round(np.abs(a),2)
    return(x,y_x,a_round)


# def mean_step_size(dta_g,dt_list): 
#     # loop through the list of step size to get 
#     # each mean value of steps 
#     list_mean_steps=[]
#     for j in range(len(dt_list)):
#         g_j = dta_g["g"+str(j)]
#         list_mean_steps.append(np.round(g_j,4))

        

