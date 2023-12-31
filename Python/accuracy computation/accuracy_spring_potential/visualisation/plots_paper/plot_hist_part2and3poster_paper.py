import pandas as pd
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as cl
from settings import *
from pot_definition import *
plt.rc('xtick', labelsize=13) 
plt.rc('ytick', labelsize=13) 

import matplotlib.pyplot as plt
import sys
sys.path.append("Python/accuracy/accuracy_1dim")




###################

### Plot parameters: 
font_size = 15
line_w=3
b=1
a=5
x0=0.5
c=0.1

# myblue = "#0077BB"
# mycyan = "#33BBEE"
# mygreen = "#009988"
# myorange="#EE7733"
# myred="#CC3311"
# mymagenta="#EE3377"
# mygrey="#BBBBBB"

# myblue = "#4477AA"
# mycyan = "#66CCEE"
# mygreen = "#228833"
# myorange="#CCBB44"
# myred="#EE6677"
# mymagenta="#AA3377"
# mygrey="#BBBBBB"

# myblue = (0,119/235,187/235)
# myred=(187/235,85/235,102/235)
# myyellow=(221/235,170/235,51/235)
# mygrey=(187/235,187/235,187/235)
# mygreen="#66BB55"
# mymagenta="#7733DD"

myblue = (0,119/235,187/235)
myred=(187/235,85/235,102/235)
myyellow=(221/235,170/235,51/235)
mygrey=(187/235,187/235,187/235)
mygreen="#66BB55"
mymagenta="#7733DD"


plt.rc('xtick', labelsize=font_size) 
plt.rc('ytick', labelsize=font_size) 

def plot_one_distr_over(df_noada,df_tr,df_re,tau,range_bins,h,T):

    ## Set up the plots 
    Nt=int(T*1/h)+1

    fig, ((axs))= plt.subplots(1,1,figsize=(20,8))# plt.figure(figsize=(4,4))
    fig.subplots_adjust(left=0.1,bottom=0.1, right=0.9, top=0.9, wspace=0.4, hspace=0.4)
    # txt = 'M={:.2f}, m={:.2f}, a={:d}, b={:.2f}, c={:.2f}, $x_0$={:.2f}, $N_t$={:.2f}, $\\tau$={:.2f}, h={:.3f}'.format(round(M,2),round(m,2),a,b,round(c,2),round(x0,2),Nt, tau, h)
    # fig.suptitle(txt, fontsize=font_size)

    # List of time
    counti = df_noada['count'].max()
    # * Dataframe using hessian
    df_noada_i=df_noada[df_noada['count']==counti]
    # * Dataframe using first derivative 
    df_tr_i=df_tr[df_tr['count']==counti]
    # * Dataframe using no adaptivity 
    df_re_i=df_re[df_re['count']==counti]

    nbins=100

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
    axs.plot(midx_sde,rho,linewidth=line_w*2,label='Invariant distribution $\\rho_\infty(x)$',color=mygrey)
    # axs.plot(midx_re,histogram_re,"-.",linewidth=line_w*1.3,label='Naive time rescaled\noverdamped SDE',color=mygreen)
    axs.plot(midx_tr,histogram_tr,"-.",linewidth=line_w*1.3,label='Sampling the\ntransformed SDE',color=myblue)
    axs.plot(midx_sde,histogram_sde,":",linewidth=line_w*1.7,label='Sampling the SDE',color=myred)

    # Legend 
    axs.legend(loc='upper left',fancybox=True,shadow=True,fontsize=font_size)
    axs.set_xlabel("x",fontsize=font_size)
    plt.show()

###########################################################################################
####################### PLOT NAIVE TIME RESCALING ########################################
###########################################################################################
## set up others values 
pot="spring"


if True==False:
    h=0.001
    T=50
    Nt=int(T*1/h)+1
    M=2
    m=0.001
    tau=0.1
    n_samples=100000
    range_bins=[-4,2]
    path="Stepupyourgame/data/python/accuracy_spring_potential_overdamped/"
    ## Parameters 
    list_param = str(pot)+'-tau='+str(tau)+'-M='+str(M)+'m='+str(m)+"-Nt="+str(Nt)+"-ns="+str(n_samples)
    df_noada=pd.read_pickle("Stepupyourgame/data/python/accuracy_spring_potential_overdamped/saved_pickles_over/plot_hist/dta_noada_"+list_param)
    df_tr=pd.read_pickle("Stepupyourgame/data/python/accuracy_spring_potential_overdamped/saved_pickles_over/plot_hist/dta_tr_"+list_param)
    df_re=pd.read_pickle("Stepupyourgame/data/python/accuracy_spring_potential_overdamped/saved_pickles_over/plot_hist/dta_re_"+list_param)

    ## Set up the plots 

    fig, ((axs))= plt.subplots(1,1,figsize=(7,6))# plt.figure(figsize=(4,4))
    fig.subplots_adjust(left=0.1,bottom=0.1, right=0.9, top=0.9, wspace=0.4, hspace=0.4)
    # txt = 'M={:.2f}, m={:.2f}, a={:d}, b={:.2f}, c={:.2f}, $x_0$={:.2f}, $N_t$={:.2f}, $\\tau$={:.2f}, h={:.3f}'.format(round(M,2),round(m,2),a,b,round(c,2),round(x0,2),Nt, tau, h)
    # fig.suptitle(txt, fontsize=font_size)

    # List of time
    counti = df_noada['count'].max()
    # * Dataframe using hessian
    df_noada_i=df_noada[df_noada['count']==counti]
    # * Dataframe using first derivative 
    df_tr_i=df_tr[df_tr['count']==counti]
    # * Dataframe using no adaptivity 
    df_re_i=df_re[df_re['count']==counti]

    nbins=150

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
    axs.plot(midx_sde,rho,linewidth=line_w*2,label='Invariant distribution $\\rho_\infty(x)$',color=mygrey)
    axs.plot(midx_re,histogram_re,"--",linewidth=line_w*1.3,label='Sampling the naive\ntime rescaled SDE',color=mygreen)
    # axs.plot(midx_tr,histogram_tr,"--",linewidth=line_w*1.3,label='Sampling the\ntransformed SDE',color=myblue)
    axs.plot(midx_sde,histogram_sde,":",linewidth=line_w*1.7,label='Sampling the SDE',color=myred)

    # Legend 
    axs.legend(loc='upper left',fancybox=True,shadow=True,fontsize=font_size)
    axs.set_xlabel("x",fontsize=font_size)
    plt.show()


###########################################################################################
####################### PLOT PART 3 TRANSFORMED ########################################
###########################################################################################

if True==False:
    # line_w=5

    nbins=100
    ## Set up the plots 
    fig, ((axs))= plt.subplots(1,1,figsize=(7,6))# plt.figure(figsize=(4,4))
    fig.subplots_adjust(left=0.1,bottom=0.1, right=0.9, top=0.9, wspace=0.4, hspace=0.4)

    ######################### SMALL H ###########################
    #############################################################

    ## SMALL H parameters
    T=50
    tau=0.1
    n_samples=100000
    h=0.001
    Nt= int(T*1/h)+1
    x=0
    M=2
    m=0.001
    range_bins=[-4,2]
    nbins=100
    range_int=[-5,4]
    path="Stepupyourgame/data/python/accuracy_spring_potential_overdamped/"

    list_param = str(pot)+'-tau='+str(tau)+'-M='+str(M)+'m='+str(m)+"-Nt="+str(Nt)+"-ns="+str(n_samples) #+"-h="+str(h)
    df_noada=pd.read_pickle(path+"/saved_pickles_over/plot_hist/dta_noada_"+list_param)
    df_tr=pd.read_pickle(path+"/saved_pickles_over/plot_hist/dta_tr_"+list_param)
    df_re=pd.read_pickle(path+"/saved_pickles_over/plot_hist/dta_re_"+list_param)


    # List of time
    counti = df_noada['count'].max()
    # * Dataframe using hessian
    df_noada_i=df_noada[df_noada['count']==counti]
    # * Dataframe using first derivative 
    df_tr_i=df_tr[df_tr['count']==counti]
    # * Dataframe using no adaptivity 
    df_re_i=df_re[df_re['count']==counti]


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
    axs.plot(midx_sde,rho,linewidth=line_w*2,label='Invariant distribution $\\rho_\infty(x)$',color=mygrey)
    # axs.plot(midx_re,histogram_re,"-.",linewidth=line_w*1.3,label='Naive time rescaled\noverdamped SDE',color=mygreen)
    axs.plot(midx_tr,histogram_tr,"--",linewidth=line_w*1.3,label='Sampling the\ntransformed SDE',color=myblue)
    axs.plot(midx_sde,histogram_sde,":",linewidth=line_w*1.7,label='Sampling the SDE',color=myred)
    axs.set_xlabel("x",fontsize=font_size)

    # Legend 
    axs.legend(loc='upper left',ncols=1,fancybox=True,shadow=True,fontsize=font_size*0.9) #bbox_to_anchor=(-0.9, 1.225)
    axs.set_xlabel("x",fontsize=font_size)

    plt.show()


if True==False:

    ## Set up the plots 
    fig, ((axs))= plt.subplots(1,1,figsize=(7,6))# plt.figure(figsize=(4,4))
    fig.subplots_adjust(left=0.1,bottom=0.1, right=0.9, top=0.9, wspace=0.4, hspace=0.4)

    ######################### LARGE H ###########################
    #############################################################

    # LARGE H parameters
    T=70
    tau=0.1
    n_samples=500000
    h=0.05
    Nt= int(T*1/h)+1
    m=0.001
    M=1.5
    range_bins=[-4,2]
    nbins=150
    range_int=[-5,4]
    path="Stepupyourgame/data/python/accuracy_spring_potential_overdamped/"

    list_param = str(pot)+'-tau='+str(tau)+'-M='+str(M)+'m='+str(m)+"-Nt="+str(Nt)+"-ns="+str(n_samples)+"-h="+str(h)
    df_noada_largeh=pd.read_pickle(path+"/saved_pickles_over/plot_hist/dta_noada_"+list_param)
    df_tr_largeh=pd.read_pickle(path+"/saved_pickles_over/plot_hist/dta_tr_"+list_param)
    df_re_largeh=pd.read_pickle(path+"/saved_pickles_over/plot_hist/dta_re_"+list_param)

    # List of time
    counti = df_noada_largeh['count'].max()
    # * Dataframe using hessian
    df_noada_i=df_noada_largeh[df_noada_largeh['count']==counti]
    # * Dataframe using first derivative 
    df_tr_i=df_tr_largeh[df_tr_largeh['count']==counti]
    # * Dataframe using no adaptivity 
    df_re_i=df_re_largeh[df_re_largeh['count']==counti]

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
    axs.plot(midx_sde,rho,linewidth=line_w*2,label='Invariant distribution\n$\\rho_\infty(x)$',color=mygrey)
    # axs.plot(midx_re,histogram_re,"-.",linewidth=line_w*1.3,label='Naive time rescaled\noverdamped SDE',color=mygreen)
    axs.plot(midx_tr,histogram_tr,"--",linewidth=line_w*1.3,label='Sampling the\ntransformed SDE',color=myblue)
    axs.plot(midx_sde,histogram_sde,":",linewidth=line_w*1.7,label='Sampling the SDE',color=myred)

    # Legend 
    axs.legend(loc='upper left',ncols=1,fancybox=True,shadow=True,fontsize=font_size*0.9) #bbox_to_anchor=(-0.9, 1.225)
    axs.set_xlabel("x",fontsize=font_size)

    plt.show()

###########################################################################################
####################### PLOT PART 3 HISTOGRAM OF G(X) #####################################
###########################################################################################
if True==False:
    ### plot the step size 
    fig, ((axs))= plt.subplots(1,1,figsize=(7,6))# plt.figure(figsize=(4,4))
    fig.subplots_adjust(left=0.1,bottom=0.1, right=0.9, top=0.9, wspace=0.4, hspace=0.4)
   
    # LARGE H
    T=70
    tau=0.1
    n_samples=500000
    h=0.05
    Nt= int(T*1/h)+1
    m=0.001
    M=2
    Nt=int(1/h*T)+1
    path="Stepupyourgame/data/python/accuracy_spring_potential_overdamped/"
    list_param = str(pot)+'-tau='+str(tau)+'-M='+str(M)+'m='+str(m)+"-Nt="+str(Nt)+"-ns="+str(n_samples)+"-h="+str(h)
    df_noada_largeh=pd.read_pickle(path+"/saved_pickles_over/plot_hist/dta_noada_"+list_param)
    df_tr_largeh=pd.read_pickle(path+"/saved_pickles_over/plot_hist/dta_tr_"+list_param)
    df_re_largeh=pd.read_pickle(path+"/saved_pickles_over/plot_hist/dta_re_"+list_param)

    ## Transformed 
    var =df_tr_largeh['g']
    mean_tr=np.mean(var)
    lowerbound=np.min(var)
    upperbound=np.max(var)
    nbins=100
    histogram,bins = np.histogram(var,bins=nbins,range=[lowerbound,upperbound], density=True)
    midx = (bins[0:-1]+bins[1:])/2
    axs.bar(midx,histogram,width=(upperbound-lowerbound)/nbins,color=myblue,alpha=0.5)

    # ## Rescaled 
    # var =df_re['g']
    # mean_re=np.mean(var)
    # lowerbound=np.min(var)
    # upperbound=np.max(var)
    # nbins=100
    # histogram,bins = np.histogram(var,bins=nbins,range=[lowerbound,upperbound], density=True)
    # midx = (bins[0:-1]+bins[1:])/2
    # axs[1].bar(midx,histogram,width=(upperbound-lowerbound)/nbins,color="blue",label="Rescaled",alpha=0.5)
    
    # 
    axs.set_xlabel("$g(x)$", fontsize=font_size)
    # axs[1].legend()
    txt = str(round(mean_tr,2)) #+" Rescaled mean g(x): "+str(round(mean_re,3))
    axs.axvline(x=round(mean_tr,2),color=myred,linewidth=line_w, label="$\\frac{1}{N} \\sum_{j=1}^{N} g(X^j)=$"+txt)

    axs.legend(fancybox=True,shadow=True,fontsize=font_size)

    # axs.set_title(txt)
    plt.show()
    # fig.savefig('Python/accuracy/accuracy_1dim/saved_figures/'+pot+'/hist_stepsize'+str(pot)+'tau'+str(tau)+'h'+str(h)+'M='+str(M)+'m'+str(m)+'T'+str(T)+'Nt'+str(Nt)+'.png')
