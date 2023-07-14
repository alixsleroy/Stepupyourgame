## Import the package to run the samples 
from sample_over_noada import sample_noada
from sample_over_rescaled import sample_re
from sample_over_transformed import sample_tr
from settings import *
from useful_tools import *

def check_methods(Nt,tau,n_samples,h,x,range_bins,T):
    #######################################################
    ##### Obtain data to plot with the bond problem
    #######################################################
    dta_re = dta_format_over(sample_re(x,n_samples,Nt,h,tau))
    dta_tr = dta_format_over(sample_tr(x,n_samples,Nt,h,tau))
    dta_noada = dta_format_over(sample_noada(x,n_samples,Nt,h,tau))
    plot_one_distr_over(dta_noada,dta_tr,dta_re,tau,range_bins,h,T)
    list_param = str(pot)+'-tau='+str(tau)+'-M='+str(M)+'m='+str(m)+"-Nt="+str(Nt)+"-ns="+str(n_samples)+"-h="+str(h)
    dta_noada.to_pickle("Python/accuracy/accuracy_1dim/saved_pickles_over/plot_hist/dta_noada_"+list_param)
    dta_tr.to_pickle("Python/accuracy/accuracy_1dim/saved_pickles_over/plot_hist/dta_tr_"+list_param)
    dta_re.to_pickle("Python/accuracy/accuracy_1dim/saved_pickles_over/plot_hist/dta_re_"+list_param)



def sample_listdt(T,tau,n_samples,x,dt_list,pot):

    ####################################################
    ## Obtain order of a list of sample for different dt
    ####################################################

    # List of datasets to save the information on the sample values
    dta_noada = pd.DataFrame()
    dta_transfo = pd.DataFrame()
    dta_rescale = pd.DataFrame()

    # List of datasets to save the information on the   values of the function g
    dta_g_transfo=pd.DataFrame()
    dta_g_rescale=pd.DataFrame()


    for j in range(len(dt_list)):
        h=dt_list[j]
        Nt=int(T*1/h)+1 #set up the number of steps we will take

        ## compute the non adaptive value
        try1 =sample_noada(x,n_samples,Nt,h,tau) 
        dt = dta_format_over(try1)
        dta_noada["x"+str(j)] = dt["x"]
        

        ## compute the non adaptive value
        dt = dta_format_over(sample_tr(x,n_samples,Nt,h,tau))
        dta_transfo["x"+str(j)] = dt["x"]
        dta_g_transfo["g"+str(j)]=dt["g"]

        ## Comment out rescaled as interested in transform vs no ada
        # ## compute the non adaptive value
        # dt = dta_format_over(sample_re(x,n_samples,Nt,h,tau))
        # dta_rescale["x"+str(j)] = dt["x"]
        # dta_g_rescale["g"+str(j)]=dt["g"]

    # save the values of x
    list_param = str(pot)+'-tau='+str(tau)+'-M='+str(M)+'m='+str(m)+"-T="+str(T)+"-ns="+str(n_samples)
    dta_noada.to_pickle("Python/accuracy/accuracy_1dim/saved_pickles_over/dta_noada_"+list_param)
    dta_transfo.to_pickle("Python/accuracy/accuracy_1dim/saved_pickles_over/dta_tr_"+list_param)
    # dta_rescale.to_pickle("Python/accuracy/accuracy_1dim/saved_pickles_over/dta_re_"+list_param)

    # save the values of the function g
    dta_g_transfo.to_pickle("Python/accuracy/accuracy_1dim/saved_pickles_over/dta_g_tr_"+list_param)
    # dta_g_rescale.to_pickle("Python/accuracy/accuracy_1dim/saved_pickles_over/dta_g_re_"+list_param)


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
    true_mom_1 = np.round(integrate.quad(lambda q: np.exp(-U(q)/tau)*(q), a,b)[0],16)/norm
    true_mom_2 = np.round(integrate.quad(lambda q: np.exp(-U(q)/tau)*q*q, a,b)[0],16)/norm
    true_mom_3 = np.round(integrate.quad(lambda q: np.exp(-U(q)/tau)*(q)**3, a,b)[0],16)/norm
    true_mom_4 = np.round(integrate.quad(lambda q: np.exp(-U(q)/tau)*q*q*q*q, a,b)[0],16)/norm

    # print("true moment 1:\n")
    # print(true_mom_1)
    for j in range(len(dt_list)):

        x = dta_noada["x"+str(j)]

        # compute first moment
        mom_1 =np.sum((x))/len(x)
        mom_1=np.abs(mom_1-true_mom_1)
        mom_sd_1=np.std(np.abs(x))/np.sqrt(len(x))*1.96 #(Z for alpha 0.05)
        mom1_list.append(mom_1)
        mom_1_plussd.append(mom_1+mom_sd_1)

        #compute second moment
        mom_2 = np.sum(np.power(np.abs(x),2))/len(x)
        mom_2 = np.abs(mom_2-true_mom_2)
        mom_sd_2=np.std(np.power(np.abs(x),2))/np.sqrt(len(x))*1.96 #(Z for alpha 0.05)
        mom2_list.append(mom_2)
        mom_2_plussd.append(mom_2+mom_sd_2)

        # compute third moment
        mom_3 = np.sum(np.power((x),3))/len(x)
        mom_3=np.abs(mom_3-true_mom_3)
        mom_sd_3=np.std(np.power(np.abs(x),3))/np.sqrt(len(x))*1.96 #(Z for alpha 0.05)
        mom3_list.append(mom_3)
        mom_3_plussd.append(mom_3+mom_sd_3)

        # compute fourth moment 
        mom_4 = np.sum(np.power(np.abs(x),4))/len(x)
        mom_4=np.abs(mom_4-true_mom_4)
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

        

