##### Build a main to compute accuracy plot with different schemes and different examples problem

# ## import useful packages
# import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.append("Python/Underdamped langevin/compare 3 schemes/accuracy_1dim")
# import scipy.integrate as integrate
# from numba import jit,njit,vectorize, float64, int32
# import numba as nb
# import time as time


## Import the package to run the samples 

from sample_under_noada import sample_noada
from sample_under_rescaled import sample_re
from sample_under_transformed import sample_tr

## Import the settings 
from settings import *
from useful_tools import *

global T
global gamma
global tau
global h
global n_samples
global Nt
    
T=100
gamma = 0.5
tau=0.1
n_samples=100000
h=0.1
Nt=int(T*1/h)+1

def main():

    ## Setting 

    T=100
    gamma = 0.5
    tau=0.1
    n_samples=1000
    h=0.1
    Nt=int(T*1/h)+1

    #######################################################
    ##### Obtain data to plot with the bond problem
    #######################################################
    dta_re = dta_format_under(sample_re(n_samples,gamma,tau,Nt,h))
    dta_tr = dta_format_under(sample_tr(n_samples,gamma,tau,Nt,h))
    dta_noada = dta_format_under(sample_noada(n_samples,gamma,tau,Nt,h))
    # dta_re.to_pickle("Python/Underdamped langevin/compare 3 schemes/accuracy_1dim/saved_pickles/dta_re")
    plot_one_distr_under(df_rescale=dta_re,df_noada=dta_noada,df_transfo=dta_tr,tau=tau)
    
    #######################################
    ## Obtain order of accuracy
    ########################################

    # ## set up the list of time steps
    # dt_list=[ np.round(2**(-j),5) for j in range(1,8,1)]

    # ## set up others values 
    # T=100
    # gamma = 0.5
    # tau=0.1
    # n_samples=10

    # ## set the list of accuracy
    # dta_noada = pd.DataFrame()
    # dta_transfo = pd.DataFrame()
    # dta_rescale = pd.DataFrame()

    # for j in range(len(dt_list)):
    #     h=dt_list[j]
    #     Nt=int(T*1/h)+1 #set up the number of steps we will take

    #     ## compute the non adaptive value
    #     dt = dta_format_under(sample_noada(n_samples,gamma,tau,Nt,h))
    #     dta_noada["q"+str(j)] = dt["q"]

    #     ## compute the non adaptive value
    #     dt = dta_format_under(sample_tr(n_samples,gamma,tau,Nt,h))
    #     dta_transfo["q"+str(j)] = dt["q"]
        
    #     ## compute the non adaptive value
    #     dt = dta_format_under(sample_re(n_samples,gamma,tau,Nt,h))
    #     dta_rescale["q"+str(j)] = dt["q"]

    
    # dta_noada.to_pickle("Python/Underdamped langevin/compare 3 schemes/accuracy_1dim/saved_pickles/dta_noada_bond")
    # dta_transfo.to_pickle("Python/Underdamped langevin/compare 3 schemes/accuracy_1dim/saved_pickles/dta_tr_bond")
    # dta_rescale.to_pickle("Python/Underdamped langevin/compare 3 schemes/accuracy_1dim/saved_pickles/dta_re_bond")

# 

main()





