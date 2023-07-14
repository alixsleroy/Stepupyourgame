# from numba import jit,njit,vectorize, float64, int32, boolean
# import numba as nb
# import numpy as np

from pot_defintion_openmp import *

####################
#  Numerical method#
####################  

def one_traj_noada(x,Nt,dt,tau):
    """
    Run the simulation for one sample path
    Input
    -----
    Ntot: int
        Number of counts to take to get to Tf with dt
    dt: float 
        Value of time increment. Is 1/N.
    tau: float 
        Is in the multiplicative term of the SDE in sqrt(2 \tau dt). 
    Return
    ------
    x: float
        Value of X(T) as approximated by the numerical scheme chosen
    """

    t = 0
    count=0
    # set up vectors to save results
    ctxg_list = np.zeros(5)

    for count in range(Nt):
        # Compute the values of f, g, g' and g*dt 
        #########################################
        f = -Up(x)

        # No adaptivity   
        ###############
        t+=dt

        # Compute the values of next count 
        #################################
        b1 = np.random.normal(0,1)
        x+=f*dt+np.sqrt(tau*dt*2)*b1

    #****************************
    #* Save (x) and update time *
    #****************************
    ctxg_list[0]=count
    ctxg_list[1]=t
    ctxg_list[2]=x
    ctxg_list[3]= 1 #value of g not used =1
    ctxg_list[4]= 0 #value of g' not used = 0 
    
    return (ctxg_list)

####################################################
#  Numerical method ran over M statistical samples #
####################################################

def sample_noada(x,n_samples,Nt,dt,tau): # Function is compiled and runs in machine code

    """
    Input
    -------
    x : float   
        initial value
    n_samples: int
        Number of sample to draw
    Nt: int 
        number of run
    dt: float
        Size of the time discretization 
    tau: float
        Value of the temperature of the DW SDE (+ sqrt(2*tau)*dW)
    Return
    -------
    y_final: np.array
        Array of shape (M,). Sample of numerical approximation of the DW SDE at time T
    
    """
    #set up the matrix to save the results 
    simctxdt_list=np.zeros((n_samples,6))
    for j in range(n_samples):
        ctxg_list =one_traj_noada(x,Nt,dt,tau)
        simctxdt_list[j,1:6]=ctxg_list
        simctxdt_list[j,0]=j
    return simctxdt_list
