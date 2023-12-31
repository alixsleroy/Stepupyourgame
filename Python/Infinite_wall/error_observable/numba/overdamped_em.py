from numba import jit,njit,vectorize, float64, int32
import numba as nb
import numpy as np


def getU(x):
    """
    potential is the infinite double well
    """

    return (1/(2*x**2)+x*x)
    
# define the gradV function 
@njit(float64(float64))
def getminusdU(x):
    """
    Derivative of the potential with infinite double well:
    x: float 
    """

    return -1/(x*x*x)+2*x

@njit(float64(float64,float64,float64,float64))
def e_m_method(y0,s,b1,dt):
    """
    The Euler-Maruyama scheme applied to the overdamped langevin with infinite double well:
    Input
    -----
    y0: float
        value of y at t_n
    s: float
        value of additive noise constant
    b1: float
        brownian increment 
    dt: float
        time increment

    Output
    ------
    y1: float 
        value of y at time t+h

    """
    y1=y0 - getminusdU(y0)*dt+s*b1
    return y1    


@njit(float64(float64,float64,float64))
def run_num(Ntot,dt,s):
    """
    Run the simulation for one sample path
    Input
    -----
    Ntot: int
        Number of steps to take to get to Tf with dt
    dt: float 
        Value of time increment. Is 1/N.
    s: float 
        Is sqrt(2 \tau dt). 
    Return
    ------
    y0: float
        Value of X(T) as approximated by the numerical scheme chosen
        
    """
    y0 = 1 #set up 1 as initial conditions
    for jj in range(Ntot): # Run until T= Tsec
        b1 = np.random.normal(0,1)
        y1 = e_m_method(y0,s,b1,dt)
        y0=y1 
    return (y0)



@njit(parallel=True)
def nsample(n_samples,T,dt,tau): # Function is compiled and runs in machine code
    """
    Input
    -------
    n_samples: int
        Number of sample to draw
    T: int 
        Final time
    dt: float
        Size of the time discretization 
    tau: float
        Value of the temperature of the DW SDE (+ sqrt(2*tau)*dW)

    Return
    -------
    y_final: np.array
        Array of shape (M,). Sample of numerical approximation of the DW SDE at time T
    
    """
    N = int(np.round(1/dt,6))  #size of the time steps
    Ntot = N*T #total number of steps to take to arrive at T in steps of dt 
    y_final = [] #np.zeros(n_samples)
    s = np.sqrt(2*tau*dt)
    for i in range(n_samples):
        yf =run_num(Ntot,dt,s)
        y_final.append(yf)
    y_final=np.array(y_final)
    return y_final

# @njit(parallel=True)
# def IDW_nsampleN(n_samples,T,Ntot,tau): # Function is compiled and runs in machine code
#     """
#     Input
#     -------
#     n_samples: int
#         Number of sample to draw
#     T: int 
#         Final time
#     dt: float
#         Size of the time discretization 
#     tau: float
#         Value of the temperature of the DW SDE (+ sqrt(2*tau)*dW)
#     method: function
#         Numerical scheme used for the DW SDE
#     Return
#     -------
#     y_final: np.array
#         Array of shape (M,). Sample of numerical approximation of the DW SDE at time T
    
#     """
#     # N = int(np.round(1/dt,6))  #size of the time steps
#     dt = np.round(T/Ntot,6) #total number of steps to take to arrive at T in steps of dt 
#     y_final = [] #np.zeros(n_samples)
#     s = np.sqrt(2*tau*dt)
#     for i in range(n_samples):
#         yf =run_num(Ntot,dt,s)
#         y_final.append(yf)
#     y_final=np.array(y_final)
#     return y_final


# y_compile = IDW_nsample(10**6,3,0.1,10) # compile the function

#%time ytest=y_compile = DW_sde_fast(1000,3,10,0.01,20) # compile the function
