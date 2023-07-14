"""
Pass functions on the method and on the potential used as an argument


"""

from numba import jit,njit,vectorize, float64, int32
import numba as nb
import numpy as np


def getU(x):
    """
    potential is the infinite double well
    """

    return x*x
    
# define the gradV function 
@njit(float64(float64))
def getminusdU(x):
    """
    Derivative of the potential with infinite double well:
    x: float 
    """

    return 2*x

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
    y_final = [] 
    s = np.sqrt(2*tau*dt)
    for i in range(n_samples):
        yf =run_num(Ntot,dt,s)
        y_final.append(yf)
    y_final=np.array(y_final)
    return y_final


### Visualisse distribution

def plot_dist(y,tau,title,ax):
    """
    Input
    -------
    y: np.array
        Samples to visualise
    title: str 
        title of the plot we wish to see
    ax: axis
        On which axis we wish to plot it
    tau: float
        Value of the size of the noise


    Return
    -------
    None: simply plot the distribution
    
    """

    ax.set_title(title)
    #Plot 1
    histogram,bins = np.histogram(y,bins=250,range=[-1,1], density=True)
    midx = (bins[0:-1]+bins[1:])/2
    ax.plot(midx,histogram,label='experiment')
    rho = np.exp(- (getU(midx)/tau))
    rho = rho / ( np.sum(rho) * (midx[1]-midx[0]) ) 
    # rho=[rho[i] if i>50 else 0 for i in range(len(rho))]
    ax.plot(midx,rho,'--',label='Truth') 
    ax.legend()