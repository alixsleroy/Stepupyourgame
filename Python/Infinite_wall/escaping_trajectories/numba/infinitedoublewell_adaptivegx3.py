"""
This python file designed a numerical method for SDEs applied to the case of the infinite double well SDE.
The euler maruyama numerical methods is applied to the transformed SDE using the adaptive function g(x)
The adaptive function is determined by the potential function as defined in the report 

This file includes: 
- plot_distr(y,tau,dt,n_samples,T,title,ax): a function to plot the true distribution alongside the histogram of the generated sample 
- U(x): a function of the infinite double well potential 
- dU(x): a function of the infinite double well potential derivative
- g3(x,dtbounds): the adaptive function used here
- e_m_ada3(y0,s,b1,dt,dtbounds): a function that computes the next value of the increment using euler maruyama on the transformed SDE
- run_num_ada3(Ntot,dt,s,dtbounds): a function that goes through all the iterations 
- IDW_nsample_ada3(n_samples,T,dt,tau,dtbounds): a function that runs the EM increments for a number of sample
"""


from numba import jit,njit,vectorize, float64, int32
import numba as nb
import numpy as np

def U(x):
    """
    potential of -the infinite double well
    """

    return (1/(2*x**2)+x*x)
    
# define the gradV function 
@njit(float64(float64))
def dU(x):
    """
    Compute the potential of the infinite double well:
    x: float 
    """

    return 1/(x*x*x)-2*x

# define the adaptive function 
@njit(float64[:](float64,float64,float64,float64))
def g3(x,h,dtmin,dtmax):
    """
    Compute the value of the adaptive function choosen:
    x: float 
    """
    M=h/dtmin
    m=h/dtmax
    x3=np.power(x,3)

    # value of function f
    f=np.abs(1/x3-2*x)
    fprime=np.abs(3/(x3*x)+2)

    #compute gx
    gx_num=1/M*(f+m)+1
    gx_den =f+m 
    gx=np.round(gx_num/gx_den,6)

    #compute gx prime 
    gxprime=np.round(-fprime/(gx_den*gx_den),6)

    #return
    re=np.array([gx,gxprime])
    return re

@njit(float64(float64,float64,float64,float64))
def e_m_ada3(y0,s,b1,dt):
    """
    The Euler-Maruyama scheme applied to the infinite double well
    y0: float
        value of y at t_n
    tau: float
        value of the temperature 
    b1: float
        brownian increment 
    dt: float
        time increment
    """
    re=g3(y0,dt,0.0001,0.05)
    gy=re[0]
    nablag=re[1]
    y1=y0+(gy*dU(y0)+nablag)*dt+np.sqrt(gy)*s*b1
    return y1    


# @njit(nb.types.UniTuple(nb.float64,2)(float64,float64,float64,float64,float64))
# def run_num(N,dt,s,T,n_esc):

@njit(float64(float64,float64,float64))
def run_num_ada3(Ntot,dt,s):
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
    yf: float
        Value of X(T) as approximated by the numerical scheme chosen
        
    """
    y0 = 1
    for jj in range(Ntot): # Run until T= Tsec
        b1 = np.random.normal(0,1,1)[0]
        y1 = e_m_ada3(y0,s,b1,dt)
        y0=y1 
    return (y0)



@njit(parallel=True)
def IDW_nsample_ada3(n_samples,T,dt,tau): # Function is compiled and runs in machine code
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
    method: function
        Numerical scheme used for the DW SDE
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
        yf =run_num_ada3(Ntot,dt,s)
        y_final.append(yf)
    y_final=np.array(y_final)
    return y_final

ytest= IDW_nsample_ada3(10**2,3,0.001,10) # compile the function


#%time ytest=y_compile = DW_sde_fast(1000,3,10,0.01,20) # compile the function
def plot_dist(y,tau,dt,n_samples,T,title,ax):
    ax.set_title(str(title)+", $\\tau$="+str(tau)+", h="+str(dt)+", \n N="+str(n_samples)+", T="+str(T))

    #Plot 1
    histogram,bins = np.histogram(y,bins=100,range=[-2,5], density=True)

    midx = (bins[0:-1]+bins[1:])/2
    histogram=(histogram/np.sum(histogram))
    ax.plot(midx,histogram,label='q-Experiment')


    ### true distribution 
    # rho = np.exp(- U(midx)/tau)
    # rho = rho / ( np.sum(rho)* (midx[1]-midx[0]) ) # Normalize rho by dividing by its approx. integral
    # ax1.plot(midx,rho,'--',label='Truth')

    rho = np.exp(- (U(midx)/tau))
    rho = rho / ( np.sum(rho) * (midx[1]-midx[0]) ) 
    rho=(rho/np.sum(rho))*2
    rho=[rho[i] if i>50 else 0 for i in range(len(rho))]
    ax.plot(midx,rho,'--',label='Truth') 
    ax.legend()

