

"""
This python file designed a numerical method for SDEs applied to the case of the underdamped langevin dynamic (two dimensional SDE). 
The underdamped system has a potential that is in q^2. 
Because it is two-dimensional, it is necessary to apply an other method than just Euler-Maruyama, so we use the BAOAB method. 
The numerical methods is applied to the transformed SDE using the adaptive function g(x)
The adaptive function is the exponential bounded between dtbounds[0] and dtbounds[1]

This file includes: 
- plot_distr(y,tau,dt,n_samples,T,title,ax): a function to plot the true distribution alongside the histogram of the generated sample 
- F(x): a function of the potential 
- nablaU(x): a function of the potential derivative
- g(x): the adaptive function used here
- A_ada(qp,h): step A of the method
- B_ada(qp,h): step B
- O_ada(qp,h,gamma,beta): step O
- one_traj_ada(qp,T,h,gamma,beta): compute all the steps for the required number of increment to reach T
- method_baoab_ada(n_samples,T,gamma,beta,h): compute the BAOAB method for n_samples 
"""


import numpy as np
import matplotlib.pyplot as plt
import scipy.integrate as integrate
from numba import jit,njit,vectorize, float64, int32
import numba as nb
import time as time

def F(q):
    return q**2/2


@njit(float64(float64))
def nablaU(q):
    return q


@njit(float64(float64))
def g(x): #,dtmin, dtmax, R):
    dtmin=0.005
    dtmax=0.01
    R=1
    y=(dtmax-dtmin)/dtmax*(1-np.exp(-np.abs(x)*R))+dtmin/dtmax
    return y


@njit(float64[:](float64[:],float64))
def A_ada(qp,h):
    q=qp[0]
    p=qp[1]
    gq = qp[2]
    ## fixed point method for g((qn+1+qn)/2)
    g_half = g(q+0.5*h*p*gq)
    g_half = g(q+0.5*h*p*g_half)
    g_half = g(q+0.5*h*p*g_half)
    g_half = g(q+0.5*h*p*g_half)
    gq = g_half
    q = q+p*gq*h
    qp=np.array([q,p])
    return (qp)

@njit(float64[:](float64[:],float64))
def B_ada(qp,h):
    q=qp[0]
    p=qp[1]
    gq=g(q)
    p = p-gq*nablaU(q)*h
    qp_gq=np.array([q,p,gq])
    return (qp_gq)

@njit(float64[:](float64[:],float64,float64,float64))
def O_ada(qp,h,gamma,beta):
    q=qp[0]
    p=qp[1]
    dB = np.random.normal(0,1,1)[0]
    gq=g(q)
    alpha =np.exp(-gamma*h*gq)
    p = alpha*p+ np.sqrt((1-alpha*alpha)/beta)*dB
    qp_gq=np.array([q,p,gq])
    return (qp_gq)

@njit(float64[:](float64[:],float64,float64,float64,float64))
def one_traj_ada(qp,T,h,gamma,beta):
    t=0
    h_half=h/2
    while t<T:
        qp_gq=B_ada(qp,h_half)
        qp=A_ada(qp_gq,h_half)
        qp_gq=O_ada(qp,h_half,gamma,beta)
        qp=A_ada(qp_gq,h_half)
        qp_gq=B_ada(qp,h_half)
        qp=qp_gq[:2]
        gq=qp_gq[2]
        t=np.round(t+gq*h,7)
    return (qp)

@njit(parallel=True)
def method_baoab_ada(n_samples,T,gamma,beta,h):
    qp_list=np.zeros((n_samples,2))
    qipi = np.array([1.0,1.0]) #np.random.normal(0,1,2) #initial conditions
    for j in nb.prange(n_samples):
        qfpf = one_traj_ada(qipi,T,h,gamma,beta)
        qp_list[j,::]=qfpf
    return(qp_list)

#compile the method
qp = method_baoab_ada(1,1,0.1,0.5,0.01)

