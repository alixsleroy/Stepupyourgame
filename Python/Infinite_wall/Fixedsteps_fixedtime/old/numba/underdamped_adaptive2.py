

"""
This python file designed a numerical method for SDEs applied to the case of the underdamped langevin dynamic (two dimensional SDE). 
The underdamped system has a potential that is in q^2. 
Because it is two-dimensional, it is necessary to apply an other method than just Euler-Maruyama, so we use the BAOAB method. 
The numerical methods is applied to the transformed SDE using the adaptive function g(x)
The adaptive function is the function designed for the derivative and g(x) to stay bounded

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


# @njit(float64(float64,float64,float64[:]))
# def g(x,h,dtbounds):
#     """
#     Compute the value of the adaptive function choosen:
#     x: float 
#     """
#     dtmin=dtbounds[0]
#     dtmax=dtbounds[1]

#     M=h/dtmin
#     m=h/dtmax

#     x3=np.power(x,3)

#     # value of function f, f' and f^2
#     f=np.abs(nablaU(x))
#     f2 = f*f

#     #compute the value of phi(f(x)) = \sqrt{f(x)^2}
#     phif2 = f2*f2

#     # value of m^2
#     m2 = m*m

#     #compute gx
#     gx_den=np.sqrt(phif2+m2)
#     gx_num = gx_den/M + 1 
#     gx=gx_num/gx_den

#     #return
#     re=gx
#     return re

@njit(float64[:](float64,float64,float64[:]))
def g4(x,h,dtbounds):
    """
    Compute the value of the adaptive function choosen:
    x: float 
    """
    dtmin=dtbounds[0]
    dtmax=dtbounds[1]

    M=h/dtmin
    m=h/dtmax

    x3=np.power(x,3)

    # value of function f, f' and f^2
    f=(1/x3-2*x)
    fprime=-(3/(x3*x)+2)
    f2 = f*f

    #compute the value of phi(f(x)) = \sqrt{f(x)^2}
    phif = np.sqrt(f2)
    phif2 = f2*f2

    # value of m^2
    m2 = m*m

    #compute gx
    gx_den=np.sqrt(phif2+m2)
    gx_num = gx_den/M + 1 
    gx=gx_num/gx_den

    #compute gx prime 
    gxp_num= -f*fprime
    gxp_den = gx_den*gx_den*gx_den
    gxprime= gxp_num/gxp_den

    #round number to avoid having too large number 
    gx =gx
    gxprime = gxprime

    #return
    re=np.array([gx,gxprime])
    re=np.array([1,1])
    return re


@njit(float64[:](float64[:],float64,float64[:]))
def A_ada(qp_gprime,h,dtbounds):
    q=qp_gprime[0]
    p=qp_gprime[1]
    gq = qp_gprime[2]

    ## fixed point method for g((qn+1+qn)/2)
    g_half = g4(q+0.5*h*p*gq,h,dtbounds)[0]
    g_half = g4(q+0.5*h*p*g_half,h,dtbounds)[0]
    g_half = g4(q+0.5*h*p*g_half,h,dtbounds)[0]
    g_half = g4(q+0.5*h*p*g_half,h,dtbounds)[0]
    gq = g_half
    q = q+p*gq*h
    qp=np.array([q,p])
    return (qp)

@njit(float64[:](float64[:],float64,float64[:],float64))
def B_ada(qp_gprime,h,dtbounds,beta):
    q=qp_gprime[0]
    p=qp_gprime[1]
    g_gprime=g4(q,h,dtbounds)
    gq=g_gprime[0]
    gprime=g_gprime[1]
    p = p-(gq*nablaU(q)+gprime/beta)*h
    qp_gq=np.array([q,p,gq])
    return (qp_gq)

@njit(float64[:](float64[:],float64,float64[:],float64,float64))
def O_ada(qp,h,dtbounds,gamma,beta):
    q=qp[0]
    p=qp[1]
    dB = np.random.normal(0,1,1)[0]
    gq_gprime=g4(q,h,dtbounds)
    alpha =np.exp(-gamma*h*gq_gprime[1])
    p = alpha*p+ np.sqrt((1-alpha*alpha)/beta)*dB
    qp_gq=np.array([q,p,gq_gprime[0]])
    return (qp_gq)

@njit(float64[:](float64[:],float64,float64,float64[:],float64,float64))
def one_traj_ada(qp,T,h,dtbounds,gamma,beta):
    t=0
    h_half=h/2
    tcount=0
    while t<T:
        qp_gq=B_ada(qp,h_half,dtbounds,beta)
        qp=A_ada(qp_gq,h_half,dtbounds)
        qp_gq=O_ada(qp,h_half,dtbounds,gamma,beta)
        qp=A_ada(qp_gq,h_half,dtbounds)
        qp_gq=B_ada(qp,h_half,dtbounds,beta)
        qp=qp_gq[:2]
        gq=qp_gq[2]
        t=np.round(t+gq*h,7)
        tcount=tcount+1
    qp_t=np.append(qp,tcount)
    return (qp_t)
    

@njit(parallel=True)
def method_baoab_ada2(n_samples,T,gamma,beta,h,dtbounds):
    qpt_list=np.zeros((n_samples,3))
    qipi = np.array([1.0,1.0]) #np.random.normal(0,1,2) #initial conditions
    for j in nb.prange(n_samples):
        qfpftf = one_traj_ada(qipi,T,h,dtbounds,gamma,beta)
        qpt_list[j,::]=qfpftf
    return(qpt_list)

#compile the method
print(method_baoab_ada2(10,10,0.1,0.5,0.1,np.array([0.1,0.1])))



