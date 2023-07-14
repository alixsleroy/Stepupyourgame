import numpy as np
import matplotlib.pyplot as plt
import scipy.integrate as integrate
from numba import jit,njit,vectorize, float64, int32
import numba as nb
from settings import *

# # SQUARE POTENTIAL 
# def U(x):
#     res = x[0]*x[0]+x[1]*x[1]
#     return res

# def rhox(x,tau): 
#     beta=1/tau
#     rho = np.exp(-beta*(x*x))
#     return rho

# # just to check that the code works
# @njit(float64[:](float64[:]))
# def Up(x):
#     Up=np.zeros(2)
#     Upx = 2*x[0]
#     Upy = 2*x[1]
#     Up[0]=Upx
#     Up[1]=Upy
#     return Up

###################################################
# Entropic potential barrier in molecular dynamic #
###################################################

 
def rhox(x,tau): 
    beta=1/tau
    rho = np.sqrt(10*np.power(x,4)+1)*np.exp(-beta*0.001*(x*x-9)*(x*x-9))
    return rho

# 
def U(x):
    res = 100*x[1]*x[1]/(1+10*np.power(x[0],4))+0.001*np.power((x[0]*x[0]-9),2)
    return res


@njit(float64[:](float64[:]))
def Up(x):
    Up=np.zeros(2)
    Upx = (-4000*x[1]*x[1]*x[0]*x[0]*x[0]*np.power((1+10*np.power(x[0],4)),-2)+0.004*(x[0]*x[0]-9)*x[0])
    Upy = 200*x[1]/(1+10*np.power(x[1],4))
    Up[0]=Upx
    Up[1]=Upy
    return Up

@njit(float64(float64[:]))
def getg(x): 
    x2 = np.power(x[0],2)
    Aden=A*x2/(1+x2)
    g=1./(1./m+Aden)
    return g

@njit(float64[:](float64[:]))
def getgprime(x):
    gp=np.zeros(2)
    gpx=(2*A*m*m*x[0])/np.power((x[0]*x[0]*(A*m+1)+1),2)
    gpy=0
    gp[0]=gpx
    gp[1]=gpy
    return gp

