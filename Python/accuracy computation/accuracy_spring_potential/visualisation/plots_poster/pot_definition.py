import numpy as np
import matplotlib.pyplot as plt
import scipy.integrate as integrate
from numba import jit,njit,vectorize, float64, int32
import numba as nb
from settings import *

if pot=="bond":
    ###########################
    # BOND PROBLEM DEFINITION #
    ##########################
    def U(x):
        res = (0.5/(x*x)+x*x)
        return res

    @njit(float64(float64))
    def Up(x):
        res = -1/(x*x*x)+2*x
        return res

    @njit(float64(float64))
    def getg(x): #,dtmin, dtmax, R):
        x6 = np.power(x,6)
        xi = np.sqrt(m*m+x6)
        g=1./(1./M+1./xi)
        return g

    @njit(float64(float64))
    def getgprime(x):
        x5 = np.power(x,5)
        x6 = x5*x
        xi = np.sqrt(x6+m*m)
        g =1./(1./M + 1./xi)
        gprime= 3*x5*np.power(g,2)/np.power(xi,3)
        return gprime

########################
# # STUPID SQUARRED POT  #
# ########################
elif pot=="square":
    def U(x):
        res = x*x
        return res

    @njit(float64(float64))
    def Up(x):
        res = 2*x
        return res

    @njit(float64(float64))
    def getg(x): #,dtmin, dtmax, R):
        x2 = np.abs(2*x)
        xi = np.sqrt(m*m+x2)
        g=(1./M+1./xi)
        return g

    @njit(float64(float64))
    def getgprime(x):
        x2 = np.abs(2*x)
        xi = np.sqrt(m*m+x2)
        gprime= -4*x/np.power(xi,3/2)
        return gprime

########################
# V(x) = x^4  #
########################
elif pot=="power4":
    def U(x):
        res = x*x*x*x
        return res

    @njit(float64(float64))
    def Up(x):
        res = 4*x*x*x
        return res

    @njit(float64(float64))
    def getg(x): #,dtmin, dtmax, R):
        x6 = np.power(x,6)
        xi = np.sqrt(m*m+16*x6)
        g=(1./M+1./xi)
        return g

    @njit(float64(float64))
    def getgprime(x):
        x5=np.power(x,5)
        x6=x*x5
        xi = np.sqrt(m*m+16*x6)
        gprime= -48*x5/np.power(xi,3/2)
        return gprime
    
########################
# V(x) =  spring potential  #
########################
elif pot=="spring":

    def U(x):
        res = (a**1.5*b**0.5*x0*np.arctan((a/b)**0.5*(x-x0))+(a*b*(a*x0*(x-x0)-b))/(a*(x-x0)**2+b)+c*(x-x0)**2+2*c*(x-x0)*x0)*0.5
        return res
    
    @njit(float64(float64))
    def Up(x):
        wx =b/(b/a+(x-x0)**2)
        res = (wx*wx+c)*x
        return res
    

    #  USING ONLY THE WHOLE FUNCTION 

    # @njit(float64(float64))
    # def getg(x): 
    #     wx =b/(b/a+(x-x0)**2)
    #     f = (wx*wx+c)*x
    #     xi = f*f + m*m
    #     g = 1/M+1/np.sqrt(xi)
    #     return g

    # @njit(float64(float64))
    # def getgprime(x):
    #     wx =b/(b/a+(x-x0)**2)
    #     f = (wx*wx+c)*x
    #     fp = (a*a*b*b*(a*(-3*x*x+2*x0*x+x0*x0)+b))/np.power((a*(x-x0)*(x-x0)+b),3)
    #     gprime= -f*fp/np.power((f*f+m*m),1.5)
    #     return gprime
    
    # USING ONLY THE SPRING PART TO REDUCE THE STEP SIZE 
    # @njit(float64(float64))
    # def getg(x): 
    #     wx =b/(b/a+(x-x0)**2)
    #     f = wx*wx
    #     xi = f*f + m*m
    #     g = 1/M+1/np.sqrt(xi)
    #     return g
    
    # @njit(float64(float64))
    # def getgprime(x):
    #     wx =b/(b/a+(x-x0)**2)
    #     f = wx*wx
    #     fp = -4*b*b*(x-x0)/np.power((b/a+(x-x0)*(x-x0)),3)
    #     gprime= -f*fp/np.power((f*f+m*m),1.5)
    #     return gprime
    
    ## Using the different defintion of the function
    @njit(float64(float64))
    def getg(x): 
        wx =(b/a+(x-x0)**2)/b
        f = wx*wx
        xi = f+m
        g = 1/(1/M+1/np.sqrt(xi))
        return g
    
    @njit(float64(float64))
    def getgprime(x):
        wx =(b/a+(x-x0)**2)/b
        f = wx*wx
        fp = 4*(x-x0)*((b/a)+(x-x0)**2)/b**2
        xi=np.sqrt(f+m*m)
        gprime= M**2*fp/(2*xi*(xi+M)**2)
        return gprime
    ## something is not working with that definition of the potential 
    ## not too sure what is not working
    ## would need to see 
