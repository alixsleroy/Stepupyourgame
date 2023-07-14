import numpy as np

m=0.001
M=1.5
b=0.1
a=10
x0=.5
c=0.10

#############################
# V(x) =  spring potential  #
#############################
def U(x):
    res = (a**1.5*b**0.5*x0*np.arctan((a/b)**0.5*(x-x0))+(a*b*(a*x0*(x-x0)-b))/(a*(x-x0)**2+b)+c*(x-x0)**2+2*c*(x-x0)*x0)*0.5
    return res

def Up(x):
    wx =b/(b/a+(x-x0)**2)
    res = (wx*wx+c)*x
    return res

def getg(x): 
    wx =(b/a+(x-x0)**2)/b
    f = wx*wx
    xi = f+m
    g = 1/(1/M+1/np.sqrt(xi))
    return g

def getgprime(x):
    wx =(b/a+(x-x0)**2)/b
    f = wx*wx
    fp = 4*(x-x0)*((b/a)+(x-x0)**2)/b**2
    xi=np.sqrt(f+m*m)
    gprime= M**2*fp/(2*xi*(xi+M)**2)
    return gprime
