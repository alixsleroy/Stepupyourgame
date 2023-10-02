import scipy.integrate as integrate
import numpy as np
import re 

global a
global b
global x0
global c
global tau
global h
global dtlist 

# Spring-M=1.500000-m=0.001000-Ns=100-a=10.000000-b=0.100000-c=0.100000-x0=0.500000

# ## hard problem

#Spring-M=1.500000-m=0.001000-Ns=100-a=10.000000-b=0.100000-c=0.100000-x0=0.500000
##################################################################################
a=  2.75
b=  0.1
x0= 0.5
c=  0.1
tau=0.1
gamma=1.

# dtlist = np.array([np.exp(-4.5), np.exp(-4.21), np.exp(-3.93), np.exp(-3.64), np.exp(-3.36), np.exp(-3.07), np.exp(-2.79), np.exp(-2.5) , np.exp(-2.21), np.exp(-1.93), np.exp(-1.64), np.exp(-1.36), np.exp(-1.07), np.exp(-0.79), np.exp(-0.5)])
dtlist = np.array([np.exp(-3.93),np.exp(-3.36),np.exp(-2.79),np.exp(-2.21),np.exp(-1.64),np.exp(-1.07),np.exp(-0.5)])
def U(x):
    res = (a**1.5*b**0.5*x0*np.arctan((a/b)**0.5*(x-x0))+(a*b*(a*x0*(x-x0)-b))/(a*(x-x0)**2+b)+c*(x-x0)**2+2*c*(x-x0)*x0)*0.5
    return res



# # ## easy problem on eddie 
# pot="spring"
# a=  1.0
# b=  1.0
# x0= 0.5
# c=  0.1
# tau=0.1
# # # dtlist = np.array([0.01,0.03,0.05,0.07,0.09,0.1,0.2,0.3,0.4])
# # range_int=[-10,10]

# # easier problem
# a=  1.0
# b=  0.1
# x0= 0.5
# c=  0.1
# tau=0.1
# # dtlist = np.array([0.002,0.006,0.018,0.05,0.13,0.36])

# dtlist = np.array([0.009,0.01,0.02,0.03,0.04,0.05,0.06,0.07,0.08,0.09,0.1,0.2,0.3,0.4,0.5,0.6])


#######################################################
##### read file from c code
#######################################################
def openCfile(file):
    """
    Open the C txt file in order to obtain a matrix of results 
    -----------
    Input
    -----------
    file: txt file
        a file containing the results from the C simulation

    Return
    -----------
    mat: list of list
        A matrix containing the results of the simulations 
    """
    with open(file) as f:
        cols = f.readlines() #columns in the txt file
    n_col = len(cols) #number of columns in the text file
    mat=[] # matrix 
    for i in range(n_col): # for each columns 
        elems_i=cols[i].split(" ") #split the elements using " "
        col_i=[] #create an empty column i 
        for elem in elems_i: #for each element of the list 
            if elem!="\n" and elem!=" ": #compare each elements and discard " " and "\n"
                col_i.append(float(elem)) #append elems that are floats to the vector of interest
        mat.append(col_i) #create the matrix
    mat=np.array(mat)
    return(mat) #return the value of the matrix.

def get_slope(accuracy_list,dt_list):
    #######################################
    ## Obtain the slope of the error decay
    #######################################
    logx1=np.log(dt_list[0])
    logx2=np.log(dt_list[-1])
    logy1=np.log(accuracy_list[0])
    logy2=np.log(accuracy_list[-1])
    a=(logy1-logy2)/(logx1-logx2)
    b=logy1-a*logx1
    x=np.linspace(logx1,logx2,1000)
    y_x=a*x+b
    a_round=np.round(np.abs(a),2)
    return(x,y_x,a_round)

# def get_slope(accuracy_list,dt_list):
#     #######################################
#     ## Obtain the slope of the error decay
#     #######################################
#     logx1=dt_list[0]
#     logx2=dt_list[-1]
#     logy1=(accuracy_list[0])
#     logy2=(accuracy_list[-1])
#     a=(logy1-logy2)/(logx1-logx2)
#     b=logy1-a*logx1
#     x=np.linspace(logx1,logx2,1000)
#     y_x=a*x+b
#     a_round=np.round(np.abs(a),2)
#     return(x,y_x,a_round)
