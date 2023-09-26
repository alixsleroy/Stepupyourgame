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

# a=  1.0
# b=  1
# x0= 0.5
# c=  0.1
# tau=0.1
# dtlist = np.array([0.01,0.03,0.05,0.07,0.09,0.1,0.2,0.3,0.4])

a=  2.75
b=  0.1
x0= 0.5
c=  0.1
tau=0.1
#dtlist = np.array([0.009,0.01,0.02,0.03,0.04,0.05,0.06,0.07,0.08,0.09,0.1,0.2,0.3,0.4,0.5,0.6])
#dtlist = np.array([0.005,0.01,0.05,0.1,0.5])
# dtlist = np.array([0.009,0.02,0.04,0.06,0.08,0.1,0.2,0.4,0.6])
dtlist = np.array([np.exp(-4.5),np.exp(-4.),np.exp(-3.5),np.exp(-3.),np.exp(-2.5),np.exp(-2.),np.exp(-1.5),np.exp(-1.),np.exp(-.5)])

dtlist = np.array([np.exp(-4.5), np.exp(-4.21), np.exp(-3.93), np.exp(-3.64), np.exp(-3.36), np.exp(-3.07), np.exp(-2.79), np.exp(-2.5) , np.exp(-2.21), np.exp(-1.93), np.exp(-1.64), np.exp(-1.36), np.exp(-1.07), np.exp(-0.79), np.exp(-0.5)])

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

def U(x):
    res = (a**1.5*b**0.5*x0*np.arctan((a/b)**0.5*(x-x0))+(a*b*(a*x0*(x-x0)-b))/(a*(x-x0)**2+b)+c*(x-x0)**2+2*c*(x-x0)*x0)*0.5
    return res

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

def moment_list(dt_list,tau,dta_noada,range_int):
    ####################################################
    ## Obtain the error on the moments
    ####################################################

    mom1_list=[]
    mom2_list=[]
    mom3_list=[]
    mom4_list=[]

    mom_1_plussd=[]
    mom_2_plussd=[]
    mom_3_plussd=[]
    mom_4_plussd=[]


    ## When no access to the true moment
    a=range_int[0]
    b=range_int[1]
    norm=np.round(integrate.quad(lambda q: np.exp(-U(q)/tau), a,b)[0],16)
    true_mom_1 = np.round(integrate.quad(lambda q: np.exp(-U(q)/tau)*(q), a,b)[0],16)/norm
    true_mom_2 = np.round(integrate.quad(lambda q: np.exp(-U(q)/tau)*q*q, a,b)[0],16)/norm
    true_mom_3 = np.round(integrate.quad(lambda q: np.exp(-U(q)/tau)*(q)**3, a,b)[0],16)/norm
    true_mom_4 = np.round(integrate.quad(lambda q: np.exp(-U(q)/tau)*q*q*q*q, a,b)[0],16)/norm

    # print("true moment 1:\n")
    # print(true_mom_1)
    for j in range(len(dt_list)):

        x = dta_noada["x"+str(j)]

        # compute first moment
        mom_1 =np.sum((x))/len(x)
        mom_1=np.abs(mom_1-true_mom_1)
        mom_sd_1=np.std(np.abs(x))/np.sqrt(len(x))*1.96 #(Z for alpha 0.05)
        mom1_list.append(mom_1)
        mom_1_plussd.append(mom_1+mom_sd_1)

        #compute second moment
        mom_2 = np.sum(np.power(np.abs(x),2))/len(x)
        mom_2 = np.abs(mom_2-true_mom_2)
        mom_sd_2=np.std(np.power(np.abs(x),2))/np.sqrt(len(x))*1.96 #(Z for alpha 0.05)
        mom2_list.append(mom_2)
        mom_2_plussd.append(mom_2+mom_sd_2)

        # compute third moment
        mom_3 = np.sum(np.power((x),3))/len(x)
        mom_3=np.abs(mom_3-true_mom_3)
        mom_sd_3=np.std(np.power(np.abs(x),3))/np.sqrt(len(x))*1.96 #(Z for alpha 0.05)
        mom3_list.append(mom_3)
        mom_3_plussd.append(mom_3+mom_sd_3)

        # compute fourth moment 
        mom_4 = np.sum(np.power(np.abs(x),4))/len(x)
        mom_4=np.abs(mom_4-true_mom_4)
        mom_sd_4=np.std(np.power(np.abs(x),4))/np.sqrt(len(x))*1.96 #(Z for alpha 0.05)
        mom4_list.append(mom_4)
        mom_4_plussd.append(mom_4+mom_sd_4)

    return(mom1_list,mom2_list,mom3_list,mom4_list,mom_1_plussd,mom_2_plussd,mom_3_plussd,mom_4_plussd)

