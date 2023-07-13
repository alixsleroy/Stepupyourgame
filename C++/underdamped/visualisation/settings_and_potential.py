
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

a=  0.1
b=  1.0
x0= 0.1
c=  0.1
tau=0.1
# dtlist = np.array([0.001,0.003,0.005,0.007,0.009,0.01,0.02])

#######################################################
##### read file from c code
#######################################################
def openCfile_qp(file):
    with open(file) as f:
        cols = f.readlines() #columns in the txt file
    n_col = len(cols) #number of columns in the text file
    # when fill_p is 1, then fill in the matrix q
    fill_p=0
    fill_g=0
    # vector of res
    vals_q=[] #create an empty column i 
    vals_p=[] #create an empty column i
    vals_g=[] #create an empty column i  
    for i in range(n_col): # for each columns 
        if cols[i]=='q\n':
            fill_q=1
            i+=1
        if cols[i]=='p\n':
            fill_p=1
            fill_q=0
            i+=1
        if cols[i]=='g\n':
            fill_g=1
            fill_q=0
            fill_p=0
            i+=1
        # clean up the cols 
        elems_i=cols[i].split(" ") #split the elements using " "
        for elem in elems_i: #for each element of the list 
            if elem!="\n" and elem!=" ": #compare each elements and discard " " and "\n"
                if fill_q==1:
                    vals_q.append(float(elem)) #append elems that are floats to the vector of interest
                elif fill_p==1:
                    vals_p.append(float(elem))
                elif fill_g==1:
                    vals_g.append(float(elem))
    return np.array(vals_q),np.array(vals_p),np.array(vals_g)

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
