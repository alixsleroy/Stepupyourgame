# mpiexec -n 4 python "Python/accuracy/accuracy_openmp/test_openmp.py"
## Run the accuracy function through open MP and save the pickle file

# initialise path - change for eddie
import os
import sys
# nrank = sys.argv[0]
os.chdir("/home/s2133976/OneDrive/ExtendedProject/Code/Weak SDE approximation/Python/accuracy/")

# usual packages 
import pandas as pd
import numpy as np

# import necessary files.py with function 
from sample_over_noada_openmp import *
from sample_over_rescaled_openmp import *
from sample_over_transformed_openmp import *

### initialize mpi.
import mpi4py.rc
mpi4py.rc.initialize = False  
from mpi4py import MPI


## MPI intialise
MPI.Init()
################

### the actual program.
comm = MPI.COMM_WORLD   # the global communicator.
rank = comm.Get_rank()  # get process ID within comm.
# max_rank=comm.Get
# print(max_rank)

# Parameters of the simulations 
T=50
tau=0.1
n_samples=1000 #0000
# n_samples per rank 
n_samples2_rank = int(n_samples/12)
x=0
dt_list= [0.0005,0.001,0.01,0.05]


## Initialise necessary variables. 
data = None

## useful function 
def dta_format_over(mat):
    dta = pd.DataFrame(mat,columns=["sim","count","t","x","g","gp"])
    return dta

####################################################
## Obtain order of a list of sample for different dt
####################################################

# List of datasets to save the information on the sample values
dta_xnoada = pd.DataFrame()
dta_xtr = pd.DataFrame()
dta_xre = pd.DataFrame()

# List of datasets to save the information on the values of the function g
dta_gtr=pd.DataFrame()
dta_gre=pd.DataFrame()

for j in range(len(dt_list)):

    h=dt_list[j]
    Nt=int(T*1/h)+1 #set up the number of steps we will take

    ## compute the non adaptive sample
    dt_noada =sample_noada(x,n_samples2_rank,Nt,h,tau)
    x_noada = dt_noada[:,3]
    
    # ## compute the transformed sample
    dt_tr = sample_tr(x,n_samples2_rank,Nt,h,tau)
    x_tr = dt_tr[:,3]
    g_tr = dt_tr[:,4]

    # ## compute the rescaled sample
    dt_re = sample_re(x,n_samples2_rank,Nt,h,tau)
    x_re = dt_re[:,3]
    g_re = dt_re[:,4]

    ## Gather data from all rank 
    new_xnoada=comm.gather(x_noada,root=0)
    new_xtr=comm.gather(x_tr,root=0)
    new_xre=comm.gather(x_re,root=0)
    new_gtr=comm.gather(g_tr,root=0)
    new_gre=comm.gather(g_re,root=0)

    if rank==0:
        # Get a flat array with all the samples from all the ranks for each info of interest
        new_xnoada=np.hstack(new_xnoada)
        new_xtr = np.hstack(new_xtr)
        new_xre = np.hstack(new_xre)
        new_gtr = np.hstack(new_gtr)
        new_gre = np.hstack(new_gre)

        #save these info in a dataframe in the same format as before 
        dta_xnoada["x"+str(j)] = new_xnoada
        dta_xtr["x"+str(j)] = new_xtr
        dta_xre["x"+str(j)] = new_xre
        dta_gtr["x"+str(j)] = new_gtr
        dta_gre["x"+str(j)] = new_gre


if rank==0:
    print(dta_xnoada.head())
    print(dta_xtr.head())
    print(dta_xre.head())
    print(dta_gtr.head())
    print(dta_gre.head())

    # save the values of x
    list_param = 'spring-tau='+str(tau)+'-M='+str(M)+'m='+str(m)+"-T="+str(T)+"-ns="+str(n_samples)
    dta_xnoada.to_pickle("accuracy_openmp/data/dta_noada_"+list_param)
    dta_xtr.to_pickle("accuracy_openmp/data/dta_tr_"+list_param)
    dta_xre.to_pickle("accuracy_openmp/data/dta_re_"+list_param)

    # save the values of the function g
    dta_gtr.to_pickle("accuracy_openmp/data/dta_g_tr_"+list_param)
    dta_gre.to_pickle("accuracy_openmp/data/dta_g_re_"+list_param)

### denote end of mpi calls.
MPI.Finalize()
###  
