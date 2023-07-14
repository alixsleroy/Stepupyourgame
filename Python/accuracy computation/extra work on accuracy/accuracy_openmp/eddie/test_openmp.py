# mpiexec -n 4 python "Python/accuracy/accuracy_openmp/test_openmp.py"
import pandas as pd
import os
import numpy as np

# initialise path 
os.chdir("/home/s2133976/OneDrive/ExtendedProject/Code/Weak SDE approximation/Python/accuracy/")


### initialize mpi.
import mpi4py.rc
mpi4py.rc.initialize = False  
from mpi4py import MPI
MPI.Init()
###

### the actual program.

comm = MPI.COMM_WORLD   # the global communicator.

rank = comm.Get_rank()  # get process ID within comm.


## lowercase version for generic objects, uses pickle under the hood.

data = None
next_h=1
j=0

list_of_data=[]

while j<9:
    j+=1
    # if rank!=0:
    data = "rank="+str(rank)+"-j="+str(j)+"prout"
        # comm.send(data, dest=0, tag=1) #send to the rank 0 the information it needs to receive
    
    newdata=comm.gather(data,root=0)
    if rank==0:
        list_of_data.append(newdata)
        


if rank==0:
    for el in list_of_data:
        print(el)
    dta = pd.DataFrame(list_of_data,columns=["rank0","rank1","rank2","rank3"])

    dta.to_pickle("accuracy_openmp/data/data")

### denote end of mpi calls.
MPI.Finalize()
###  
