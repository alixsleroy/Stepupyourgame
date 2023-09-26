from pot_definition import *
import pandas as pd

@njit(float64[:](float64[:],float64,float64,float64,float64))
def one_traj_re(qp,Nt,dt,gamma,tau):
    t=0
    q=qp[0]
    p=qp[1]

    #useful things to compute for first step
    f = -Up(q)
    g=getg(q)
    gdt=g*dt

    #set up vectors to save results 
    tqpg_list = np.zeros(4)

    for count in range(Nt):
        #**********
        #* STEP B *
        #**********
        p +=0.5*gdt*f 

        #**********
        #* STEP A *
        #**********
        q += 0.5*gdt*p

        #**********
        #* STEP O *
        #**********
        dB = np.random.normal(0,1)
        C =np.exp(-gdt*gamma) 
        p = C*p+ np.sqrt((1.-C*C)*tau)*dB

        #**********
        #* STEP A *
        #**********
        q += 0.5*gdt*p

        #**********
        #* STEP B *
        #**********
        # Need to compute new quantities for f and g 
        f = -Up(q)
        g=getg(q)
        gdt=g*dt

        p +=0.5*gdt*f 
        t+=gdt
        

    #*********************************
    #* Save (p,q) and update time and*
    #*********************************
    tqpg_list[0]=t
    tqpg_list[1]=q
    tqpg_list[2]=p
    tqpg_list[3]=g
            
    return(tqpg_list)

@njit(parallel=True)
def sample_re(n_samples,gamma,tau,Nt,h):
    nsample_pertraj = 1 
    tqpg_list=np.zeros((n_samples*nsample_pertraj,5))
    qipi = np.abs(np.random.normal(0,1,2)) #initial conditions np.array([2.0,0.0]) 
    for j in nb.prange(n_samples):
        qp_samples = one_traj_re(qipi,Nt,h,gamma,tau)
        tqpg_list[j,1:5]=qp_samples
        tqpg_list[j,0]=j
    return(tqpg_list)



