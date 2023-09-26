
from pot_definition import *


@njit(float64[:](float64[:],float64,float64,float64,float64))
def one_traj_noada(qp,Nt,dt,gamma,tau):
    t=0
    q=qp[0]
    p=qp[1]

    #useful things to compute for first step
    f = -Up(q)

    #set up vectors to save results 
    tqpg_list = np.zeros(4)

    for count in range(Nt):
        #**********
        #* STEP B *
        #**********
        p +=0.5*dt*f 

        #**********
        #* STEP A *
        #**********
        q += 0.5*dt*p

        #**********
        #* STEP O *
        #**********
        dB = np.random.normal(0,1)
        C =np.exp(-dt*gamma) 
        p = C*p+ np.sqrt((1.-C*C)*tau)*dB

        #**********
        #* STEP A *
        #**********
        q += 0.5*dt*p

        #**********
        #* STEP B *
        #**********
        f = -Up(q)
        p +=0.5*dt*f 
 
        t+= dt 
   

    #*********************************
    #* Save (p,q) and update time and*
    #*********************************
    tqpg_list[0]=t
    tqpg_list[1]=q
    tqpg_list[2]=p
    tqpg_list[3]=1
            
    return(tqpg_list)

@njit(parallel=True)
def sample_noada(n_samples,gamma,tau,Nt,h):
    nsample_pertraj = 1 
    tqpg_list=np.zeros((n_samples*nsample_pertraj,5))
    qipi = np.abs(np.random.normal(0,1,2)) #initial conditions np.array([2.0,0.0]) 
    for j in nb.prange(n_samples):
        qp_samples = one_traj_noada(qipi,Nt,h,gamma,tau)
        tqpg_list[j,1:5]=qp_samples
        tqpg_list[j,0]=j
    return(tqpg_list)

