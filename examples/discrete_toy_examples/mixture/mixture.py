import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys, time, pickle
from IPython.display import clear_output

sys.path.insert(1, '../../../discrete_mixflows/')
from concrete import *
from aux import *

########################
########################
# target specification #
########################
########################
########################
########################
# target specification #
########################
########################
eps=1e-16
mu=np.array([4,15]) # the two modes
K1,K2=20,2
def aux_gausslp(aux): return -0.5*(np.arange(0,20)-mu[aux][:,np.newaxis])**2-0.5*np.log(2*np.pi)
def aux_gausslp_1d(y,x): return -0.5*(y-mu[x])**2-0.5*np.log(2*np.pi)

def lp(x,axis=None):
    # compute the univariate log joint and conditional target pmfs
    #
    # inputs:
    #    x    : (2,d) array with state values
    #    axis : int, full conditional to calculate; returns joint if None
    # outputs:
    #   ext_lprb : if axis is None, (d,) array with log joint; else, (d,K_{axis+1}) array with d conditionals

    y=x[0,:]
    aux=x[1,:]

    if axis==None: return -0.5*(y-mu[aux])**2-0.5*np.log(2*np.pi)#1.
    if axis==0: return aux_gausslp(aux)
    if axis==1:
        wlpy0=aux_gausslp_1d(y,0)+np.log(0.5)
        wlpy1=aux_gausslp_1d(y,1)+np.log(0.5)
        m=np.maximum(wlpy0,wlpy1)
        lpy=m+np.log(np.exp(wlpy0-m)+np.exp(wlpy1-m))
        #out=np.log1p(-eps)*np.ones((y.shape[0],2))
        #out[y>=10,0]=np.log(eps)
        #out[y<10,1]=np.log(eps)
        #out=0.5*np.ones((y.shape[0],2))
        out=np.ones((y.shape[0],2))
        out[:,0]=wlpy0
        out[:,1]=wlpy1
        return out-lpy[:,np.newaxis]
    raise Exception("Axis out of bounds - there aren't that many variables")

# evaluate target density
x=np.zeros((2,40),dtype=int)
x[0,:20]=np.arange(0,20)
x[0,20:]=np.arange(0,20)
x[1,20:]=np.ones(20,dtype=int)

mylp=np.exp(lp(x))
prbs=np.zeros((20,2))
prbs[:,0]=mylp[:20]
prbs[:,1]=mylp[20:]
prbs=prbs/np.sum(prbs)


########################
########################
#    Concrete          #
########################
########################
# simulation settings:
max_iters = 10001
temps     = np.array([0.1,0.5,1.,5.])
depths    = np.array([10,50,100])
layers    = np.array([32,64,128,256])
sim_size  = temps.shape[0]*depths.shape[0]*layers.shape[0]
flows     = [0 for i in range(sim_size)]
losses    = np.zeros((sim_size,max_iters))
cpu_times = np.zeros(sim_size)


i=-1
print('Training '+str(sim_size)+' flows')
for temp in temps:
    for depth in depths:
        for width in layers:
            i=i+1
            print('Training flow '+str(i+1)+'/'+str(sim_size))
            print('Temperature: '+str(temp))
            print('Depth: '+str(depth))
            print('Width: '+str(width))
            t0 = time.perf_counter()
            tmp_flow,tmp_loss=trainRealNVP(
                temp=temp,depth=depth,lprbs=np.log(prbs).flatten(),width=width,max_iters=max_iters,lr=1e-3,seed=2023
            )
            cpu_times[i]=time.perf_counter()-t0
            flows[i]=tmp_flow
            losses[i,:]=tmp_loss

            # save results
            pkl_save(flows,'mixture_flows')
            pkl_save(losses,'mixture_losses')
            pkl_save(cpu_times,'mixture_cpu_times')
            clear_output(wait=True)

        # end for
    # end for
# end for
clear_output(wait=True)
print('Done!')
print('Total training time: '+str(cpu_times.sum())+' seconds')
