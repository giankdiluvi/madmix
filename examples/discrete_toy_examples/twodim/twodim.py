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
np.random.seed(2023)
K1=4
K2=5
prbs=np.random.rand(K1,K2)
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
            pkl_save(flows,'twodim_flows')
            pkl_save(losses,'twodim_losses')
            pkl_save(cpu_times,'twodim_cpu_times')
            clear_output(wait=True)

        # end for
    # end for
# end for
clear_output(wait=True)
print('Done!')
print('Total training time: '+str(cpu_times.sum())+' seconds')
