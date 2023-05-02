import numpy as np
import argparse
import sys, time, pickle

sys.path.insert(1, 'src/')
sys.path.insert(1, '../src/')
sys.path.insert(1, '../../src/')
sys.path.insert(1, '../../../src/')
from concrete import *
from aux import *



#########################
#########################
#   arg parse options   #
#########################
#########################
parser = argparse.ArgumentParser(description="Fit a RealNVP normalizing flow with Concrete relaxed nodes for an Ising model")

parser.add_argument('-M', type = float, default = 5,
    help = 'Number of particles in Ising model')
parser.add_argument('--beta', type = float, default = 1,
    help = 'Inverse temperature of Ising model')
parser.add_argument('--temp', type = float, default = 0.1,
    help = 'Temperature of Concrete relaxation')
parser.add_argument('--depth', type = float, default = 10,
    help = 'Depth of the RealNVP flow')
parser.add_argument('--width', type = float, default = 32,
    help = 'Width of the RealNVP flow')
parser.add_argument('--lr', type = float, default = 0.001,
    help = 'Learning rate for the Adam optimizer')
parser.add_argument('--max_iters', type = float, default = 100001,
    help = 'Number of iterations of the Adam optimizer')
parser.add_argument('--outpath', type = str, default = '',
help = 'Path where results will be saved (in pkl format)')
parser.add_argument('--idx', type = float, default = 0,
    help = 'Index at which to save results in pkl file')

args = parser.parse_args()

# save options
M         = int(args.M)
beta      = args.beta
temp      = args.temp
depth     = int(args.depth)
width     = int(args.width)
lr        = args.lr
max_iters = int(args.max_iters)
outpath   = args.outpath
idx       = int(args.idx)
#########################
#########################


########################
########################
# target specification #
########################
########################
logZ1=np.log(2*np.cosh(beta)) # same for any value of x_2 and x_(M-1)
def lp(x,axis=None):
    # compute the univariate log joint and conditional target pmfs of the Ising model
    #
    # inputs:
    #    x    : (M,d) array with state values
    #    axis : int, variable to condition on; returns joint if None
    # outputs:
    #   ext_lprb : if axis is None, (d,) array with log joint; else, (d,2) array with d conditionals

    xc=np.copy(x)
    xc[xc==0]=-1 # internally lowest x=0, but here we need lowest x=-1
    if axis==None:
        tmp_x=np.vstack((np.zeros((1,xc.shape[1])),np.copy(xc))) # add row with 0's at start of x
        lag_x=np.vstack((np.copy(xc),np.zeros((1,xc.shape[1])))) # add row with 0's at end of x
        return beta*np.sum(tmp_x*lag_x,axis=0)
    if axis==0: return np.vstack((-xc[1,:],xc[1,:])).T-logZ1#np.log(2*np.cosh(beta*xc[axis+1,:]))
    if axis==M-1: return np.vstack((-xc[-2,:],xc[-2,:])).T-logZ1#np.log(2*np.cosh(beta*xc[axis-1,:]))
    if axis>=M: raise Exception("Axis out of bounds")
    logZm=np.log(2*np.cosh(beta*(xc[axis-1,:]+xc[axis+1,:])))
    return np.vstack((-xc[axis-1,:]-xc[axis+1,:],xc[axis-1,:]+xc[axis+1,:])).T-logZm[:,np.newaxis]

if M<13: # possible to compute normalizing constant and estimate exact target
    x=idx_unflattenBinary(np.arange(0,2**M),M)
    lprbs=lp(x)
    prbs=np.exp(lprbs)
    prbs=prbs/np.sum(prbs)
else: # import gibbs samples and use them to estimate probabilities
    prbs=pkl_load('gibbs_density')



########################
########################
#    Settings          #
########################
########################
print('Training a RealNVP normalizing flow with a Concrete relaxation for an Ising model')
print()
print('Ising model settings:')
print('Number of particles M: '+str(M))
print('Inverse temperature: '+str(beta))
print()
print('Flow settings:')
print('Relaxation temperature: '+str(temp))
print('Flow depth: '+str(depth))
print('Flow width: '+str(width))
print()
print('Optimizer settings:')
print('Max number of iters: '+str(max_iters))
print('Learning rate: '+str(lr))
print()


########################
########################
#    Concrete          #
########################
########################
print('Starting optimization')
t0 = time.perf_counter()
tmp_flow,tmp_loss=trainRealNVP(
    temp=temp,depth=depth,lprbs=np.log(prbs),width=width,max_iters=max_iters,lr=lr,seed=2023
)
cpu_time=time.perf_counter()-t0
print('Done!')
print('Total training time: '+str(cpu_time)+' seconds')
print()


print('Cacheing results')
pkl_save(tmp_flow, outpath+'cache/0'+str(idx)+'_ising_flows')
pkl_save(tmp_loss, outpath+'cache/0'+str(idx)+'_ising_losses')
print()


print('Saving results')
# load files
flows     = pkl_load(outpath+'ising_flows')
losses    = pkl_load(outpath+'ising_losses')
cpu_times = pkl_load(outpath+'ising_cpu_times')

# update files
flows[idx]     = tmp_flow
losses[idx,:]  = tmp_loss
cpu_times[idx] = cpu_time

# save updated files
pkl_save(flows, outpath+'ising_flows')
pkl_save(losses, outpath+'ising_losses')
pkl_save(cpu_times, outpath+'ising_cpu_times')
print('Done!')
