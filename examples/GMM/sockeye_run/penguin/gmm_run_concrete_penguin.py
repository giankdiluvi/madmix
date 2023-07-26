import numpy as np
import torch
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
parser = argparse.ArgumentParser(description="Fit a RealNVP normalizing flow with Concrete relaxed nodes to a GMM with Palmer penguins data set")

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
pred_x  = pkl_load('pred_x')
pred_w = pkl_load('pred_w')
pred_mu = pkl_load('pred_mu')
pred_sigma = pkl_load('pred_sigma')

# convert gibbs output to torch tensors
xs_concrete = torch.from_numpy(pred_x)
ws_concrete = torch.from_numpy(pred_w)
mus_concrete = torch.from_numpy(pred_mu)
sigmas_concrete = torch.from_numpy(pred_sigma)

N,K,D = pred_x.shape[1], pred_mu.shape[1], pred_mu.shape[2] # 333, 3, 4
print(N,K,D)
tau0=0.1



########################
########################
#    Settings          #
########################
########################
print('Training a RealNVP normalizing flow with a Concrete relaxation for a GMM with the Palmer penguin data set')
print()
print('Mixture settings:')
print('Mixture size K: '+str(K))
print('Number of observations N: '+str(N))
print('Prior precision tau0: '+str(tau0))
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
conc_sample=gmm_concrete_sample(xs_concrete,ws_concrete,mus_concrete,sigmas_concrete,temp)
tmp_flow,tmp_loss=trainGMMRealNVP(
    temp=temp,depth=depth,N=N,K=K,D=D,tau0=tau0,sample=conc_sample,width=width,max_iters=max_iters,lr=lr,seed=2023,verbose=True
)

cpu_time=time.perf_counter()-t0
print('Done!')
print('Total training time: '+str(cpu_time)+' seconds')
print()

print('Cacheing results')
pkl_save(tmp_flow, outpath+'cache/0'+str(idx)+'_gmm_flows')
pkl_save(tmp_loss, outpath+'cache/0'+str(idx)+'_gmm_losses')
print()


print('Saving results')
# load files
flows     = pkl_load(outpath+'gmm_flows')
losses    = pkl_load(outpath+'gmm_losses')
cpu_times = pkl_load(outpath+'gmm_cpu_times')

# update files
flows[idx]     = tmp_flow
losses[idx,:]  = tmp_loss
cpu_times[idx] = cpu_time

# save updated files
pkl_save(flows, outpath+'gmm_flows')
pkl_save(losses, outpath+'gmm_losses')
pkl_save(cpu_times, outpath+'gmm_cpu_times')
print('Done!')
