import numpy as np
import torch
import argparse
import sys, time, pickle

sys.path.insert(1, 'src/')
sys.path.insert(1, '../src/')
sys.path.insert(1, '../../src/')
sys.path.insert(1, '../../../src/')
import dequantization
from concrete import *
from aux import *



#########################
#########################
#   arg parse options   #
#########################
#########################
parser = argparse.ArgumentParser(description="Fit a RealNVP normalizing flow with dequantized label indices to a GMM with waveform data set")

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

N,K,D = pred_x.shape[1], pred_mu.shape[1], pred_mu.shape[2] # 333, 3, 4
tau0=0.1



########################
########################
#    Settings          #
########################
########################
print('Training a RealNVP normalizing flow with dequantized label indices for a GMM with the waveform data set')
print()
print('Mixture settings:')
print('Mixture size K: '+str(K))
print('Number of observations N: '+str(N))
print('Prior precision tau0: '+str(tau0))
print()
print('Flow settings:')
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
dequant_sample=dequantization.gmm_dequant_sample(pred_x,pred_w,pred_mu,pred_sigma)
tmp_flow,tmp_loss=dequantization.train_dequant_gmm(
    depth=depth,N=N,K=K,D=D,tau0=tau0,sample=dequant_sample,width=width,max_iters=max_iters,lr=lr
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
flows     = pkl_load(outpath+'gmm_flows_dequant')
losses    = pkl_load(outpath+'gmm_losses_dequant')
cpu_times = pkl_load(outpath+'gmm_cpu_times_dequant')

# update files
flows[idx]     = tmp_flow
losses[idx,:]  = tmp_loss
cpu_times[idx] = cpu_time

# save updated files
pkl_save(flows, outpath+'gmm_flows_dequant')
pkl_save(losses, outpath+'gmm_losses_dequant')
pkl_save(cpu_times, outpath+'gmm_cpu_times_dequant')
print('Done!')
