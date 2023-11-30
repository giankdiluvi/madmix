import numpy as np
import torch
import argparse
import sys, time, pickle

sys.path.insert(1, 'src/')
sys.path.insert(1, '../src/')
sys.path.insert(1, '../../src/')
sys.path.insert(1, '../../../src/')
import argmax_flows as amf
from concrete import *
from aux import *



#########################
#########################
#   arg parse options   #
#########################
#########################
parser = argparse.ArgumentParser(description="Fit a RealNVP normalizing flow with argmaxed label indices to a Spike and Slab model with superconductivity data")

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
pred_pi     = pkl_load('pred_pi')
pred_beta   = pkl_load('pred_beta')
pred_theta  = pkl_load('pred_theta')
pred_sigma2 = pkl_load('pred_sigma2')
pred_tau2   = pkl_load('pred_tau2')

K = pred_pi.shape[1]



########################
########################
#    Settings          #
########################
########################
print('Training a RealNVP normalizing flow with argmaxed label indices for a Spike and Slab model with the superconductivity data set')
print()
print('Regression settings:')
print('Number of covariates K: '+str(K))
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
training_sample = amf.argmax_sassample_gen(pred_pi,pred_beta,pred_theta,pred_sigma2,pred_tau2)
tmp_flow,tmp_loss=amf.train_argmax_discrete(depth=depth,sample=training_sample,width=width,max_iters=max_iters,lr=lr)

cpu_time=time.perf_counter()-t0
print('Done!')
print('Total training time: '+str(cpu_time)+' seconds')
print()

print('Cacheing results')
pkl_save(tmp_flow, outpath+'cache/0'+str(idx)+'_sas_flows')
pkl_save(tmp_loss, outpath+'cache/0'+str(idx)+'_sas_losses')
print()


print('Saving results')
# load files
flows     = pkl_load(outpath+'sas_flows_argmax')
losses    = pkl_load(outpath+'sas_losses_argmax')
cpu_times = pkl_load(outpath+'sas_cpu_times_argmax')

# update files
flows[idx]     = tmp_flow
losses[idx,:]  = tmp_loss
cpu_times[idx] = cpu_time

# save updated files
pkl_save(flows, outpath+'sas_flows_argmax')
pkl_save(losses, outpath+'sas_losses_argmax')
pkl_save(cpu_times, outpath+'sas_cpu_times_argmax')
print('Done!')
