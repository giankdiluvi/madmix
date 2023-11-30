import numpy as np
import argparse
import sys, time, pickle

sys.path.insert(1, 'src/')
sys.path.insert(1, '../src/')
sys.path.insert(1, '../../src/')
sys.path.insert(1, '../../../src/')
import argmax_flows as amf
import concrete
from aux import pkl_save, pkl_load



#########################
#########################
#   arg parse options   #
#########################
#########################
parser = argparse.ArgumentParser(description="Fit a RealNVP normalizing flow to an Ising model with argmaxed nodes")

parser.add_argument('-M', type = str, default = 'small',
    help = 'Number of particles. One of small (M=5) or large (M=50)')
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
M         = args.M
depth     = int(args.depth)
width     = int(args.width)
lr        = args.lr
max_iters = int(args.max_iters)
outpath   = args.outpath
idx       = int(args.idx)
filename  = 'ising' # to cache files
#########################
#########################




########################
########################
#  load target sample  #
########################
########################
sample = pkl_load('gibbs_samples_'+str(M)+'M')


########################
########################
#    Settings          #
########################
########################
print('Training a RealNVP normalizing flow for an Ising model with argmaxed nodes')
print()
print('M: '+M)
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
training_sample = amf.argmax_discsample_gen(sample)
tmp_flow,tmp_loss=amf.train_argmax_discrete(depth=depth,sample=training_sample,width=width,max_iters=max_iters,lr=lr)
cpu_time=time.perf_counter()-t0
print('Done!')
print('Total training time: '+str(cpu_time)+' seconds')
print()

print('Cacheing results')
pkl_save(tmp_flow, outpath+'cache/0'+str(idx)+'_'+filename+'_flows_argmax')
pkl_save(tmp_loss, outpath+'cache/0'+str(idx)+'_'+filename+'_losses_argmax')
print()

print('Saving results')
# load files
flows     = pkl_load(outpath+filename+'_flows_argmax')
losses    = pkl_load(outpath+filename+'_losses_argmax')
cpu_times = pkl_load(outpath+filename+'_cpu_times_argmax')

# update files
flows[idx]     = tmp_flow
losses[idx,:]  = tmp_loss
cpu_times[idx] = cpu_time

# save updated files
pkl_save(flows, outpath+filename+'_flows_argmax')
pkl_save(losses, outpath+filename+'_losses_argmax')
pkl_save(cpu_times, outpath+filename+'_cpu_times_argmax')
print('Done!')
