import numpy as np
import argparse
import sys, time, pickle

sys.path.insert(1, '../madmix/')
from concrete import *
from aux import *



#########################
#########################
#   arg parse options   #
#########################
#########################
parser = argparse.ArgumentParser(description="Fit a RealNVP normalizing flow with Concrete relaxed nodes")

parser.add_argument('--target', type = str, default = 'onedim', choices=['onedim', 'twodim', 'mixture'],
    help = 'Target distribution to learn')
parser.add_argument('--temp', type = float, default = 0.1,
    help = 'Temperature of Concrete relaxation')
parser.add_argument('--depth', type = int, default = 10,
    help = 'Depth of the RealNVP flow')
parser.add_argument('--width', type = int, default = 32,
    help = 'Width of the RealNVP flow')
parser.add_argument('--lr', type = float, default = 0.001,
    help = 'Learning rate for the Adam optimizer')
parser.add_argument('--max_iters', type = int, default = 100001,
    help = 'Number of iterations of the Adam optimizer')

args = parser.parse_args()

# save options
target=args.target
temp=args.temp
depth=args.depth
width=args.width
lr=args.lr
max_iters=args.max_iters
#########################
#########################




########################
########################
# target specification #
########################
########################
if target=='onedim':
    prbs=np.array([0.07393561, 0.20446061, 0.13502975, 0.02906925, 0.03245506,
        0.10743913, 0.00507227, 0.16699829, 0.12041089, 0.12512913])
    prbs=prbs/np.sum(prbs)

if target=='twodim':
    # flattened version of the 2d array
    prbs=np.array([0.04394993, 0.12153859, 0.08026644, 0.01727979, 0.01929243,
           0.06386561, 0.00301514, 0.09926967, 0.07157647, 0.07438117,
           0.06229286, 0.06843638, 0.05384315, 0.02063433, 0.04925781,
           0.02212277, 0.04612981, 0.0246133 , 0.05336854, 0.00486582])
    prbs=prbs/np.sum(prbs)

if target=='mixture':
    # flattened version of the 2d array
    prbs=np.array([6.69152124e-05, 2.76535889e-50, 2.21592750e-03, 5.48304095e-44,
           2.69955234e-02, 3.99941983e-38, 1.20985542e-01, 1.07319346e-32,
           1.99471437e-01, 1.05941120e-27, 1.20985542e-01, 3.84730504e-23,
           2.69955234e-02, 5.13989443e-19, 2.21592750e-03, 2.52613930e-15,
           6.69152124e-05, 4.56736700e-12, 7.43360863e-07, 3.03794594e-09,
           3.03794594e-09, 7.43360863e-07, 4.56736700e-12, 6.69152124e-05,
           2.52613930e-15, 2.21592750e-03, 5.13989443e-19, 2.69955234e-02,
           3.84730504e-23, 1.20985542e-01, 1.05941120e-27, 1.99471437e-01,
           1.07319346e-32, 1.20985542e-01, 3.99941983e-38, 2.69955234e-02,
           5.48304095e-44, 2.21592750e-03, 2.76535889e-50, 6.69152124e-05])
    prbs=prbs/np.sum(prbs)


########################
########################
#    Concrete          #
########################
########################
print('Training a RealNVP normalizing flow with a Concrete relaxation')
print('Temperature: '+str(temp))
print('Depth: '+str(depth))
print('Width: '+str(width))
t0 = time.perf_counter()
tmp_flow,tmp_loss=trainRealNVP(
    temp=temp,depth=depth,lprbs=np.log(prbs),width=width,max_iters=max_iters,lr=1e-3,seed=2023
)
cpu_time=time.perf_counter()-t0
print('Done!')
print('Total training time: '+str(cpu_time)+' seconds')


i=0
print('Training '+str(sim_size)+' flows')
for temp in temps:
    for depth in depths:
        for width in layers:
            print('Training flow '+str(i+1)+'/'+str(sim_size))
            print('Temperature: '+str(temp))
            print('Depth: '+str(depth))
            print('Width: '+str(width))
            t0 = time.perf_counter()
            tmp_flow,tmp_loss=trainRealNVP(
                temp=temp,depth=depth,lprbs=np.log(prbs),width=width,max_iters=max_iters,lr=1e-3,seed=2023
            )
            cpu_times[i]=time.perf_counter()-t0
            flows[i]=tmp_flow
            losses[i,:]=tmp_loss

            # save results
            pkl_save(flows,'onedim_flows')
            pkl_save(losses,'onedim_losses')
            pkl_save(cpu_times,'onedim_cpu_times')
            print()
            i=i+1
        # end for
    # end for
# end for

print('Done!')
print('Total training time: '+str(cpu_times.sum())+' seconds')
