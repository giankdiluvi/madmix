import numpy as np


def LogSumExp(w):
    """
    LogSumExp trick
    
    inputs:
        w : (N,d) array, exponents
        
    outputs:
        w' : (d,) array, log(sum(exp(w)))
    """
    
    wmax = np.amax(w,axis=0)
    return wmax + np.log(np.sum(np.exp(w-wmax[np.newaxis,:]),axis=0))