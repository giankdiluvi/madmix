import numpy as np
import pickle


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


"""
########################################
########################################
pickle dump and load wrappers
########################################
########################################

Inputs:
    obj  : Python object to save to a pkl file
    file : str, name of pkl file (without extension) to save obj to

Outputs (pkl_load):
    obj : Python object from file.pkl
"""
def pkl_save(obj,file):
    with open(file+'.pkl', 'wb') as f:
        pickle.dump(obj, f)
    f.close()

def pkl_load(file):
    with open(file+'.pkl', 'rb') as f:
        obj = pickle.load(f)
    f.close()
    return obj
