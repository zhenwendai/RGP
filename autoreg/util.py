
from numpy.lib.stride_tricks import as_strided

def get_conv_1D(arr, win):
    assert win>0
    arr = arr.copy()
    if win==1:
        return arr
    else:
        return as_strided(arr, shape=(arr.shape[0]-win+1,win)+arr.shape[1:], strides=(arr.strides[0],)+arr.strides)