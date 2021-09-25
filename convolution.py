import numpy as np
import math

def conv2d(inp, fil, strides=1, padding="valid"):
    
    f = fil.shape[0] # filter size

    if padding.lower() == "same":
        p = math.floor((f - 1) / 2) # padding needed to get the same size output
        inp = np.pad(inp, p)

    n = inp.shape[0] # matrix size

    s = list(range(0, n - f + 1, strides)) # all chunk starts  
    e = list(range(f, n + 1, strides)) # all chunk ends

    o_s = math.floor(((n - f) / strides) + 1) # outputs shape
    out = np.zeros((o_s, o_s))

    for row in range(len(s)):
        for col in range(len(e)):

            # get chunk that is the shape of the filter
            chunk = inp[s[row]:e[row], s[col]:e[col]]

            # element-wise multiplication and summation
            out[row, col] = np.sum(chunk * fil)
            
    return out


def conv2d_transpose(inp, fil, stride=1, padding='valid'):

    n = inp.shape[0] # input size
    f = fil.shape[0] # filter size
    
    o_s = stride * (n-1) + f # output shape for valid padding
    out = np.zeros((o_s, o_s))

    s = list(range(0, o_s - 2, stride)) # chunk starts
    e = list(range(f, o_s + 1, stride)) # chunk ends

    for row in range(len(s)):
        for col in range(len(e)):
            # add to selected chunk
            out[s[row]:e[row], s[col]:e[col]] += inp[row, col] * fil

    # output shape -> (n * s, n * s)
    if padding.lower() == 'same':
        s = stride
        l = math.floor((f - s)/2) # padding left
        r = f - s - l # padding right
        t = math.floor((f - s)/2) # padding top
        b = f - s - l # padding bottom

    out = out[l:-r, t:-b]
    
    return out