import numpy as np
import torch


def top10acc(y_true, y_predict):
    s = 0
    for i in range(len(y_true)):
        #print(y_true[i], y_predict[i])
        if y_true[i] in y_predict[i]:
            s+=1
    return s/len(y_true)


def letter2index(s):
    d = {'A':5, 'C':1, 'G':2, 'N':3, 'T':4}
    result = []
    for l in s:
        result.append(d[l])
    return np.array(result)
