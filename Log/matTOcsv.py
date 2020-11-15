import scipy.io
import numpy as np
import pandas as pd

def MatCsv(file):
    data = scipy.io.loadmat(file)

    for i in data:
        if '__' not in i and 'readme' not in i:
            np.savetxt(("file.csv"),data[i],delimiter=',')

def MatCsv2(file):
    mat = scipy.io.loadmat(file)
    mat = {k:v for k, v in mat.items() if k[0] != '_'}
    data = pd.DataFrame({k: pd.Series(v[0]) for k, v in mat.items()})
    data.to_csv("example.csv")

MatCsv2("BASEHOCK.mat")