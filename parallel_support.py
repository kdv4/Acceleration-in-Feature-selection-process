import pandas as pd
import numpy as np
from numpy import asarray
import re
import sys

def Write_XL(OPfile,category):
	IPfile=pd.read_csv("output/centroids.txt",sep=" ",header=None)
	IPfile=IPfile.iloc[:,:-1]
	IPfile.columns=category
	IPfile.to_excel(OPfile,index=False)
	
def Write_TXT(filepath,dataset):
    with open(filepath,"w") as fp:
    	dataset.iloc[:,:].to_string(fp,index=False,header=False)
    
def init_centroid(dataset,n):
	samples=dataset.sample(n)
	Write_TXT("input/initCoord.txt",samples)
	
	

