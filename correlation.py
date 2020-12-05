import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# from sklearn.model_selection import train_test_split
from scipy.stats import pearsonr as peterson
from statsmodels.stats.outliers_influence import variance_inflation_factor
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning) 

file='cars.csv'

def visualize_corrlation(category,importances,indices):
    #Visualize correlation
    plt.title('Correlation Graph')
    plt.barh(range(len(indices)), importances[indices], color='g', align='center')
    plt.yticks(range(len(indices)), [category[i] for i in indices])
    plt.xlabel('Relative Importance')
    plt.show()
    
def cal_vif(x):
    thresh = 5 
    output= pd. DataFrame() 
    k = x.shape[1] 
    vif = [variance_inflation_factor(x.values, i) for i in range(x.shape[1])] 
    for i in range(1,k):
        # print('Iteration no ',i) 
        # print(vif) 
        a = np.argmax(vif) 
        #print('Max vif is for variable no : ',a) 
        if(vif[a]<=thresh):
            break 
        if(i==1):
            output = x.drop(x.columns[a], axis=1)
            vif = [variance_inflation_factor(output.values, j) for j in range(output.shape[1])] 
        elif(i>1):
            output = output.drop(output.columns[a], axis=1)
            vif = [variance_inflation_factor (output.values,j) for j in range (output.shape[1])] 
    return(output)

if __name__ == "__main__":
    dataset= pd.read_csv(file)
    X= dataset.iloc[:,1:9]