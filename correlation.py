import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import mutual_info_regression, mutual_info_classif
from sklearn.feature_selection import SelectKBest, SelectPercentile
from sklearn.preprocessing import LabelEncoder
from scipy.io import loadmat
# %matplotlib inline

# Load Dataset
dataset= pd.read_csv('cars.csv')
# print(dataset.head())

#This is use to convert Text dat to Numeric Data
car_encoder = LabelEncoder()
origin_encoder = LabelEncoder()

X_car= dataset.iloc[:, 0].values
X_car = car_encoder.fit_transform(X_car)
X_origin=dataset.iloc[:,8].values
X_origin=origin_encoder.fit_transform(X_origin)

#Here MPG is predicting value
X= dataset.iloc[:,1:9]
X['Car'] = X_car
X['Origin']=X_origin
y= dataset.iloc[:,0].values

#Full data consist of data and output(MPG) both
full_data= X.copy()
full_data['MPG']= y
print(full_data.head(2))

#Here we will are Finding correlation of data
importances = full_data.drop("MPG", axis=1).apply(lambda x: x.corr(full_data.MPG))
indices = np.argsort(importances)
print(importances[indices])

#Visualize correlation
names=['weight','displacement','cylinders','horsepower','Origin','acceleration','model year', 'car']
plt.title('Miles Per Gallon')
plt.barh(range(len(indices)), importances[indices], color='g', align='center')
plt.yticks(range(len(indices)), [names[i] for i in indices])
plt.xlabel('Relative Importance')
plt.show()