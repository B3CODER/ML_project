from statistics import correlation
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn import metrics

gold_dataset = pd.read_csv('gld_price_data.csv')
print(gold_dataset.head())
print(gold_dataset.isnull().sum())
print(gold_dataset.describe())
print(gold_dataset.info())

correlation = gold_dataset.corr()
plt.figure(figsize=(10,10))
sns.heatmap(correlation , cbar=True , square= True , fmt='.1f' , annot=True , cmap ='Blues')
plt.show()

print(correlation['GLD'])

sns.distplot(gold_dataset['GLD'] , color= 'green')
plt.show()

X = gold_dataset.drop(['GLD','Date'],axis=1)
Y=gold_dataset['GLD']

print(X)
print(Y)

# Training the dataset
X_train ,X_test ,Y_train,Y_test = train_test_split(X,Y,test_size=0.2,random_state=2)

print(X.shape,X_train.shape,X_test.shape)

# model Training
# Random forest regression 
regressor = RandomForestRegressor(n_estimators=100)

# train the model
regressor.fit(X_train,Y_train)

# model evalution 
test_data_prediction =regressor.predict(X_test)
print(test_data_prediction)

# R squared value
erro_score =metrics.r2_score(Y_test , test_data_prediction)
print(erro_score)

Y_test =list(Y_test)
plt.plot(Y_test , color ='blue' ,label = 'Actual Value')
plt.plot(test_data_prediction , color ='green' ,label = 'Predicted Value')
plt.title('Actual Price vs Predicted Price')
plt.xlabel('Nubers of values')
plt.ylabel('GLD Price')
plt.legend()
plt.show()