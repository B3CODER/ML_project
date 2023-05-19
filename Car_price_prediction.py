import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso
from sklearn import metrics

from Diabetes import X_test, X_train, Y_test, Y_train

car_dataset = pd.read_csv('car data.csv')
print(car_dataset.head())
print(car_dataset.shape)
print(car_dataset.isnull().sum())
print(car_dataset.info())
print(car_dataset.describe())

print(car_dataset.Fuel_Type.value_counts())
print(car_dataset.Seller_Type.value_counts())
print(car_dataset.Transmission.value_counts())

car_dataset.replace({'Fuel_Type' :{'Petrol' : 0 ,'Diesel' : 1 , 'CNG' : 2}},inplace=True)
car_dataset.replace({'Seller_Type' :{'Dealer' : 0 ,'Individual' : 1 }},inplace=True)
car_dataset.replace({'Transmission' :{'Manual' : 0 ,'Automatic' : 1 }},inplace=True)

print(car_dataset.head())

# slplitting the data into training data and set data
X=car_dataset.drop(['Car_Name','Selling_Price'],axis=1)
Y=car_dataset['Selling_Price']

print(X)
print(Y)

X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.1,random_state=2)

#Model Training 
# linear regression model

lin_reg_model = LinearRegression()
lin_reg_model.fit(X_train,Y_train)

# model evalution 
training_data_prediction = lin_reg_model.predict(X_train)
error_score = metrics.r2_score(Y_train,training_data_prediction)

print("R squared value : ",error_score)

# visualize the actual prices and Predicted prices
# plt.scatter(Y_train ,training_data_prediction)
# plt.show()

test_data_prediction = lin_reg_model.predict(X_test)
error_score = metrics.r2_score(Y_test,test_data_prediction)

print("R squared value : ",error_score)

plt.scatter(Y_test ,test_data_prediction)
plt.show()

# lesso regression
lasso_reg_model = Lasso()
lasso_reg_model.fit(X_train,Y_train)

# model evalution 
training_data_prediction = lasso_reg_model.predict(X_train)
error_score = metrics.r2_score(Y_train,training_data_prediction)

print("R squared value : ",error_score)
plt.scatter(Y_train ,training_data_prediction)
plt.show()

test_data_prediction = lasso_reg_model.predict(X_test)
error_score = metrics.r2_score(Y_test,test_data_prediction)

print("R squared value : ",error_score)

plt.scatter(Y_test ,test_data_prediction)
plt.show()


