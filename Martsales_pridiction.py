import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn import metrics

big_mart_data = pd.read_csv('Big_mart.csv')
print(big_mart_data.head())
# big_mart_data.info()
# big_mart_data.isnull().sum()


# Mean --> average
# Mode --> more repeated value
big_mart_data['Item_Weight'].mean()
big_mart_data['Item_Weight'].fillna(big_mart_data['Item_Weight'].mean(),inplace=True)

big_mart_data['Outlet_Size'].mode()


New_Outlet_size =big_mart_data.pivot_table(values ='Outlet_Size' ,columns='Outlet_Type' , aggfunc =lambda x: x.mode()[0])
print(New_Outlet_size)

miss_values = big_mart_data['Outlet_Size'].isnull() 
print(miss_values)

big_mart_data.loc[miss_values, 'Outlet_Size'] = big_mart_data.loc[miss_values,'Outlet_Type'].apply(lambda x: New_Outlet_size[x])
print(big_mart_data.isnull().sum())

big_mart_data.describe()

plt.figure(figsize=(6,6))
sns.distplot(big_mart_data['Item_Weight'])
# plt.show()

plt.figure(figsize=(6,6))
sns.displot(big_mart_data['Item_Visibility'])
# plt.show()

plt.figure(figsize=(6,6))
sns.displot(big_mart_data['Item_MRP'])
# plt.show()

plt.figure(figsize=(6,6))
sns.displot(big_mart_data['Item_Outlet_Sales'])
# plt.show()

plt.figure(figsize=(6,6))
sns.countplot(x='Outlet_Establishment_Year' ,data =big_mart_data)
# plt.show()

plt.figure(figsize=(30,6))
sns.countplot(x='Item_Type' , data =big_mart_data)
# plt.show()

plt.figure(figsize=(6,6))
sns.countplot(x='Outlet_Size', data=big_mart_data)
# plt.show()


print(big_mart_data.head())

print(big_mart_data['Item_Fat_Content'].value_counts())

big_mart_data.replace({'Item_Fat_Content' : {'low fat' : 'Low Fat','LF' : 'Low Fat','reg': 'Regular'}},inplace=True)

print(big_mart_data['Item_Fat_Content'].value_counts())

encoder = LabelEncoder()
big_mart_data['Item_Identifier'] = encoder.fit_transform(big_mart_data['Item_Identifier'])

big_mart_data['Item_Fat_Content'] = encoder.fit_transform(big_mart_data['Item_Fat_Content'])

big_mart_data['Item_Type'] = encoder.fit_transform(big_mart_data['Item_Type'])

big_mart_data['Outlet_Identifier'] = encoder.fit_transform(big_mart_data['Outlet_Identifier'])

big_mart_data['Outlet_Size'] = encoder.fit_transform(big_mart_data['Outlet_Size'])

big_mart_data['Outlet_Location_Type'] = encoder.fit_transform(big_mart_data['Outlet_Location_Type'])

big_mart_data['Outlet_Type'] = encoder.fit_transform(big_mart_data['Outlet_Type'])

big_mart_data.head()

X = big_mart_data.drop(columns='Item_Outlet_Sales', axis=1)
Y = big_mart_data['Item_Outlet_Sales']

X_train ,Y_train , X_test , Y_test = train_test_split(X,Y, test_size=  0.3 , random_state=2)
print(X.shape , X_train.shape , X_test.shape)


from xgboost import XGBRegressor

regressor = XGBRegressor
regressor.fit(X_train ,Y_train)

trainig_data_prediction = regressor.predict(X_train)
r2_train =metrics.r2_score(Y_train , trainig_data_prediction)
print('R Squared value = ', r2_train)


test_data_prediction = regressor.predict(X_test)
r2_test =metrics.r2_score(Y_test , test_data_prediction)
print('R Squared value = ', r2_test)