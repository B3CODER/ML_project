import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics
import matplotlib.pyplot as plt

insurance_data = pd.read_csv('insurance.csv')
print(insurance_data.head())
print(insurance_data.isnull().sum())

sns.set()
plt.figure(figsize=(6,6))
sns.displot(insurance_data['age'])
plt.title('Age Distribution')
plt.show()


plt.figure(figsize=(6,6))
sns.countplot(x='sex', data=insurance_data)
plt.title('Sex Distribution')
plt.show()

# bmi distribution
plt.figure(figsize=(6,6))
sns.distplot(insurance_data['bmi'])
plt.title('BMI Distribution')
plt.show()

plt.figure(figsize=(6,6))
sns.countplot(x='children' , data =insurance_data)
plt.title('Children')
plt.show()

plt.figure(figsize=(6,6))
sns.countplot(x='smoker', data=insurance_data)
plt.title('smoker')
plt.show()

# Encoding the categorical features
insurance_data.replace({'sex':{'male':0,'female':1}} , inplace=True)
insurance_data.replace({'smoker':{'yes':0 , 'no': 1}}, inplace=True)
insurance_data.replace({'region':{'southeast':0 , 'southwest':1 ,'northeast':2 , 'northwest':3}} , inplace=True)
X =insurance_data.drop(columns='charges' , axis =1)
Y=insurance_data['charges']

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=2)
print(X.shape, X_train.shape, X_test.shape)

regressor =LinearRegression()
regressor.fit(X_train ,Y_train)

training_data_prediction = regressor.predict(X_train)
r2_train = metrics.r2_score(Y_train, training_data_prediction)
print('R squared vale : ', r2_train)

test_data_prediction =regressor.predict(X_test)
r2_test = metrics.r2_score(Y_test, test_data_prediction)
print('R squared vale : ', r2_test)

input_data = (31,1,25.74,0,1,0)

# changing input_data to a numpy array
input_data_as_numpy_array = np.asarray(input_data)

# reshape the array
input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)

prediction = regressor.predict(input_data_reshaped)
print(prediction)

print('The insurance cost is USD ', prediction[0])