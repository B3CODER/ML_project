import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 
from sklearn.model_selection import train_test_split
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_percentage_error ,r2_score
from sklearn.metrics import mean_absolute_error ,mean_squared_error 

dataset = pd.read_csv("50_Startups.csv")
# print(dataset.head())
print(dataset.describe())
print(dataset.isnull().sum())

# print(dataset.info())

c= dataset.corr()
sns.heatmap(c,annot =True ,cmap='Blues')
# plt.show()

# Outlier is a lot of data points/values that differ from actual data values.
# Outliers can be detected using a box plot provided by seaborn.

outliers =['Profit']
plt.rcParams['figure.figsize'] = [8,8]

sns.boxplot(data = dataset[outliers] ,orient ="v" , palette = "Set2" ,width =0.6)
plt.title("Outliers Variable Distribution")
plt.ylabel("Profit Range")
plt.xlabel("Continuous Variable")


sns.boxplot(x='State' ,y ='Profit' , data =dataset)
# plt.show()

# this part you have to explore 
sns.distplot(dataset['Profit'],bins=5,kde=True)
# plt.show()


X = dataset.iloc[:, :-2].values
y = dataset.iloc[:, 4].values
# print(X)
labelencoder = LabelEncoder()
X[:,2] =labelencoder.fit_transform(X[:,2])
X1 = pd.DataFrame(X)
X1.head()


x_train,x_test,y_train,y_test = train_test_split(X,y,train_size=0.7,random_state=0) # performs the split

print(X1.shape , x_train.shape ,x_test.shape)

model1 = LinearRegression()
model1.fit(x_train ,y_train)

y_pred = model1.predict(x_test)
print(y_pred)


testing_data_model_score = model1.score(x_test, y_test)
print("Model Score/Performance on Testing data",testing_data_model_score)
 
training_data_model_score = model1.score(x_train, y_train)
print("Model Score/Performance on Training data",training_data_model_score)

# model2 = GradientBoostingRegressor(n_estimators =500 ,learning_rate = 0.3)
# model2.fit(x_train,y_train)

# y_pred1 = model2.predict(x_test)

# accuracy_score =mean_absolute_percentage_error(y_pred1 ,y_test)
# r2 = r2_score(y_pred1 ,y_test)
# print(accuracy_score ,r2)


# model3=RandomForestRegressor(n_estimators=500,max_depth=17)

# model3.fit(x_train,y_train)
# pred=model3.predict(x_test)
# mape=mean_absolute_percentage_error(pred,y_test)
# r2=r2_score(pred,y_test)
# print(mape,r2)


# Evaluating Performance Based On Metrics

MAE =mean_absolute_error(y_pred , y_test)
print("Mean Absolute error " , MAE)
print()

MSE = mean_squared_error(y_pred ,y_test)
print("Mean squarred error is " , MSE*100)
print()

RMSE = np.sqrt(mean_squared_error(y_pred ,y_test))
print("root mean squarred error is " , RMSE*100)
print()

r2Score =r2_score(y_pred ,y_test)
print("R2 Score is  ",  r2Score*100)
