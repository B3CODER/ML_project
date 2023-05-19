from itertools import count
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

wine_dataset = pd.read_csv('winequality-red.csv')
print(wine_dataset.head())
print(wine_dataset.describe())
print(wine_dataset.shape)
print(wine_dataset.info())

sns.catplot(x='quality' , data = wine_dataset ,kind= 'count')
plt.show()

# plt.figure(figsize=(8,8))
# sns.barplot(x='quality' , y='volatile acidity' ,data= wine_dataset)
# plt.show()
# sns.barplot(x='quality' , y='citric acid' ,data= wine_dataset)
# plt.show()
# sns.barplot(x='quality' , y='volatile acidity' ,data= wine_dataset)
# plt.show()
# sns.barplot(x='quality' , y='alcohol' ,data= wine_dataset)
# plt.show()


correlation = wine_dataset.corr()
# plt.figure(figsize=(9,9))
# sns.heatmap(correlation , cbar =True ,square=True ,fmt='.1f',annot = True , cmap='Blues')
# plt.show()

X= wine_dataset.drop('quality' ,axis=1)
Y=wine_dataset['quality'].apply(lambda y_value : 1 if y_value>=7 else 0)
print(Y)

X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.2,random_state=2)
print(Y.shape,Y_train.shape,Y_test.shape)


model = RandomForestClassifier()
model.fit(X_train,Y_train)
X_test_prediction = model.predict(X_test)
test_data_accuracy = accuracy_score(X_test_prediction ,Y_test)
print(test_data_accuracy)

# building prediction system
input_data = (8.0,0.59,0.16,1.8,0.065,3.0,16.0,0.9962,3.42,0.92,10.5)
data_as_numpy = np.asarray(input_data)
input_reshaped = data_as_numpy.reshape(1,-1)

prediction = model.predict(input_reshaped)
print(prediction)
