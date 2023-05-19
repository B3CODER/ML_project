import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import accuracy_score


# Data collection and analysis

diabetes_dataset= pd.read_csv('diabetes.csv')
print(diabetes_dataset.head())

print(diabetes_dataset.describe())

print(diabetes_dataset['Outcome'].value_counts())
print(diabetes_dataset.groupby('Outcome').mean())

X = diabetes_dataset.drop(columns='Outcome',axis=1)
Y= diabetes_dataset['Outcome']

# Data Standarization
standarized_data=StandardScaler().fit_transform(X)
print(standarized_data)
print(standarized_data.std())

X=standarized_data
Y= diabetes_dataset['Outcome']

# train test split

X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.2,random_state=3)
print(X.shape,X_test.shape,X_train.shape)

# Trainng the model

classifier = svm.SVC(kernel='linear')

# training the SVC

classifier.fit(X_train,Y_train)

# model evalution
# Accuracy score
# accuracy score of training data 
X_train_prediction=classifier.predict(X_train)
trainig_data_accuracy_train = accuracy_score(X_train_prediction,Y_train)
print("Accuracy of the traning data ",trainig_data_accuracy_train)

# accuracy score for test data

classifier.fit(X_test,Y_test)

# X_test_prediction =classifier.predict(X_test)
# trainig_data_accuracy_test = accuracy_score(X_test_prediction,Y_test)
# print("Accuracy of the traning data ",trainig_data_accuracy_test)


# making a predictive system

input_data=(3,78,50,32,88,31,0.248,26)

# changing the input data to numpy array
data_as_numpy = np.asarray(input_data)

# reshape the array as we are prdicting for one instance
input_data_reshaped = data_as_numpy.reshape(1,-1)
# standarized the data 
std_data =StandardScaler().fit_transform(input_data_reshaped)
# print(std_data)

prediction = classifier.predict(std_data)
print(prediction)
