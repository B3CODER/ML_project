import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

features = pd.read_csv('zomato.csv')

features = features.drop(['url', 'address', 'phone', 'menu_item', 'dish_liked', 'reviews_list'], axis = 1)
print(features.head())

print(features.info())
features.drop_duplicates(inplace=True)

print(features['rate'] .unique())

def handlerate(value):
    if(value =='NEW' or value =='-'):
        return np.nan
    else:
        value =str(value).split('/')
        value =value[0]
        return float(value)

features['rate'] = features['rate'].apply(handlerate)
features['rate'].fillna(features['rate'].mean(), inplace=True)
print(features.info())

features = features.drop(['listed_in(city)'], axis = 1)

features.rename(columns = {'approx_cost(for two people)':'approx_cost', 'listed_in(type)':'Type'}, inplace = True)
print(features.head())
print(features.info())

print(features['approx_cost'].unique())

def handle_approxcost(value):
    value = str(value)
    if ',' in value:
        value = value.replace(',','')
        return float(value)
    
    else:
        return float(value)
    
features['approx_cost'] = features['approx_cost'].apply(handle_approxcost)


rest_types = features['rest_type'].value_counts(ascending  = False)
print(rest_types)


rest_types_lessthan1000 = rest_types[rest_types<1000]
def handle_resttype(value):
    if(value in  rest_types_lessthan1000):
        return 'others'
    else:
        return value
features['rest_type'] = features['rest_type'].apply(handle_resttype)
print(features['rest_type'].value_counts())

location = features['location'].value_counts(ascending =False)
location_lessthan300 = location[location<300]

def handle_location(value):
    if(value in location_lessthan300):
        return 'other'
    else:
        return value
features['location'] = features['location'].apply(handle_location)
print(features['location'].value_counts())

cuisines = features['cuisines'].value_counts(ascending =False)
cuisines_lessthan100 = cuisines[cuisines<100]


def handle_cuisines(value):
    if(value in cuisines_lessthan100):
        return 'others'
    else:
        return value
        
features['cuisines'] = features['cuisines'].apply(handle_cuisines)
print(features['cuisines'].value_counts())

print(features.head())

# Data cleaning is done
# plt.figure(figsize = (6,6))
# sns.countplot('online_order' ,data = features)

# plt.figure(figsize = (6,6))
# sns.countplot(features['book_table'], palette = 'rainbow')

# plt.figure(figsize = (10,6))
# sns.countplot(features['location'], palette = 'rainbow')

plt.figure(figsize = (6,6))
sns.boxplot(x = 'book_table', y = 'rate', data = features)

plt.figure(figsize = (6,6))
sns.boxplot(x = 'online_order', y = 'rate', data = features)


df1 = features.groupby(['location','online_order'])['name'].count()
df1.to_csv('location_online.csv')
df1 = pd.read_csv('location_online.csv')
df1 = pd.pivot_table(df1, values=None, index=['location'], columns=['online_order'], fill_value=0, aggfunc=np.sum)
print(df1)


df2 = features.groupby(['location','book_table'])['name'].count()
df2.to_csv('location_booktable.csv')
df2 = pd.read_csv('location_booktable.csv')
df2 = pd.pivot_table(df2, values=None, index=['location'], columns=['book_table'], fill_value=0, aggfunc=np.sum)
print(df2)


plt.figure(figsize =(14,8))
sns.boxplot(x='Type' ,y='rate' , data = features , palette='inferno')

# Visualizing Top Cuisines
df3 = features[['cuisines', 'votes']]
df3.drop_duplicates()
df4= df3.groupby(['cuisines'])['votes'].sum()
df4 = df4.to_frame()
df4 = df4.sort_values('votes', ascending=False)
print(df4.head())


df5 = features.groupby(['location','Type'])['name'].count()
df5.to_csv('location_Type.csv')
df5 = pd.read_csv('location_Type.csv')
df5 = pd.pivot_table(df3, values=None, index=['location'], columns=['Type'], fill_value=0, aggfunc=np.sum)
print(df5)

plt.show()

