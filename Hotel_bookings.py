# -*- coding: utf-8 -*-
"""
Created on Wed Jun 10 11:56:57 2020

@author: user
"""


#Importing libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split
import seaborn as sns

#Reading Dataset and sorting it via index
dataset = pd.read_csv('hotel_bookings.csv')
dataset.sort_index(axis = 0)
#Checking if there are incomplete cells in the data 
dataset.isnull().sum()

replacements = {'children': 0.0, 'country': 'Unknown', 'agent': 0.0, 'company': 0.0}

dataset = dataset.fillna(replacements)

dataset['meal'].replace(to_replace = 'Undefined', value = 'SC', inplace = True)

no_guests = list(dataset.loc[dataset['adults'] + dataset['children']+ dataset['babies'] == 0].index)
dataset.drop(dataset.index[no_guests], inplace = True)

# Separating the resort and city hotel information
resort = dataset.loc[(dataset['hotel'] == 'Resort Hotel')]
city = dataset.loc[(dataset['hotel'] == 'City Hotel')]

resort_per_month = resort.groupby('arrival_date_month')['hotel'].count()
resort_cancel_per_month = resort.groupby('arrival_date_month')['is_canceled'].sum()

city_per_month = city.groupby('arrival_date_month')['hotel'].count()
city_cancel_per_month = city.groupby('arrival_date_month')['is_canceled'].sum()

resort_cancel_data = pd.DataFrame({'Hotel': 'Resort Hotel',
                                'Month': list(resort_per_month.index),
                                'Bookings': list(resort_per_month.values),
                                'Cancellations': list(resort_cancel_per_month.values)})

city_cancel_data = pd.DataFrame({'Hotel': 'City Hotel',
                                'Month': list(city_per_month.index),
                                'Bookings': list(city_per_month.values),
                                'Cancellations': list(city_cancel_per_month.values)})

full_cancel_data = pd.concat([resort_cancel_data, city_cancel_data], ignore_index = True)
full_cancel_data['cancel_percent'] = full_cancel_data['Cancellations'] / full_cancel_data['Bookings'] * 100

ordered_months = ['January', 'February', 'March', 'April', 'May', 'June', 
          'July', 'August', 'September', 'October', 'November', 'December']
full_cancel_data['Month'] = pd.Categorical(full_cancel_data['Month'], categories = ordered_months, ordered = True)

# show figure:
plt.figure(figsize=(12, 8))
sns.barplot(x = 'Month', y = 'cancel_percent' , hue = 'Hotel',
            hue_order = ['City Hotel', 'Resort Hotel'], data = full_cancel_data)
plt.title('Cancellations per month', fontsize = 16)
plt.xlabel('Month', fontsize = 16)
plt.xticks(rotation=45)
plt.ylabel('Cancelations [%]', fontsize = 16)
plt.legend(loc = 'upper right')
plt.show()

# Finding the percentage of cancellations
resort_cancellations = resort['is_canceled'].sum()
resort_percentage_cancel = round(resort_cancellations / resort.shape[0] * 100)

city_cancellations = city['is_canceled'].sum()
city_percentage_cancel = round(city_cancellations / city.shape[0] * 100)


#Finding average price per person in each hotel
resort_avg_price = round((resort['adr']/resort['adults']).mean())
city_avg_price = round((city['adr']/(city['adults'] + city['children'])).mean())


dataset.drop(columns = ['arrival_date_year', 'assigned_room_type', 'booking_changes', 'reservation_status', 'country', 'reservation_status_date', 'days_in_waiting_list'], axis = 1, inplace = True)

#Seperating numerical variables and categorical variables
numerical_features = list(dataset.select_dtypes(exclude = [object]))
categorical_features = list(dataset.select_dtypes(include = [object]))
numerical_features.remove('is_canceled')

#Changing the dataset into dependent and independent variables
X = dataset.drop(columns = ['is_canceled'], axis = 1)#Independent vector
print(X)
y = dataset.iloc[:,1].values#Dependent vector
print(y)

#Pre processing the data
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
numeric_transformer = SimpleImputer(strategy = 'constant')
categoric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy = 'constant', fill_value = 'Unknown')),
    ('onehot', OneHotEncoder(handle_unknown = 'ignore'))])
preprocessor = ColumnTransformer(transformers = [('numeric', numeric_transformer, numerical_features),
                                               ('categorical', categoric_transformer, categorical_features)])

X = preprocessor.fit_transform(X)


#Splitting the dataset into training and testing dataset
X_train, X_test, Y_train, Y_test = train_test_split(X,y, test_size = 0.2, random_state = 0)

from sklearn.ensemble import RandomForestClassifier
classifier  = RandomForestClassifier(n_estimators = 300, n_jobs = -1, random_state = 0)
classifier.fit(X_train, Y_train)

y_pred = classifier.predict(X_test)

#Finding the confusion matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(Y_test, y_pred)


from sklearn.metrics import accuracy_score
accuracy_score(Y_test, y_pred)

sns.heatmap(cm, annot = True, linewidths = 0.5, square = True, cmap = 'YlGnBu', fmt = '.0f')
plt.ylabel('Actual label')
plt.xlabel('Predicted label')
plt.title('Accuracy Score: {0}'.format(round(accuracy_score(Y_test, y_pred) * 100, 2)))


# Evaluate the performance of the classifier
from sklearn.metrics import accuracy_score, mean_squared_error, classification_report
accuracy = round(accuracy_score(Y_test, y_pred) * 100, 2)
mse = mean_squared_error(Y_test, y_pred)
rmse = np.sqrt(mean_squared_error(Y_test, y_pred))
report = classification_report(Y_test, y_pred)

print('The accuracy of the classifier is {0}%'.format(accuracy))
print('\nThe calculated RMSE is {0}'.format(rmse))
print('\nThe classification report is as follows:\n')
print(report)

from scipy.stats import sem, t
from scipy import mean
confidence = 0.95


"""from matplotlib.colors import ListedColormap
X_set, y_set = X_train, y_train
X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.01),
                     np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.01))
plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
             alpha = 0.75, cmap = ListedColormap(('red', 'green')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                c = ListedColormap(('black', 'yellow'))(i), label = j)
plt.title('Logistic Regression (Training set)')
plt.xlabel('Age')
plt.ylabel('Estimated Salary')
plt.legend()
plt.show()

X_set, y_set = X_train, y_train
X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.01),
                     np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.01))
plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
             alpha = 0.75, cmap = ListedColormap(('red', 'green')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                c = ListedColormap(('black', 'yellow'))(i), label = j)
plt.title('Logistic Regression (Training set)')
plt.xlabel('Age')
plt.ylabel('Estimated Salary')
plt.legend()
plt.show()"""