# load libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# load data
train_data = pd.read_csv('/Users/donglinxiong/Desktop/kagglex/Train.csv')
test_data = pd.read_csv('/Users/donglinxiong/Desktop/kagglex/Test.csv')

# inspect data for missing values and located missing values rows and columns in the dataset
missing_values_train = train_data.isnull().sum()
missing_columns_train = missing_values_train[missing_values_train > 0]

#inspect the missing values in Test data
missing_values_test = test_data.isnull().sum()
missing_columns_test = missing_values_test[missing_values_test > 0]
print("This is missing columns in test dataset:\n", missing_columns_test)
print("This is missing columns in train dataset:\n", missing_columns_train)


# Locate rows with missing values
rows_with_missing_train = train_data[train_data.isnull().any(axis=1)]
rows_with_missing_test = test_data[test_data.isnull().any(axis=1)]


# Display the rows with missing values
print("Rows with missing values:\n", rows_with_missing_train)
print("Rows with missing values:\n", rows_with_missing_test)


# turn column 'model_year' into 'car_age' by subtracting 'model_year' from 2024
train_data['car_age'] = 2024 - train_data['model_year']
test_data['car_age'] = 2024 - test_data['model_year']

# drop 'model_year' column
train_data.drop(['model_year'], axis=1, inplace=True)
test_data.drop(['model_year'], axis=1, inplace=True)

# change test data column price/model_year to price/car_age
test_data.rename(columns={'price/model_year': 'price/car_age'}, inplace=True)

#Add two column in the train data set.
# round the price/car_age and price/mileage to 2 decimal places

# Calculate the new columns
train_data['price/car_age'] = train_data['price'] / train_data['car_age']
train_data['price/mileage'] = train_data['price'] / train_data['milage']

# Round the new columns to two decimal places
train_data['price/car_age'] = train_data['price/car_age'].round(2)
train_data['price/mileage'] = train_data['price/mileage'].round(2)

# Verify the changes
print(train_data[['price/car_age', 'price/mileage']].head())



# change test data column price/milage to price/mileage
test_data.rename(columns={'price/milage': 'price/mileage'}, inplace=True)
test_data.rename(columns={'price/year': 'price/car_age'}, inplace=True)

# Fill missing values in 'price/car_age' with the mean of the column
train_data['price/car_age'].fillna(train_data['price/car_age'].mean(), inplace=True)

# Ensure there are no more missing values in the dataset
print(train_data.isnull().sum())


# save train data to csv file named 'train_clean_data.csv'
train_data.to_csv('/Users/donglinxiong/Desktop/kagglex/train_clean_data.csv', index=False)
test_data.to_csv('/Users/donglinxiong/Desktop/kagglex/test_clean_data.csv', index=False)


# drop column price/car_age and price/mileage
train_data.drop(['price/car_age', 'price/mileage'], axis=1, inplace=True)
test_data.drop(['price/car_age', 'price/mileage'], axis=1, inplace=True)


#Machine Learning Model
# Define the features and target variable
x = train_data[['brand', 'model', 'milage', 'fuel_type', 'engine', 'transmission',
                'accident','car_age']]
y = train_data['price']


# create and train the model
model = LinearRegression()
model.fit(x, y)

# make predictions
predictions = model.predict(test_data[['brand', 'model', 'milage', 'fuel_type', 'engine', 'transmission',
                                       'accident','car_age']])
# save the predictions to csv file
predictions_df = pd.DataFrame(predictions, columns=['price'])
predictions_df.to_csv('/Users/donglinxiong/Desktop/kagglex/predictions.csv', index=False)

# narrow down the features to 'brand', 'model', 'milage', 'fuel_type', 'engine', 'transmission', 'accident','car_age'

