#load libraries
import pandas as pd
import numpy as np

#load data from the xlsx file
train_data = pd.read_csv('/Users/donglinxiong/Desktop/kagglex/train_clean_data.csv')
test_data = pd.read_csv('/Users/donglinxiong/Desktop/kagglex/test_clean_data.csv')

#inspect the data
print(train_data.head())
print(test_data.head())

# fill test_data price column with 0
test_data['price'] = 0

#check the data types of the columns in both dataset
print(train_data.dtypes)
print(test_data.dtypes)

# remove column in both dataset named 'model_year'
train_data.drop(['model_year'], axis=1, inplace=True)