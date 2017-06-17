#This code is used to do experiments on the training data, by splitting it into train-development sets.
#Idea is to pick the best performing model-feature combination and use that to predict on test data.

import numpy as np
import pandas as pd
from sklearn import linear_model

#Training and testing files
input_file = "train.csv"

#Read the csv file as a data frame object
df = pd.read_csv(input_file, header = 0)

#Gets the header for the file as a list
original_headers = list(df.columns.values)

#replace NaN values with 0
def removeNaNEtc(nparray):
  nparray[np.isnan(nparray)] = 0
  return nparray

#Just checking if there are columns with infinite values
def checking_columns(numpy_array):
  print("Columns with infinite vals")
  for i in range(0,6):
   if not np.all(np.isfinite(numpy_array[:,i])):
    print(i)
 
  #checking if any column has NaN vals.
  print("Columns with NaN vals")
  for i in range(0,6):
   if not np.all(np.isnan(numpy_array[:,i])):
    print(i)


#Choosing only specific columns instead of everything. 
features_array = [] #2,3,4,5,6,7,8,9
features_array.extend(range(2,11))
#features_array.extend([13,15,16,19,22,23,25,31,31,85,86,87,88,291])
features_array.append(291)
print(features_array)

numpy_array = df.ix[:,features_array].as_matrix()
#print(numpy_array.shape)
#print("Total columns: ", numpy_array.shape[1])

#Remove all NAn values in feature array and replace them with zeroes
numpy_array = removeNaNEtc(numpy_array)

#splitting the data into features and prediction.
print(numpy_array.shape) 
all_data = numpy_array[:,0:-1]
all_preds = numpy_array[:,-1]

#Train-test split
train_data = numpy_array[0:30000,0:-1]
test_data = numpy_array[30001:,0:-1]
train_preds = numpy_array[0:30000,-1]
test_preds = numpy_array[30001:,-1]

#Exploring multiple models:
for i in [linear_model.LinearRegression(), linear_model.Lasso(alpha = 0.1), linear_model.Lasso(alpha = 0.01), 
linear_model.Lasso(alpha = 10), linear_model.Ridge(alpha = 0.1), linear_model.Ridge(alpha = 0.01), linear_model.Ridge(alpha = 10), 
linear_model.BayesianRidge()]:
   regr = i
   regr.fit(train_data,train_preds)
   # learned coefficients
   #print('Coefficients: \n', regr.coef_)
   # MSE
   #print("Mean squared error: %.2f" % np.mean((regr.predict(test_data) - test_preds) ** 2))
   n = len(test_preds)
   print("****", i, "********")
   print("RMSLE: ", np.sqrt((1/n) * sum(np.square(np.log10(regr.predict(test_data) +1) - (np.log10(test_preds) +1)))))
   print("*************")
'''

#Just a few print statements to check stuff
#print(original_headers)
#print(len(original_headers))
#print(len(df))

#Just a few example print statements to access the data frame
#print(df['sport_count_5000'])
#print(df.columns.get_loc('price_doc'))
#print(len(df._get_numeric_data()))

#Converting the data frame to a numpy array.
#numpy_array = df.as_matrix()

'''
