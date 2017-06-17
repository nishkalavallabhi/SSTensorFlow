import numpy as np
import pandas as pd
from sklearn import linear_model

input_file = "train.csv/train.csv"

test_file = "test.csv"

#Read the csv file as a data frame object
df = pd.read_csv(input_file, header = 0)

df_test = pd.read_csv(test_file, header = 0)

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
numpy_array = df.ix[:,[2,3,4,5,8,9,291]].as_matrix()
numpy_array_test = df_test.ix[:,[2,3,4,5,8,9]].as_matrix()
#print(numpy_array.shape)
#print("Total columns: ", numpy_array.shape[1])

numpy_array = removeNaNEtc(numpy_array)
numpy_array_test = removeNaNEtc(numpy_array_test)

#splitting the data into features and prediction. 
train_data = numpy_array[:,1:5]
train_preds = numpy_array[:,6]

test_data = numpy_array_test[:,1:5]
#test_preds = numpy_array_test[:,6]

#Regression model
regr = linear_model.LinearRegression()
regr.fit(train_data,train_preds)

# learned coefficients
#print('Coefficients: \n', regr.coef_)
# MSE
#print("Mean squared error: %.2f" % np.mean((regr.predict(test_data) - test_preds) ** 2))

predictions = regr.predict(test_data)
print(len(predictions))
print(len(df_test['id']))

fw = open("output-1.csv","w")
fw.write("id,price_doc")
fw.write("\n")

for i in range(0,len(predictions)):
   fw.write(str(df_test['id'][i]) + "," + str(predictions[i]))
   fw.write("\n")

fw.close()

#n = len(test_preds)
#print("RMSLE: ", np.sqrt((1/n) * sum(np.square(np.log10(regr.predict(test_data) +1) - (np.log10(test_preds) +1)))))

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


#splitting into train and test splits.
train_data = numpy_array[0:30000,0:5]
test_data = numpy_array[30001:,0:5]
train_preds = numpy_array[0:30000,6]
test_preds = numpy_array[30001:,6]
'''
