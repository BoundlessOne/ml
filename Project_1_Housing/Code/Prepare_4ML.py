import numpy as np
import pandas as pd
import os
from load import *
from splitdata import save_me_csv
from pandas.plotting import scatter_matrix
from sklearn.impute import SimpleImputer

# C:\Users\Boundless\PycharmProjects\ML\Project_1_Housing\datasets\housing\Data_Index_bck

def load_data(File=None):
    if File is None:
        print("No input provided!")
        path_name = str(input("Input Path:"))
        file_name = str(input("Input file name"))
        file_call = os.path.join(path_name, file_name)
        data = pd.read_csv(file_call)
    else:
        print("God is love")
        data = pd.read_csv(File)
    return data

data = load_data("C:\\Users\\Boundless\\PycharmProjects\\ML\\Project_1_Housing\\datasets\\housing\\Data_Index_bck\\Train_Set.csv")
# successful import of stratified training set

# We are about to transform the data : But we don't want to transform the predictors and the labels the same way,
# so we will instead drop the median house value and make a copy so we can explore a bit and compare

d_data = data.drop("median_house_value", axis = 1)
data_labels = data["median_house_value"].copy

# d_data.dropna(subset=["total_bedrooms"]) # Option 1, remove the entire district (row) with a missing bedroom value
# d_data.drop("total_bedrooms", axis=1) # Option 2, nuclear, remove the entire attribute

median = d_data["total_bedrooms"].median() # Option 3, replace missing values with median - step 1
d_data["total_bedrooms"].fillna(median, inplace=True)
imputer = SimpleImputer(strategy="median")
# we will run it on the whole thing to ensure no missing values, but we therefore need to drop non numerical data
d_data_num = d_data.drop("ocean_proximity", axis = 1) # remove entire attribute, no non-numerical data allowed

imputer.fit(d_data_num)

# The Imputer just calculates the median value of each attribute, which we can also do with .median.values -> confirm
# print(imputer.statistics_)
# print(d_data_num.median().values)

X = imputer.transform(d_data_num)
# outputs numpy array of transformed features (must restore dataframe format)
d_tr = pd.DataFrame(X, columns=d_data_num.columns, index=d_data_num.index)
print(d_tr)

