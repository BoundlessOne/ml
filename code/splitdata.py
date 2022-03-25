import numpy as np
import pandas as pd
import os
from load import *
from sklearn.model_selection import train_test_split


def split_train_test(data, test_ratio):
    shuffled_Indices = np.random.permutation(len(data))
    test_Set_size = int(len(data) * test_ratio)
    test_Indices = shuffled_Indices[:test_Set_size]
    train_Indices = shuffled_Indices[test_Set_size:]
    return data.iloc[train_Indices], data.iloc[test_Indices]


data = load_housing_data()
#train_set, test_set = split_train_test(data, 0.2) #this method is inconsistent (resampling produces different results)

train_set, test_set = train_test_split(data, test_size=0.2, random_state=42)
#the above is purely random.
#what if we need to ensure representative sampling of a particularly strong variable?
#median income is one such variable. Lets cut it into representative categories

#brief chek
print(len(train_set))
print(len(test_set))
print(len(data))

data["income_cat"] = pd.cut(data["median_income"], bins=[0., 1.5, 3.0, 4.5, 6., np.inf], labels=[1,2,3,4,5])

data.hist(column='income_cat')