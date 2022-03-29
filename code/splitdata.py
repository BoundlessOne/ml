import numpy as np
import pandas as pd
import os
from load import *
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedShuffleSplit

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
#print(len(train_set))
#print(len(test_set))
#print(len(data))

data["income_cat"] = pd.cut(data["median_income"], bins=[0., 1.5, 3.0, 4.5, 6., np.inf], labels=[1, 2, 3, 4, 5])
#print(type(data["income_cat"]))
#print(data["income_cat"], data["median_income"])
#data["income_cat"].value_counts()
#data["income_cat"].hist()
#plt.show()

# stratified sampling based on category, such that no one category is over represented and our model does not overtrain
# certain categories

split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
#print(split)
for train_index, test_index in split.split(data, data["income_cat"]):
    strat_train_set = data.loc[train_index]
    strat_test_set = data.loc[test_index]

#print(strat_test_set["income_cat"].value_counts() / len(strat_test_set))

#the sampling bias comparison of overall, stratified, and random sampling reveals that the stratified sampling meth0d
# yields an order of magnitude less sampling error.

#remove the income_cat to return data to original state

#for set_ in (strat_train_set, strat_test_set):
    set_.drop("income_cat", axis=1, inplace=True)

