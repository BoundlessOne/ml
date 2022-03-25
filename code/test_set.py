import numpy as np
import pandas as pd
import os
from data_cleaning import *

def split_train_test(data, test_ratio):
    shuffled_Indices = np.random.permutation(len(data))
    test_Set_size = int(len(data) * test_ratio)
    test_Indices = shuffled_Indices[:test_Set_size]
    train_Indices = shuffled_Indices[test_Set_size:]
    return data.iloc[train_Indices], data.iloc[test_Indices]

split_train_test(load_housing_data())