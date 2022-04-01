import numpy as np
import pandas as pd
import os
from load import *
from splitdata import save_me_csv
from pandas.plotting import scatter_matrix

def load_data():
    path_name = str(input("Input Path:"))
    file_name = str(input("Input file name"))
    file_call = os.path.join(path_name, file_name)
    data = pd.read_csv(file_call)
    return data

# data = load_data()
# successful import of stratified training set

# C:\Users\Boundless\PycharmProjects\ML\Project_1_Housing\datasets\housing\Data_Index_bck