from sklearn.linear_model import LinearRegression
from load import load_data
import pandas as pd


dpath = 'C:\\Users\\Boundless\\PycharmProjects\\ML\\Project_1_Housing\\datasets\\housing\\prepared_data.csv'
lpath =

pata = load_data(path)
pata_labels = load_data()
print(pata)
print(pata.columns)


lin_reg = LinearRegression()
lin_reg.fit(pata, pata_labels)

