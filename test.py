import pandas as pd
import numpy as np
from CreateDatasetClass import CreateDataset

diabetes_dataset = pd.read_csv(
    'C:/Users/wayne/Documents/FSU YSP/IRP_Machine Learning/Final Project/dataset_diabetes/diabetic_data.csv')

pd.options.display.max_columns = None  # enable display of all columns

# remove encounter_id, patient_nbr, race, weight, payer_code, diag_1, diag_2, diag_3 columns due to missing data / irrelevance
diabetes_dataset.drop(['encounter_id', 'patient_nbr', 'race', 'weight', 'payer_code', 'diag_1', 'diag_2', 'diag_3'],
                      axis=1, inplace=True)

# convert strings or other data types to integers

for column in diabetes_dataset.columns:
    uniques = diabetes_dataset[column].unique()
    print(uniques)
    dict_unique = dict(enumerate(uniques))
    dict_unique = dict([(value, key) for key, value in dict_unique.items()])
    diabetes_dataset[column] = [dict_unique[value] for value in diabetes_dataset[column]]


# print(diabetes_dataset)
'''
with_med = pd.read_csv('with_med_specialty_dataset.csv')
print(len(with_med['medical_specialty'].unique()))
for value in with_med['medical_specialty'].unique():
    count = with_med['medical_specialty'].value_counts()[value]
    print(f'{value}: {count}')
    if count < 110:
        with_med = with_med[with_med['medical_specialty'] != value]

print(len(with_med['medical_specialty'].unique()))

'''

# print(len(diabetes_dataset['time_in_hospital'].unique()))

# cleve_hd_pd = pd.read_csv('heart_disease_cleveland.csv')
# print(len(cleve_hd_pd[cleve_hd_pd.columns[12]].unique()))


