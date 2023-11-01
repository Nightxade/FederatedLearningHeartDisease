import pandas as pd

diabetes_dataset = pd.read_csv(
    'C:/Users/wayne/Documents/FSU YSP/IRP_Machine Learning/Final Project/dataset_diabetes/diabetic_data.csv')

pd.options.display.max_columns = None  # enable display of all columns

# remove encounter_id, patient_nbr, race, weight, payer_code, diag_1, diag_2, diag_3 columns due to missing data / irrelevance
diabetes_dataset.drop(['encounter_id', 'patient_nbr', 'race', 'weight', 'payer_code', 'diag_1', 'diag_2', 'diag_3'],
                      axis=1, inplace=True)


# remove medical_specialty values with small frequencies
# for value in diabetes_dataset['medical_specialty'].unique():
#     count = diabetes_dataset['medical_specialty'].value_counts()[value]
#     if count < 110:
#         diabetes_dataset = diabetes_dataset[diabetes_dataset['medical_specialty'] != value]

# convert strings or other data types to integers
for column in diabetes_dataset.columns:
    uniques = diabetes_dataset[column].unique() # get unique values
    # remove columns with all 0's / one unique value
    if len(uniques) == 1:
        diabetes_dataset.drop([column], axis=1, inplace=True)
        continue
    dict_unique = dict(enumerate(uniques)) # number unique values
    dict_unique = dict([(value, key) for key, value in dict_unique.items()])  # swap key and value pairs to make it (unique value, number)
    diabetes_dataset[column] = [dict_unique[value] for value in diabetes_dataset[column]]  # replace values with numbers

# print(diabetes_dataset)


# create dataset with medical_specialty data and one without medical_specialty data
# with_med_specialty_dataset = diabetes_dataset[diabetes_dataset['medical_specialty'] != 1]
# without_med_specialty_dataset = diabetes_dataset[diabetes_dataset['medical_specialty'] == 1]

# remove columns with all 0's / one unique value in each dataset
# for column in with_med_specialty_dataset:
#     if len(with_med_specialty_dataset[column].unique()) == 1:
#         with_med_specialty_dataset.drop([column], axis=1, inplace=True)
#
# for column in without_med_specialty_dataset:
#     if len(without_med_specialty_dataset[column].unique()) == 1:
#         without_med_specialty_dataset.drop([column], axis=1, inplace=True)

# print(without_med_specialty_dataset['medical_specialty'])
# print(with_med_specialty_dataset['medical_specialty'])

# convert datasets to .csv files
# without_med_specialty_dataset.to_csv('without_med_specialty_dataset.csv')
# with_med_specialty_dataset.to_csv('with_med_specialty_dataset.csv')

# convert dataset to .csv file
diabetes_dataset.to_csv('processed_diabetes_dataset.csv')

