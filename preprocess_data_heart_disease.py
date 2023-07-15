import numpy as np
import pandas as pd

# load data
file_path = '../heart_disease/processed.cleveland.data'
data = np.loadtxt(fname=file_path, dtype=str, delimiter=',')

# remove rows with missing data
processed_data = []
for i in range(data.shape[0]):
    if '?' not in data[i]:
        processed_data.append(data[i])

data_np = np.array(processed_data).astype(float)
for i in range(len(data_np[0])):
    column = data_np[:, i]
    # print(column)
    max_value, min_value = np.max(column), np.min(column)
    # print(max_value, min_value)
    average = (max_value + min_value)/2
    average_log = int(np.log10(average))
    # print(average_log)
    if average_log <= 0:
        continue
    data_np[:, i] = [a / 10**average_log for a in column]

data_temp = []
for i in range(len(data_np)):
    if data_np[i, -1] != 0:
        data_np[i, -1] = 1
#         data_temp.append(data_np[i])
# data_np = np.array(data_temp)

print(data_np)

pd.DataFrame(data_np).to_csv('heart_disease_cleveland.csv', header=None, index=False)
