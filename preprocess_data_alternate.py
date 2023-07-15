import numpy as np
import pandas as pd

dataset = np.loadtxt(fname='../liver_disorders/bupa.data', delimiter=',')
dataset = dataset[:, :-1]  # remove last column

data_np = np.array(dataset).astype(float)
for i in range(len(data_np[0])):
    column = data_np[:, i]
    # print(column)
    # max_value, min_value = np.max(column), np.min(column)
    # # print(max_value, min_value)
    # average = (max_value + min_value)/2
    average = np.mean(column)
    average_log = int(np.log10(average))
    # print(average_log)
    if average_log <= 0:
        continue
    data_np[:, i] = [a / 10**average_log for a in column]

pd.DataFrame(data_np).to_csv('liver_disorders.csv', header=None, index=False)

print(dataset)
print(data_np)
print(data_np.shape)
