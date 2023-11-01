import torch
import pandas as pd
import torchmetrics.functional.nominal as tfn
import matplotlib.pyplot as plt

# create tensor using with_ms csv
with_ms_pd = pd.read_csv('with_med_specialty_dataset.csv', index_col=0)
with_ms_tn = torch.tensor(with_ms_pd.values)

# rotate tensor (switch rows and columns)
with_ms_tn = torch.rot90(with_ms_tn, 1)

# increase tensor printing size maximum
torch.set_printoptions(profile="full")

#
with_ms_tn_len = len(with_ms_tn)
# print(with_ms_tn[0])
coefficients = torch.zeros(with_ms_tn_len)
for i in range(with_ms_tn_len):
    try:
        coefficients[i] = tfn.pearsons_contingency_coefficient(with_ms_tn[i], with_ms_tn[0])
    except:
        print("error ", i)
print(coefficients)

# print(tfn.pearsons_contingency_coefficient_matrix(with_ms_tn))

# a = 0
# for i in range(len(with_ms_tn[0])):
#     a = a + with_ms_tn[0][i]
#
# print(a)

coefficient_matrix = torch.zeros(with_ms_tn_len, with_ms_tn_len)
for i in range(with_ms_tn_len):
    for j in range(with_ms_tn_len):
        try:
            coefficient_matrix[i][j] = tfn.cramers_v(with_ms_tn[i], with_ms_tn[j])
        except:
            print("error ", i, ", ", j)

print(coefficient_matrix)

# coefficients = torch.flatten(coefficient_matrix)

x = torch.zeros(with_ms_tn_len)
count = 1
for i in range(len(x)):
    x[i] = count
    count += 1

print(with_ms_pd.columns)

plt.matshow(coefficient_matrix.numpy())
plt.show()
