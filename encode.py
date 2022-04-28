import sys
import numpy as np
import pandas as pd

data_set = pd.read_csv("Usable_data_sets/training_data_set.csv")
data_set.rename(columns={list(data_set)[0]:"OldIndex"}, inplace=True)
FEATURES_NAMES = list(data_set)[2:]
COUNTER = 0
NUM_FEATURES = 12

def get_values(index):

	values = np.empty(NUM_FEATURES)

	for i, feature in zip(range(NUM_FEATURES), FEATURES_NAMES):
		values[i] = data_set.at[index, feature]

	return values

def normalize(lst_values):

	soma = sum(lst_values)
	normalize_values = np.array(lst_values)
	normalize_values /= soma

	return normalize_values

def padding(values, num_qubits):
	print(num_qubits**2)
	if len(values) == num_qubits**2:
		print("PAD não necessário")
	else:
		values_pad = np.zeros(num_qubits**2)
		for i in range(NUM_FEATURES):
			values_pad[i] = values[i]

	return values_pad

def main():
	values = get_values(0)
	values = normalize(values)
	values = padding(values, 4)

if __name__ == '__main__':
	main()