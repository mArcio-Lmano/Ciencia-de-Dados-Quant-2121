import sys
import numpy as np
import pandas as pd

identifier = sys.argv[1]
data_set = pd.read_csv(f"Usable_data_sets/{identifier}_data_set.csv")
data_set.rename(columns={list(data_set)[0]:"OldIndex"}, inplace=True)
FEATURES_NAMES = list(data_set)[2:]
COUNTER = 0
NUM_FEATURES = 12
NUM_SAMPLES = len(data_set)

def get_values(index):

	values = np.empty(NUM_FEATURES)

	for i, feature in zip(range(NUM_FEATURES), FEATURES_NAMES):
		values[i] = data_set.at[index, feature]

	return values

def normalize(lst_values):

	soma_sq = sum([x**2 for x in lst_values])
	normalize_values = np.array(lst_values)
	normalize_values /= np.sqrt(soma_sq)

	return normalize_values

def padding(values, num_qubits):
	if len(values) == num_qubits**2:
		print("PAD não necessário")
	else:
		values_pad = np.zeros(num_qubits**2)
		for i in range(NUM_FEATURES):
			values_pad[i] = values[i]

	return values_pad

def main():

    global COUNTER
    norm_data_set = np.empty([NUM_SAMPLES, 4**2])
    while COUNTER < NUM_SAMPLES:
        values = get_values(0)
        values = normalize(values)
        values = padding(values, 4)
        norm_data_set[COUNTER] = values
        COUNTER += 1

    np.savetxt(f"Encode_data/amp_enc_data_set_{identifier}.csv", norm_data_set, delimiter=";")
    print(f"Amplitude Enconding guardado em \"Encode_data/amp_enc_data_set_{identifier}.csv\"")

if __name__ == '__main__':
	main()
