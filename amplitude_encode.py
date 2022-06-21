import sys
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

identifier = sys.argv[2]
PCA_num_feature = int(sys.argv[1])
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

	if data_set.at[index, "CLASSE"] == "R":
		classe = 1 
	elif data_set.at[index, "CLASSE"] == "W":
		classe = -1
		
	return values, classe

def normalize(array):
    r = array.copy()
    for i, datapoint in enumerate(array):
        soma_sq = sum([x for x in datapoint])
        r[i] /= soma_sq
    return r

def normalize_amp(array):
    r = array.copy()
    for i, datapoint in enumerate(array):
        soma_sq = sum([x**2 for x in datapoint])
        r[i] /= np.sqrt(soma_sq)
    return r

def padding(values, num_qubits):
    if len(values[0]) == num_qubits**2:
        print("PAD não necessário")
    else:
        for datapoint in values:
            datapoint_pad = np.zeros(num_qubits**2)
            for i in range(NUM_FEATURES):
                datapoint_pad[i] = datapoint[i]
            datapoint = datapoint_pad
    return values

def main():
    data_set_features = data_set.loc[:, FEATURES_NAMES]
    classes = data_set.loc[:, ["CLASSE"]].values
    np_data_set_features = data_set_features.to_numpy()

    if PCA_num_feature**2 < 13: 
        norm_data_set_features = normalize(np_data_set_features)
        pca = PCA(n_components=PCA_num_feature**2)
        np_data_set_features = pca.fit_transform(norm_data_set_features)
        
    classes_encode = [-1 if classe == "W" else 1 for classe in classes]
    np_data_set_features_std = normalize_amp(np_data_set_features)
    np_data_set_features_std = padding(np_data_set_features_std, PCA_num_feature)

    np.savetxt(f"Encode_data/amp_enc_data_set_{identifier}_values.csv", np_data_set_features_std, delimiter=";")
    np.savetxt(f"Encode_data/amp_enc_data_set_{identifier}_classes.csv", classes_encode, delimiter=";")
    print(f"Amplitude Enconding guardado em \"Encode_data/amp_enc_data_set_{identifier}_values.csv\"")
    print(f"Amplitude Enconding guardado em \"Encode_data/amp_enc_data_set_{identifier}_classes.csv\"")

if __name__ == '__main__':
	main()
