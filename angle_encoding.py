import sys
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


identifier = sys.argv[2]
PCA_num_features = int(sys.argv[1])
data_set = pd.read_csv(f"Usable_data_sets/{identifier}_data_set.csv")
data_set.rename(columns={list(data_set)[0]:"OldIndex"}, inplace=True)
FEATURES_NAMES = list(data_set)[2:]
COUNTER = 0
NUM_FEATURES = 12
NUM_SAMPLES = len(data_set)

def normalize_ang(array):
    r = array.copy()
    for i, elem in enumerate(array):
        maxi = max([abs(x) for x in elem])
        for j, num in enumerate(elem):
            r[i][j] = num * np.pi / maxi
    return r            

def normalize(array):
    r = array.copy()
    for i, datapoint in enumerate(array):
        soma_sq = sum([x for x in datapoint])
        r[i] /= soma_sq
    return r

def main():

    data_set_features = data_set.loc[:, FEATURES_NAMES]
    classes = data_set.loc[:,["CLASSE"]].values

    np_data_set_features = data_set_features.to_numpy() 

    if PCA_num_features != 12: 
        np_data_set_features = normalize(np_data_set_features)
        #print(np_data_set_features)
        pca = PCA(n_components=PCA_num_features)
        np_data_set_features = pca.fit_transform(np_data_set_features)

    classes_encode = [-1 if classe == "W" else 1 for classe in classes]
    np_data_set_features_std = normalize_ang(np_data_set_features)
    
    np.savetxt(f"Encode_data/ang_enc_data_set_{identifier}_values.csv", np_data_set_features_std, delimiter=";")
    np.savetxt(f"Encode_data/ang_enc_data_set_{identifier}_classes.csv", classes_encode, delimiter=";")
    print(f"Amplitude Enconding guardado em \"Encode_data/ang_enc_data_set_{identifier}_values.csv\"")
    print(f"Amplitude Enconding guardado em \"Encode_data/ang_enc_data_set_{identifier}_classes.csv\"")        

if __name__ == '__main__':
    main()

