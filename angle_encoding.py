import sys
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


identifier = sys.argv[1]
PCA_num_features = int(sys.argv[2])
data_set = pd.read_csv(f"Usable_data_sets/{identifier}_data_set.csv")
data_set.rename(columns={list(data_set)[0]:"OldIndex"}, inplace=True)
FEATURES_NAMES = list(data_set)[2:]
COUNTER = 0
NUM_FEATURES = 12
NUM_SAMPLES = len(data_set)

def normalize(array):
    r = array.copy()
    for i, elem in enumerate(array):
        maxi = max(elem)
        for j, num in enumerate(elem):
            r[i][j] = num * np.pi / maxi

    return r            
def main():

    data_set_features = data_set.loc[:, FEATURES_NAMES]
    classes = data_set.loc[:,["CLASSE"]].values

    #data_set_features_std = StandardScaler().fit_transform(data_set_features)
    rdc_data_set_features = data_set_features.to_numpy() 
    #if PCA_num_features != 12:

    print(PCA_num_features)
    if PCA_num_features != 12: 
        pca = PCA(n_components=PCA_num_features)
        rdc_data_set_features = pca.fit_transform(data_set_features)
    print(rdc_data_set_features)
    rdc_data_set_features_std = normalize(rdc_data_set_features)
    classes_encode = [-1 if classe == "W" else 1 for classe in classes]

    np.savetxt(f"Encode_data/ang_enc_data_set_{identifier}_values.csv", rdc_data_set_features_std, delimiter=";")
    np.savetxt(f"Encode_data/ang_enc_data_set_{identifier}_classes.csv", classes_encode, delimiter=";")
    print(f"Amplitude Enconding guardado em \"Encode_data/ang_enc_data_set_{identifier}_values.csv\"")
    print(f"Amplitude Enconding guardado em \"Encode_data/ang_enc_data_set_{identifier}_classes.csv\"")        

if __name__ == '__main__':
    main()

