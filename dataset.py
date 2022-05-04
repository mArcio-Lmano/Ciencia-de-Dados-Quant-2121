import sys
import numpy as np
import pandas as pd
from random import sample

# GLOBAL VARIABLES
## Num of data points in each data set
WHITE_LEN = 4898
RED_LEN = 1599
## Completed data sets
white_wine_ds = pd.read_csv("Data_set_completed/winequality-white.csv", delimiter=";") 
red_wine_ds = pd.read_csv("Data_set_completed/winequality-red.csv", delimiter=";")

def sample_ds(num_samples):
    """
    IN ::
	    num_samples -> Numero de samples que queresmo para cada classe do nosso 
	    data set de treino

    OUT ::
	    csv_file_trainning -> Ficheiro csv onde cria um data set de treino
	    csv_file_teste -> Ficheiro csv onde cria um data set de teste
    """
    test_num_samples = round(num_samples*0.3)

    white_range = [x for x in range(0, WHITE_LEN)]
    red_range = [x for x in range(0, RED_LEN)]

    #Indexes for the new trainning and testing data set, last test_num_samples indexes are for the TEST fase
    white_wine_samples = sample(white_range, num_samples+test_num_samples)
    red_wine_samples = 	sample(red_range, num_samples+test_num_samples)

    #Creatting a TRAINING data set for white wine
    new_data_frame_white_trainning = white_wine_ds.iloc[white_wine_samples[:num_samples]]
    new_data_frame_white_trainning.insert(0,"CLASSE", "W", True)

    #Creatting a TEST data set for white wine
    new_data_frame_white_test = white_wine_ds.iloc[white_wine_samples[-test_num_samples:]]
    new_data_frame_white_test.insert(0, "CLASSE", "W", True)

    #Creatting a TRAINING data set for red wine
    new_data_frame_red_trainning = red_wine_ds.iloc[red_wine_samples[:num_samples]]
    new_data_frame_red_trainning.insert(0,"CLASSE", "R", True)

    #Creatting a TEST data set for red wine
    new_data_frame_red_test = red_wine_ds.iloc[red_wine_samples[-test_num_samples:]]
    new_data_frame_red_test.insert(0, "CLASSE", "R", True)

    #Creatting TRAINING data set (WHITE+RED) and shuffling
    trainning_data_set = pd.concat([new_data_frame_white_trainning, new_data_frame_red_trainning])
    trainning_data_set = trainning_data_set.sample(frac=1)

    #Creatting TEST data set (WHITE+RED) and shuffling
    test_data_set = pd.concat([new_data_frame_white_test, new_data_frame_red_test])
    test_data_set = test_data_set.sample(frac=1)

    return trainning_data_set, test_data_set

def save_csv(dataframe, name):
    dataframe.to_csv(name, index=True)

if __name__ == '__main__':
    if sys.argv[1].lower() == "print":
        print(6*"#"+" White Wine "+6*"#"+"\n")
        print(white_wine_ds)
        print("\n")
        print(6*"#"+" Red Wine "+6*"#"+"\n")
        print(red_wine_ds)
    elif sys.argv[1].lower() == "sampler":
        trainning_data_set, test_data_set = sample_ds(int(sys.argv[2]))
        print(6*"#"+" Trainning "+6*"#"+"\n")
        print(trainning_data_set)
        print("\n")
        print(6*"#"+" Testing "+6*"#"+"\n")
        print(test_data_set)
        save_csv(trainning_data_set, "Usable_data_sets/trainning_data_set.csv")
        save_csv(test_data_set, "Usable_data_sets/test_data_set.csv")

    else:
        print("ERRO")
