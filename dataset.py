import sys
import numpy as np
import pandas as pd
from random import sample

WHITE_LEN = 4898

RED_LEN = 1599

white_wine_ds = pd.read_csv("winequality-white.csv", delimiter=";") 
red_wine_ds = pd.read_csv("winequality-red.csv", delimiter=";")


def sample_ds(num_samples):
	"""
	IN ::
		num_samples -> Numero de samples que queresmo para cada classe do nosso 
		data set de treino

	OUT ::
		csv_file_training -> Ficheiro csv onde cria um data set de treino
		csv_file_teste -> Ficheiro csv onde cria um data set de teste
	"""
	test_num_samples = round(num_samples*0.3)

	white_range = [x for x in range(0, WHITE_LEN)]
	red_range = [x for x in range(0, RED_LEN)]
	#print(white_range)

	white_wine_samples = sample(white_range, num_samples+test_num_samples)
	white_str = (num_samples+test_num_samples)*"W"

	red_wine_samples = 	sample(red_range, num_samples+test_num_samples)
	red_str = (num_samples+test_num_samples)*"R"

	#columns_names_new_data_set = np.append(np.array(["CLASSE"]), white_wine_ds.columns.to_numpy()) 

	# for sample_index in white_wine_samples :
	new_data_frame = white_wine_ds.iloc[white_wine_samples]
	#print(new_data_frame)
	#print(white_wine_samples)
	new_data_frame = new_data_frame["CLASSE"] = [s for s in 16*"W"]


	#print(columns_names_new_data_set) 



if __name__ == '__main__':
	if sys.argv[1].lower() == "print":
		print("White Wine")
		print(white_wine_ds)
		print(3*"\n")
		print("Red Wine")
		print(red_wine_ds)
	elif sys.argv[1].lower() == "sampler":
		sample_ds(int(sys.argv[2]))


