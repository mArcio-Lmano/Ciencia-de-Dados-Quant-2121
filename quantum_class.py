from qiskit import QuantumCircuit, ClassicalRegister, QuantumRegister, Aer, execute, transpile
from qiskit.circuit import Parameter,ParameterVector
from qiskit.visualization import plot_histogram
import matplotlib.pyplot as plt
import numpy as np

DIR = "Encode_data/amp_enc_data_set_training.csv"

def get_encode(file, type_ENC):

	"""
	Funcao responsavel pelo encoding
	"""

	if type_ENC.lower() == "amplitude":

		data_ENC = np.genfromtxt(file, delimiter=";")

	else:
		return None

	return data_ENC


def Embedding(x=None, n_qubits=1):

    qc = QuantumCircuit(n_qubits,name="S")

    if x is not None:
        qc.initialize(x)

    return qc

def Ansatz(theta=None, n_qubits=1):

    qc = QuantumCircuit(n_qubits,name="U($\theta$)")

    if theta is not None:
        ### TO DO ###
        pass
    
    return qc

def Measurement(qc, n_qubits=1):

    qc.measure_all()
    ### change me at your will ###
    
    return qc

def main():
	soma = 0
	data = get_encode(DIR, "amplitude")
	data_point = data[0]
	for data_point in data:
		qc = Embedding(data_point, n_qubits=4)
		qc.draw()
	qc = Embedding(data_point, n_qubits=4)
	qc.draw()

if __name__ == '__main__':
	main()s