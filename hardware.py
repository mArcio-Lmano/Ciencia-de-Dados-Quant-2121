####### Imports
from qiskit import QuantumCircuit, ClassicalRegister, QuantumRegister, Aer, execute, transpile, assemble, IBMQ
from qiskit.circuit import Parameter,ParameterVector
from qiskit.visualization import plot_histogram
from qiskit.circuit.library import EfficientSU2
from qiskit.opflow import Z, X, I, StateFn, CircuitStateFn, SummedOp
from qiskit.opflow.gradients import Gradient, NaturalGradient, QFI, Hessian
from qiskit_machine_learning.circuit.library import RawFeatureVector
from sklearn.preprocessing import OneHotEncoder
from qiskit.algorithms.optimizers import SPSA, GradientDescent, QNSPSA, ADAM
from qiskit.tools.monitor import job_monitor
from qiskit_ibm_runtime import QiskitRuntimeService, Sampler


import numpy as np
import itertools as itr
import matplotlib.pyplot as plt
#######

####### Globals
COUNTER = 0
DIR_val_train = "Encode_data/amp_enc_data_set_trainning_values.csv"
DIR_cls_train = "Encode_data/amp_enc_data_set_trainning_classes.csv"

DIR_val_test = "Encode_data/amp_enc_data_set_test_values.csv"
DIR_cls_test = "Encode_data/amp_enc_data_set_test_classes.csv"

#IBMQ.save_account("07916ec631273d08971f67f1267677801b440be43215767571a165abe0ac621415a17cc5a357e2e2fffa511a6fd3748eb0c46d35ca79d8752b97788fd71f390a", overwrite=True)
#IBMQ.load_account()


# Save an IBM Cloud account.
#QiskitRuntimeService.save_account(channel="ibm_cloud", token="MY_IBM_CLOUD_API_KEY", instance="MY_IBM_CLOUD_CRN")

# Save an IBM Quantum account.
#QiskitRuntimeService.save_account(channel="ibm_quantum", token="07916ec631273d08971f67f1267677801b440be43215767571a165abe0ac621415a17cc5a357e2e2fffa511a6fd3748eb0c46d35ca79d8752b97788fd71f390a")
service = QiskitRuntimeService()
print(service.programs())

backend = service.backend("ibmq_lima")
#######

####### Circuito
n_qubits = 4
encoding = RawFeatureVector(16)
ansatz = EfficientSU2(n_qubits, entanglement='full', reps=4, 
						insert_barriers=True, name="U(\u03B8)", 
						parameter_prefix="\u03B8")
#######

####### Funcoes Importantes

def get_encode(file):
    """
     Funcao responsavel pelo encoding (amplitude)
        Input :: 
        ### file : File dir
        Output :: 
        #### data_enc : Valores para o encode (numpy array)
    """
    return np.genfromtxt(file, delimiter=";")

def circuit_parameters(encoding, ansatz, x, thetas):
    """
     Cria um dicionario para dar assign ao valores do encoding e dos paramateros
    a serem otimizados
        Input ::
        ### encoding : Circuito de encoding a ser usado (Com os devidos parametros)
        ### ansatz : Ansatz a ser usado (Com os devidos parametros)
        ### x : Valor dos datapoints para o encode
        ### thetas : Valor dos parametros optimizados a cada treino
        Ouput : 
        #### parameters : Dicionario com os parametros do encoding e anstaz (python dict)
    """
    parameters = {}
    for i, p in enumerate(list(encoding.parameters)):
        parameters[p] = x[i]
    for i, p in enumerate(ansatz.ordered_parameters):
        parameters[p] = thetas[i]
    
    return parameters

def train(qc, encoding, ansatz, train_data, train_labels, lr=None, initial_point=None):
    """
     Funcao de responsavel pelo treino/optimizacao dos parameters das gates do ansatz
	Input : 
	### qc :
	### encoding :
	### ansatz :
	### train_data :
	### train_labels :
	### lr = None :
	### initial_point = None :
	Output : 
	#### opt_var :
	#### opt_value :
	#### parameters :
	#### costs :
	#### evaluations :
    """
    # Callback function for optimiser for plotting purposes
    def store_intermediate_result(evaluation, parameter, cost, stepsize, accept):
        evaluations.append(evaluation)
        parameters.append(parameter)
        costs.append(cost)

    # Set up the optimization

    parameters = []
    costs = []
    evaluations = []

    if lr is not None:
        optimizer = SPSA(maxiter=200 , learning_rate=lr, perturbation=0.01, callback=store_intermediate_result)
    else:
        optimizer = SPSA(maxiter=200, callback=store_intermediate_result)

    if initial_point is not None:
        initial_point = initial_point
    else:      
        initial_point = np.random.random(ansatz.num_parameters)

    objective_function = lambda variational: cost_function(qc,encoding, ansatz,train_data, train_labels, variational)
    # Run the optimization
    opt_var, opt_value, _ = optimizer.optimize(len(initial_point), objective_function, initial_point=initial_point)

    return opt_var, opt_value , parameters, costs, evaluations 

def basis_states_probs(counts, shots=1024, n_qubits=1):
    """
     Retorna as probabilidades de cada estado
	Input : 
	### counts : 
	### shots = 1024 :
	### n_qubits = 1 :
	Output : 
	#### probs : 
    """
    probs = []
    basis_states = [np.binary_repr(i,width=n_qubits) for i in range(2**n_qubits)]

    for b in basis_states:
        c = counts.get(b)
        if c is None:
            probs.append(0)
        else:
            probs.append(counts[b]/shots)
    
    return probs


def classification(counts, shots=1024, label=True):
	
    def count_ones(string):
        r = 0
        for char in string:
            if char == "1":
                r+=1
        return r
    
    def label_assign(exp_val):
        if exp_val >= 0:
            r = 1
        else:
            r = -1  
        return r

    probs = basis_states_probs(counts, n_qubits=n_qubits)
    states = ["".join(seq) for seq in itr.product("01", repeat=n_qubits)]
    
    exp_val = 0
    for state, prob in zip(states, probs):
        exp_val += ((-1)**(count_ones(state)%2))*prob

    if label:   
        #print(exp_val)
        return label_assign(exp_val)
    else:
        return exp_val
	
def cost_function(qc, encoding, ansatz, X, Y, thetas):

    def loss_function(y,y_hat):
        return (y-y_hat)*(y-y_hat)

    #thetas -> variational parameters
    #X dataset
    #Y labels

    #build circuits for each datapoint
    circuits = [qc.assign_parameters(circuit_parameters(encoding, ansatz, x, thetas)) for x in X]
    
	    
    job =  execute(circuits,backend = backend,shots = 1024)
    job_monitor(job)
    results = job.result()
    
    #print(results)
    predictions = [classification(results.get_counts(c)) for c in circuits]
    #print(predictions)
    cost = np.mean(np.array([loss_function(y,y_hat) for (y,y_hat) in zip(Y, predictions)]))
    print(f"Cost {cost}")
    return cost


def main(backend, **kwarg):
    # Trainning data points
    train_data = get_encode(DIR_val_train)
    train_labels = np.genfromtxt(DIR_cls_train, delimiter=";")

    # Test data points
    test_data = get_encode(DIR_val_test)
    test_labels = np.genfromtxt(DIR_cls_test, delimiter=";")

    # train_labels_oh = train_labels
    # test_labels_oh = test_labels

    qc = encoding.compose(ansatz)
    qc.measure_all()

    qcs = []
    for data_point in train_data:
        enc = encoding.assign_parameters(data_point)
        qcs += [enc.compose(ansatz)]

    with Sampler(circuits=qcs, options = {"backend_name": "ibmq_qasm_simulator"}) as sampler:
        results = sampler(circuit_indices=range(0,len(train_data)), shots =1024)
        print(results)

#    program_inputs = {
#        'circuits': ReferenceCircuits.bell(),
#        'circuit_indices': [0]
#        }
#
#    options = {'backend_name': 'i'}
#    job = service.run(
#        program_id="sampler",
#        options=options,
#        inputs=program_inputs)
#

    #opt_var, opt_value , parameters, costs, evaluations = train(qc, encoding, ansatz, train_data, train_labels)
    print(parameters)

if __name__ == "__main__":
    main(backend)
