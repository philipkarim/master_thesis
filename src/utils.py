"""
Expressions utilized throughout the scripts
"""
# Common imports
from xml.etree import ElementInclude
import numpy as np
import matplotlib.pyplot as plt
import os
import qiskit as qk
import torch
import sys
from qiskit.quantum_info.operators import Operator, Pauli
import copy
import matplotlib.pyplot as plt
import seaborn as sns

import matplotlib
matplotlib.rcParams['mathtext.fontset'] = 'stix'
matplotlib.rcParams['font.family'] = 'STIXGeneral'

sns.set_style("darkgrid")

FIGWIDTH=4.71935 #From latex document
FIGHEIGHT=FIGWIDTH/1.61803398875

params = {'text.usetex' : True,
          'font.size' : 10,
          'font.family' : 'lmodern',
          'figure.figsize' : [FIGWIDTH, FIGHEIGHT],
          #'text.latex.unicode': True,
          }
plt.rcParams.update(params)


def accuracy_score(y, y_pred):
    """
    Computes the accuracy score
    """
    y=np.ravel(y)
    y_pred=np.ravel(y_pred)
    numerator=np.sum(y == y_pred)

    return numerator/len(y)

def hard_labels(y_array, treshold):
    """
    Rewriting the soft predictions in probabillity space
    into hard predictions, of 0 or 1
    """
    for i in range(len(y_array)):
        y_array[i]=np.where(y_array[i]>treshold, 1, 0)
    
    return y_array

def sigmoid(x):
    """
    Sigmoid function
    """
    return 1 / (1 + np.exp(-x))


def data_path(DATA_ID, dat_id):
    """
    Helper functions
    """
    return os.path.join(DATA_ID, dat_id)

def plotter(*args, x_axis,x_label, y_label):
    """
    Just a function to plot functions.

    Args: *args:    arguments passed as x1 data, y1 data, 
                    label 1, x2 data, y2 data, label 2...
          x_label:  Name of x axis(string)
          y_label:  Name of y axis(string)
    """
    #sns.set_style("darkgrid")
    
    if len(args)>1:
        for i in range(0, int(len(args)),2):
            plt.plot(x_axis, args[i], label=args[i+1])
            plt.legend()
        plt.xlabel(x_label,fontsize=12)
        plt.ylabel(y_label,fontsize=12)
    else:
        plt.plot(x_axis, args[0])
        plt.xlabel(x_label,fontsize=12)
        plt.ylabel(y_label,fontsize=12)
    plt.show()

    return

def run_circuit(qc_circ, statevector_test=True,shots=1024, multiple_circuits=False ,parallel=False, backend="statevector_simulator", histogram=False):
    """
    Function to run circuits including different options like paralell and statevectors

    Args:
            qc_circ(object):        Circuit
            statevector_test(bool): Run statevector?
            shots(int):             Number of simulations if not statevector
            multiple_circuits(bool):Simulate list of circuit
            parallell(bool):        Run in parallell yes or no
            backend(str):           Backend to use
            histogram(bool):        Plot historgram?
    """
    if parallel==True:
        #Based on the following https://quantumcomputing.stackexchange.com/questions/13707/how-many-shots-are-executed-when-using-the-ibmqjobmanager
        qobj_list = [qk.compile(qc, qk.Aer.get_backend(backend)) for qc in qc_circ]
        job_list = [backend.run(qobj) for qobj in qobj_list]

        while job_list:
            for job in job_list:
                if job.status() in JOB_FINAL_STATES:
                    job_list.remove(job)
                    print(job.result().get_counts())

    elif multiple_circuits==True:
        job = qk.execute(qc_circ,
                    backend=qk.Aer.get_backend(backend),
                    shots=shots,
                    #seed_simulator=10,
                    optimization_level=0
                    )
        results = job.result()
        results= results.get_counts()


        prediction=[]
        for qc in range(len(results)):
            prediction_value = 0
            for key,value in results[qc].items():
                #print(key, value)
                if key == '1':
                    prediction_value += value
            prediction_value/=shots
            prediction.append(prediction_value)

    else:
        if statevector_test==False:
            backendtest = qk.Aer.get_backend(backend)            
            job = qk.execute(qc_circ,
                        backend=backendtest,
                        shots=shots,
                        seed_simulator=10,
                        optimization_level=0,
                        max_credits=1000
                        )
            results = job.result()            
            results = results.get_counts(qc_circ)

            if histogram==True:
                qk.visualization.plot_histogram(results)
                plt.show()

            prediction = 0
            
            for key,value in results.items():
                #print(key, value)
                if key == '1':
                    prediction += value
            prediction/=shots
            
            return prediction

        else:
            backendtest = qk.Aer.get_backend(backend)
            result = backendtest.run(qc_circ).result()
            psi=result.get_statevector()
            probs_qubit_0 = psi.probabilities([0])
            
            return probs_qubit_0[1]


class pauli_terms:
    """
    Class for handling Pauligates
    """
    def __init__(self, theta=None, label=None):
        self.theta=theta
        self.label=label
    
    def gate_to_matrix(self):
        if self.label=='rx':
            return self.rx_gate()
        elif self.label=='ry':
            return self.ry_gate()
        elif self.label=='rz':
            return self.rz_gate()
        else:
            return "Something is wrong"

    def rx_gate(self):
        return np.exp(-1j*self.get_X()*self.theta/2)

    def ry_gate(self):
        return np.exp(-1j*self.get_Y()*self.theta/2)

    def rz_gate(self):
        return np.exp(-1j*self.get_Z()*self.theta/2)

    def get_I(self):
        """
        Returns an I matrix
        """
        return np.array([[1,0],[0,1]])

    def get_X(self):
        """
        Returns an X matrix
        """
        return np.array([[0,1], [1,0]])

    def get_Y(self):
        """
        Returns an Y matrix
        """
        return np.array([[0,-1.j], [1.j,0]])

    def get_Z(self):
        """
        Returns an Z matrix
        """
        return np.array([[1,0],[0,-1]])
    
def get_M(unitary_gate):
    """
    Args:
        An unitary gate as a matrix

    Returns: (Matrix) the hermitian matrix M in e^(iM(w))
    """

    return np.log(unitary_gate)/1j

def decomposing_to_pauli(H):
    """
    This functions takes simple hermitian matrices and decomposes them into coefficients of Pauli terms

    Args:
        H(np.array)     Hermitian matrix

    Return: (np.array) size 4 containing coefficients of [I,X,Y,Z] terms

    Example: Works for RX(pi) then it decomposes into x gates, might round the smallest numbers? e-33

    """
    pauli_terms=np.array([[[1,0],[0,1]], [[0,1], [1,0]], [[0,-1.j], [1.j,0]], [[1,0], [0,-1]]])
    decomp=np.zeros(len(pauli_terms), dtype = 'complex')
    
    for i in range(len(pauli_terms)):
        decomp[i]=np.trace(pauli_terms[i]@H)

    return np.real(decomp)/len(H)

def update_parameter_dict(dict, new_parameters):
    """
    Just updating parameters in a dict.
    This function could for sure be optimized
    """
    i=0
    for key in dict:
        dict[key]=new_parameters[i]
        i+=1

    return dict

def encoding_circ(circuit, input_qubits):
    """
    Just defining the quantum circuit encoder in FigS1. in the McArdles supplement article
    """
    data_register_enc = qk.QuantumRegister(2+input_qubits+1)
    classical_register_enc = qk.ClassicalRegister(1)
    
    qc_enc = qk.QuantumCircuit(data_register_enc, classical_register_enc)

    if circuit=='A':
        qc_enc.h(data_register_enc[0])

    elif circuit=='C':
        qc_enc.h(data_register_enc[0])
        qc_enc.s(data_register_enc[0])

    else:
        print('A or C?')
        exit()
    
    return qc_enc



#Just some testing of producing the initialstate
def create_initialstate(gates_params):
    """
    Building the ansatz circuits from the given parameters.

    Args:
         gates_params(nested list): List describing the ansatz   
    """
    qubits=np.amax(np.array(gates_params)[:,2].astype('int'))
    #print(np.max(gates_params[0][1]))
    qr = qk.QuantumRegister(qubits+1)
    circ = qk.QuantumCircuit(qr)
    
    #Looping thorugh the gates in teh ansatz
    for i in range(len(gates_params)):
        getattr(circ, str(gates_params[i][0]))(gates_params[i][1], gates_params[i][2])

    return circ

def update_parameters(old_params, new_params):
    """
    Update parameters describing some circuit with new parameters
    """
    #Looping through gates
    for i in range(len(new_params)):
        if old_params[i][0]=='rx' or old_params[i][0]=='ry' or old_params[i][0]=='rz':
            old_params[i][1]=new_params[i]
    
    return old_params

def getUtilityParameters(ansatz):
    """
    Just doing some small caluclations that helps extracting parameters during computtations
    later on. Finding the largest qubit of needed and the number of rotational gates

    Args:
           ansats(list):    List describing the trial circuit
    """
    rotational_indices=[]
    n_qubits_A=0

    #Looping through parameters
    for i in range(len(ansatz)):
        if ansatz[i][0]=='cx' or ansatz[i][0]=='cy' or ansatz[i][0]=='cz':
            if n_qubits_A<ansatz[i][1]:
                n_qubits_A=ansatz[i][1]
            else:
                pass
        else:
            rotational_indices.append(i)

        if n_qubits_A<ansatz[i][2]:
            n_qubits_A=ansatz[i][2]
            
    trace_indices=list(range(int((n_qubits_A+1)/2), n_qubits_A+1))


    return  trace_indices, rotational_indices


def get_Hamiltonian(n):
    """
    Producing Hamiltonian following a trend based on number of qubits

    Args:
            n(int): Number of qubit Hamiltonian
    """
    hamiltonian_list=[]

    for i in range(n):
        hamiltonian_list.append([[0., 'z', i]])

    for j in range(n):
        for k in range(j):
            if j==k:
                pass
            else:
                hamiltonian_list.append([[0., 'z', j], [0., 'z', k]])

    return np.array(hamiltonian_list, dtype=object)
    

def getAnsatz(anatz_number):
    """
    Just buildning some ansatzes easier based on number of qubits

    Args:
            ansatz_number(int): How many qubits are needed for the ansatz?
    """
    buils_ansatz=[]

    interval=np.arange(anatz_number*2)

    for i in range(anatz_number*2):
        buils_ansatz.append(['ry',0, i])

    for i in range(anatz_number*2):
        buils_ansatz.append(['rz',0, i])

    for i in range(anatz_number*2):
        buils_ansatz.append(['cx', interval[i-1],i])

    for i in range(anatz_number*2):
        buils_ansatz.append(['ry',0, i])

    for i in range(anatz_number*2):
        buils_ansatz.append(['rz',0, i])

    for i in range(anatz_number*2):
        buils_ansatz.append(['cx', interval[i-1] ,interval[i-2]])

    for i in range(anatz_number*2):
        if interval[i-1]<=anatz_number-1:
            ry_value=np.pi*0.5
        else:
            ry_value=0
        
        buils_ansatz.append(['ry',ry_value, interval[i-1]])

    for i in range(anatz_number*2): 
        buils_ansatz.append(['rz',0, interval[i-1]])
    
    for i in range(anatz_number):
        buils_ansatz.append(['cx',i, i+anatz_number])

    return buils_ansatz

def getCircuitMatrix(circuit, not_object=False):
    """
    Gets the unitary matrix of the circuit

    Args: 
        circuit(object):    The circuit which we want to have the unitary matrix of
        not_object(Bool):   Boolean to choose if the circuit is an object or patrameters
                            of the circuit which has to be created first
    Returns:
        (array):    Unitary matrix
    """

    if not_object==False:
        backend = qk.Aer.get_backend('unitary_simulator')

        job = qk.execute(circuit, backend)
        result = job.result()
        print(result.get_unitary(circuit, decimals=3))

    else:
        print('Not implemented yet, but just build H basicly')
        exit()

    return 


def bias_param(x, theta):
    """
    Function which computes the Hamiltonian parameters with supervised fraud dataset
    and datasample as bias

        Args:   
            x(list):        Data sample
            theta(array):   Hamiltonian parameters for 1 parameter

        Return: (float): The dot producted parameter
    """
    x=torch.tensor(x,dtype=torch.float64)
    
    return torch.dot(x, theta)

def compute_NN_nodes(input, output, layers):
    """
    Computes an estimate of the size of hidden layers according
    to the pyramid rule in the book of Timothy Masters
    """
    hidden_layers=np.zeros(layers, dtype='int')
    if layers==1:
        hidden_layers[0]=round(np.sqrt(input*output))

    elif layers==2:
        r=np.cbrt(input/output)
        hidden_layers[0]=round(output*r*r)
        hidden_layers[1]=round(output*r)
    elif layers==3:
        """
        3 layer rule not reliable, this is not from the book
        """
        base=input/(output**3)
        r=np.power(base,(1/4))

        hidden_layers[0]=round(output*r**3)
        hidden_layers[1]=round(output*r**2)
        hidden_layers[2]=round(output*r)
    else:
        sys.exit('Numbers of hidden layers not defined')

    return hidden_layers

def NN_nodes(*argv, act='tanh', sig_last=False):
    layers_nodes=[]

    for arg in argv:
        layers_nodes.append([act])
        layers_nodes.append([arg, 1])
    
    if sig_last:
        layers_nodes.append(['sigmoid'])


    return layers_nodes

def compute_gs_energy(circuit, H_final, time, backend="statevector_simulator", rz_add=True):
    """
    Function to measure and compute the energy using the evolved parameters in VarITE

    Args:
            Circuit(object):    The trial circuit containing the evolved parameters
            H_final(list):      Hamiltonian
            time(float):        Current time which is saved, this is filled automatic during computations
    """
    compute_gs_energy.counter += 1
    if rz_add:
        circ=create_initialstate(circuit[:-1])
    else:
        circ=create_initialstate(circuit)

    #Create the circuit with th eHamiltonian also
    backendtest = qk.Aer.get_backend(backend)

    E_final=0
    #1.40 bond length
    E_exact= -1.137275943080
    #states and their corresponding eigenvalue
    states=['00', 1, '01', -1, '10', -1, '11', 1]
    #print(H_final)

    for h_gate in H_final:
        copy_circ=copy.deepcopy(circ)
        for i in h_gate:
            if i[1]=='x':
                copy_circ.h(i[2])
            
            elif i[1]=='y':
                copy_circ.sdg(i[2])
                copy_circ.h(i[2])

        result = backendtest.run(copy_circ).result()

        if len(h_gate)==2:
            #Here the identity matrix has to be the first one in the list
            if h_gate!=H_final[0]:
                result_dict=result.get_counts(copy_circ)
                temp_E=0
                for state in range(0,int(len(states)),2):
                    temp_E+=(result_dict[states[state]]*states[state+1])
            
                E_final+=h_gate[0][0]*temp_E
            else:
                E_final+=h_gate[0][0]
            
        elif len(h_gate)==1:
            psi=result.get_statevector()
            prob = psi.probabilities([h_gate[0][2]])
            E_final+=h_gate[0][0]*(prob[0]-prob[1])

        else:
            temp_E=0
            result_dict=result.get_counts(copy_circ)

            if h_gate[0][0]==0:
                state_1q=['00', 1, '01', -1]
            else:
                state_1q=['11', 1, '10', -1]

            for state in range(0,int(len(state_1q)),2):
                temp_E+=(result_dict[state_1q[state]]*state_1q[state+1])

            E_final+=h_gate[0][0]*temp_E

    compute_gs_energy.computed_E.append(E_final)
    compute_gs_energy.energy_diff.append(abs(E_exact-E_final))
    compute_gs_energy.time_t.append(time)

    print(f'Iteration: {compute_gs_energy.counter}, Energy: {round(E_final, 4)}, Error: {round(abs((E_exact-E_final)/E_exact)*100, 2)}')
   