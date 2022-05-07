"""
Expressions utilized throughout the scripts
"""
# Common imports
import numpy as np
import matplotlib.pyplot as plt
import os
import qiskit as qk
import torch
import sys
from qiskit.quantum_info import DensityMatrix, partial_trace
from qiskit.quantum_info.operators import Operator, Pauli


#from qiskit.compiler import assemble
#from qiskit.backends.jobstatus import JOB_FINAL_STATES

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
    return 1 / (1 + np.exp(-x))

def getDistribution(type, stop, n):
    if type=="U":
        return np.random.uniform(0.,stop,size=n)
    elif type=="N":
        return np.random.normal(stop/2,stop,size=n)
    else:
        print("Choose U for uniform distribution or N for normal distribution. \n Shutting down")
        quit()

def data_path(DATA_ID, dat_id):
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

def run_circuit(qc_circ, statevector_test=True,shots=1024, multiple_circuits=False ,parallel=False, backend="statevector_simulator", histogram=False): #backend="qasm_simulator"
    
    if parallel==True:
        #Read at this jobmanager thing https://quantumcomputing.stackexchange.com/questions/13707/how-many-shots-are-executed-when-using-the-ibmqjobmanager
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
            #simulator = qk.AerSimulator(method='matrix_product_state') 
            
            #backendtest.set_options(max_parallel_experiments=0)
            #backendtest.set_options(statevector_parallel_threshold=1)
            #backendtest.set_options(device='CPU')
            
            job = qk.execute(qc_circ,
                        backend=backendtest,
                        shots=shots,
                        seed_simulator=10,
                        optimization_level=0,
                        max_credits=1000
                        )
            #Run or execute?
            results = job.result()
            
            results = results.get_counts(qc_circ)
            #print(results)


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
            #backendtest.set_options(device='CPU')

            #backend = qk.Aer.get_backend(method='matrix_product_state') 
            
            #backendtest.set_options(max_parallel_experiments=0)
            #backendtest.set_options(statevector_parallel_threshold=1)
            #backendtest.set_options(device='CPU')
            
            """
            job = qk.execute(qc_circ,
                        backend=backendtest,#qk.Aer.get_backend(backend),
                        shots=0,
                        optimization_level=0)
            result = job.result()
            """
            #Run or execute?
            result = backendtest.run(qc_circ).result()

            psi=result.get_statevector()
            probs_qubit_0 = psi.probabilities([0])
            
            
            return probs_qubit_0[1]

    

class pauli_terms:
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

        """
        Continue with controlled gates, if no better method is found
        """

    def rx_gate(self):
        #cos_term=np.cos(self.theta/2)
        #sin_term=np.sin(self.theta/2)
        #return np.array([[cos_term, -1j*sin_term], [-1j*sin_term ,cos_term]])
        return np.exp(-1j*self.get_X()*self.theta/2)

    def ry_gate(self):
        #cos_term=np.cos(self.theta/2)
        #sin_term=np.sin(self.theta/2)
        #return np.array([[cos_term, -sin_term], [sin_term ,cos_term]])
        return np.exp(-1j*self.get_Y()*self.theta/2)

    def rz_gate(self):
        return np.exp(-1j*self.get_Z()*self.theta/2)
        #return np.array([[np.exp(-1j*self.theta/2), 0], [0, np.exp(1j*self.theta/2)]])

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
    DOES NOT WORK FOR CONTROLLED QUBITS
    https://quantumcomputing.stackexchange.com/questions/8725/can-arbitrary-matrices-be-decomposed-using-the-pauli-basis

    This functions takes hermitian matrices and decomposes them into coefficients of Pauli terms

    Args:
        H(np.array)     Hermitian matrix

    Return: (np.array) size 4 containing coefficients of [I,X,Y,Z] terms

    Example: Works for RX(pi) then it decomposes into x gates, might round the smallest numbers? e-33

    """
    pauli_terms=np.array([[[1,0],[0,1]], [[0,1], [1,0]], [[0,-1.j], [1.j,0]], [[1,0], [0,-1]]])

    #Check that matrix H is hermitian:
    #assert (H.conj().T == H).all(), "How should I say it.. Well, your Hermitian matrix is not really.. Hermitian"

    decomp=np.zeros(len(pauli_terms), dtype = 'complex')
    for i in range(len(pauli_terms)):
        decomp[i]=np.trace(pauli_terms[i]@H)

    #decomp = decomp.astype(np.float)

    #Just normalizes it
    #Should I do it?
    #return np.real(decomp/np.sum(decomp))

    #why divide by 2?
    return np.real(decomp)/len(H)

def get_f_sigma(label):
    """
    Args:
        label of the gate

    returns:f, sigma(2 lists)
    """
    if label=='rx':
        return np.array([-0.5j,0,0])
    elif label=='ry':
        return np.array([0,-0.5j,0])
    elif label=='rz':
        return np.array([0,0,-0.5j])
    else:
        print("Something is wrong.. exiting")
        exit()
        return np.array([0,0,0,0])

        #print("Only rx,ry and rz gates are implemented")
        #exit()
    """
    Here it is possible to implement for arbitrary hermitian matrices by
    using the following approach under
    """
    #pauli_matrix=pauli_terms(gates_list[i][1], gates_list[i][0])
    #rx_as_matrix=pauli_matrix.gate_to_matrix()
    #mat_M=get_M(rx_as_matrix)
    #decomp_pauli_terms=decomposing_to_pauli(mat_M)

    return 



def update_parameter_dict(dict, new_parameters):
    """
    Just updating the parameters.
    This function could for sure be optimized
    """
    i=0
    for key in dict:
        dict[key]=new_parameters[i]
        i+=1
    return dict

def encoding_circ(circuit, input_qubits):
    """
    Just defining the quantum circuit encoder in FidS1. in the supplememnt
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
    
    #qc_enc.z(data_register_enc[-1])

    return qc_enc



#Just some testing of producing the initialstate
def create_initialstate(gates_params):
    #param_fig2=[['ry',0, 0],['ry',0, 1], ['cx', 0,1], ['cx', 1, 0], ['cx', 0, 1]]
    #gates_params=np.array(gates_params)
    #Creating the circuit
    qubits=np.amax(np.array(gates_params)[:,2].astype('int'))
    #print(np.max(gates_params[0][1]))
    qr = qk.QuantumRegister(qubits+1)
    #cr = qk.ClassicalRegister(1)

    circ = qk.QuantumCircuit(qr)
    
    #print(gates_params)
    #print(circ)

    #dict with gate in them, maybe as functions?

    #gates_params=list(gates_params)
    for i in range(len(gates_params)):
        getattr(circ, str(gates_params[i][0]))(gates_params[i][1], gates_params[i][2])
    #circ.measure_all()
    #circ.measure(0, cr)
    #print(run_circuit(circ, shots=1024, histogram=True))

    #print(circ)
    return circ


def remove_Nans(A, C):
    """
    Just removing the vectors and nan elements corresponding 
    to gates not dependent of theta
    """
    indexes=[]
    for nan in range(0,len(C)):
        if np.isnan(C[nan])==True:
            indexes.append(nan)

    #Remove nans from C
    C=np.delete(C, indexes)

    #Remove vectors corresponding to C's nans in A
    A=np.delete(A, indexes, axis=0)
    A=np.delete(A, indexes, axis=1)

    return A, C

def remove_constant_gates(V_circ, A, C):
    """
    Just removing the vectors and nan elements corresponding 
    to gates not dependent of theta
    """
    indexes=[]
    for i in range(0,len(C)):
        if V_circ[i][0]=='rx' or V_circ[i][0]=='ry' or V_circ[i][0]=='rz':
            pass
        else:
            indexes.append(i)

    #Remove nans from C
    C=np.delete(C, indexes)

    #Remove vectors corresponding to C's nans in A
    A=np.delete(A, indexes, axis=0)
    A=np.delete(A, indexes, axis=1)

    return A, C

def update_parameters(old_params, new_params):
    for i in range(len(new_params)):
        if old_params[i][0]=='rx' or old_params[i][0]=='ry' or old_params[i][0]=='rz':
            old_params[i][1]=new_params[i]
    
    return old_params

def getUtilityParameters(ansatz):
    """
    Just doing some small caluclations that will show themselves usefull
    later on
    """
    rotational_indices=[]
    n_qubits_A=0

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


def fidelity_own(A, B, validate=True):
    """
    Calculates the fidelity (pseudo-metric) between two density matrices.
    See: Nielsen & Chuang, "Quantum Computation and Quantum Information"

    Parameters
    ----------
    A : qobj
        Density matrix or state vector.
    B : qobj
        Density matrix or state vector with same dimensions as A.

    Returns
    -------
    fid : float
        Fidelity pseudo-metric between A and B.

    Examples
    --------
    >>> x = fock_dm(5,3)
    >>> y = coherent_dm(5,1)
    >>> fidelity(x,y)
    0.24104350624628332

    """
    #print(A)
    #print(B)
    sqrtmA = np.sqrt(A)

    #if sqrtmA.dims != B.dims:
    #    raise TypeError('Density matrices do not have same dimensions.')

    # We don't actually need the whole matrix here, just the trace
    # of its square root, so let's just get its eigenenergies instead.
    # We also truncate negative eigenvalues to avoid nan propagation;
    # even for positive semidefinite matrices, small negative eigenvalues
    # can be reported.
    temp_term= np.sqrt(sqrtmA * B * sqrtmA)

    return float(np.trace(temp_term)**2)


def get_Hamiltonian(n):

    Ham2=       [[[0., 'z', 0], [0., 'z', 1]], 
                [[0., 'z', 0]], [[0., 'z', 1]]]

    hamiltonian_list=[]

    for i in range(n):
        hamiltonian_list.append([[0., 'z', i]])

    
    for j in range(n):
        for k in range(j):
            if j==k:
                pass
            else:
                hamiltonian_list.append([[0., 'z', j], [0., 'z', k]])

    #print(hamiltonian_list)

    return np.array(hamiltonian_list, dtype=object)
    

def getAnsatz(anatz_number):
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


def apply_hamiltonian(psi, mini_max_cut=True):
    """
    Function that applies an operator to a quantumstate

    Args:
            - psi(QuantumCircuit): The quantum state
            - max_cut(bool):        If max cut then run the mini max cut
                                    hamiltonian with 3 nodes

            returns(float) the expectation value 
    """

    if mini_max_cut==True:
        #wm=np.array([[0,3,1], [3,0,3], [1,3,0]])
        wm=np.array([[0,3], [3,0]])

        #DM=DensityMatrix.from_instruction(psi)
        #TODO Fix trace circ list
        #PT=partial_trace(DM,[2,3])

        #Evolve qunatum state
        H_zz_op=Operator(Pauli(label='ZZ'))

        print(psi)
        evolved_state=psi.evolve(H_zz_op)
        print(evolved_state.probabilities())
        #Array of probabilities
        results=evolved_state.probabilities()

        H=0
        for i in range(len(wm)):
            for j in range(i):
                H+=wm[i][j]*results[i]*results[j]



        """
        #This works
        test_qc=qk.QuantumCircuit(2)
        test_qc.x(0)
        test_qc.x(1)
        print(test_qc)
        DM_test=DensityMatrix.from_instruction(test_qc)
        H_test = Pauli(label='XX')
        H_testop=Operator(H_test)
        evo_state_test=DM_test.evolve(H_testop)
        print(evo_state_test.to_statevector().probabilities())

        #psi=result.get_statevector()
        #    probs_qubit_0 = psi.probabilities([0])
        
        exit()
        """
        #evoliving the qauantum state 
        #print(PT)
        #print(f'Statevector: {PT.to_statevector}')
        #print(f'Operator: {PT.to_operator}')

    return H

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

def NN_nodes(*argv, act='tanh'):
    layers_nodes=[]

    for arg in argv:
        layers_nodes.append([act])
        layers_nodes.append([arg, 1])
    
    return layers_nodes

def compute_gs_energy(circuit, backend="statevector_simulator"):
    """
    Function to compute the energy using the evolved parameters in VarITE
    
    Circuit(object):    The trial circuit containing the evolved parameters
    """
    compute_gs_energy.counter += 1
    #print(compute_gs_energy.counter)
    circ=create_initialstate(circuit)

    backendtest = qk.Aer.get_backend(backend)
    result = backendtest.run(circ).result()
    psi=result.get_statevector()
    probs_qubit_0 = psi.probabilities([0])
    probs_qubit_1 = psi.probabilities([1])

    print(f'{compute_gs_energy.counter}__________________________')
    print(f'Qubit 1 {probs_qubit_0[1]}')
    print(f'Qubit 2 {probs_qubit_1[1]}')


    #print(circ)
    

   