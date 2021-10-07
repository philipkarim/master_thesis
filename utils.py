"""
Expressions utilized throughout the scripts
"""
# Common imports
from typing import ValuesView
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import random
import qiskit as qk
import cmath

random.seed(2021)

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
    sns.set_style("darkgrid")
    
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

def run_circuit(qc, shots=1024, backend="qasm_simulator"):
    job = qk.execute(qc,
                    backend=qk.Aer.get_backend(backend),
                    shots=shots,
                    seed_simulator=10
                    )
    results = job.result()
    results = results.get_counts(qc)

    prediction = 0
    for key,value in results.items():
        #print(key, value)
        if key == '1':
            prediction += value
    prediction/=shots

    return prediction


class pauli_terms:
    def __init__(self, theta, label):
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