import random
import numpy as np
import qiskit as qk
from qiskit.circuit import Parameter, ParameterVector
from qiskit.quantum_info import DensityMatrix, partial_trace, state_fidelity
import time

# Import the other classes and functions
from optimize_loss import optimize
from utils import *
from varQITE import *

import multiprocessing as mp

# Seeding the program to ensure reproducibillity
#random.seed(2021)

#Best seed=2021

#Set parameters
n_params=3         #Number of variational parameters
init_params=np.random.uniform(-1,1,size=n_params) #Distribution of the initial variational parameters
learning_rate=0.1  #Learning rate
Hamiltonian="bell state"    #theta_0 ZZ +theta_1 IZ + theta2 ZI
epochs=50           #Number of epochs
batch_size=1        #Batch size, set equal to 1

"""
Testing hamiltonian stuff
"""
#1 qubit hamiltonian
H_1q=1*np.array(([1, 0],[0, -1]))
time_tau=0.5
psi_0=np.array([1,0])


def C_first(time, Ham, psi_0):
    deno=np.exp(-2*Ham*time)*psi_0*psi_0.T
    C_start= 1/np.sqrt(np.trace(deno))
    return C_start

C_expression=C_first(time_tau, H_1q, psi_0)

#Create psi_w=V(w)|psi_in


#Define a qc cirquit creating V: U_1=e^(iH(w)) where i think M could be the gates in H? or not?
#So basicly: define \psi_in and \V each run probably.

###okay lets do this! Here is the plan:

"""
Define V and psi_in as psi tau (0). Try using V from figure

Need one gate to compute A and C, and one gate as encoder/ansatz
"""
n_qubits=4
n_cbits=1

data_register = qk.QuantumRegister(n_qubits)
classical_register = qk.ClassicalRegister(n_cbits)

#Define encoder, maybe use the same as usual?
qc_fig_4 = qk.QuantumCircuit(data_register, classical_register)


#Making figure 4+ Hademard gate at the start
for qubit in range(n_qubits):
    qc_fig_4.h(data_register[qubit])
    qc_fig_4.ry(2*np.pi*0,data_register[qubit])
    qc_fig_4.rz(2*np.pi*0,data_register[qubit])
    
#Adding a couple of controleld gates
for j in range(0,n_qubits-1):
    qc_fig_4.cx(data_register[j+1], data_register[j])

qc_fig_4.ry(np.pi/2, data_register[0])
qc_fig_4.ry(np.pi/2, data_register[1])
qc_fig_4.ry(0, data_register[2])
qc_fig_4.ry(0, data_register[3])

for z_gate in range(n_qubits):
    qc_fig_4.rz(2*np.pi*0,data_register[z_gate])

def evaluation_circuit(psi_in, A_or_C, V_circuit, U_gate):
    data_register = qk.QuantumRegister(2)
    classical_register = qk.ClassicalRegister(1)

    """
    Input states
    """
    #Input for the first qubit in figure 8
    qc_AC= qk.QuantumCircuit(data_register, classical_register)
    qc_AC.h(data_register[0])

    if A_or_C == 'C':
        qc_AC.s(data_register[0])

    #Input for the second qubit in  figure 8
    qc_AC.append(psi_in, [1])

    """
    Then create circuit in fig 8
    """
    #Define the V_circuit as a controlled gate
    V_gate=V_circuit.to_gate(label="V").control(1)
    #print(V_gate)
    #V_gate=V_gate.control(1)

    qc_AC.append(V_gate, [0,1])
    qc_AC.x(data_register[0])

    U_gate=U_gate.to_gate(label="U").control(1)
    #U_gate=U_gate.control(1)
    qc_AC.append(U_gate, [0,1])

    qc_AC.x(data_register[0])
    qc_AC.h(data_register[0])

    """
    Measuring the first qubit
    """
    qc_AC.measure(data_register[0],classical_register[0])
    print(qc_AC)
    expectation_val=run_circuit(qc_AC)

    return expectation_val


"""
Just testing the function computing the A and C expressions
"""
data_register_1q = qk.QuantumRegister(1)
classical_register_1 = qk.ClassicalRegister(1)

V_test = qk.QuantumCircuit(data_register_1q)
V_test.x(0)
V_test.h(0)

U_test= qk.QuantumCircuit(data_register_1q)
U_test.y(0)

psi_in_test = qk.QuantumCircuit(1, name='|psi_in>')  # Create a quantum circuit with one qubit
initial_state = [0,1]   # Define initial_state as |1>
psi_in_test.initialize(initial_state, 0)

#expectation=evaluation_circuit(psi_in_test, 'C', V_test, U_test)

"""
Okay lezzgo, plan is as follows:
-Write a function to return the derivative for 1 qubit without controlled gates
Easy return, then fill a matrix A and see what happens
"""
n_qubits=1

q1_test = qk.QuantumRegister(n_qubits)
V_series_test = qk.QuantumCircuit(q1_test)

theta_test=np.random.uniform(-1,1, size=1)

V_series_test.rx(2*np.pi*theta_test[0],0, label='rx')

print(V_series_test[0])

#V_series_test.ry(2*np.pi*theta_test[0],0)
#print(V_series_test.num_unitary_factors())
#print(V_series_test.width())
#print(V_series_test.qbit_argument_conversion(V_series_test))
#rint(V_series_test.num_ctrl_qubits())

#Hermitian matrix on stanby
H_test=np.array([[1,1-1.j], [1+1.j,1]])

"""
Next step: Make a gate, then decompose it to a matrix and 
send it into the thing and see if we get the correct result.
The main goal is to extract the f_i,k and the term
"""

V_series_gate=V_series_test.to_gate(label='ry')

#pauli_matrix=pauli_terms(0.5, V_series_gate.label)
pauli_matrix=pauli_terms(np.pi, 'rz')
rx_as_matrix=pauli_matrix.gate_to_matrix()
mat_M=get_M(rx_as_matrix)
decomp_pauli_terms=decomposing_to_pauli(mat_M)

#print(mat_M)
#print(decomp_pauli_terms)
test_mat=np.zeros_like(pauli_matrix.get_Y())
test_mat+=decomp_pauli_terms[0]*pauli_matrix.get_I()
test_mat+=decomp_pauli_terms[1]*pauli_matrix.get_X()
test_mat+=decomp_pauli_terms[2]*pauli_matrix.get_Y()
test_mat+=decomp_pauli_terms[3]*pauli_matrix.get_Z()

#print(test_mat)

"""
Okay now the pauli terms are good, and also f=-i/2 if it is a rotational Gate

Next task:
Make the circuit and loop over it computing A for simple rotational gates
Maybe insert the gate parameters by using a dict? bound parameters to gates?
Do the same for C
"""

#Number of parameters and gates
"""
This does not work, if a control qubit is applied to the last qubit and is the only one
should work for most of them tho 
"""
#Trying to reproduce fig2- Now we know that these params produce a bell state
#param_fig2=[['ry',0, 0],['ry',0, 1], ['cx', 1,0], ['cx', 0, 1],['ry',np.pi/2, 0],['ry',0, 1], ['cx', 0, 1]]

"""
Testing
"""
gates_str=[['rx',0],['ry', 0], ['rz', 0]] #, ['crz', 0, 1]]
V_test=[['rx',0],['ry', 0], ['rz', 0]]
H_simple=[[0.2, 'x'], [0.4, 'z'], [1-np.sqrt(0.2**2+0.4**2),'y']]
H_sup=[[0.2252, 'i'], [0.3435, 'z', 0], [-0.4347, 'z', 1], [0.5716, 'z', 0, 'z', 1], [0.0910, 'y', 0, 'y', 1], [0.0910, 'x', 0, 'x', 1]]


num_qubits= max([el[1] for el in gates_str])
n_params=len(gates_str)

theta_list=np.random.uniform(0,2*np.pi,size=n_params)
#Assertion statement
assert n_params==len(gates_str), "Number of gates and parameters do not match"

parameters=np.random.uniform(0,1,size=n_params)

param_vec = ParameterVector('Init_param', n_params)
"""
circuit.ry(param_vec[0], 0)
circuit.crx(param_vec[1], 0, 1)

bound_circuit = circuit.assign_parameters({params[0]: 1, params[1]: 2})
"""
#Creates the circuit
#for i in range(n_params):
#    gates_str[i]=qk.circuit.Parameter('rx')
#Fill circuit with gates

"""
Make a dict, containing gate and parameter
"""

qc_param = qk.QuantumCircuit(num_qubits+1)

#Initializing the parameters
#Make list of parameters:
#parameters22=[]
#for name in range(len(gates_str)):
#    parameters22.append(qk.circuit.Parameter(gates_str[name]))
#params = [qk.circuit.Parameter('rx').bind(4), qk.circuit.Parameter('rz')]

#Creates the circuit
for i in range(len(gates_str)):
    if len(gates_str[i])==2:
        getattr(qc_param, gates_str[i][0])(param_vec[i], gates_str[i][1])
    elif len(gates_str[i])==3:
        getattr(qc_param, gates_str[i][0])(param_vec[i], gates_str[i][1], gates_str[i][2])
    else:
        print("Function not implemented with double controlled gates yet")
        exit()

"""
This basicly, creates a new circuit with the existing gates, 
then the runs the copy, and when completed makes a copy of the main circuit with
new parameters. Difference between (bind_parameters and assign_parameters?)
"""
#print('Original circuit:')
#print(qc_param)
parameter_dict={param_vec[0]: 1, param_vec[1]: 2}
qc_param2=qc_param.assign_parameters(parameter_dict, inplace=False)
#print(qc_param2)

test_par=update_parameter_dict(parameter_dict, [0,3])
qc_param2=qc_param.bind_parameters(test_par)

#print(qc_param2)

#A_mat=np.copy(get_A(theta_list, gates_str))
#C_vec=np.copy(get_C(theta_list, gates_str, H_simple))

#print(A_mat)
#print(C_vec)
"""
I dont see how A and C depends on A or C except maybe from the hamiltonian?
...Anyway here it the next step:
-Implement tensor product
-Find omega cdot{w}, by inverting A.
-Find derivative of thetaC and theta A
-follow the loop in varQITE
"""

"""
New chapter.. recreate fig 2
"""
"""
PARAMETERS
"""
Hamiltonian=2
p_data=np.array([0.5, 0.5])

#Trying to reproduce fig2- Now we know that these params produce a bell state
if Hamiltonian==1:
    params= [['ry',0, 0],['ry',0, 1], ['cx', 1,0], ['cx', 0, 1],
                ['ry',np.pi/2, 0],['ry',0, 1], ['cx', 0, 1]]
                #[gate, value, qubit]
    H=        [[1., 'z', 0]]
elif Hamiltonian==2:
    params=  [['ry',0, 0], ['ry',0, 1], ['ry',0, 2], ['ry',0, 3], 
            ['cx', 3,0], ['cx', 2, 3],['cx', 1, 2], ['ry', 0, 3],
            ['cx', 0, 1], ['ry', 0, 2], ['ry',np.pi/2, 0], 
            ['ry',np.pi/2, 1], ['cx', 0, 2], ['cx', 1, 3]]
            #[gate, value, qubit]
    H=     [[1., 'z', 0], [1., 'z', 1], [-0.2, 'z', 0], 
            [-0.2, 'z', 1],[0.3, 'x', 0], [0.3, 'x', 1]]

#Transforms the parameters into arrays
#params=np.array(params)
#H=np.array(H)

"""
Rewrite this to work the way it says in the article, 1ZZ-0.2ZI..
because the coefficients must be the same for pairwise hamiltonians
"""
#[coefficient, gate, qubit]
#H1=[[1., 'z', 0]]
#H2=[[1., 'z', 0], [1., 'z', 1], [-0.2, 'z', 0], 
#    [-0.2, 'z', 1],[0.3, 'x', 0], [0.3, 'x', 1]]
#[gate, value, qubit]


#circ=create_initialstate(params)

"""
Testing
"""
#make_varQITE object
start=time.time()
varqite=varQITE(H, params, steps=10)
omega, d_omega=varqite.state_prep(gradient_stateprep=True)
#varqite.run_C2(0)
end=time.time()

print(f'Time used: {np.around(end-start, decimals=1)} seconds')

#varqite.dC_circ0(4,0)
#varqite.dC_circ1(5,0,0)
#varqite.dC_circ2(4,1,0)

"""
Investigating the tracing of subsystem b
"""

params=update_parameters(params, omega)

#Dansity matrix measure, measure instead of computing whole DM
trace_circ=create_initialstate(params)
DM=DensityMatrix.from_instruction(trace_circ)

#Rewrite this to an arbitrary amount of qubits
if Hamiltonian==1:
    PT=partial_trace(DM,[1])
    H_analytical=np.array([[0.12, 0],[0, 0.88]])

elif Hamiltonian==2:
    #What even is this partial trace? thought it was going to be [1,3??]
    PT=partial_trace(DM,[0,2])
    H_analytical= np.array([[0.10, -0.06, -0.06, 0.01], 
                            [-0.06, 0.43, 0.02, -0.05], 
                            [-0.06, 0.02, 0.43, -0.05], 
                            [0.01, -0.05, -0.05, 0.05]])

print('---------------------')
print('Analytical Gibbs state:')
print(H_analytical)
print('Computed Gibbs state:')
print(PT.data)
print('---------------------')

H_fidelity=state_fidelity(PT.data, H_analytical, validate=False)

print(f'Fidelity: {H_fidelity}')


"""
Okay here is the real next step, assuming we got the VarITE:
- Generate pv QBM and pv^QBM/dw, This last one is computed by using a parameter shift rule?
-Compute the loss
-Update the parameters in the Hamiltonian
"""

#Computing the pv_QBM

#This can be done a more productive way I am pretty sure
"""
if np.max(np.array(H)[:,2])==0:
    zero_state=np.array([1,0])
    one_state=np.array([0,1])
elif np.max(np.array(H)[:,2])==1:
    zero_zero_state=np.array([1,0,0,0])
    zero_one_state=np.array([0,1,0,0])
    one_zero_state=np.array([0,0,1,0])
    one_one_state=np.array([0,0,0,1])
"""

#print(f'dw/dø: {d_omega}')

"""
#Is this correct?
p_QBM=np.diag(PT.data)

#Hamiltonian is the number of hamiltonian params
optim=optimize(len(H), Hamiltonian) ##Do not call this each iteration, it will mess with the momentum
loss=optim.cross_entropy_new(p_data,p_QBM)

print(f'Loss: {loss}')

#Then find dL/d theta by using eq. 10
print('Updating params..')
gradient_qbm=optim.gradient_ps(H, params, d_omega, steps=2)

gradient_loss=optim.gradient_loss(p_data, p_QBM)

print(f'gradient_loss: {gradient_loss}')
print(type(gradient_loss))
new_parameters=optim.adam(np.array(H)[:,0], gradient_loss)
print(new_parameters)

np.array(H)[:,0]=new_parameters

print(H)

#Compute the dp_QBM/dtheta_i

"""


"""
Steps to next week in code:

- Rewrite code to work for arbitrary amount of qubits
    - And hamiltonians
- Find out why the imaginary og c_term works but not the real as the article says
- Compute the p^QBM and stuff like that 
- Compute the loss and stuff like that
- Optimize the code to run faster.
    - Assign parameters instead of building the circ? 
    - Maybe have the circ as self.circ?
    - Arrays instead of lists for hamiltonians and arrays
        - The update parameter function for instance
    -Search the web for optimization methods
- Read up on GANs, and see if that could be a cool thing to do
"""
