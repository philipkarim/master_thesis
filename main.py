import numpy as np
import random
import qiskit as qk
# Import the other classes and functions
from PQC import QML
from optimize_loss import optimize
from utils import *
#from varqbm import *

from varQITE import *

from qiskit.circuit import Parameter, ParameterVector


"""
Reproduce example: Trying to mimic a bell state as the article to check if the it works correct
"""

# Seeding the program to ensure reproducibillity
random.seed(2021)

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

def derivative_U(U_gate):
    qubits=U_gate.width()
    for k in range(qubits):
        #Return gate as matrix, then if size is larger than 4=control, else not?, also check if gates are equal
        #if U_gate.num_ctrl_qubits()==0:
            return np.imag(j/2)
    return         

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
#Trying to reproduce fig2- Now we know that these params produce a bell state
param_fig2=[['ry',0, 0],['ry',0, 1], ['cx', 1,0], ['cx', 0, 1],['ry',np.pi/2, 0],['ry',0, 1], ['cx', 0, 1]]

#[coefficient, gate, qubit]
H_simple=[[1., 'z', 0]]
#[gate, value, qubit]
V_test2=[['rx',0, 0],['ry', 0, 0], ['rz', 0, 0]]

"""
Testing
"""

#make_varQITE object
varqite=varQITE(H_simple, param_fig2)
varqite.state_prep()


#print(param_fig2)
#BS_circ_initial=create_initialstate(param_fig2)

#print(BS_circ_initial)

#A_mat2=np.copy(get_A2(param_fig2))
#C_vec2=np.copy(get_C2(param_fig2, H_simple))

#print(A_mat2)
#print(C_vec2)

#A_mat2, C_vec2=remove_Nans(A_mat2, C_vec2)

#print(A_mat2)
#print(C_vec2)










"""
def varQITE_state_preparation(steps_n, H_theta):

    #Basicly some input values, and then the returned values are the gibbs states.
    #Probably should make this an own function
    #And find out how to solve the differential equations

    #Input: page 6 in article algorithm first lines
    k_b=1
    temp_T=1
    tau=0.5*k_b*temp_T

    time_step=tau/steps_n

    #initialisation of w for each theta, starting with 0?
    w_dtheta=np.zeros(len(H_theta))

    for t in range(time_step, tau+1):   #+1?
        #Compute A(t) and C(t)
        A_temp=expression_A(t)
        C_temp=expression_C(t)

        A_inv_temp=np.inv(A_temp)
        
        omega_derivative=A_inv_temp@C_temp

        #Solve A* derivative of \omega=C
        #No idea how to do it
        for i in range(len(H_theta)): 
            #Compute the expression of the derivative
            dA_mat=np.copy(get_dA(theta_list, gates_str))
            dC_vec=np.copy(get_dC(theta_list, gates_str, H_simple))

            #Now we compute the derivative of omega derivated with respect to
            #hamiltonian parameter
            #dA_mat_inv=np.inv(dA_mat)
            w_dtheta_dt= A_inv_temp@(dC_vec-dA_mat@omega_derivative)#* or @?

            w_dtheta[i]+=w_dtheta_dt*time_step

            #Solve A(d d omega)=d C -(d A)*d omega(t)
            
            #Compute:
            #dw(t)=dw(t-time_step)+d d w time_step
        #compute dw
        #w(t+time_step)=w(t)dw(t)time_step

    return w(t), dw(t) 
"""

"""
Next step:
Make the ITE handle tensorproducts also, find a smart
way to implement it
"""

#Theta parameters are dependent of the hamiltonian so for each
#rotation in the fig, the theta params are equal.
#Then the omega things are initialized, and then the next can be
#computed forward.


"""
Okay here is the real next step, assuming we got the VarITE:
- Generate pw and
-Then use pw and dw/dtheta to compute pw_QBM and dpw_QBM/dtheta
-Compute the loss and stuff like that
-Update the parameters in the Hamiltonian
"""

w_t=1
dwt=1

#Find p_w_gibbs by using eq.8.5 for each configuration and tracing over the qubit thing
#which should be thaaat hard iguess, need to understand the thing
#with qubit systems
qubits_in_H=4

np.random.seed(222)
test_parameters=np.random.randn(5)
#print(test_parameters)

p_v_data=abs(np.random.randn(2**qubits_in_H))

p_w_gibbs=np.random.randn(2**qubits_in_H)

#Then compute p_v_QBM by tr(mystic A \cdot p_gibbs 8.75
p_v_QBM=abs(np.random.randn(2**qubits_in_H))

#Then compute the dp_v_QBM by using eq.10 chain rule or shift?
dp_v_QBM=1

#Then find dL/d theta by using eq. 10
gradient_theta=np.random.randn(5)

#Update the classical parameters, this could be done by using a classical optimizer
"""
Updating the parameters:
"""
optim=optimize(len(test_parameters)) ##Do not call this each iteration, it will mess with the momentum
loss=optim.cross_entropy_new(p_v_data,p_v_QBM)
#Update the parameters
#new_parameters=optim.gradient_descent_gradient_done(test_parameters, 0.01, gradient_theta)

new_parameters=optim.adam(test_parameters, gradient_theta)
