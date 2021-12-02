import numpy as np
import qiskit as qk
from qiskit.circuit import Parameter, ParameterVector
#import random

"""
Testing the update thing of params.
"""

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

gates_str=[['rx',0],['ry', 0], ['rz', 0]] #, ['crz', 0, 1]]
V_test=[['rx',0],['ry', 0], ['rz', 0]]
H_simple=[[0.2, 'x'], [0.4, 'z'], [1-np.sqrt(0.2**2+0.4**2),'y']]
H_sup=[[0.2252, 'i'], [0.3435, 'z', 0], [-0.4347, 'z', 1], [0.5716, 'z', 0, 'z', 1], [0.0910, 'y', 0, 'y', 1], [0.0910, 'x', 0, 'x', 1]]


num_qubits= max([el[1] for el in gates_str])
#num_qubits=0
n_params=len(gates_str)

theta_list=np.random.uniform(0,2*np.pi,size=n_params)
#Assertion statement
assert n_params==len(gates_str), "Number of gates and parameters do not match"


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
#parameter_dict={param_vec[0]: 1, param_vec[1]: 2}
#qc_param2=qc_param.assign_parameters(parameter_dict, inplace=False)
#print(qc_param2)

#test_par=update_parameter_dict(parameter_dict, [0,3])
#qc_param2=qc_param.bind_parameters(test_par)
qc_param2=qc_param.bind_parameters([1,2,3])
#print(qc_param2)
qc_param2=qc_param.bind_parameters([4,5,6])




#print(qc_param)
#print(qc_param2)


p= [['ry',0, 0],['ry',0, 1], ['cx', 1,0], ['cx', 0, 1],
    ['ry',np.pi/2, 0],['ry',0, 1], ['cx', 0, 1]]


qse=qk.QuantumCircuit(2)


p_vec = ParameterVector('Init_param', len(p))

for i in range(0, len(p)):
    gate=p[i][0]
    if gate == 'cx' or gate == 'cy' or gate == 'cz':
        getattr(qse, p[i][0])(p[i][1], p[i][2])
    else:
        getattr(qse, p[i][0])(p_vec[i], p[i][2])

print(qse)

print(len(qse.parameters))

"""
getattr(qse,p[i][0])(p[i][1],p[i][2])
getattr(qse,p[1][0])(p[1][1],p[1][2])
getattr(qse,p[2][0])(p[2][1],p[2][2])
getattr(qse,p[3][0])(p[3][1],p[2][2])
"""

qc_par=qse.bind_parameters([1,2,3,4])
qc_par=qse.bind_parameters([1,2,3,4])


#print(qse)

#print(qc_par)

i=1
j=2
import time

a = np.empty(shape=(2,2), dtype=object)
for i in range(2):
    for j in range(2):
        a[i,j]=qse.bind_parameters([i,j,i,j])

#a[1,0]=a[1,0].bind_parameters([1,2,2,1])
#a[1,0]=a[1,0].bind_parameters([1,3,2,1])

#a[1,0]=a[1,0].bind_parameters([1,1,2,1])
#a[1,0]=a[1,0].bind_parameters([1,1,2,1])

#for i in range(0):
#    a[1,0]=a[1,0].bind_parameters([1,1,2,1])

"""
start=time.time()
for i in range(10000):
    a[1,0]=a[1,0].bind_parameters([1,1,2,1])
end=time.time()

print(f'assign time {end-start}')

start2=time.time()
for i in range(10000):
    a[1,0]=a[1,0].bind_parameters([1,2,3,4])
end2=time.time()
"""

#print(a)
#print(a[1,0])

"""
A_dict = {
    str(j)+str(i): qse,
    str(i)+str(j): qse}

print(A_dict)
#print(A_dict["21"])
print(A_dict["12"])
A_dict["12"]=A_dict["12"].bind_parameters([1,1,1,1])
print(A_dict["12"])

#something like this?
#A_dict["12"]=A_dict["12"].bind_parameters([seld.trial_circ[:,0]])
"""

























"""
from functools import partial
from itertools import repeat
from multiprocessing import Pool, freeze_support

def func(a, b):
    return a + b

def main():
    a_args = [1,2,3]
    second_arg = 1
    with Pool() as pool:
        print("test")
        L = pool.starmap(func, [(1, 1), (2, 1), (3, 1)])
        M = pool.starmap(func, zip(a_args, repeat(second_arg)))
        N = pool.map(partial(func, b=second_arg), a_args)
        assert L == M == N

#if __name__=="__main__":
    #freeze_support()
    #main()


class test():
    def __init__(self, x):
        self.x=x
    
    def func2(self):
        a_args = [1,2,3]
        second_arg = 1
        with Pool() as pool:
            print("test")
            L = pool.starmap(func, [(1, 1), (2, 1), (3, 1)])
            M = pool.starmap(func, zip(a_args, repeat(second_arg)))
            N = pool.map(partial(func, b=second_arg), a_args)
            print(L)
            assert L == M == N
            
        
        return


xx=test(3)
xx.func2()
"""
