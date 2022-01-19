"""
alt+z to fix word wrap

Rotating the monitor:
xrandr --output DP-1 --rotate right
xrandr --output DP-1 --rotate normal

xrandr --query to find the name of the monitors

"""
import random
import numpy as np
import qiskit as qk
from qiskit.circuit import Parameter, ParameterVector
from qiskit.quantum_info import DensityMatrix, partial_trace, state_fidelity
import time
import matplotlib.pyplot as plt

# Import the other classes and functions
from optimize_loss import optimize
from utils import *
from varQITE import *

import multiprocessing as mp

# Seeding the program to ensure reproducibillity
random.seed(2021)

#Best seed=2021

both=True

if both==False:
    Hamiltonian=2
    p_data=np.array([0.12, 0.88])

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

        #Write qk.z instead of str? then there is no need to use get.atr?
        H=     [[1., 'z', 0], [1., 'z', 1], [-0.2, 'z', 0], 
                [-0.2, 'z', 1],[0.3, 'x', 0], [0.3, 'x', 1]]

    elif Hamiltonian==3:
        params= [['ry',0, 0],['ry',0, 1], ['cx', 1,0], ['cx', 0, 1],
                    ['ry',np.pi/2, 0],['ry',0, 1], ['cx', 0, 1]]
                    #[gate, value, qubit]
        H_init=np.random.uniform(low=-1.0, high=1.0, size=1)
        H=        [[H_init[0], 'z', 0]]

    elif Hamiltonian==4:
        params=  [['ry',0, 0], ['ry',0, 1], ['ry',0, 2], ['ry',0, 3], 
                ['cx', 3,0], ['cx', 2, 3],['cx', 1, 2], ['ry', 0, 3],
                ['cx', 0, 1], ['ry', 0, 2], ['ry',np.pi/2, 0], 
                ['ry',np.pi/2, 1], ['cx', 0, 2], ['cx', 1, 3]]
        
        p_data=np.array([0.5,0, 0, 0.5])
        H_init=np.random.uniform(low=-1.0, high=1.0, size=3)
        print(H_init)
        H=     [[H_init[0], 'z', 0], [H_init[0], 'z', 1], [H_init[1], 'z', 1], [H_init[2], 'z', 0]]


    #make_varQITE object
    start=time.time()
    varqite=varQITE(H, params, steps=10)
    #varqite.initialize_circuits()

    #Testing
    omega, d_omega=varqite.state_prep(gradient_stateprep=True)
    #print(d_omega)
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
    if Hamiltonian==1 or Hamiltonian==2:
        if Hamiltonian==1:
            PT=partial_trace(DM,[1])
            H_analytical=np.array([[0.12, 0],[0, 0.88]])

        elif Hamiltonian==2:
            #What even is this partial trace? thought it was going to be [1,3??]
            #PT=partial_trace(DM,[0,3])=80%
            PT=partial_trace(DM,[2,3])
            
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
elif both==True:
    pass

else:
    params1= [['ry',0, 0],['ry',0, 1], ['cx', 1,0], ['cx', 0, 1],
                ['ry',np.pi/2, 0],['ry',0, 1], ['cx', 0, 1]]
                #[gate, value, qubit]
    H1=        [[1., 'z', 0]]
    params2=  [['ry',0, 0], ['ry',0, 1], ['ry',0, 2], ['ry',0, 3], 
            ['cx', 3,0], ['cx', 2, 3],['cx', 1, 2], ['ry', 0, 3],
            ['cx', 0, 1], ['ry', 0, 2], ['ry',np.pi/2, 0], 
            ['ry',np.pi/2, 1], ['cx', 0, 2], ['cx', 1, 3]]
            #[gate, value, qubit]

    #Write qk.z instead of str? then there is no need to use get.atr?
    H2=     [[1., 'z', 0], [1., 'z', 1], [-0.2, 'z', 0], 
            [-0.2, 'z', 1],[0.3, 'x', 0], [0.3, 'x', 1]]

    ##Computing
    """
    rotational_indices1=[]
    n_qubits_params1=0
    for i in range(len(params1)):
        if params1[i][0]=='cx' or params1[i][0]=='cy' or params1[i][0]=='cz':
            if n_qubits_params1<params1[i][1]:
                n_qubits_params1=params1[i][1]
        else:
            rotational_indices1.append(i)

        if n_qubits_params1<params1[i][2]:
            n_qubits_params1=params1[i][2]

    rotational_indices2=[]
    n_qubits_params2=0
    for i in range(len(params2)):
        if params2[i][0]=='cx' or params2[i][0]=='cy' or params2[i][0]=='cz':
            if n_qubits_params2<params2[i][1]:
                n_qubits_params2=params2[i][1]
        else:
            rotational_indices2.append(i)

        if n_qubits_params2<params2[i][2]:
            n_qubits_params2=params2[i][2]
    """

    """
    Testing
    """
    print('VarQite 1')
    varqite1=varQITE(H1, params1, steps=10)
    varqite1.initialize_circuits()
    start1=time.time()
    omega1, d_omega=varqite1.state_prep(gradient_stateprep=True)
    end1=time.time()


    print('VarQite 2')
    varqite2=varQITE(H2, params2, steps=10)
    varqite2.initialize_circuits()
    start2=time.time()
    omega2, d_omega=varqite2.state_prep(gradient_stateprep=True)
    end2=time.time()
    #print(d_omega)

    print(f'Time used H1: {np.around(end1-start1, decimals=1)} seconds')
    print(f'Time used H2: {np.around(end2-start2, decimals=1)} seconds')

    print(f'omega: {omega2}')

    """
    Investigating the tracing of subsystem b
    """
    params1=update_parameters(params1, omega1)
    params2=update_parameters(params2, omega2)

    #Dansity matrix measure, measure instead of computing whole DM
    trace_circ1=create_initialstate(params1)
    trace_circ2=create_initialstate(params2)

    print(trace_circ2)

    DM1=DensityMatrix.from_instruction(trace_circ1)
    DM2=DensityMatrix.from_instruction(trace_circ2)

    PT1 =partial_trace(DM1,[1])
    H1_analytical=np.array([[0.12, 0],[0, 0.88]])

    PT2=partial_trace(DM2,[2,3])
    #Just to check that the correct parts are subtraced
    PT2_2=partial_trace(DM2,[1,3])
    PT2_3=partial_trace(DM2,[0,1])
    PT2_4=partial_trace(DM2,[0,2])

    H2_analytical= np.array([[0.10, -0.06, -0.06, 0.01], 
                            [-0.06, 0.43, 0.02, -0.05], 
                            [-0.06, 0.02, 0.43, -0.05], 
                            [0.01, -0.05, -0.05, 0.05]])

    print('---------------------')
    print('Analytical Gibbs state:')
    print(H1_analytical)
    print('Computed Gibbs state:')
    print(np.real(PT1.data))
    print('---------------------')

    
    print('---------------------')
    print('Analytical Gibbs state:')
    print(H2_analytical)
    print('Computed Gibbs state:')
    print(np.real(PT2.data))
    print('---------------------')

    H_fidelity1=state_fidelity(PT1.data, H1_analytical, validate=False)
    H_fidelity2=state_fidelity(PT2.data, H2_analytical, validate=False)
    H_fidelity2_2=state_fidelity(PT2_2.data, H2_analytical, validate=False)
    H_fidelity2_3=state_fidelity(PT2_3.data, H2_analytical, validate=False)
    H_fidelity2_4=state_fidelity(PT2_4.data, H2_analytical, validate=False)

    print(f'Fidelity: H1: {np.around(H_fidelity1, decimals=2)}, H2: '
                        f'{np.around(H_fidelity2, decimals=2)}, '
                        f'{np.around(H_fidelity2_2, decimals=2)}, '
                        f'{np.around(H_fidelity2_3, decimals=2)}, '
                        f'{np.around(H_fidelity2_4, decimals=2)}')




def train(H, ansatz, n_epochs, p_data, n_steps=10, lr=0.001, plot=True):
    print('------------------------------------------------------')

    loss_list=[]
    epoch_list=[]

    tracing_q, rotational_indices, n_qubits_ansatz=getUtilityParameters(H, ansatz)

    #print(tracing_q, rotational_indices, n_qubits_ansatz)

    optim=optimize(H, rotational_indices, tracing_q, learning_rate=lr) ##Do not call this each iteration, it will mess with the momentum

    varqite_train=varQITE(H, ansatz, steps=n_steps)
    varqite_train.initialize_circuits()


    for epoch in range(n_epochs):
        print(f'epoch: {epoch}')

        #Stops, memory allocation??? How to check
        omega, d_omega=varqite_train.state_prep(gradient_stateprep=False)
        ansatz=update_parameters(ansatz, omega)

        print(f' omega: {omega}')
        print(f' d_omega: {d_omega}')

        #Dansity matrix measure, measure instead of computing whole DM
        
        trace_circ=create_initialstate(ansatz)
        DM=DensityMatrix.from_instruction(trace_circ)

        PT=partial_trace(DM,tracing_q)

        #Is this correct?
        p_QBM=np.diag(PT.data).real.astype(float)
        #Hamiltonian is the number of hamiltonian params
        print(f'p_QBM: {p_QBM}')
        loss=optim.cross_entropy_new(p_data,p_QBM)
        print(f'Loss: {loss, loss_list}')
        
        #Appending loss and epochs
        loss_list.append(loss)
        epoch_list.append(epoch)
        #Then find dL/d theta by using eq. 10
        #print('Updating params..')
        #print(f'd_omega same? {d_omega}')
        #TODO: This is quiet high
        gradient_qbm=optim.gradient_ps(H, ansatz, d_omega, steps=n_steps)
        #print(f'gradient of qbm: {gradient_qbm}')
        gradient_loss=optim.gradient_loss(p_data, p_QBM, gradient_qbm)
        print(f'gradient_loss: {gradient_loss}')
        #print(type(gradient_loss))
        #TODO: Fix the thing to handle gates with same coefficient

        #TODO: Make the coefficients an own list, and the parameters another. 
        # That way I can use array for the cefficients. this might actually be the
        #reason for the error

        new_parameters=optim.adam(np.array(H)[:,0].astype(float), gradient_loss)
        print(f'old params: {np.array(H)[:,0].astype(float)}')
        #new_parameters=optim.gradient_descent_gradient_done(np.array(H)[:,0].astype(float), gradient_loss)
        print(f'New params {new_parameters}')
        #TODO: Try this
        #gradient_descent_gradient_done(self, params, lr, gradient):

        #print(f'new coefficients: {new_parameters}')

        #Is this only params or the whole list? Then i think i should insert params and the
        #function replace the coefficients itself

        for i in range(len(H)):
            H[i][0]=new_parameters[i]
        
        varqite_train.update_H(H)

        #print(f'Final H, lets go!!!!: {H}')

        #Compute the dp_QBM/dtheta_i
    if plot==True:
        plt.plot(epoch_list, loss_list)
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.show()
    
    return loss_list


ansatz2=  [['ry',0, 0], ['ry',0, 1], ['ry',0, 2], ['ry',0, 3], 
            ['cx', 3,0], ['cx', 2, 3],['cx', 1, 2], ['ry', 0, 3],
            ['cx', 0, 1], ['ry', 0, 2], ['ry',np.pi/2, 0], 
            ['ry',np.pi/2, 1], ['cx', 0, 2], ['cx', 1, 3]]
            #[gate, value, qubit]

Ham2=     [[1., 'z', 0], [1., 'z', 1], [-0.2, 'z', 0], 
            [-0.2, 'z', 1],[0.3, 'x', 0], [0.3, 'x', 1]]

p_data2=[0.25, 0, 0, 0.25]


ansatz1=    [['ry',0, 0],['ry',0, 1], ['cx', 1,0], ['cx', 0, 1],
                ['ry',np.pi/2, 0],['ry',0, 1], ['cx', 0, 1]]
                #[gate, value, qubit]
Ham1=       [[1., 'z', 0]]

p_data1=[0.8, 0.2]

H_U_1=np.random.uniform(low=-1.0, high=1.0, size=1)
HU_1=        [[H_U_1[0], 'z', 0]]

H_U_2=np.random.uniform(low=-1.0, high=1.0, size=4)
HU_2=   [[H_U_2[0], 'z', 0], [H_U_2[1], 'z', 1], 
        [H_U_2[2],'z', 0], [H_U_2[3], 'z', 1]]

print(H_U_2)

train(HU_2, ansatz2, 5, p_data2, n_steps=10, lr=0.05)




def multiple_simulations(n_sims, HU_2, ansatz2, epochs, target_data, l_r):
    saved_error=np.zeros((n_sims, epochs))

    for i in range(n_sims):
        saved_error[i]=train(HU_2, ansatz2, epochs, target_data, lr=l_r, plot=False)

    epochs_list=list(range(0,epochs))
    avg_list=np.mean(saved_error, axis=0)
    std_list=np.std(saved_error, axis=0)

    plt.errorbar(epochs_list, avg_list, std_list, linestyle='None', marker='^')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.show()

    return




"""
Try and fail method:

- Try to add i<j in C and dCA
- Check if the gradients are good
- Why are the2 qubit case so bad?
- Check the derivative cases of the things
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


"""
Thoughts:
- Test encoding thing, do some math?
- Okay I might know something, well basicly the code doesnt update all the params,
only half of them actually. But how does it know if it is a controlled gate or not?
"""
"""
Todays list:
    - Fix rot indices loops
    - times -0.5j standard
    - + or - in sums
    - might be due to ridge?

    - where to put H gate

    -Something wrong with C since it is not arbitrary to the X gates hmmmm


"""


"""
H2 best: Non ridge, C:-=, temp.x
H1 best: Non ridge, C:+=, without temp.x
"""


"""
Tips and tricks:

kan lage en liste med tuples og plusse på det ene elementet mens den andre er de med rot index for å slippe if cx
slik:
if gate2 == 'cx' or gate2 == 'cy' or gate2 == 'cz':
    getattr(temp_circ, gate2)(1+self.trial_circ[test][1], 1+self.trial_circ[test][2])
else:
    getattr(temp_circ, gate2)(self.trial_circ[test][1], 1+self.trial_circ[test][2])

til

for i in [(0,1), (1,2), (0,3)...]
    getattr(temp_circ, gate2)(self.trial_circ[i[1]][1]+i[0], 1+self.trial_circ[i[1]][2])

This looks neat!
"""

"""
Next list:
    - Complete initialisation of the thing med labels and such
        - Complete C
    - Go through the TODO's
    - Gradietn complete
        - dC expression
        - With initialisation
    - Reproduce results/write code to produce it
    - Numba/paralellization
    - Fix the bug, probably have something to do with omega at index 2,3 and 7 being equal.
        - Okay listen up fam, I think I have some kind of idea to the source of the bug. Basicly V=U_N..U_1, but
        that means U_1 is applied first which makes sense for why C is reversed?
        - The key might be to know why C should be reversed
        - Maybe mixed the arguments some places?
    - Fix H to deal with multiple same coefficients
    - Run multiple circuits in paralell instead of separate
    - Do classical BM
    - Gradient too high, why? Normalize 0,1 instead of pi? learning rate?
    - Always increases, within the righ/left? where it is printed, might be that the parameters is set by running the method, 
    or that it should be copied some place
"""