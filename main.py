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
import seaborn as sns

sns.set_style("darkgrid")

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
#elif both==True:
#    pass

else:
    params1= [['ry',0, 0],['ry',0, 1], ['cx', 1,0], ['cx', 0, 1],
                ['ry',np.pi/2, 0],['ry',0, 1], ['cx', 0, 1]]
                #[gate, value, qubit]
    #H1=        [[1., 'z', 0]]
    H1=        [[[1., 'z', 0]]]

    params2=  [['ry',0, 0], ['ry',0, 1], ['ry',0, 2], ['ry',0, 3], 
            ['cx', 3,0], ['cx', 2, 3],['cx', 1, 2], ['ry', 0, 3],
            ['cx', 0, 1], ['ry', 0, 2], ['ry',np.pi/2, 0], 
            ['ry',np.pi/2, 1], ['cx', 0, 2], ['cx', 1, 3]]
            #[gate, value, qubit]

    #Write qk.z instead of str? then there is no need to use get.atr?
    #H2=     [[1., 'z', 0], [1., 'z', 1], [-0.2, 'z', 0], 
    #        [-0.2, 'z', 1],[0.3, 'x', 0], [0.3, 'x', 1]]
    
    H2=     [[[1., 'z', 0], [1., 'z', 1]], [[-0.2, 'z', 0]], 
        [[-0.2, 'z', 1]], [[0.3, 'x', 0]], [[0.3, 'x', 1]]]
    
    #H2=     [[[1., 'z', 0]], [[1., 'z', 1]], [[-0.2, 'z', 0]], 
    #    [[-0.2, 'z', 1]], [[0.3, 'x', 0]], [[0.3, 'x', 1]]]


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




def train(H, ansatz, n_epochs, p_data, n_steps=10, lr=0.01, optim_method='Amsgrad', plot=True):
    print('------------------------------------------------------')

    loss_list=[]
    epoch_list=[]

    tracing_q, rotational_indices=getUtilityParameters(ansatz)

    #print(tracing_q, rotational_indices, n_qubits_ansatz)

    optim=optimize(H, rotational_indices, tracing_q, learning_rate=lr, method=optim_method) ##Do not call this each iteration, it will mess with the momentum

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
        gradient_qbm=optim.gradient_ps(H, ansatz, d_omega)
        #print(f'gradient of qbm: {gradient_qbm}')
        gradient_loss=optim.gradient_loss(p_data, p_QBM, gradient_qbm)
        print(f'gradient_loss: {gradient_loss}')
        #print(type(gradient_loss))
        #TODO: Fix the thing to handle gates with same coefficient

        #TODO: Make the coefficients an own list, and the parameters another. 
        # That way I can use array for the cefficients. this might actually be the
        #reason for the error

        H_coefficients=np.zeros(len(H))

        for ii in range(len(H)):
            H_coefficients[ii]=H[ii][0][0]

        print(f'Old params: {H_coefficients}')
        new_parameters=optim.adam(H_coefficients, gradient_loss)
        #new_parameters=optim.gradient_descent_gradient_done(np.array(H)[:,0].astype(float), gradient_loss)
        print(f'New params {new_parameters}')
        #TODO: Try this
        #gradient_descent_gradient_done(self, params, lr, gradient):

        #print(f'new coefficients: {new_parameters}')

        #Is this only params or the whole list? Then i think i should insert params and the
        #function replace the coefficients itself

        for i in range(len(H)):
            for j in range(len(H[i])):
                H[i][j][0]=new_parameters[i]
        
        varqite_train.update_H(H)

        #print(f'Final H, lets go!!!!: {H}')

        #Compute the dp_QBM/dtheta_i
    
    del optim
    del varqite_train

    if plot==True:
        plt.plot(epoch_list, loss_list)
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.show()
    
    return loss_list, p_QBM


ansatz2=  [['ry',0, 0], ['ry',0, 1], ['ry',0, 2], ['ry',0, 3], 
            ['cx', 3,0], ['cx', 2, 3],['cx', 1, 2], ['ry', 0, 3],
            ['cx', 0, 1], ['ry', 0, 2], ['ry',np.pi/2, 0], 
            ['ry',np.pi/2, 1], ['cx', 0, 2], ['cx', 1, 3]]
            #[gate, value, qubit]

Ham2=     [[1., 'z', 0], [1., 'z', 1], [-0.2, 'z', 0], 
            [-0.2, 'z', 1],[0.3, 'x', 0], [0.3, 'x', 1]]

p_data2=[0.5, 0, 0, 0.5]


ansatz1=    [['ry',0, 0],['ry',0, 1], ['cx', 1,0], ['cx', 0, 1],
                ['ry',np.pi/2, 0],['ry',0, 1], ['cx', 0, 1]]
                #[gate, value, qubit]
Ham1=       [[1., 'z', 0]]

p_data1=[0.8, 0.2]

np.random.seed(2021)

#H_U_1=np.random.uniform(low=-1.0, high=1.0, size=1)
#HU_1=        [[H_U_1[0], 'z', 0]]

H_U_2=np.random.uniform(low=-1.0, high=1.0, size=3)

HU_2=[[[H_U_2[0],'z', 0], [H_U_2[0], 'z', 1]], 
      [[H_U_2[1],'z', 0]], [[H_U_2[2], 'z', 1]]]

#print(H_U_2)

#train(HU_2, ansatz2, 30, p_data2, n_steps=10, lr=0.1)

#OMega isnt trained why?


def multiple_simulations(n_sims, ansatz2, epochs, target_data, l_r, steps):
    saved_error=np.zeros((n_sims, epochs))
    
    qbm_list=[]
    np.random.seed(10)

    for i in range(n_sims):
        print(f'Seed: {i} of {n_sims}')
        H_U_2=np.random.uniform(low=-1., high=1., size=3)
        print(H_U_2)
        HU_2=   [[[H_U_2[0],'z', 0], [H_U_2[0], 'z', 1]], 
                [[H_U_2[1],'z', 0]], [[H_U_2[2], 'z', 1]]]
        saved_error[i], dist=train(HU_2, ansatz2, epochs, target_data, n_steps=steps, lr=l_r, plot=False)
        qbm_list.append(dist)
    

    epochs_list=list(range(0,epochs))
    avg_list=np.mean(saved_error, axis=0)
    std_list=np.std(saved_error, axis=0)

    min_error=1000
    max_error=0
    best_index=0
    for j in range(n_sims):
        print(f'saved error: {saved_error[j][-1]}')
        if min_error>saved_error[j][-1]:
            print(f'saved error: {saved_error[j][-1]}')
            min_error=saved_error[j][-1]
            best_pbm=qbm_list[j]
            best_index=j

        if max_error<saved_error[j][-1]:
            worst_pbm=qbm_list[j]
            max_error=saved_error[j][-1]

    print(f'best_pbm {best_pbm}')
    print(f'worst pbm {worst_pbm}')
    print('---------------------')
    print(f'avg_list {avg_list}')
    print(f'std_list {std_list}')
    print(f'error {saved_error}')
    print('---------------------')


    bell_state=[0.5,0,0,0.5]
    barWidth = 0.25
 
    # Set position of bar on X axis
    br1 = np.arange(len(bell_state))
    br2 = [x + barWidth for x in br1]
    br3 = [x + barWidth for x in br2]
    
    # Make the plot
    plt.bar(br1, bell_state, color ='r', width = barWidth,
            edgecolor ='grey', label ='Bell state')
    plt.bar(br2, worst_pbm, color ='g', width = barWidth,
            edgecolor ='grey', label ='Worst trained')
    plt.bar(br3, best_pbm, color ='b', width = barWidth,
            edgecolor ='grey', label ='Best trained')
    plt.xlabel('Sample')
    plt.ylabel('Probability')
    plt.xticks([r + barWidth for r in range(len(bell_state))],['00', '01', '10', '11'])
    plt.legend()
    plt.tight_layout()
    plt.savefig(str(l_r*1000)+'_bar.png')
    plt.clf()
    #plt.show()

    plt.errorbar(epochs_list, avg_list, std_list)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Bell state: Mean loss with standard deviation using 10 seeds')
    plt.savefig('lr'+str(l_r*1000)+'_mean.png')
    plt.clf()

    #plt.show()

    plt.plot(epochs_list, saved_error[best_index])
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Bell state: Best of 10 seeds')
    plt.savefig(str(l_r*1000)+'_best.png')
    plt.clf()

    #plt.show()

    for k in range(len(saved_error)):
        plt.plot(epochs_list, saved_error[k])
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Bell state: Random seeds')
    plt.savefig(str(l_r*1000)+'_all.png')

    return
multiple_simulations(10, ansatz2, 10, p_data2, l_r=0.1, steps=10)
#exit()
#multiple_simulations(10, ansatz2, 50, p_data2, l_r=0.1, steps=10)
#multiple_simulations(3, ansatz2, 20, p_data2, l_r=0.01, steps=10)
#multiple_simulations(10, ansatz2, 50, p_data2, l_r=0.001, steps=10)



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
Next list:
    - Complete initialisation of the thing med labels and such
        - Complete C
    - Go through the TODO's
    - Gradietn with initialisation
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

    - Okay I think I know: Just implement the gradients by ignoring the V_circ when it does not have a derivative and move the
    pauli gate over to the other side. Basicly do just the same thing as in C for 99 and 98 percent lol

    - CV from scikit not ridge CV

    - Why si gradient higher when the loss is lower? Might have a wrong sign. Not always like that
    Why is loss still good after a run? Something isnt reset, the gradient lacks after the run
        - I think this is due to some running of the grads in the algorithm scheme maybe?
        Try transposing it?
    
    - Check why they are the same depending on the coefficients in the gradient loop?
    - Find out what is pulling the predictions so high
    - Normalizing the quantum gates between -1 and 1?

    - Check on the parameters of adam, maybe better with ridge?

    -Params always same size, maybe try with amsgrad with + instead of minus in the x thing

    - Thoughts: dA is quiet high, and the inverse have some values which are quiet low high

    - Should probably normalize the shit
    -Gå gjennom dA step by step og finn ut hvorfor den er drithøy

    - Noe henger igjen som object fra tidligere

    - Try testing if the shit works correct now, remember to update qiskit
    - Implement the correct shit
"""
