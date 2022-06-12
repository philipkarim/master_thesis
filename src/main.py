import copy
import numpy as np
import qiskit as qk
from qiskit.quantum_info import DensityMatrix, partial_trace, state_fidelity
import time
import matplotlib.pyplot as plt
import torch

# Import the classes and functions
from simulations import *
from optimize_loss import optimize
from utils import *
from varQITE import *
from franke import franke

def train(H_operator, ansatz, n_epochs, target_data, n_steps=10, lr=0.1, optim_method='Adam', plot=True):
    """
    Training shell with self written optimizers. Do not use this, instead use the functions in simulations.py
    """
    print('------------------------------------------------------')
    
    init_params=np.array(copy.deepcopy(ansatz))[:, 1].astype('float')

    loss_list=[]
    epoch_list=[]
    norm_list=[]
    tracing_q, rotational_indices=getUtilityParameters(ansatz)

    optim=optimize(H_operator, rotational_indices, tracing_q, learning_rate=lr, method=optim_method) ##Do not call this each iteration, it will mess with the momentum

    varqite_train=varQITE(H_operator, ansatz, steps=n_steps, symmetrix_matrices=False)
    
    time_intit=time.time()
    varqite_train.initialize_circuits()
    print(f'initialization time: {time.time()-time_intit}')
    for epoch in range(n_epochs):
        print(f'epoch: {epoch}')

        ansatz=update_parameters(ansatz, init_params)
        omega, d_omega=varqite_train.state_prep(gradient_stateprep=False)
        ansatz=update_parameters(ansatz, omega)

        trace_circ=create_initialstate(ansatz)
        DM=DensityMatrix.from_instruction(trace_circ)
        PT=partial_trace(DM,tracing_q)
        p_QBM=np.diag(PT.data).real.astype(float)
        
        print(f'p_QBM: {p_QBM}')
        loss=optim.cross_entropy_new(target_data,p_QBM)
        print(f'Loss: {loss, loss_list}')
        norm=np.linalg.norm((target_data-p_QBM), ord=1)
        #Appending loss and epochs
        norm_list.append(norm)
        loss_list.append(loss)
        epoch_list.append(epoch)

        gradient_qbm=optim.gradient_ps(H_operator, ansatz, d_omega)

        gradient_loss=optim.gradient_loss(target_data, p_QBM, gradient_qbm)
        H_coefficients=np.zeros(len(H_operator))

        for ii in range(len(H_operator)):
            H_coefficients[ii]=H_operator[ii][0][0]

        new_parameters=optim.adam(H_coefficients, gradient_loss)

        for i in range(len(H_operator)):
            for j in range(len(H_operator[i])):
                H_operator[i][j][0]=new_parameters[i]
        
        varqite_train.update_H(H_operator)
    
    del optim
    del varqite_train

    if plot==True:
        plt.plot(epoch_list, loss_list)
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.show()
    
    return np.array(loss_list), np.array(norm_list), p_QBM


def multiple_simulations(n_sims, initial_H, ans, epochs, target_data,opt_met , l_r, steps, names):
    """
    Computing multiple seeds, do not use this. Rather use function final_seed_sim() in simulations.py, 
    that one is better, utilizing pytorch and fork() paralellization using multiple cores.
    """
    saved_error=np.zeros((n_sims, epochs))
    l1_norm=np.zeros((n_sims, epochs))
    
    qbm_list=[]

    for i in range(n_sims):
        print(f'Seed: {i} of {n_sims}')
        H_init_val=np.random.uniform(low=-1.0, high=1.0, size=len(initial_H))
        print(H_init_val)
        
        for term_H in range(len(initial_H)):
            for qub in range(len(initial_H[term_H])):
                initial_H[term_H][qub][0]=H_init_val[term_H]
        
        time_1epoch=time.time()
        saved_error[i], l1_norm[i], dist=train(initial_H, copy.deepcopy(ans), epochs, target_data, n_steps=steps, lr=l_r, optim_method=opt_met, plot=False)
        qbm_list.append(dist)
        time_1epoch_end=time.time()

        print(f'Time for one loop: {time_1epoch_end-time_1epoch}')
    
    epochs_list=list(range(0,epochs))
    avg_list=np.mean(saved_error, axis=0)
    std_list=np.std(saved_error, axis=0)

    avg_list_norm=np.mean(l1_norm, axis=0)
    std_list_norm=np.std(l1_norm, axis=0)

    #Plotting distributions and plots
    if len(target_data)==4:
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
        plt.savefig(str(l_r*1000)+str(len(target_data))+names+'_bar.pdf')
        plt.clf()
        #plt.show()

    plt.errorbar(epochs_list, avg_list, std_list)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    #plt.title('Bell state: Mean loss with standard deviation using 10 seeds')
    plt.savefig('lr'+str(l_r*1000)+str(len(target_data))+names+'_mean.pdf')
    plt.clf()

    #plt.show()
    if len(target_data)==4:
        plt.plot(epochs_list, saved_error[best_index])
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        #plt.title('Bell state: Best of 10 seeds')
        plt.savefig(str(l_r*1000)+str(len(target_data))+names+'_best.pdf')
        plt.clf()

    #plt.show()
    for k in range(len(l1_norm)):
        plt.plot(epochs_list, l1_norm[k])

    plt.xlabel('Epoch')
    plt.ylabel('L1 norm')
    #plt.title('Bell state: Random seeds')
    plt.savefig(str(l_r*1000)+str(len(target_data))+names+'_all.pdf')
    plt.clf()


    for k in range(len(saved_error)):
        plt.plot(epochs_list, saved_error[k])

    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    #plt.title('Bell state: Random seeds')
    plt.savefig(str(l_r*1000)+str(len(target_data))+names+'_all.pdf')
    plt.clf()

        # Plot the subplots
    # Plot 1
    fig, axs = plt.subplots(2, sharex=True)
    #fig.suptitle('QBM training- Target: '+str(target_data))    #fig.xlabel('Epoch')
    fig.suptitle('QBM training- Target: Bell state')    #fig.xlabel('Epoch')

    #plt.figure(figsize=[11, 9])
    axs[1].errorbar(epochs_list, avg_list, std_list)
    axs[1].set(ylabel='Loss', xlabel='Epoch')
    axs[0].errorbar(epochs_list, avg_list_norm, std_list_norm)
    axs[0].set(ylabel='L1 Distance')
    fig.savefig('lr'+str(l_r*1000)+str(len(target_data))+names+'_both.pdf')

    return 

def main():
    #Seeding the simulator
    np.random.seed(2202)
    torch.manual_seed(2202)
    
    #Extra qubit to account for the phase derivatives. Does not seem to be 
    rz_add=False

    #Define parameters
    number_of_seeds=10          #Number of seeds, when doing multiple simulations
    learningRate=0.01           #Learning rate
    ite_steps=10                #Imaginary time steps
    epochs=50                   #Epochs
    optimizing_method='RMSprop' #Optimization technique

    if rz_add==False:
        #Ansatzes
        Ham1=       [[[1., 'z', 0]]]
        ansatz1=    [['ry',0, 0],['ry',0, 1], ['cx', 1,0], ['cx', 0, 1],
                    ['ry',np.pi/2, 0],['ry',0, 1], ['cx', 0, 1]]
            
        Ham2=       [[[0., 'z', 0], [0., 'z', 1]], 
                    [[0., 'z', 0]], [[0., 'z', 1]]]
        
        ansatz2=    [['ry',0, 0], ['ry',0, 1], ['ry',0, 2], ['ry',0, 3], 
                    ['cx', 3,0], ['cx', 2, 3],['cx', 1, 2], ['ry', 0, 3],
                    ['cx', 0, 1], ['ry', 0, 2], ['ry',np.pi/2, 0], 
                    ['ry',np.pi/2, 1], ['cx', 0, 2], ['cx', 1, 3]]

        Ham2_fidelity=      [[[1., 'z', 0], [1., 'z', 1]], [[-0.2, 'z', 0]], 
                            [[-0.2, 'z', 1]], [[0.3, 'x', 0]], [[0.3, 'x', 1]]]

        ansatz_gen_dis=[['ry',0, 0], ['ry',0, 1], ['ry',0, 2], ['ry', 0, 3],
                        ['rz',0, 0], ['rz',0, 1], ['rz',0, 2], ['rz', 0, 3], 
                        ['cx', 1,0], ['ry',np.pi/2, 0],        ['cx', 2, 1],
                        ['rz',0, 0], ['ry', np.pi/2, 1],       ['cx', 3, 2],
                        ['rz',0, 1], ['ry', 0, 2],['ry',0, 3], ['rz', 0, 2],
                        ['rz',0, 3], ['cx', 0, 2],['cx', 1, 3]]
    else:
        Ham1=       [[[1., 'z', 0]]]
        ansatz1=    [['ry',0, 0],['ry',0, 1], ['cx', 1,0], ['cx', 0, 1],
                    ['ry',np.pi/2, 0],['ry',0, 1], ['cx', 0, 1], ['rx', 0, 2]]

        Ham2=       [[[0., 'z', 0], [0., 'z', 1]], 
                    [[0., 'z', 0]], [[0., 'z', 1]]]
        ansatz2=    [['ry',0, 0], ['ry',0, 1], ['ry',0, 2], ['ry',0, 3], 
                    ['cx', 3,0], ['cx', 2, 3],['cx', 1, 2], ['ry', 0, 3],
                    ['cx', 0, 1], ['ry', 0, 2], ['ry',np.pi/2, 0], 
                    ['ry',np.pi/2, 1], ['cx', 0, 2], ['cx', 1, 3], ['rz', 0, 4]]
    

    #Target distributions
    p_data1=np.array([0.5, 0.5])
    p_data2=np.array([0.5, 0, 0, 0.5])

    Ham1=np.array(Ham1, dtype=object)
    Ham2=np.array(Ham2, dtype=object)

    #fraud_detection(1, ansatz2, 30, ite_steps, 0.1, optimizing_method, m1=0.7, m2=0.99)#000509_40_samples_both_sets')
    #fraud_detection(1, ansatz2, 30, ite_steps, 0.01, optimizing_method, m1=0.7, m2=0.99, network_coeff=layers)#000509_40_samples_both_sets')
    #fraud_detection(1, ansatz2, 30, ite_steps, 0.01, optimizing_method, network_coeff)#000509_40_samples_both_sets')
    #quantum_mnist(1, ansatz2, epochs, ite_steps, 0.01, optimizing_method, network_coeff=layers, nickname='network_24_3_4samples')
    #quantum_mnist(1, ansatz2, epochs, ite_steps, 0.1, optimizing_method, nickname='reg_bias_4samples')

    #multiple_simulations(number_of_seeds, Ham1, ansatz1, epochs, p_data1, optimizing_method,l_r=0.1, steps=ite_steps, names='H1_latest_10_seeds')
    #multiple_simulations(number_of_seeds, Ham2, ansatz2, epochs, p_data2, optimizing_method,l_r=0.1, steps=ite_steps, names='H2_latest_10_seeds')
    """Run these"""
    #multiple_simulations(number_of_seeds, Ham3, ansatz3, epochs, p_data3, optimizing_method,l_r=0.1, steps=ite_steps, names='H3_no_seed_new')
    #multiple_simulations(number_of_seeds, Ham4, ansatz4, epochs, p_data4, optimizing_method,l_r=0.1, steps=ite_steps, names='H4_no_seed')
    """
    """
    #learningrate_investigation(1, Ham1, ansatz1, 15, p_data1, optimizing_method,l_r=0.1, steps=ite_steps)
    #learningrate_investigation(number_of_seeds, Ham1, ansatz1, epochs, p_data1, optimizing_method,l_r=0.005, steps=ite_steps, name='09')
    #learningrate_investigation(number_of_seeds, Ham1, ansatz1, epochs, p_data1, optimizing_method,l_r=0.002, steps=ite_steps, name='09')
    #multiple_simulations(number_of_seeds, Ham1, ansatz1, epochs, p_data1, optimizing_method,l_r=0.1, steps=ite_steps, names='H1_10_seed_50_epoch')
    #multiple_simulations(1, Ham1, ansatz1, 100, p_data1, optimizing_method,l_r=0.1, steps=ite_steps, names='test_trash_run_dont_save')
    #multiple_simulations(number_of_seeds, Ham1, ansatz1, epochs, p_data1, optimizing_method,l_r=0.002, steps=ite_steps)
    
    #find_hamiltonian(ansatz2, ite_steps, learningRate, optimizing_method)

    #multiple_simulations(1, Ham2, ansatz_gen_dis, 25, p_data2, optimizing_method,l_r=learningRate, steps=10, names='test_trash_run_dont_save')
    #multiple_simulations(number_of_seeds, Ham1, ansatz1, epochs, p_data1, optimizing_method,l_r=learningRate, steps=ite_steps)
    #multiple_simulations(number_of_seeds, Ham2, ansatz2, epochs, p_data2, optimizing_method,l_r=learningRate, steps=ite_steps)
    
    """
    Fidelity simulations
    """
    #rz true and symmetric false gives best, 98.5 and 99.98
    #sim_plot_fidelity(ite_steps, rz_add=rz_add, name='Fidelity_dynamic_lmb_without_rz_new')#, 'Final_fidelity')#, 'after_statevector')#, 'fidelity_H1_H2_new_0_001minC')
    #sim_plot_fidelity(ite_steps, rz_add=True, name='Fidelity_dynamic_lmb_with_rz')#, 'Final_fidelity')#, 'after_statevector')#, 'fidelity_H1_H2_new_0_001minC')

    #sim_lambda_fidelity_search(ite_steps, np.logspace(-12,0,13), rz_add=False, name='without_rz_ab_new')
    #sim_lambda_fidelity_search(ite_steps, np.logspace(-12,0,13), rz_add=True, name='with_rz_ab')

    """
    Generative learning
    """
    ## Remember to save all arrays in case i run it for a real quantum computer, then save.
    #learning_rate_search(Ham1, ansatz1, epochs, p_data1, n_steps=ite_steps, lr=0.1, name='H1_ab_new', optim_method='SGD', plot=False)
    #learning_rate_search(Ham2, ansatz2, epochs, p_data2, n_steps=ite_steps, lr=0.1, name='H2_ab_new', optim_method='SGD', plot=False)
    
    #Use that learning rate to plot for various optimization methods, rms prop, adam, amsgrad, and sgd, each with different momemntum, maybe 2 or 3 momentums, same color of same thing

    #exhaustive_gen_search_paralell(Ham1, ansatz1, epochs, p_data1, n_steps=ite_steps)
    #exhaustive_gen_search_paralell(Ham2, ansatz2, epochs, p_data2, n_steps=ite_steps)

    #final_seed_sim(Ham2, ansatz2, epochs, p_data2, n_steps=ite_steps)
    #final_seed_sim(Ham1, ansatz1, epochs, p_data1, n_steps=ite_steps)

    #train_sim(Ham1, ansatz1, epochs, p_data1, n_steps=ite_steps,lr=0.1, optim_method='Amsgrad', m1=0.7, m2=0.99)
    #train(Ham1, ansatz1, epochs, p_data1, n_steps=ite_steps, lr=0.1, optim_method='Amsgrad', plot=False)
 
    #exit()
    #fraud_detection(1, ansatz2, n_epochs=100, lr=0.01, opt_met=optimizing_method, layers=test_layers)#[[[8,1],[8,1]], [0, 1]])#000509_40_samples_both_sets')
    #fraud_detection(1, ansatz2, n_epochs=100, lr=0.01, opt_met=optimizing_method, layers=test_layers)#[[[8,1],[8,1]], [0, 1]])#000509_40_samples_both_sets')
    #quantum_mnist(1, ansatz2, n_epochs=100, lr=0.01, optim_method=optimizing_method, layers=test_layers, QBM=True)#[[[8,1],[8,1]], [0, 1]])#000509_40_samples_both_sets')

    #train_sim(Ham1, ansatz1, epochs, p_data1, n_steps=ite_steps,lr=0.1, optim_method='RMSprop', m1=0.99, m2=0.99)
    
    #TODO: I think this kind of actually works, but the learning rate have to be small enough/correct
    #Ham22=[[[0., 'z', 0], [0., 'z', 1]], [[0., 'z', 0]], [[0., 'z', 1]], [[0, 'x', 0]],[[0, 'x', 1]]]

    #train_sim(Ham2, ansatz2, epochs, p_data2, n_steps=ite_steps,lr=0.01, optim_method='RMSprop', m1=0.99, m2=0.99, rz_add=rz_add)

    #train(Ham2, ansatz2, epochs, p_data2, n_steps=ite_steps, lr=0.1, optim_method='Amsgrad', plot=False)


    #Then use the generative, learning thing for q3? With 10 seeds to see that everyone converges
    #Run it again with real computer?
    
    """
    Discriminative learning- Fraud dataset
    """    
    #TODO: Make code with fraud regular- With bias variance?

    #TODO: Make code with network- With bias variance?
    #print(compute_NN_nodes(64, 3, 1))
    #print(compute_NN_nodes(64, 3, 2))
    """This is the one"""
    #fraud_sim(1, ansatz2, 50, ite_steps, 0.01, optimizing_method)#000509_40_samples_both_sets')
    #fraud_detection(1, ansatz2, 1, 0.01, optimizing_method, 0.99, 0, v_q=1, layers=NN_nodes(8,2), ml_task='classification', directory='test', name='more_test', samp_400=False)
    #quantum_mnist(1, ansatz2, 1, 0.01, optimizing_method, 0.99, 0, v_q=2, layers=NN_nodes(8,2), ml_task='classification', directory='test2', name='more_test2', samp_400=True)
    #quantum_mnist(1, ansatz2, 1, 0.01, optimizing_method, 0.99, 0, v_q=2, layers=None, ml_task='classification', directory='test3', name='more_test3', samp_400=True)
    #quantum_mnist(1, ansatz2, 1, 0.01, optimizing_method, 0.99, 0, v_q=2, layers=NN_nodes(123,19), ml_task='classification', directory='test4', name='more_test4', samp_400=False, big_mnist=True)
    #franke(1, ansatz2, 1, 0.01, optimizing_method, 0.99, 0, v_q=1, layers=NN_nodes(11,6), directory='test5', name='more_test5')


    #fraud_detection(1, ansatz2, n_epochs=100, lr=0.01, opt_met=optimizing_method, layers=None)#[[[8,1],[8,1]], [0, 1]])#000509_40_samples_both_sets')
    #TODO: What to do about the learning rates and stuff like that?
    #TODO: Layers and node tests?
    #TODO: Test for less samples?

    #TODO: Create code with Franke function to test
    #TODO: And also with mnist

    #Run with all 3 hamiltonians

    ###TASKS:
    # Start plot of franke function
    # Run with MNIST 4 samples?
    # Find best seed values
    # Plot the results created
    #Insert lots of results and write a bit on them and everything
    #Then just write methods part asap, like, just write out 5 pages of methods all night
    #should not take that long time since method part is just what I have been  doing during
    #the implementation
    #Preprocessing maybe?


    """
    Discriminative learning- Franke Function
    """
    #TODO: Fix the neural network thing with activation functions and number of neurons

    #franke(1, ansatz2, 100, learningRate, optimizing_method, m1=0.99, m2=0, directory=direc, name=name_file)

    """
    Discriminative learning- MNIST
    """


    """
    Classical Boltzmann machine
    """
    #fraud_detection(1, ansatz2, n_epochs=100, lr=0.1, opt_met=optimizing_method, samp_400=True, QBM=False)#[[[8,1],[8,1]], [0, 1]])#000509_40_samples_both_sets')
    #quantum_mnist(1, ansatz2, n_epochs=100, lr=0.01, optim_method=optimizing_method, layers=None, QBM=False, samp_400=True, big_mnist=True)#[[[8,1],[8,1]], [0, 1]])#000509_40_samples_both_sets')
    franke(1, ansatz2, 100, learningRate, optimizing_method, m1=0.99, m2=0, directory='frank_plot', name='frank_plot', QBM=False)



if __name__ == "__main__":
    main()
