from matplotlib.colors import makeMappingArray
import numpy as np
import random
import qiskit as qk

# Import the other classes and functions
from PQC import QML
from optimize_loss import optimize
from utils import *
#from varqbm import *

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

###okay lets fucking do this! Here is the plan:

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
n_qubits=2
q1_test = qk.QuantumRegister(n_qubits)
V_series_test = qk.QuantumCircuit(q1_test)

theta_test=np.random.uniform(-1,1, size=1)

V_series_test.crx(2*np.pi*theta_test[0],0,1)

#V_series_test.ry(2*np.pi*theta_test[0],0)

print(V_series_test.num_unitary_factors())

print(V_series_test.width())


#print(V_series_test.qbit_argument_conversion(V_series_test))

print(V_series_test)

#rint(V_series_test.num_ctrl_qubits())

def derivative_U(U_gate):
    qubits=U_gate.width()
    for k in range(qubits):
        if U_gate.num_ctrl_qubits()==0:
            return np.imag(j/2)* something
    return 








"""
#Define V

qc_ansatz = qk.QuantumCircuit(data_register, classical_register)

#Creating the ansatz circuit:
for i in range(n_qubits):
    qc_.rz(2*np.pi*sample[qubit],data_register[qubit])


tot_gates=0
for block in range(blocks):
    for rot_y in range(n_qubits):
        if rot_y+n_qubits*block<n_parameters:
            qc_ansatz.ry(2*np.pi*thetas[rot_y+n_qubits*block], data_register[rot_y])
            tot_gates+=1
        if rot_y!=0 and tot_gates<len(thetas)-reminder_gates-1:
            qc_ansatz.cx(data_register[rot_y-1], data_register[rot_y])

#Entangling the qubits before measuring
c3h_gate = XGate().control(3)
qc_ansatz.append(c3h_gate, data_register)
"""
"""
Define expression A and C as functions, maybe use a class? or parameter shift rule
(The derivative of U can be computed by the formula below)
"""




#Creating some qunatum cirquit object
#qc=QML(Hamiltonian,X.shape[1], 1, n_params, backend="qasm_simulator", shots=1024)



"""
def train(circuit, n_epochs, n_batch_size, initial_thetas,lr, X_tr, y_tr, X_te, y_te):
    
     Train(and validation) function that runs the simulation

     Args:
             circuit:        Quantum circuit (object from the QML class)
             n_epochs:       Number of epochs(integer) 
             n_batch_size:   Batch size (integer)
             initial_thetas: Initialisation values of the variational parameters
                             (List or array)
             lr:             Learning rate (float)
             X_tr:           Training data (2D array or list)
             y_tr:           True labels of the training data(list or array)
             X_te:           Test data (2D array or list)
             y_tr:           True labels of the test data(list or array)
     Returns:
             loss_train,     Loss of train set per epoch (list)
             accuracy_train, Accuracy of train set per epoch (list)
             loss_test:      Loss of test set per epoch (list)
             accuracy_test:  Accuracy of test set per epoch (list)

    #Creating optimization object
    optimizer=optimize(lr, circuit)
    #Splits the dataset into batches
    batches=len(X_tr)//n_batch_size
    #Adds another batch if it has a reminder
    if len(X_tr)%n_batch_size!=0:
         batches+=1
    #Reshapes the data into batches
    theta_params=initial_thetas.copy()

    #Defines a list containing all the prediction for each epoch
    prediction_epochs_train=[]
    loss_train=[]
    accuracy_train=[]

    prediction_epochs_test=[]
    loss_test=[]
    accuracy_test=[]

    temp_list=[]
    #Train parameters
    for epoch in range(n_epochs):
        for batch in range(batches):
            batch_pred=circuit.predict(X_reshaped[batch],theta_params)
            temp_list+=batch_pred
            theta_params=optimizer.gradient_descent(theta_params, batch_pred, y_tr[batch:batch+n_batch_size], X_reshaped[batch])
    
        #Computes loss and predicts on the test set with the new parameters
        train_loss=optimizer.binary_cross_entropy(temp_list,y_tr)
        test_pred=circuit.predict(X_te,theta_params)
        test_loss=optimizer.binary_cross_entropy(test_pred,y_te)
    
        #Tresholding the probabillities into hard label predictions
        temp_list=hard_labels(temp_list, 0.5)
        test_pred=hard_labels(test_pred,0.5)

        #Computes the accuracy scores
        acc_score=accuracy_score(y_tr,temp_list)
        acc_score_test=accuracy_score(y_te, test_pred)
    
        print(f"Epoch: {epoch}, loss:{train_loss}, accuracy:{acc_score}")
        #Appends the results to lists
        loss_train.append(train_loss)
        accuracy_train.append(acc_score)
        prediction_epochs_train.append(temp_list)
        loss_test.append(test_loss)#
        accuracy_test.append(acc_score_test)
        prediction_epochs_test.append(test_pred)

        temp_list.clear()

    return loss_train, accuracy_train, loss_test, accuracy_test
    """


"""
cmd+c+k to comment
cmd+c+u to uncomment
"""
# #This import is just because of some duplicate of mpi or armadillo on the computer
# import os
# os.environ['KMP_DUPLICATE_LIB_OK']='True'
# #Importing packages 
# import numpy as np
# import matplotlib.pyplot as plt
# import random

# from sklearn import datasets
# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import MinMaxScaler
# from sklearn.utils import shuffle

# #Import the other classes and functions
# from PQC import QML
# from optimize_loss import optimize
# from utils import *

# #Seeding the program to ensure reproducibillity
# random.seed(2021)

# #Set parameters
# n_params=10         #Number of variational parameters
# learning_rate=0.01  #Learning rate
# init_params=np.random.uniform(0.,0.01,size=n_params) #Distribution of the initial variational parameters
# ansatz=0            #Choose ansatz: 0 represent ansatz 1 in the report, 1 represent ansatz 2 in the report
# epochs=50           #Number of epochs
# batch_size=1        #Batch size, set equal to 1

# #Choose type of investigation, if all parameters are chosen prehand, 
# # set all to false
# inspect_distribution=False
# inspect_lr_param=False
# regular_run_save=False

# #Choose a datset
# dataset="iris"

# #Handling the data
# if dataset=="iris":
#     iris = datasets.load_iris()
#     X = iris.data
#     y = iris.target
#     idx = np.where(y < 2) # we only take the first two targets.

#     X = X[idx,:]
#     X=np.squeeze(X, axis=0)
#     y = y[idx]
    
#     path="Results/saved_data/iris/"

# else:
#     print("No datset chosen\nClosing the program")
#     quit()

# X, y = shuffle(X, y, random_state=0)

# #Splitting the dataset into a trainset and validation set. 
# # (The validation set is named test set in the code)
# X_train,X_test, y_train, y_test = train_test_split(X,y,test_size=0.25, stratify=y)

# #Scaling the data
# scaler2 = MinMaxScaler()
# scaler2.fit(X_train)
# X_train = scaler2.transform(X_train)
# X_test = scaler2.transform(X_test)

# #Creating some qunatum cirquit object
# qc=QML(ansatz,X.shape[1], 1, n_params, backend="qasm_simulator", shots=1024)
# qc_2=QML(1,X.shape[1], 1, 30, backend="qasm_simulator", shots=1024)

# def train(circuit, n_epochs, n_batch_size, initial_thetas,lr, X_tr, y_tr, X_te, y_te):
#     """
#     Train(and validation) function that runs the simulation

#     Args:
#             circuit:        Quantum circuit (object from the QML class)
#             n_epochs:       Number of epochs(integer) 
#             n_batch_size:   Batch size (integer)
#             initial_thetas: Initialisation values of the variational parameters
#                             (List or array)
#             lr:             Learning rate (float)
#             X_tr:           Training data (2D array or list)
#             y_tr:           True labels of the training data(list or array)
#             X_te:           Test data (2D array or list)
#             y_tr:           True labels of the test data(list or array)
#     Returns:
#             loss_train,     Loss of train set per epoch (list)
#             accuracy_train, Accuracy of train set per epoch (list)
#             loss_test:      Loss of test set per epoch (list)
#             accuracy_test:  Accuracy of test set per epoch (list)
#     """
#     #Creating optimization object
#     optimizer=optimize(lr, circuit)
#     #Splits the dataset into batches
#     batches=len(X_tr)//n_batch_size
#     #Adds another batch if it has a reminder
#     if len(X_tr)%n_batch_size!=0:
#         batches+=1
#     #Reshapes the data into batches
#     X_reshaped=np.reshape(X_tr,(batches,n_batch_size,X_tr.shape[1]))
#     theta_params=initial_thetas.copy()

#     #Defines a list containing all the prediction for each epoch
#     prediction_epochs_train=[]
#     loss_train=[]
#     accuracy_train=[]
    
#     prediction_epochs_test=[]
#     loss_test=[]
#     accuracy_test=[]
    
#     temp_list=[]
#     #Train parameters
#     for epoch in range(n_epochs):
#         for batch in range(batches):
#             batch_pred=circuit.predict(X_reshaped[batch],theta_params)
#             temp_list+=batch_pred
#             theta_params=optimizer.gradient_descent(theta_params, batch_pred, y_tr[batch:batch+n_batch_size], X_reshaped[batch])
        
#         #Computes loss and predicts on the test set with the new parameters
#         train_loss=optimizer.binary_cross_entropy(temp_list,y_tr)
#         test_pred=circuit.predict(X_te,theta_params)
#         test_loss=optimizer.binary_cross_entropy(test_pred,y_te)
        
#         #Tresholding the probabillities into hard label predictions
#         temp_list=hard_labels(temp_list, 0.5)
#         test_pred=hard_labels(test_pred,0.5)

#         #Computes the accuracy scores
#         acc_score=accuracy_score(y_tr,temp_list)
#         acc_score_test=accuracy_score(y_te, test_pred)
        
#         print(f"Epoch: {epoch}, loss:{train_loss}, accuracy:{acc_score}")
#         #Appends the results to lists
#         loss_train.append(train_loss)
#         accuracy_train.append(acc_score)
#         prediction_epochs_train.append(temp_list)
#         loss_test.append(test_loss)
#         accuracy_test.append(acc_score_test)
#         prediction_epochs_test.append(test_pred)

#         temp_list.clear()

#     return loss_train, accuracy_train, loss_test, accuracy_test


# def investigate_distribution(type, folder_name):
#     """
#     Function that investigate the best initialisation of the varational parameters
#     and also tests if uniform- or normal distribution is best
    
#     Needs to be ran at an unix system bases computer due to the fork commands

#     args: 
#             type: "U" or "N" for uniform or normal
#     """
#     pid = os.fork()
#     if pid:
#         train_loss, train_accuracy, test_loss, test_accuracy=train(qc, epochs, batch_size, getDistribution(type, 0.01, n_params), learning_rate, X_tr=X_train,y_tr=y_train, X_te=X_test,y_te=y_test)
#         np.save(data_path(folder_name, "train"+type+"0_01.npy"), np.array(train_loss))
#         np.save(data_path(folder_name, "test"+type+"0_01.npy"), np.array(test_loss))

#     else:
#         pid1 = os.fork()
#         if pid1:
#             train_loss, train_accuracy, test_loss, test_accuracy=train(qc, epochs, batch_size, getDistribution(type, 0.1, n_params), learning_rate, X_tr=X_train,y_tr=y_train, X_te=X_test,y_te=y_test)
#             np.save(data_path(folder_name, "train"+type+"0_1.npy"), np.array(train_loss))
#             np.save(data_path(folder_name, "test"+type+"0_1.npy"), np.array(test_loss))
#         else:
#             pid2=os.fork()
#             if pid2:
#                 train_loss, train_accuracy, test_loss, test_accuracy=train(qc, epochs, batch_size, getDistribution(type, 0.25, n_params), learning_rate, X_tr=X_train,y_tr=y_train, X_te=X_test,y_te=y_test)
#                 np.save(data_path(folder_name, "train"+type+"0_25.npy"), np.array(train_loss))
#                 np.save(data_path(folder_name, "test"+type+"0_25.npy"), np.array(test_loss))            
#             else:
#                 train_loss, train_accuracy, test_loss, test_accuracy=train(qc, epochs, batch_size, getDistribution(type, 0.001, n_params), learning_rate, X_tr=X_train,y_tr=y_train, X_te=X_test,y_te=y_test)
#                 np.save(data_path(folder_name, "train"+type+"0_001.npy"), np.array(train_loss))
#                 np.save(data_path(folder_name, "test"+type+"0_001.npy"), np.array(test_loss))
    
#     return

# def investigate_lr_params(folder_name, folder_name2, lr_list, n_params_list):
#     """
#     Function that investigate the best learning rate and learning rate
    
#     Needs to be ran at an unix system bases computer due to the fork commands

#     args: 
#             folder_name:    place of saving the folder
#             lr_list:        list of learning rates
#             n_params_list:  list of number of parameters to run 
#     """
#     pid = os.fork()
#     if pid:
#         for i in lr_list:
#             for j in n_params_list:
#                 print("Ansatz 0: ", j, i)
#                 qc_lr_n=QML(0,X.shape[1], 1, j, backend="qasm_simulator", shots=1024)
#                 train_loss, train_accuracy, test_loss, test_accuracy=train(qc_lr_n, epochs, batch_size, np.random.uniform(0.,0.1,size=j), i, X_tr=X_train,y_tr=y_train, X_te=X_test,y_te=y_test)
#                 np.save(data_path(folder_name, "lr_params/train_lr"+str(i)+"_n"+str(j)+".npy"), np.array(train_loss))
#                 np.save(data_path(folder_name, "lr_params/test_lr"+str(i)+"_n"+str(j)+".npy"), np.array(test_loss))

#     else:
#         for ii in lr_list:
#             for jj in n_params_list:
#                 print("Ansatz 1: ",jj, ii)
#                 qc_2_lr_n=QML(1,X.shape[1], 1, jj, backend="qasm_simulator", shots=1024)
#                 train_loss, train_accuracy, test_loss, test_accuracy=train(qc_2_lr_n, epochs, batch_size, np.random.uniform(0.,0.1,size=jj), ii, X_tr=X_train,y_tr=y_train, X_te=X_test,y_te=y_test)
#                 np.save(data_path(folder_name2, "lr_params/train_lr"+str(ii)+"_n"+str(jj)+".npy"), np.array(train_loss))
#                 np.save(data_path(folder_name2, "lr_params/test_lr"+str(ii)+"_n"+str(jj)+".npy"), np.array(test_loss))
    
#     return

# #Below is just some code handling the runs of functions
# #and saving the predictions to file
# if ansatz==0:
#     folder="ansatz_0/"
# else:
#     folder="ansatz_1/"

# file_folder=data_path(path,folder)

# if inspect_distribution==True:
#     file_folder=data_path(path,folder)
#     investigate_distribution("N", file_folder)

# elif inspect_lr_param==True:
#     file_folder2=data_path(path,"ansatz_1/")
#     #Learning rates and number of paraeters to investigate
#     lrs=[0.1, 0.01, 0.001, 0.0001]
#     n_par=[10, 14, 18, 22, 26, 30]
#     investigate_lr_params(file_folder, file_folder2, lrs, n_par)

# elif regular_run_save==True:
#     #Paralellize the code to save some time
#     pid = os.fork()
#     if pid:
#         train_loss, train_accuracy, test_loss, test_accuracy =train(qc, epochs, batch_size, 
#                                                             np.random.uniform(0.,0.01,size=30), learning_rate, X_tr=X_train,
#                                                             y_tr=y_train, X_te=X_test,y_te=y_test)
#         np.save(path+"ansatz_1/trainloss_optimal.npy", np.array(train_loss))
#         np.save(path+"ansatz_1/testloss_optimal.npy", np.array(test_loss))
#         np.save(path+"ansatz_1/trainacc_optimal.npy", np.array(train_accuracy))
#         np.save(path+"ansatz_1/testacc_optimal.npy", np.array(test_accuracy))
#     else:
#         train_loss, train_accuracy, test_loss, test_accuracy =train(qc_2, epochs, batch_size, 
#                                                             init_params, learning_rate, X_tr=X_train,
#                                                             y_tr=y_train, X_te=X_test,y_te=y_test)
#         np.save(path+"ansatz_0/trainloss_optimal.npy", np.array(train_loss))
#         np.save(path+"ansatz_0/testloss_optimal.npy", np.array(test_loss))
#         np.save(path+"ansatz_0/trainacc_optimal.npy", np.array(train_accuracy))
#         np.save(path+"ansatz_0/testacc_optimal.npy", np.array(test_accuracy))
        
# else:
#     train_loss, train_accuracy, test_loss, test_accuracy =train(qc, epochs, batch_size, 
#                                                             init_params, learning_rate, X_tr=X_train,
#                                                             y_tr=y_train, X_te=X_test,y_te=y_test)

