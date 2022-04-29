"""
alt+z to fix word wrap

Rotating the monitor:
xrandr --output DP-1 --rotate right
xrandr --output DP-1 --rotate normal

xrandr --query to find the name of the monitors
"""
import copy
import numpy as np

#Import scikit learn modules
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

# Import the other classes and functions
from varQITE import *
from utils import *
from NN_class import *
from train_supervised import train_model
from BM import *


def franke_function(x, y, noise_sigma=0):
    """
    Frankes function
    """
    x_nine = 9 * x
    y_nine = 9 * y
    first = 0.75 * np.exp(-(x_nine - 2)**2 * 0.25 - (y_nine - 2)**2 * 0.25)
    second = 0.75 * np.exp(-(x_nine + 1)**2 / 49 - (y_nine + 1)**2 * 0.1)
    third = 0.5 * np.exp(-(x_nine - 7)**2 * 0.25 - (y_nine - 3)**2 * 0.25)
    fourth = - 0.2 * np.exp(-(x_nine - 4)**2 - (y_nine - 7)**2)
    if noise_sigma != 0:
        rand = np.random.normal(0, noise_sigma)

        return first + second + third + fourth + rand
    else:
        return first + second + third + fourth

def design_matrix(x, y, d):
    """Function for setting up a design X-matrix with rows [1, x, y, x², y², xy, ...]
    Input: x and y mesh, argument d is the degree.
    """

    if len(x.shape) > 1:
    # reshape input to 1D arrays (easier to work with)
        x = np.ravel(x)
        y = np.ravel(y)

    N = len(x)
    p = int((d+1)*(d+2)/2)	# number of elements in beta
    X = np.ones((N,p))

    for n in range(1,d+1):
        q = int((n)*(n+1)/2)
        for m in range(n+1):
            X[:,q+m] = x**(n-m)*y**m

    return X

def plot_franke(N=20, file_title='real_franke'):
    deg = 2
    x = np.linspace(0, 1, N); y = np.linspace(0, 1, N)
    x, y = np.meshgrid(x, y)
    x = np.ravel(x); y = np.ravel(y)
    X = design_matrix(x, y, deg)
    Y = franke_function(x, y, noise_sigma=0.1)

    fig = plt.figure() 
    ax = plt.axes(projection ='3d') 
    
    # Creating color map
    #my_cmap = plt.get_cmap('cividis')
    #my_cmap = plt.get_cmap('magma')
    my_cmap = plt.get_cmap('YlGnBu')
    
    # Creating plot
    trisurf = ax.plot_trisurf(X[:,1], X[:,2], Y,cmap = my_cmap,
                            linewidth = 0.2,antialiased = True,
                            edgecolor = 'grey') 
    
    # Adding labels
    ax.set_xlabel('x'); ax.set_ylabel('y'); ax.set_zlabel('z')
    #plt.tight_layout()
    plt.savefig('results/disc_learning/franke/'+file_title+'.pdf')
    plt.clf


def franke(H_num, ansatz, n_epochs, lr, 
            opt_met, v_q=1, ml_task='regression', 
            m1=0.9, m2=0.999, layers=None, 
            directory='', name=None, QBM=True, init_ww='xavier_normal'):
    """
    Function to run regression of the franke function with the variational Boltzmann machine

    Args:
            initial_H(array):   The Hamiltonian which will be used, the parameters 
                                will be initialized within this function

            ansatz(array):      Ansatz whill be used in the VarQBM

            network_coeff(list): layerwise [input, output, bias], 0 if no bias, 1 with bias

    Returns:    Scores on how the BM performed
    """

    #Importing the data
    N=20

    #Bias variance tradeoff with complexity?
    deg = 5
    x = np.linspace(0, 1, N); y = np.linspace(0, 1, N)
    x, y = np.meshgrid(x, y)
    x = np.ravel(x); y = np.ravel(y)
    X = design_matrix(x, y, deg)
    Y = franke_function(x, y, noise_sigma=0.1)

    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, random_state=42)

    #Now it is time to scale the data
    #MinMax data due two the values of the qubits will give the target value
    scaler=MinMaxScaler()
    scaler.fit(X_train)
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)
    
    y_train=np.reshape(y_train,(-1,1))
    y_test=np.reshape(y_test,(-1,1))
    
    target_scaler=MinMaxScaler()
    target_scaler.fit(np.reshape(y_train,(-1,1)))
    y_train = target_scaler.transform(y_train)
    y_test = target_scaler.transform(y_test)

    #X_train=X_train[[0]]
    #y_train=y_train[[0]]
    #X_test=X_test[[0]]
    #y_test=y_test[[0]]

    y_train=np.ravel(y_train)
    y_test=np.ravel(y_test)

    data_franke=[X_train, y_train, X_test, y_test]
    params_franke=[n_epochs, opt_met, lr, m1, m2]

    if QBM ==True:
        train_model(data_franke, H_num, ansatz, params_franke, visible_q=v_q, task=ml_task, folder=directory, network_coeff=layers, nickname=name, init_w=init_ww)
    else:
        #Does not work
        best_params=None
        #best_params=gridsearch_params(data_fraud, 10)
        train_rbm(data_franke, best_params)
        #rbm_plot_scores(data_fraud, name='fraud2')
