import matplotlib.pyplot as plt
import numpy as np
import matplotlib

matplotlib.rcParams['mathtext.fontset'] = 'stix'
matplotlib.rcParams['font.family'] = 'STIXGeneral'

#print(plt.rcParams.keys())

FIGWIDTH=4.71935 #From latex document
FIGHEIGHT=FIGWIDTH/1.61803398875

params = {'text.usetex' : True,
          'font.size' : 10,
          'font.family' : 'lmodern',
          'figure.figsize' : [FIGWIDTH, FIGHEIGHT],
          #'text.latex.unicode': True,
          }
plt.rcParams.update(params)

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
    plt.tight_layout()
    plt.savefig('results/disc_learning/franke/'+file_title+'.pdf')
    plt.clf

plot_franke()