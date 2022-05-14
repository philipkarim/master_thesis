# Common imports
import os
import numpy as np
import pandas as pd
from pandas import DataFrame
import matplotlib.pyplot as plt
#from scipy import *
import seaborn as sns
from matplotlib.ticker import ScalarFormatter
from statisticalhandling import block

sns.set_style("darkgrid")

CB91_Blue = '#2CBDFE'
CB91_Green = '#47DBCD'
CB91_Pink = '#F3A0F2'
CB91_Violet = '#661D98'
CB91_Amber = '#F5B14C'

color_list = [CB91_Blue, CB91_Pink, CB91_Green, CB91_Amber, CB91_Violet]
#plt.rcParams['axes.prop_cycle'] = plt.cycler(color=color_list)

def data_path(DATA_ID, dat_id):
    return os.path.join(DATA_ID, dat_id)

def plottsigma():    
    #Filenames
    fn_noint=['N=1D=1HN=4lr=100']

    #Folders
    folder= ["Results/no_interaction/step_size/gibbs/normal_distribution/"]

    file = np.loadtxt(data_path(folder[0], fn_noint[0]))

 
    plt.plot(file[:,1], file[:,0])
    plt.xlabel(r'$\sigma$',fontsize=12)
    plt.ylabel(r'$\langle E_L \rangle (a.u.)$',fontsize=12)
    plt.show()

    return

def plottsteps():    
    #Filenames
    fn_noint=['N=1D=1HN=2lr=350']
    fn_int=['N=2D=2HN=2lr=1']

    #Folders
    folder_noint= ["Results/no_interaction/step_size/bruteforce/normal_distribution/", "Results/no_interaction/step_size/importance/normal_distribution/"]
    folder_int= ["Results/interaction/step_size/bruteforce/normal_distribution/", "Results/interaction/step_size/importance/normal_distribution/"]

    no_int_bf = np.loadtxt(data_path(folder_noint[0], fn_noint[0]))
    no_int_is = np.loadtxt(data_path(folder_noint[1], fn_noint[0]))
    int_bf =    np.loadtxt(data_path(folder_int[0], fn_int[0]))
    int_is =    np.loadtxt(data_path(folder_int[1], fn_int[0]))

    plt.plot(no_int_bf[:,1], no_int_bf[:,0])
    plt.ylim(0.497,0.503)
    plt.xlabel('Step size',fontsize=12)
    plt.ylabel(r'$\langle E_L \rangle (a.u.)$',fontsize=12)
    plt.show()
    
    plt.plot(no_int_bf[::-1,1], no_int_bf[::-1,2])
    plt.xlabel('Step size',fontsize=12)
    plt.ylabel('Acceptance rate',fontsize=12)
    plt.show()

    plt.plot(no_int_is[:,1], no_int_is[:,0])
    plt.ylim(0.497,0.503)
    plt.xlabel('Time step',fontsize=12)
    plt.ylabel(r'$\langle E_L \rangle (a.u.)$',fontsize=12)
    plt.show()
    
    plt.plot(no_int_is[:,1], (no_int_is[:,2]))
    plt.xlabel('Time step',fontsize=12)
    plt.ylabel('Acceptance rate',fontsize=12)
    plt.show()

    plt.plot(int_bf[:,1], int_bf[:,0])
    plt.xlabel('Step size',fontsize=12)
    plt.ylabel(r'$\langle E_L \rangle (a.u.)$',fontsize=12)
    plt.show()
    
    plt.plot(int_bf[:,1], (int_bf[:,2]))
    plt.xlabel('Step size',fontsize=12)
    plt.ylabel('Acceptance rate',fontsize=12)
    plt.show()

    plt.plot(int_is[:,1], int_is[:,0])
    plt.xlabel('Time step',fontsize=12)
    plt.ylabel(r'$\langle E_L \rangle (a.u.)$',fontsize=12)
    plt.show()

    plt.plot(int_is[:,1], (int_is[:,2]))
    plt.xlabel('Time step',fontsize=12)
    plt.ylabel('Acceptance rate',fontsize=12)
    plt.show()
    return

def plot_distributions():
    #Folders
    folder= ["Results/no_interaction/distribution_investigation/normal_distribution/", "Results/no_interaction/distribution_investigation/uniform_distribution/"]
    
    #Filenames
    fn_bf=['HN=2lr=11', 'HN=2lr=15', 'HN=2lr=110', 'HN=2lr=1250']

    infile1 = np.loadtxt(data_path(folder[0], fn_bf[0]))
    infile2 = np.loadtxt(data_path(folder[0], fn_bf[1]))
    infile3 = np.loadtxt(data_path(folder[0], fn_bf[2]))
    infile4 = np.loadtxt(data_path(folder[0], fn_bf[3]))

    #print((infile1[:,0]).size())
    x=np.linspace(0,len(infile1), len(infile1))

    plt.plot(x, infile1, label='(0, 0.001)')
    plt.plot(x, infile2, label='(0, 0.005)')
    plt.plot(x, infile3, label='(0, 0.01)')
    #plt.plot(x, infile4, label='(0, 0.25)')

    #plt.plot(infile3[:,0], x, label='(0, 0.25)')
    plt.xlabel('RBM cycles',fontsize=12)
    plt.ylabel(r'$\langle E_L \rangle (a.u.)$',fontsize=12)
    #plt.grid()
    plt.legend()
    plt.show()

    infile5 = np.loadtxt(data_path(folder[1], fn_bf[0]))
    infile6 = np.loadtxt(data_path(folder[1], fn_bf[1]))
    infile7 = np.loadtxt(data_path(folder[1], fn_bf[2]))
    infile8 = np.loadtxt(data_path(folder[1], fn_bf[3]))
    
    plt.plot(x, infile5, label='(-0.001, 0.001)')
    plt.plot(x, infile6, label='(-0.005, 0.005)')
    plt.plot(x, infile7, label='(-0.01, 0.01)')
    plt.plot(x, infile8, label='(-0.25, 0.25)')
   
    plt.xlabel('RBM cycles',fontsize=12)
    plt.ylabel(r'$\langle E_L \rangle(a.u.) $',fontsize=12)
    #plt.grid()
    plt.legend()
    #plt.legend(loc=(0.67, 0.67))
    plt.show()
    
    return

def plot_lr_nodes():
    #Folders
    folder_noint= ["Results/no_interaction/nodes_and_lr/bruteforce/", "Results/no_interaction/nodes_and_lr/importance/", "Results/no_interaction/nodes_and_lr/gibbs/"]
    folder_int= ["Results/interaction/nodes_and_lr/bruteforce/", "Results/interaction/nodes_and_lr/importance/", "Results/interaction/nodes_and_lr/gibbs/"]

    #Filenames
    fn_noint=['N=1D=1energies', 'N=1D=1specs']
    fn_int=['N=2D=2energies', 'N=2D=2specs']

    
    #No interaction
    energy_bf_noint= data_path(folder_noint[0], fn_noint[0])
    specs_bf_noint =np.transpose(np.loadtxt(data_path(folder_noint[0], fn_noint[1])))
    energy_is_noint= data_path(folder_noint[1], fn_noint[0])
    specs_is_noint=np.transpose(np.loadtxt(data_path(folder_noint[1], fn_noint[1])))
    energy_gibbs_noint= data_path(folder_noint[2], fn_noint[0])
    specs_gibbs_noint =np.transpose(np.loadtxt(data_path(folder_noint[2], fn_noint[1])))
    #Interaction
    energy_bf_int= data_path(folder_int[0], fn_int[0])
    specs_bf_int=np.transpose(np.loadtxt(data_path(folder_int[0], fn_int[1])))
    energy_is_int= data_path(folder_int[1], fn_int[0])
    specs_is_int=np.transpose(np.loadtxt(data_path(folder_int[1], fn_int[1])))
    energy_gibbs_int= data_path(folder_int[2], fn_int[0])
    specs_gibbs_int=np.transpose(np.loadtxt(data_path(folder_int[2], fn_int[1])))
    
    #Switch these files
    spec_file=specs_gibbs_int
    file=energy_gibbs_int
    length_energies=2**17
    
    axis_lr=[]
    axis_nodes=[]
    axis_heat_lr=[]
    axis_heat_nodes=[]
    energy_mean=[]
    energy_std=[]

    n_runs=len(spec_file[0])

    #Using the blocking method on one part at the time of the files 
    for i in range(0, n_runs):
        axis_nodes.append(str(spec_file[0][i]))
        axis_lr.append(str(spec_file[1][i]))
        if str(spec_file[0][i])[:-2] not in axis_heat_nodes:
            node_int=str(spec_file[0][i])
            axis_heat_nodes.append(node_int[:-2])
        if str(spec_file[1][i]) not in axis_heat_lr:
            axis_heat_lr.append(str(spec_file[1][i]))
        specs_is_int_test= np.loadtxt(file, skiprows=i*length_energies\
                                      +i, max_rows=length_energies)
        mean_e, std_v=block(np.transpose(specs_is_int_test))
        energy_mean.append(mean_e)
        energy_std.append(std_v)
    
    #Creating matrix, used for heatmap
    lr_frequency= axis_lr.count(axis_lr[0])
    nodes_frequency= axis_nodes.count(axis_nodes[0])

    heatmap_mean=np.zeros((lr_frequency, nodes_frequency))
    heatmap_std=np.full_like(heatmap_mean,0)

    index=0
    for lr in range(0, nodes_frequency):
        for hn in range(0, lr_frequency):
            heatmap_mean[hn][lr]=energy_mean[index]
            heatmap_std[hn][lr]=energy_std[index]
            index+=1
    
    df_mean = DataFrame(heatmap_mean, index=axis_heat_nodes, columns=axis_heat_lr)
    df_std = DataFrame(heatmap_std, index=axis_heat_nodes, columns=axis_heat_lr)

    sns.heatmap(df_mean, annot=True, fmt='.2f', cbar_kws={'label': r'$\langle E_L \rangle (a.u.)$'})
    plt.xlabel('Learning rate',fontsize=12)
    plt.ylabel('Hidden nodes',fontsize=12)
    plt.show()

    sns.heatmap(df_std, annot=True, fmt='.2g', cbar_kws={'label': 'Error by blocking method (a.u.)'})
    plt.xlabel('Learning rate',fontsize=12)
    plt.ylabel('Hidden nodes',fontsize=12)
    plt.show()
    
    return

def computefinalresults():
    #Folders
    folder_noint= ["Results/no_interaction/bruteforce/normal_distribution/", "Results/no_interaction/importance/normal_distribution/", "Results/no_interaction/gibbs/normal_distribution/"]
    folder_int= ["Results/interaction/bruteforce/normal_distribution/", "Results/interaction/importance/normal_distribution/", "Results/interaction/gibbs/normal_distribution/"]

    #Filenames
    fn=['N=1D=1', 'N=1D=2', 'N=1D=3', 'N=2D=2']

    #No interaction
    bf_noint_1d= data_path(folder_noint[0], fn[0])
    is_noint_1d= data_path(folder_noint[1], fn[0])
    gibbs_noint_1d= data_path(folder_noint[2], fn[0])
    bf_noint_2d= data_path(folder_noint[0], fn[1])
    is_noint_2d= data_path(folder_noint[1], fn[1])
    gibbs_noint_2d= data_path(folder_noint[2], fn[1])
    bf_noint_3d= data_path(folder_noint[0], fn[2])
    is_noint_3d= data_path(folder_noint[1], fn[2])
    gibbs_noint_3d= data_path(folder_noint[2], fn[2])
    #Interaction
    bf_int= data_path(folder_int[0], fn[3])
    is_int= data_path(folder_int[1], fn[3])
    gibbs_int= data_path(folder_int[2], fn[3])
    
    #Can be plotted much more elegantly, fix this if it is time    
    mean_e, std_v=block(np.loadtxt(bf_noint_1d))
    print(mean_e, std_v)
    mean_e, std_v=block(np.loadtxt(is_noint_1d))
    print(mean_e, std_v)
    mean_e, std_v=block(np.loadtxt(gibbs_noint_1d))
    print(mean_e, std_v)
    mean_e, std_v=block(np.loadtxt(bf_noint_2d))
    print(mean_e, std_v)
    mean_e, std_v=block(np.loadtxt(is_noint_2d))
    print(mean_e, std_v)
    mean_e, std_v=block(np.loadtxt(gibbs_noint_2d))
    print(mean_e, std_v)
    mean_e, std_v=block(np.loadtxt(bf_noint_3d))
    print(mean_e, std_v)
    mean_e, std_v=block(np.loadtxt(is_noint_3d))
    print(mean_e, std_v)
    mean_e, std_v=block(np.loadtxt(gibbs_noint_3d))
    print(mean_e, std_v)
    mean_e, std_v=block(np.loadtxt(bf_int))
    print(mean_e, std_v)
    mean_e, std_v=block(np.loadtxt(is_int))
    print(mean_e, std_v)
    mean_e, std_v=block(np.loadtxt(gibbs_int))
    print(mean_e, std_v)

    
    return

#plottsigma()
#plot_distributions()
#plottsteps()
#plot_lr_nodes()
computefinalresults()



