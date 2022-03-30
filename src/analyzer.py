
from cProfile import label
from random import sample
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


#sns.set_style("darkgrid")
#plt.style.use('science')
#x=np.load('results/arrays/learningrate0.507.npy', allow_pickle=True)


import seaborn as sns
import matplotlib
matplotlib.rcParams['mathtext.fontset'] = 'stix'
matplotlib.rcParams['font.family'] = 'STIXGeneral'

sns.set_style("darkgrid")
#print(plt.rcParams.keys())

FIGWIDTH=4.71935 #From latex document
FIGHEIGHT=FIGWIDTH/1.61803398875

params = {'text.usetex' : True,
          'font.size' : 10,
          'font.family' : 'lmodern',
          'figure.figsize' : [FIGWIDTH, FIGHEIGHT],
          'figure.dpi' : 1000.0,
          #'text.latex.unicode': True,
          }
plt.rcParams.update(params)


def plot_fraud():
    acc_train=np.load('results/fraud/acc_train_5050000509_40_samples_both_sets.npy', allow_pickle=True)
    acc_test=np.load('results/fraud/acc_test_5050000509_40_samples_both_sets.npy', allow_pickle=True)
    loss_train=np.load('results/fraud/loss_train_5050000509_40_samples_both_sets.npy', allow_pickle=True)
    loss_test=np.load('results/fraud/loss_test_5050000509_40_samples_both_sets.npy', allow_pickle=True)

    plt.plot(list(range(len(acc_train))), acc_train, label='Train set')
    plt.plot(list(range(len(acc_test))), acc_test, label='Test set')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    #plt.savefig('30 epochs.png')
    plt.show()

    plt.plot(list(range(len(loss_train))), loss_train, label='Train set')
    plt.plot(list(range(len(loss_test))), loss_test,  label='Test set')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    #plt.savefig('30 epochs.png')
    plt.show()

def plot_multiple_samples():
    """
    Plots multiple data investigating the optimal parameters
    """
    acc_train_0001=np.load('results/fraud/acc_train_5050000109_10_samples.npy', allow_pickle=True)
    loss_train_0001=np.load('results/fraud/loss_train_5050000109_10_samples.npy', allow_pickle=True)

    acc_train_001=np.load('results/fraud/acc_train_505000109_10_samples.npy', allow_pickle=True)
    loss_train_001=np.load('results/fraud/loss_train_505000109_10_samples.npy', allow_pickle=True)

    acc_train01=np.load('results/fraud/acc_train_50500109_10_samples.npy', allow_pickle=True)
    loss_train01=np.load('results/fraud/loss_train_50500109_10_samples.npy', allow_pickle=True)

    acc_train_000107=np.load('results/fraud/acc_train_5050000107_10_samples.npy', allow_pickle=True)
    loss_train_000107=np.load('results/fraud/loss_train_5050000107_10_samples.npy', allow_pickle=True)

    plt.plot(list(range(len(acc_train_0001))), acc_train_0001, label=r'$\gamma=0.001$, $m_1=0.9$, $m_2=0.999$')
    plt.plot(list(range(len(acc_train_001))), acc_train_001, label=r'$\gamma=0.01$, $m_1=0.9$, $m_2=0.999$')
    plt.plot(list(range(len(acc_train01))), acc_train01, label=r'$\gamma=0.1$, $m_1=0.9$, $m_2=0.999$')
    plt.plot(list(range(len(acc_train_000107))), acc_train_000107, label=r'$\gamma=0.001$, $m_1=0.7$, $m_2=0.99$')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    #plt.savefig('30 epochs.png')
    plt.show()

    plt.plot(list(range(len(loss_train_0001))), loss_train_0001, label=r'$\gamma=0.001$, $m_1=0.9$, $m_2=0.999$')
    plt.plot(list(range(len(loss_train_001))), loss_train_001,  label=r'$\gamma=0.01$, $m_1=0.9$, $m_2=0.999$')
    plt.plot(list(range(len(loss_train01))), loss_train01, label=r'$\gamma=0.1$, $m_1=0.9$, $m_2=0.999$')
    plt.plot(list(range(len(loss_train_000107))), loss_train_000107,  label=r'$\gamma=0.001$, $m_1=0.7$, $m_2=0.99$')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    #plt.savefig('30 epochs.png')
    plt.show()

def plot_lr_search():
    loss=np.load('results/generative_learning/arrays/SGDloss_lr0.1m10m20loss.npy', allow_pickle=True)
    lr=np.load('results/generative_learning/arrays/SGDloss_lr0.1m10m20lr_exp.npy', allow_pickle=True)

    loss_A1=np.load('results/generative_learning/arrays/SGDloss_lr0.1m10m20loss_A2.npy', allow_pickle=True)
    lr_A1=np.load('results/generative_learning/arrays/SGDloss_lr0.1m10m20lr_exp_A2.npy', allow_pickle=True)


    #print(lr)
    #print(lr_A1)
    sma=1
    skip=0

    derivatives = [0] * (sma + 1)
    for i in range(1 + sma, len(lr)):
        derivative = (loss[i] - loss[i - sma]) / sma
        derivatives.append(derivative)

    print(min(derivatives), derivatives.index(min(derivatives)), lr[77])

    derivatives2 = [0] * (sma + 1)
    for i in range(1 + sma, len(lr_A1)):
        derivative2 = (loss_A1[i] - loss_A1[i - sma]) / sma
        derivatives2.append(derivative2)

    print(min(derivatives2), derivatives2.index(min(derivatives2)), lr_A1[derivatives2.index(min(derivatives2))])

    plt.figure()
    plt.ylabel("d/loss")
    plt.xlabel("learning rate (log scale)")
    plt.plot(lr[skip:], derivatives[skip:])
    plt.xscale('log')
    plt.tight_layout()
    plt.savefig('results/generative_learning/SGDdLVSlr.png')
    plt.clf


    plt.figure(figsize=[FIGWIDTH/2, FIGHEIGHT])
    plt.plot(range(0,len(lr_A1)), lr_A1)
    #plt.plot(range(0,len(lr)), lr, label=r'$H_2$')
    plt.xlabel('Iterations')
    plt.ylabel('Learning rate')
    #plt.legend()
    plt.tight_layout()
    plt.savefig('results/generative_learning/SGDitVSloss_hs.png')
    #plt.show()
    plt.clf

    plt.figure(figsize=[FIGWIDTH/2, FIGHEIGHT])
    plt.plot(lr_A1, loss_A1, label=r'$H_1$')
    plt.plot(lr, loss, label=r'$H_2$')
    plt.xlabel('Learning rate')
    plt.ylabel('Loss')
    plt.xscale("log")
    plt.legend()
    plt.tight_layout()
    plt.savefig('results/generative_learning/SGDlrVSloss_exp_SGDitVSloss_hs.png')
    #plt.show()

def plot_optim_search():
    """
    Plotting exhaustive search of optimap parameters and learning parameters
    """
    arrays_loss=[]
    arrays_norm=[]
    all=False
    if all:
        labels=[['Adam','0.2','0.9','0.999'],['Adam','0.1','0.9','0.999'], ['Adam','0.05','0.9','0.999'], 
                ['Amsgrad','0.2','0.9','0.999'], ['Amsgrad','0.1','0.9','0.999'],['Amsgrad','0.05','0.9','0.999'],['Amsgrad','0.1','0.7','0.99'],
                ['RMSprop','0.2','0.99','0'], ['RMSprop','0.1','0.99','0'], ['RMSprop','0.05','0.99','0'], ['RMSprop','0.1','0.7','0'],
                ['SGD','0.2','0','0'],  ['SGD','0.1','0','0'],  ['SGD','0.05','0','0']]

    else:
        labels=[['Adam','0.2','0.9','0.999'],['Adam','0.1','0.9','0.999'], ['Adam','0.05','0.9','0.999'], 
            ['Amsgrad','0.2','0.9','0.999'], ['Amsgrad','0.1','0.9','0.999'],['Amsgrad','0.05','0.9','0.999'],
            ['RMSprop','0.2','0.99','0'], ['RMSprop','0.1','0.99','0'], ['RMSprop','0.05','0.99','0'],
            ['SGD','0.2','0','0'],  ['SGD','0.1','0','0'],  ['SGD','0.05','0','0']]
        #labels=[['Adam','0.2','0.9','0.999'],['Adam','0.1','0.9','0.999'], ['Adam','0.05','0.9','0.999']]

    
    """
    names_loss=['Adamloss_lr0.1m10.9m20.999lossH1_real', 'Adamloss_lr0.2m10.9m20.999lossH1_real', 'Adamloss_lr0.05m10.9m20.999lossH1_real',
                'Amsgradloss_lr0.1m10.7m20.99lossH1_real', 'Amsgradloss_lr0.1m10.9m20.999lossH1_real', 'Amsgradloss_lr0.2m10.9m20.999lossH1_real',
                'Amsgradloss_lr0.05m10.9m20.999lossH1_real', 'RMSproploss_lr0.1m10.7m20lossH1_real', 'RMSproploss_lr0.1m10.99m20lossH1_real',
                'RMSproploss_lr0.2m10.99m20lossH1_real', 'RMSproploss_lr0.05m10.99m20lossH1_real', 'SGDloss_lr0.1m10m20lossH1_real',
                'SGDloss_lr0.2m10m20lossH1_real', 'SGDloss_lr0.05m10m20lossH1_real']


    names_norm=['Adamloss_lr0.1m10.9m20.999normH1_real', 'Adamloss_lr0.2m10.9m20.999normH1_real', 'Adamloss_lr0.05m10.9m20.999normH1_real',
                'Amsgradloss_lr0.1m10.7m20.99normH1_real', 'Amsgradloss_lr0.1m10.9m20.999normH1_real', 'Amsgradloss_lr0.2m10.9m20.999normH1_real',
                'Amsgradloss_lr0.05m10.9m20.999normH1_real', 'RMSproploss_lr0.1m10.7m20normH1_real', 'RMSproploss_lr0.1m10.99m20normH1_real',
                'RMSproploss_lr0.2m10.99m20normH1_real', 'RMSproploss_lr0.05m10.99m20normH1_real', 'SGDloss_lr0.1m10m20normH1_real',
                'SGDloss_lr0.2m10m20normH1_real', 'SGDloss_lr0.05m10m20normH1_real']
    """
    for i in labels:
        arrays_loss.append(np.load('results/generative_learning/arrays/search/'+i[0]+'loss_lr'+i[1]+'m1'+i[2]+'m2'+i[3]+'lossH1_real'+'.npy', allow_pickle=True))
        arrays_norm.append(np.load('results/generative_learning/arrays/search/'+i[0]+'loss_lr'+i[1]+'m1'+i[2]+'m2'+i[3]+'normH1_real'+'.npy', allow_pickle=True))

    epoch=range(len(arrays_loss[0]))

    plt.figure()
    
    #plt.rcParams['legend.fontsize'] = 12

    colors = ["tab:blue","tab:orange","tab:green","tab:red", "tab:purple", "tab:olive"]
    ticks = ["x","1",".","s", "D"]

    cat=[]
    for i in labels:
        temp=[]
        if i[0]=='Adam':
            temp.append(0)
        elif i[0]=='Amsgrad':
            temp.append(1)
        elif i[0]=='RMSprop':
            temp.append(2)
        elif i[0]=='SGD':
            temp.append(3)
        
        if i[1]=='0.2':
            temp.append(0)
        elif i[1]=='0.1':
            temp.append(1)
        elif i[1]=='0.05':
            temp.append(2)
        
        cat.append(temp)

    
    
    plt.figure()
    for j, i in enumerate(arrays_loss):
        plt.plot(epoch, i, color=colors[cat[j][0]],marker=ticks[cat[j][1]],label=labels[j][0]+', '+r'$\gamma=$'+labels[j][1], linewidth=1, ms=2)
    
    plt.xlabel('Iterations')
    plt.ylabel('Loss')
    #plt.yscale("log")

    plt.legend(fontsize=5, prop={'size': 8})
    #plt.legend()
    plt.tight_layout()
    plt.savefig('results/generative_learning/test_loss.png')
    plt.clf

    plt.figure()
    for j,i in enumerate(arrays_norm):
        plt.plot(epoch, i, color=colors[cat[j][0]],marker=ticks[cat[j][1]],label=labels[j][0]+', '+r'$\gamma=$'+labels[j][1], linewidth=1, ms=2)

    #plt.plot(range(0,len(lr)), lr, label=r'$H_2$')
    plt.xlabel('Iterations')
    plt.ylabel('Norm')
    #plt.yscale("log")
    plt.legend(fontsize=5, prop={'size': 8})

    #plt.legend()
    plt.tight_layout()
    plt.savefig('results/generative_learning/test_norm.png')
    plt.clf
    


#plot_lr_search()
#plot_fraud()
#plot_multiple_samples()
plot_optim_search()