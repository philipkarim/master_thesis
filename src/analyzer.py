#Importing necessary packages
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import precision_recall_fscore_support, accuracy_score, confusion_matrix
import seaborn as sns
import matplotlib

#Figure parameters, fit well into the latex document
matplotlib.rcParams['mathtext.fontset'] = 'stix'
matplotlib.rcParams['font.family'] = 'STIXGeneral'
sns.set_style("darkgrid")
FIGWIDTH=4.71935 #From latex document
FIGHEIGHT=FIGWIDTH/1.61803398875

params = {'text.usetex' : True,
          'font.size' : 10,
          'font.family' : 'lmodern',
          'figure.figsize' : [FIGWIDTH, FIGHEIGHT],
          #'text.latex.unicode': True,
          }
plt.rcParams.update(params)


def plot_fraud():
    """
    Plot the fraud functions
    """
    acc_train=np.load('results/fraud/acc_train_5050000509_40_samples_both_sets.npy', allow_pickle=True)
    acc_test=np.load('results/fraud/acc_test_5050000509_40_samples_both_sets.npy', allow_pickle=True)
    loss_train=np.load('results/fraud/loss_train_5050000509_40_samples_both_sets.npy', allow_pickle=True)
    loss_test=np.load('results/fraud/loss_test_5050000509_40_samples_both_sets.npy', allow_pickle=True)

    plt.plot(list(range(len(acc_train))), acc_train, label='Train set')
    plt.plot(list(range(len(acc_test))), acc_test, label='Test set')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    #plt.savefig('30 epochs.pdf')
    plt.show()

    plt.plot(list(range(len(loss_train))), loss_train, label='Train set')
    plt.plot(list(range(len(loss_test))), loss_test,  label='Test set')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    #plt.savefig('30 epochs.pdf')
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
    #plt.savefig('30 epochs.pdf')
    plt.show()

    plt.plot(list(range(len(loss_train_0001))), loss_train_0001, label=r'$\gamma=0.001$, $m_1=0.9$, $m_2=0.999$')
    plt.plot(list(range(len(loss_train_001))), loss_train_001,  label=r'$\gamma=0.01$, $m_1=0.9$, $m_2=0.999$')
    plt.plot(list(range(len(loss_train01))), loss_train01, label=r'$\gamma=0.1$, $m_1=0.9$, $m_2=0.999$')
    plt.plot(list(range(len(loss_train_000107))), loss_train_000107,  label=r'$\gamma=0.001$, $m_1=0.7$, $m_2=0.99$')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    #plt.savefig('30 epochs.pdf')
    plt.show()

def plot_lr_search():
    """
    Plots the search of the learning rate
    """
    loss=np.load('results/generative_learning/arrays/older/SGDloss_lr0.1m10m20loss_A2.npy', allow_pickle=True)
    lr=np.load('results/generative_learning/arrays/older/SGDloss_lr0.1m10m20lr_exp_A2.npy', allow_pickle=True)

    loss_A1=np.load('results/generative_learning/arrays/older/SGDloss_lr0.1m10m20loss_A1.npy', allow_pickle=True)
    lr_A1=np.load('results/generative_learning/arrays/older/SGDloss_lr0.1m10m20lr_exp_A1.npy', allow_pickle=True)

    print(len(loss_A1))
    print(loss_A1[:len(loss_A1)])
    print(loss[:len(loss)])

    sma=1
    skip=0

    #Plot derivatives
    derivatives = [0] * (sma + 1)
    for i in range(1 + sma, len(lr)):
        derivative = (loss[i] - loss[i - sma]) / sma
        derivatives.append(derivative)

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
    plt.savefig('results/generative_learning/SGDdLVSlr_ab_new.pdf')
    plt.clf

    plt.figure(figsize=[FIGWIDTH/2, FIGHEIGHT])
    plt.plot(range(0,len(lr_A1)), lr_A1)
    #plt.plot(range(0,len(lr)), lr, label=r'$H_2$')
    plt.xlabel('Iteration')
    plt.ylabel('Learning rate')
    #plt.legend()
    plt.tight_layout()
    plt.savefig('results/generative_learning/SGDitVSloss_hs_ab_new.pdf')
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
    plt.savefig('results/generative_learning/SGDlrVSloss_exp_SGDitVSloss_hs_ab_new.pdf')
    #plt.show()

def plot_optim_search():
    """
    Plot exhaustive search of optimal parameters and learning parameters
    """
    arrays_loss=[]
    arrays_norm=[]
    all='all'
    #keyw='H2_rms_search_ab'
    #folder='H2_rms'
    folder='H2_ab'
    keyw='H2_ab'
    half_size=False

    if all!='RMS':
        folder=keyw

    if all=='all':
        labels=[['Adam','0.2','0.9','0.999'], ['Adam','0.1','0.9','0.999'],['Adam','0.05','0.9','0.999'], 
                ['Amsgrad','0.2','0.9','0.999'], ['Amsgrad','0.1','0.9','0.999'],['Amsgrad','0.05','0.9','0.999'],
                ['RMSprop','0.2','0.99','0'], ['RMSprop','0.1','0.99','0'], ['RMSprop','0.05','0.99','0'],
                ['SGD','0.2','0','0'],  ['SGD','0.1','0','0'],  ['SGD','0.05','0','0']]

    elif all=='RMS':
        lr='0.1'
        labels=[['RMSprop',lr,'0.999','0'], ['RMSprop',lr,'0.99','0'],['RMSprop',lr,'0.9','0'],
         ['RMSprop',lr,'0.8','0'], ['RMSprop',lr,'0.7','0'], ['RMSprop',lr,'0.6','0']]

    elif all=='RMS_vs_ams':
        labels=[['RMSprop','0.1','0.999','0'], ['RMSprop','0.1','0.99','0'],['RMSprop','0.1','0.9','0'],
                ['RMSprop','0.1','0.8','0'], ['RMSprop','0.1','0.7','0'], ['RMSprop','0.1','0.6','0'],

                ['Amsgrad','0.1','0.9','0.999'], ['Amsgrad','0.1','0.9','0.99'],['Amsgrad','0.1','0.8','0.999'],
                ['Amsgrad','0.1','0.8','0.99'], ['Amsgrad','0.1','0.7','0.999'], ['Amsgrad','0.1','0.7','0.99'],
                ['Amsgrad','0.1','0.7','0.9'], ['Amsgrad','0.1','0.6','0.9']]

    else:
        labels=[['Adam','0.2','0.9','0.999'],['Adam','0.1','0.9','0.999'], ['Adam','0.05','0.9','0.999'], ['Adam','0.01','0.9','0.999'],
            ['Amsgrad','0.2','0.9','0.999'], ['Amsgrad','0.1','0.9','0.999'],['Amsgrad','0.05','0.9','0.999'], ['Amsgrad','0.01','0.9','0.999'],
            ['RMSprop','0.2','0.99','0'], ['RMSprop','0.1','0.99','0'], ['RMSprop','0.05','0.99','0'], ['RMSprop','0.01','0.99','0'],
            ['SGD','0.2','0','0'], ['SGD','0.1','0','0'], ['SGD','0.05','0','0'], ['SGD','0.01','0','0']]
    
    for i in labels:
        arrays_loss.append(np.load('results/generative_learning/arrays/search/'+folder+'/'+i[0]+'loss_lr'+i[1]+'m1'+i[2]+'m2'+i[3]+'loss'+keyw+'.npy', allow_pickle=True))
        arrays_norm.append(np.load('results/generative_learning/arrays/search/'+folder+'/'+i[0]+'loss_lr'+i[1]+'m1'+i[2]+'m2'+i[3]+'norm'+keyw+'.npy', allow_pickle=True))

    epoch=range(len(arrays_loss[0]))

    colors = ["tab:blue","tab:orange","tab:green","tab:red", "tab:purple", "tab:olive"]
    ticks = ["x","1",".","s", "D"]
    linestyle=['solid','dashed','dotted', 'dashdot']

    cat=[]

    if all!='all':
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
            elif i[1]=='0.01':
                temp.append(3)
            cat.append(temp)

    elif all=='RMS':
        for i in labels:
            temp=[]
            if i[2]=='0.999':
                temp.append(0)
            elif i[2]=='0.99':
                temp.append(1)
            elif i[2]=='0.9':
                temp.append(2)
            elif i[2]=='0.8':
                temp.append(3)
            elif i[2]=='0.7':
                temp.append(4)
            elif i[2]=='0.6':
                temp.append(5)

            if i[1]=='0.2':
                temp.append(0)
            elif i[1]=='0.1':
                temp.append(1)
            elif i[1]=='0.05':
                temp.append(2)
            
            cat.append(temp)
    else:
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
            
            if i[1]=='0.5':
                temp.append(3)
            elif i[1]=='0.2':
                temp.append(0)
            elif i[1]=='0.1':
                temp.append(1)
            elif i[1]=='0.05':
                temp.append(2)
            cat.append(temp)

    
    if all!='RMS_vs_ams':
        if half_size:
            plt.figure(figsize=[FIGWIDTH/2, FIGHEIGHT])
        else:
            plt.figure()
        if all!='RMS':
            for j, i in enumerate(arrays_loss):
                plt.plot(epoch, i, color=colors[cat[j][0]],linestyle=linestyle[cat[j][1]],label=labels[j][0]+', '+r'$\gamma=$'+labels[j][1])#, linewidth=1, ms=3)
        
        else:
            for j, i in enumerate(arrays_loss):
                plt.plot(epoch, i,label=r'$m=$'+labels[j][2])
        
        plt.xlabel('Iteration')
        plt.ylabel('Loss')
        #plt.yscale('log')
        plt.ylim(0.693, 0.695)
        plt.legend(prop={'size': 7.3}, loc="upper right", ncol=2) #bbox_to_anchor=[0, 1]       #plt.legend()
        plt.tight_layout()
        plt.savefig('results/generative_learning/'+keyw+'_loss_log_portion.pdf')
        plt.clf

        if half_size:
            plt.figure(figsize=[FIGWIDTH/2, FIGHEIGHT])
        else:
            plt.figure()
        if all!='RMS':
            for j,i in enumerate(arrays_norm):
                plt.plot(epoch, i, color=colors[cat[j][0]],linestyle=linestyle[cat[j][1]],label=labels[j][0]+', '+r'$\gamma=$'+labels[j][1], linewidth=1, ms=2)
        else:
            for j, i in enumerate(arrays_norm):
                plt.plot(epoch, i,label=r'$m=$'+labels[j][2])

        plt.xlabel('Iteration')
        #plt.yscale('log')
        plt.ylabel(r'L\textsubscript{1}-norm')
        #plt.ylim(0, 0.01)
        plt.legend(prop={'size': 4.4}, loc="upper right", ncol=2) #bbox_to_anchor=[0, 1]       #plt.legend()
        plt.tight_layout()
        plt.savefig('results/generative_learning/'+keyw+'_norm_log.pdf')
        plt.clf
    else:
        plt.figure()
        for j, i in enumerate(arrays_loss):
            if labels[j][0]=='RMSprop':
                plt.plot(epoch, i,label=r'$m=$'+labels[j][2])
        plt.xlabel('Iteration')
        plt.ylabel('Loss')
        plt.legend(fontsize=4, prop={'size': 6}, loc=0)
        plt.tight_layout()
        plt.savefig('results/generative_learning/test_rms_loss.pdf')
        plt.clf

        plt.figure()
        for j, i in enumerate(arrays_norm):
            if labels[j][0]=='RMSprop':
                plt.plot(epoch, i,label=r'$m=$'+labels[j][2])
        plt.xlabel('Iteration')
        plt.ylabel(r'L\textsubscript{1}-norm')
        plt.legend(fontsize=4, prop={'size': 6}, loc=0)
        plt.tight_layout()
        plt.savefig('results/generative_learning/test_rms_norm.pdf')
        plt.clf

        plt.figure()
        for j, i in enumerate(arrays_loss):
            if labels[j][0]=='Amsgrad':
                plt.plot(epoch, i,label=r'$m_1=$'+labels[j][2]+','+r'$m_2=$'+labels[j][3])
        plt.xlabel('Iteration')
        plt.ylabel('Loss')
        plt.legend(fontsize=4, prop={'size': 6}, loc=0)
        plt.tight_layout()
        plt.savefig('results/generative_learning/test_ams_loss.pdf')
        plt.clf

        plt.figure()
        for j, i in enumerate(arrays_norm):
            if labels[j][0]=='Amsgrad':
                plt.plot(epoch, i,label=r'$m_1=$'+labels[j][2]+','+r'$m_2=$'+labels[j][3])
        plt.xlabel('Iteration')
        plt.ylabel(r'L\textsubscript{1}-norm')
        plt.legend(prop={'size': 6}, loc=0)
        plt.tight_layout()
        plt.savefig('results/generative_learning/test_ams_norm.pdf')
        plt.clf

def plot_finale_seeds(std=False, plot_pbm=False):
    """
    Plott finale results of data generation
    """
    keyw='H1_ab'
    dir='H1_10seeds'
    n_seeds=10
    arrays_loss=[]
    array_norm=[]
    for i in range(n_seeds):
        arrays_loss.append(np.load('results/generative_learning/arrays/search/'+dir+'/RMSproploss_lr0.1m10.99m20loss'+keyw+'_10seedseed'+str(i)+'.npy', allow_pickle=True)[:21])
        array_norm.append(np.load('results/generative_learning/arrays/search/'+dir+'/RMSproploss_lr0.1m10.99m20norm'+keyw+'_10seedseed'+str(i)+'.npy', allow_pickle=True)[:21])
    
    epoch=range(len(arrays_loss[0]))

    if std==False:
        plt.figure()
        for j, i in enumerate(arrays_loss):
            plt.plot(epoch, i)
        plt.xlabel('Iteration')
        plt.ylabel('Loss')
        plt.tight_layout()
        plt.savefig('results/generative_learning/'+keyw+'_loss_10seeds.pdf')
        plt.clf()
        
        plt.figure()
        for j, i in enumerate(array_norm):
            plt.plot(epoch, i)
        plt.xlabel('Iteration')
        plt.ylabel(r'L\textsubscript{1}-norm')
        plt.tight_layout()
        plt.savefig('results/generative_learning/'+keyw+'_norm_10seeds.pdf')
        plt.clf()
    else:
        avg_list=np.mean(np.array(arrays_loss), axis=0)
        std_list=np.std(np.array(arrays_loss), axis=0)

        avg_list_norm=np.mean(np.array(array_norm), axis=0)
        std_list_norm=np.std(np.array(array_norm), axis=0)


        plt.figure()
        plt.errorbar(epoch, avg_list, std_list, ecolor='gray', capsize=5)
        plt.xlabel('Iteration')
        plt.ylabel('Loss')
        plt.tight_layout()
        plt.savefig('results/generative_learning/'+keyw+'_loss_10seeds_w_std.pdf')
        plt.clf()

        plt.figure()
        plt.errorbar(epoch, avg_list_norm, std_list_norm)
        plt.xlabel('Iteration')
        plt.ylabel(r'L\textsubscript{1}-norm')
        plt.tight_layout()
        plt.savefig('results/generative_learning/'+keyw+'_norm_10seeds_w_std.pdf')
        plt.clf()
        

        plt.figure()
        fig, axs = plt.subplots(2, sharex=True)
        axs[1].errorbar(epoch, avg_list, std_list, ecolor='tab:red', capsize=1.2, capthick=0.5, elinewidth=0.7, barsabove=False, fmt = 'o', markersize=0.9)
        axs[1].set(ylabel='Loss', xlabel='Iteration')
        axs[0].errorbar(epoch, avg_list_norm, std_list_norm, ecolor='tab:red', capsize=1.2, capthick=0.5, elinewidth=0.7, barsabove=False, fmt = 'o', markersize=0.9)
        axs[0].set(ylabel=r'L\textsubscript{1}-norm')
        plt.tight_layout()
        fig.savefig('results/generative_learning/'+keyw+'_sub_10seeds.pdf')
        plt.clf()

    #Plot histogram of best and worst sampling probability
    if plot_pbm:
        min_error=1000
        max_error=0.0
        pqbm_list=[]
        for i in range(n_seeds):
            pqbm_list.append(np.load('results/generative_learning/arrays/search/'+dir+'/RMSproploss_lr0.1m10.99m20pqbm_list'+keyw+'_10seedseed'+str(i)+'.npy', allow_pickle=True))

        for j in range(n_seeds):
            if min_error>array_norm[j][-1]:
                min_error=array_norm[j][-1]
                best_pbm=pqbm_list[j][-1]

            if max_error<array_norm[j][-1]:
                worst_pbm=pqbm_list[j][-1]
                max_error=array_norm[j][-1]

        print(f'best_pbm {best_pbm}')
        print(f'worst pbm {worst_pbm}')

        bell_state=[0.5,0,0,0.5]
        barWidth = 0.25

        plt.figure()
        # Set position of bar on X axis
        br1 = np.arange(len(bell_state))
        br2 = [x + barWidth for x in br1]
        br3 = [x + barWidth for x in br2]
        
        # Make the plot
        plt.bar(br1, bell_state, color ='tab:red', width = barWidth,
                edgecolor ='grey', label ='Bell state')
        plt.bar(br2, worst_pbm, color ='tab:green', width = barWidth,
                edgecolor ='grey', label ='Worst trained')
        plt.bar(br3, best_pbm, color ='tab:blue', width = barWidth,
                edgecolor ='grey', label ='Best trained')
        plt.xlabel('Sample')
        plt.ylabel('Probability')
        plt.xticks([r + barWidth for r in range(len(bell_state))],['00', '01', '10', '11'])
        plt.legend(loc="upper center")
        plt.tight_layout()
        plt.savefig('results/generative_learning/'+keyw+'RMS_lr01m099bar_10seeds.pdf')
        plt.clf()


"""
Discriminative learning plotter functions
"""
def plot_activation_functions():
    """
    Activation functions
    """
    labels=[['identity','Identity'],['sig','Sigmoid'],['relu','RELU'], ['leaky','Leaky RELU'],['tanh','Tanh'], ['sig_out','Sigmoid output'], ['tanh_out','Tanh output']]
    loss_list=[]
    name_start='loss_train12_2_'

    for i in range(len(labels)):
        loss_list.append(np.load('results/disc_learning/fraud/activations/'+name_start+labels[i][0]+'.npy', allow_pickle=True))

    epoch=range(len(loss_list[0]))
    
    plt.figure()
    for j, i in enumerate(loss_list):
        plt.plot(epoch, i, label=labels[j][1])
        
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.tight_layout()
    plt.legend(prop={'size': 9})
    plt.savefig('results/disc_learning/assets/loss_activations_12_2.pdf')
    plt.clf()

def plot_bias():
    """
    Plot bias
    """
    loss_list=[]
    name_start='loss_train12_2_'

    loss_list.append(np.load('results/disc_learning/fraud/bias/'+name_start+'bias_identity.npy', allow_pickle=True))
    loss_list.append(np.load('results/disc_learning/fraud/bias/'+name_start+'nobias_identity.npy', allow_pickle=True))

    epoch=range(len(loss_list[0]))

    plt.figure()
    plt.plot(epoch, loss_list[0], label='Bias')
    plt.plot(epoch, loss_list[1], label='No bias')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.tight_layout()
    plt.legend(prop={'size': 9 }, loc="lower right")
    plt.savefig('results/disc_learning/assets/bias_12_2_50_samples.pdf')
    plt.clf()

def plot_NN_sizes():
    """
    Plot NN tests
    """
    fraud=True
    if fraud:
        labels=[['8_5_001','[8, 5]'], ['4_4','[4, 4]'],['6','[6]'], ['6_6','[6, 6]'],['12_12','[12, 12]']]
        folder='NN_sizes_fraud'
        nickname='H2_fraud/H2_NNsizes'
    else:
        labels=[['23_8_001_m','[23, 8]'], ['4_4_m','[4, 4]'], ['8_8_m','[8, 8]'],['16_16_m','[16, 16]']]
        folder='NN_sizes_mnist'
        nickname='H2_mnist/H2_NNsizes'

    name_start='loss_trainH1_'
    loss_list=[]

    for i in range(len(labels)):
        loss_list.append(np.load('results/disc_learning/'+folder+'/'+name_start+labels[i][0]+'.npy', allow_pickle=True))
    epoch=range(len(loss_list[0]))

    plt.figure()
    for j, i in enumerate(loss_list):
        plt.plot(epoch, i, label=labels[j][1])
        
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.tight_layout()
    plt.legend(prop={'size': 8.25}, loc="upper right", ncol=1)
    plt.savefig('results/disc_learning/assets/'+nickname+'.pdf')
    plt.clf()

def plot_lr():
    """
    Plot learning rates
    """
    labels=[['01','0.1'],['001','0.01']]#,['0001','0.001']]
    #switch between two colors, different forms of lines on H1 and H2
    fraud=True
    if fraud:
        name_start='loss_trainH1_8_5_'
        nickname='H2_fraud/H2_lr_fraud_no_network'
        folder='lr_fraud_no_network'

        name_start=name_start[:-4]+'no_network_'
    else:
        name_start='loss_trainH1_23_8_'
        nickname='H2_mnist/H2_lr_mnist'
        folder='lr_mnist'

    loss_list=[]
    for i in range(len(labels)):
        loss_list.append(np.load('results/disc_learning/'+folder+'/'+name_start+labels[i][0]+'.npy', allow_pickle=True))
    epoch=range(len(loss_list[0]))
    
    plt.figure()
    for j, i in enumerate(loss_list):
        plt.plot(epoch, i, label=r'$\gamma=$'+labels[j][1])
        
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.tight_layout()
    plt.legend()
    plt.savefig('results/disc_learning/assets/'+nickname+'.pdf')
    plt.clf()

def plot_gen(name_start, labels, end):
    loss_list=[]
    for i in range(len(labels)):
        loss_list.append(np.load('results/disc_learning/fraud/optimizer/'+name_start+labels[i][0]+'.npy', allow_pickle=True))
    epoch=range(len(loss_list[0]))
    
    plt.figure(figsize=[FIGWIDTH, FIGHEIGHT/2])
    for j, i in enumerate(loss_list):
        if j%2==0:
            pass
            #plt.plot(epoch, i, label=labels[j][1])
        else:
            plt.plot(epoch, i, label=labels[j][1])
        
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.tight_layout()
    plt.legend(prop={'size': 6.5})#, loc="lower left")
    plt.savefig('results/disc_learning/assets/'+name_start+end+'.pdf')
    plt.clf()

def plot_gen_sub(name_start, labels, end):
    """
    Plot initialisation and stuff like that
    """
    loss_list=[]
    for i in range(len(labels)):
        loss_list.append(np.load('results/disc_learning/fraud/initialisation/'+name_start+labels[i][0]+'.npy', allow_pickle=True))

    epoch=range(len(loss_list[0]))
    
    plt.figure()
    fig, axs = plt.subplots(2, sharex=True)
    axs[1].errorbar(epoch, x1, y1, ecolor='tab:red', capsize=1.2, capthick=0.5, elinewidth=0.7, barsabove=False, fmt = 'o', markersize=0.9)
    axs[1].set(ylabel='Loss', xlabel='Iteration')
    axs[0].plot(epoch, x2, y2, ecolor='tab:red', capsize=1.2, capthick=0.5, elinewidth=0.7, barsabove=False, fmt = 'o', markersize=0.9)
    axs[0].set(ylabel='Loss')
    plt.tight_layout()
    plt.savefig('results/disc_learning/assets/'+name_start+end+'.pdf')

def format_labels(x, pos):
    """
    function to format label, from web
    """
    return "e$^{%i}$" % np.log(x)

def final_scores(prediction, target, CM, name):
    """
    Final score computations
    """
    precision, recal, beta, temp= precision_recall_fscore_support(target, prediction, average='macro')

    print(precision, recal, beta)
    print(f'Accuracy {accuracy_score(target, prediction)}')

    if CM: 
        cf_matrix = confusion_matrix(target, prediction, normalize='all')
        cf_matrix/=100
        plt.figure()
        ax = sns.heatmap(cf_matrix, annot=True,
            fmt='.2%', cmap='Blues', cbar=False)

        ## Ticket labels - List must be in alphabetical order
        if name[0]=='m' or name[0]=='d':
            ax.set_xlabel('\nPredicted labels')
            ax.set_ylabel('True labels ')
            ax.xaxis.set_ticklabels(['0','1', '2', '3'])
            ax.yaxis.set_ticklabels(['0','1', '2', '3'])
        else:
            ax.set_xlabel('\nPredicted labels')
            ax.set_ylabel('True labels ')
            ax.xaxis.set_ticklabels(['False','True'])
            ax.yaxis.set_ticklabels(['False','True'])

        ## Display the visualization of the Confusion Matrix.
        plt.tight_layout()
        plt.savefig('results/disc_learning/assets/final_runs/CM'+name+'.pdf')
        plt.clf()


def final_fraud(CM=False):
    """
    Computations of final run
    """
    path='results/disc_learning/final_runs/'
    net='final_run_fraud_network/'
    nonet='final_run_fraud_no_network/'

    l_tr2=np.load(path+net+'loss_trainH1_8_5_400_50_f_001.npy', allow_pickle=True)    
    l_tr5=np.load(path+nonet+'loss_trainH1_nonet_400_50_f.npy', allow_pickle=True)
    l_te2=np.load(path+net+'loss_testH1_8_5_400_50_f_001.npy', allow_pickle=True)
    l_te5=np.load(path+nonet+'loss_testH1_nonet_400_50_f.npy', allow_pickle=True)
    
    plt.plot(list(range(len(l_tr5))), l_tr5, label='Tr 45')
    plt.plot(list(range(len(l_te5))), l_te5, label='Te 45')

    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.tight_layout()
    #plt.savefig('check_results_fr_nonet.pdf')
    plt.clf()
    
    pred_nonet=np.load(path+nonet+'predictions_testH1_nonet_400_50_f.npy', allow_pickle=True)
    targ_nonet=np.load(path+nonet+'targets_testH1_nonet_400_50_f.npy', allow_pickle=True)

    pred=np.load(path+net+'predictions_testH1_8_5_400_50_f_001.npy', allow_pickle=True)
    targ=np.load(path+net+'targets_testH1_8_5_400_50_f_001.npy', allow_pickle=True)
    
    #pred=np.load(path+nonet+'predictions_testH1_nonet_400_50_f.npy', allow_pickle=True)
    #targ=np.load(path+nonet+'targets_testH1_nonet_400_50_f.npy', allow_pickle=True)

    acc=[]
    for j in range(len(pred)):
        target_list_net=[]
        for i in targ[j]:
            if i[0]==1:
                target_list_net.append(0)
            elif i[1]==1:
                target_list_net.append(1)
        acc.append(accuracy_score(pred[j], target_list_net))
    
    print(f'Acccuracy: {acc.index(max(acc))}')
    print(acc)


    target_list_net=[]
    target_list_nonet=[]

    pred_net=pred
    targ_net=targ

    pred_net=pred_net[acc.index(max(acc))]
    targ_net=targ_net[acc.index(max(acc))]
    
    #pred_net=pred_net[np.where(l_te2 == np.amin(l_te2))[0]][0]
    #targ_net=targ_net[np.where(l_te2 == np.amin(l_te2))[0]][0]

    pred_nonet=pred_nonet[np.where(l_te5 == np.amin(l_te5))[0]][0]
    targ_nonet=targ_nonet[np.where(l_te5 == np.amin(l_te5))[0]][0]
    
    for i in targ_net:
        if i[0]==1:
            target_list_net.append(0)
        else:
            target_list_net.append(1)

    for i in targ_nonet:
        if i[0]==1:
            target_list_nonet.append(0)
        else:
            target_list_nonet.append(1)

    target_list_net=np.array(target_list_net)
    target_list_nonet=np.array(target_list_nonet)

    final_scores(pred_net, target_list_net, CM, 'f_net_001')
    final_scores(pred_nonet, target_list_nonet, CM, 'f_nonet')


def final_franke():
    """
    Final frankes function
    """
    l_tr1=np.load('results/temp_results_final_runs/loss_trainH1_11_6_400_50_franke_0001.npy', allow_pickle=True)
    l_te1=np.load('results/temp_results_final_runs/loss_testH1_11_6_400_50_franke_0001.npy', allow_pickle=True)

    plt.figure()
    plt.plot(list(range(len(l_tr1))), l_tr1, label='Train')
    plt.plot(list(range(len(l_te1))), l_te1, label='Test')

    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.tight_layout()
    plt.savefig('franke_loss_final_check_mse.pdf')
    plt.clf()

    """
    fig = plt.figure() 
    ax = plt.axes(projection ='3d') 

    my_cmap = plt.get_cmap('YlGnBu')
    
    # Creating plot
    trisurf = ax.plot_trisurf(X[:,1], X[:,2], Y,cmap = my_cmap,
                            linewidth = 0.2,antialiased = True,
                            edgecolor = 'grey') 
    
    # Adding labels
    ax.set_xlabel('x'); ax.set_ylabel('y'); ax.set_zlabel('z')
    #plt.savefig('results/disc_learning/franke/'+file_title+'.pdf')
    plt.clf
    """


def final_digit(CM=False):
    """
    Plot digit datasets
    """
    path='results/disc_learning/final_runs/'
    net='final_run_digit_network/'
    nonet='final_run_digit_no_network/'
    
    l_tr2=np.load(path+net+'loss_trainH1_23_8_400_50_d.npy', allow_pickle=True)
    l_te2=np.load(path+net+'loss_testH1_23_8_400_50_d.npy', allow_pickle=True)

    l_tr5=np.load(path+nonet+'loss_trainH1_nonet_400_50_d.npy', allow_pickle=True)
    l_te5=np.load(path+nonet+'loss_testH1_nonet_400_50_d.npy', allow_pickle=True)

    plt.plot(list(range(len(l_tr2))), l_tr2, label='Train 45')
    plt.plot(list(range(len(l_te2))), l_te2, label='Test 45')

    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    #plt.savefig('check_results_di.pdf')
    plt.clf()

    pred_net=np.load(path+net+'predictions_testH1_23_8_400_50_d.npy', allow_pickle=True)
    targ_net=np.load(path+net+'targets_testH1_23_8_400_50_d.npy', allow_pickle=True)

    pred_nonet=np.load(path+nonet+'predictions_testH1_nonet_400_50_d.npy', allow_pickle=True)
    targ_nonet=np.load(path+nonet+'targets_testH1_nonet_400_50_d.npy', allow_pickle=True)
    
    pred_net=np.load(path+net+'predictions_testH1_23_8_500_40_d.npy', allow_pickle=True)
    targ_net=np.load(path+net+'targets_testH1_23_8_500_40_d.npy', allow_pickle=True)

    #Find accuracy
    #print(pred_net)
    #print(targ_net)

    acc=[]
    for j in range(len(pred_net)):
        target_list_net=[]
        for i in targ_net[j]:
            if i[0]==1:
                target_list_net.append(0)
            elif i[1]==1:
                target_list_net.append(1)
            elif i[2]==1:
                target_list_net.append(2)
            else:
                target_list_net.append(3)
        acc.append(accuracy_score(pred_net[j], target_list_net))
    
    print(f'Acccuracy: {max(acc)}')

    exit()

    target_list_net=[]
    target_list_nonet=[]

    pred_net=pred_net[acc.index(max(acc))]
    targ_net=targ_net[acc.index(max(acc))]

    #pred_net=pred_net[np.where(l_te2 == np.amin(l_te2))[0]][0]
    #targ_net=targ_net[np.where(l_te2 == np.amin(l_te2))[0]][0]


    pred_nonet=pred_nonet[np.where(l_te5 == np.amin(l_te5))[0]][0]
    targ_nonet=targ_nonet[np.where(l_te5 == np.amin(l_te5))[0]][0]

    for i in targ_net:
        if i[0]==1:
            target_list_net.append(0)
        elif i[1]==1:
            target_list_net.append(1)
        elif i[2]==1:
            target_list_net.append(2)
        else:
            target_list_net.append(3)

    for i in targ_nonet:
        if i[0]==1:
            target_list_nonet.append(0)
        elif i[1]==1:
            target_list_nonet.append(1)
        elif i[2]==1:
            target_list_nonet.append(2)
        else:
            target_list_nonet.append(3)

    target_list_net=np.array(target_list_net)
    target_list_nonet=np.array(target_list_nonet)

    final_scores(pred_net, target_list_net, CM, 'd_net')
    final_scores(pred_nonet, target_list_nonet, CM, 'd_nonet')


def final_mnist(CM):
    """
    Final mnist computations
    """
    path='results/disc_learning/final_runs/'
    net='final_run_mnist_network/'
    
    l_tr2=np.load(path+net+'loss_trainH1_32_32_400_50_mnist.npy', allow_pickle=True)
    l_te2=np.load(path+net+'loss_testH1_32_32_400_50_mnist.npy', allow_pickle=True)
    
    l_tr5=np.load(path+net+'loss_trainH1_123_19_400_50_mnist.npy', allow_pickle=True)
    l_te5=np.load(path+net+'loss_testH1_123_19_400_50_mnist.npy', allow_pickle=True)

    plt.plot(list(range(len(l_tr2))), l_tr2, label='r45,32')
    plt.plot(list(range(len(l_tr5))), l_tr5, label='r45,12')
    plt.plot(list(range(len(l_te2))), l_te2, label='e45,32')
    plt.plot(list(range(len(l_te5))), l_te5, label='e45,12')

    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.tight_layout()
    #plt.savefig('check_results_mnist.pdf')
    plt.clf()
    
    pred=np.load(path+net+'predictions_testH1_32_32_400_50_mnist.npy', allow_pickle=True)
    targ=np.load(path+net+'targets_testH1_32_32_400_50_mnist.npy', allow_pickle=True)
    target_list=[]

    acc=[]
    for j in range(len(pred)):
        target_list_net=[]
        for i in targ[j]:
            if i[0]==1:
                target_list_net.append(0)
            elif i[1]==1:
                target_list_net.append(1)
            elif i[2]==1:
                target_list_net.append(2)
            else:
                target_list_net.append(3)
        acc.append(accuracy_score(pred[j], target_list_net))

    print(f'Acccuracy: {max(acc)}')
    
    #pred=pred[np.where(l_te2 == np.amin(l_te2))[0]][0]
    #targ=targ[np.where(l_te2 == np.amin(l_te2))[0]][0]

    pred=pred[acc.index(max(acc))]
    targ=targ[acc.index(max(acc))]

    for i in targ:
        if i[0]==1:
            target_list.append(0)
        elif i[1]==1:
            target_list.append(1)
        elif i[2]==1:
            target_list.append(2)
        else:
            target_list.append(3)

    target_list=np.array(target_list)

    final_scores(pred, target_list, CM, 'm_net')




#plot_NN_sizes()
#plot_lr()
#plot_activation_functions()
#plot_bias()
#plot_finale_seeds(True, False)
#plot_lr_search()
#plot_fraud()
#plot_multiple_samples()
plot_optim_search()
#plot_three_sub()
#genereal_plotter('results/disc_learning/mnist/loss_trainnetwork_24_3_4samples.npy', 'mnist_12_sample_24_3_lr001_nosig')
#plot_gen('loss_trainsig_12_2_lr', [['001_ams','AMSgrad', 0],['001_H3_ams','AMSgrad', 1],['001','RMSProp', 0], ['001_H3','RMSProp', 1]], 'optim_sub')
#plot_gen('loss_train12_2_sig_', [['HN','He N'], ['HU','He U'], ['XN','Xavier N'],['XU','Xavier U']], 'initialisation')

#final_fraud(True)
#final_digit(True)
#final_mnist(True)
#final_franke()