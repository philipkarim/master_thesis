
from cProfile import label
from random import sample
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.ticker as tick
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
    loss=np.load('results/generative_learning/arrays/SGDloss_lr0.1m10m20lossH2_ab.npy', allow_pickle=True)
    lr=np.load('results/generative_learning/arrays/SGDloss_lr0.1m10m20lr_expH2_ab.npy', allow_pickle=True)

    loss_A1=np.load('results/generative_learning/arrays/SGDloss_lr0.1m10m20lossH1_ab.npy', allow_pickle=True)
    lr_A1=np.load('results/generative_learning/arrays/SGDloss_lr0.1m10m20lr_expH1_ab.npy', allow_pickle=True)


    #print(lr)
    #print(lr_A1)
    sma=1
    skip=0

    derivatives = [0] * (sma + 1)
    for i in range(1 + sma, len(lr)):
        derivative = (loss[i] - loss[i - sma]) / sma
        derivatives.append(derivative)

    #print(min(derivatives), derivatives.index(min(derivatives)), lr[77])

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
    plt.savefig('results/generative_learning/SGDdLVSlr_ab.pdf')
    plt.clf


    plt.figure(figsize=[FIGWIDTH/2, FIGHEIGHT])
    plt.plot(range(0,len(lr_A1)), lr_A1)
    #plt.plot(range(0,len(lr)), lr, label=r'$H_2$')
    plt.xlabel('Iteration')
    plt.ylabel('Learning rate')
    #plt.legend()
    plt.tight_layout()
    plt.savefig('results/generative_learning/SGDitVSloss_hs_ab.pdf')
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
    plt.savefig('results/generative_learning/SGDlrVSloss_exp_SGDitVSloss_hs_ab.pdf')
    #plt.show()

def plot_optim_search():
    """
    Plotting exhaustive search of optimap parameters and learning parameters
    """
    arrays_loss=[]
    arrays_norm=[]
    all='RMS'
    keyw='H2_rms_search_ab'
    folder='H2_rms'

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
        #labels=[['Adam','0.2','0.9','0.999'],['Adam','0.1','0.9','0.999'], ['Adam','0.05','0.9','0.999']]

    
    for i in labels:
        arrays_loss.append(np.load('results/generative_learning/arrays/search/'+folder+'/'+i[0]+'loss_lr'+i[1]+'m1'+i[2]+'m2'+i[3]+'loss'+keyw+'.npy', allow_pickle=True)[0:50])
        arrays_norm.append(np.load('results/generative_learning/arrays/search/'+folder+'/'+i[0]+'loss_lr'+i[1]+'m1'+i[2]+'m2'+i[3]+'norm'+keyw+'.npy', allow_pickle=True)[0:50])

    epoch=range(len(arrays_loss[0]))

    
    #plt.rcParams['legend.fontsize'] = 12

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
                #plt.plot(epoch, i, color=colors[cat[j][0]],marker=ticks[cat[j][1]],label=r'$\gamma=$'+labels[j][1]+r'$m=$'+labels[j][2], linewidth=1, ms=2)
        
        plt.xlabel('Iteration')
        plt.ylabel('Loss')
        #plt.yscale("log")

        plt.legend(prop={'size': 7.5}, loc="upper right", ncol=2) #bbox_to_anchor=[0, 1]       #plt.legend()
        plt.tight_layout()
        plt.savefig('results/generative_learning/'+keyw+'_loss.pdf')
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
                #plt.plot(epoch, i, color=colors[cat[j][0]],marker=ticks[cat[j][1]],label=r'$\gamma=$'+labels[j][1]+r'$m=$'+labels[j][2], linewidth=1, ms=2)
    
        #plt.plot(range(0,len(lr)), lr, label=r'$H_2$')
        plt.xlabel('Iteration')
        plt.ylabel(r'L\textsubscript{1}-norm')
        #plt.yscale("log")
        #plt.legend(fontsize=4, prop={'size': 6})
        #plt.legend(prop={'size': 6})
        plt.legend(prop={'size': 7.5}, loc="upper right", ncol=2) #bbox_to_anchor=[0, 1]       #plt.legend()

        #plt.legend()
        plt.tight_layout()
        plt.savefig('results/generative_learning/'+keyw+'_norm.pdf')
        plt.clf
    else:
        plt.figure()
        for j, i in enumerate(arrays_loss):
            if labels[j][0]=='RMSprop':
                plt.plot(epoch, i,label=r'$m=$'+labels[j][2])
        plt.xlabel('Iteration')
        plt.ylabel('Loss')
        plt.legend(fontsize=4, prop={'size': 6}, loc=0)
        #plt.legend()
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
        #plt.legend()
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
        #plt.legend()
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
        #plt.legend(prop={'size': 6})
        #plt.legend()
        
        #plt.legend(bbox_to_anchor=(1.05, 1.0), loc='upper left')
        plt.tight_layout()
        plt.savefig('results/generative_learning/test_ams_norm.pdf')
        plt.clf

def plot_three_sub():
    """
    Plotting exhaustive search of optimap parameters and learning parameters
    """
    arrays_loss=[]
    keyw='H2_real_new'

    labels=[['Adam','0.2','0.9','0.999'],['Adam','0.1','0.9','0.999'], ['Adam','0.05','0.9','0.999'],
            ['Amsgrad','0.2','0.9','0.999'], ['Amsgrad','0.1','0.9','0.999'],['Amsgrad','0.05','0.9','0.999'],
            ['RMSprop','0.2','0.99','0'], ['RMSprop','0.1','0.99','0'], ['RMSprop','0.05','0.99','0'],
            ['SGD','0.2','0','0'], ['SGD','0.1','0','0'], ['SGD','0.05','0','0']]
    
    for i in labels:
        arrays_loss.append(np.load('results/generative_learning/arrays/search/'+keyw+'/'+i[0]+'loss_lr'+i[1]+'m1'+i[2]+'m2'+i[3]+'loss'+keyw+'.npy', allow_pickle=True))

    epoch=range(len(arrays_loss[0]))

    
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
        elif i[1]=='0.01':
            temp.append(3)

        cat.append(temp)

        #plt.figure()
    fig, axs = plt.subplots(3, sharex=True, sharey=False)

    #plt.figure()
    for j, i in enumerate(arrays_loss):
        axs[cat[j][1]].plot(epoch, i, color=colors[cat[j][0]],marker=ticks[cat[j][1]],label=labels[j][0], linewidth=1, ms=2)
        #axs[1].plot(epoch, i, color=colors[cat[j][0]],marker=ticks[cat[j][1]],label=r'$\gamma=$'+labels[j][1]+r'$m=$'+labels[j][2], linewidth=1, ms=2)
        #axs[2].plot(epoch, i, color=colors[cat[j][0]],marker=ticks[cat[j][1]],label=r'$\gamma=$'+labels[j][1]+r'$m=$'+labels[j][2], linewidth=1, ms=2)
    
    fontsz=8
    axs[0].set_title('$\gamma=$'+labels[0][1], fontsize=fontsz)
    axs[1].set_title('$\gamma=$'+labels[1][1], fontsize=fontsz)
    axs[2].set_title('$\gamma=$'+labels[2][1], fontsize=fontsz)

    plt.xlabel('Iteration')
    axs[0].set_ylabel('Loss')
    axs[1].set_ylabel('Loss')
    axs[2].set_ylabel('Loss')

    axs[0].legend(prop={'size': 6}, bbox_to_anchor=[1, 1], loc="upper left")   #plt.legend()
    plt.tight_layout()
    
    #plt.subplots_adjust(left=0.1,bottom=0.1, right=0.9, top=0.9, wspace=0.4, hspace=0.4)
    plt.subplots_adjust(hspace=0.4)

    plt.savefig('results/generative_learning/loss_3_sub.pdf')
    plt.clf()

def plot_finale_seeds(std=False, plot_pbm=False):
    """
    Plotting exhaustive search of optimap parameters and learning parameters
    """
    keyw='H2_ab'
    dir='H2_10seeds'
    n_seeds=10
    arrays_loss=[]
    array_norm=[]
    for i in range(n_seeds):
        arrays_loss.append(np.load('results/generative_learning/arrays/search/'+dir+'/RMSproploss_lr0.1m10.99m20loss'+keyw+'_10seedseed'+str(i)+'.npy', allow_pickle=True))
        array_norm.append(np.load('results/generative_learning/arrays/search/'+dir+'/RMSproploss_lr0.1m10.99m20norm'+keyw+'_10seedseed'+str(i)+'.npy', allow_pickle=True))
    
    epoch=range(len(arrays_loss[0]))
    
    #colors = ["tab:blue","tab:orange","tab:green","tab:red", "tab:purple", "tab:olive"]

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
    #labels=[['identity','Identity'],['sig','Sigmoid'], ['leaky','Leaky RELU'],['tanh','Tanh'], ['sig_out','Sigmoid out'], ['tanh_out','Tanh out']]
    labels=[['identity','Identity'],['sig','Sigmoid'],['relu','RELU'], ['leaky','Leaky RELU'],['tanh','Tanh'], ['sig_out','Sigmoid output'], ['tanh_out','Tanh output']]

    loss_list=[]
    name_start='loss_train12_2_'
    #name_start='loss_test8_2_'

    for i in range(len(labels)):
        loss_list.append(np.load('results/disc_learning/fraud/activations/'+name_start+labels[i][0]+'.npy', allow_pickle=True))

    epoch=range(len(loss_list[0]))
    
    plt.figure()
    for j, i in enumerate(loss_list):
        plt.plot(epoch, i, label=labels[j][1])
        
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.tight_layout()
    #plt.legend()
    plt.legend(prop={'size': 9})
    plt.savefig('results/disc_learning/assets/loss_activations_12_2.pdf')
    plt.clf()

def plot_bias():
    
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
    #plt.legend()
    plt.legend(prop={'size': 9 }, loc="lower right")
    plt.savefig('results/disc_learning/assets/bias_12_2_50_samples.pdf')
    plt.clf()

def plot_NN_sizes():
    labels=[['4_2','[4, 4]'],['4_3','[4, 4, 4]'], ['4_4','[4, 4, 4, 4]'],
            ['8_2_sig','[8, 8]'],['8_3','[8, 8, 8]'],
            ['16_2','[16, 16]'], ['32_2','[32, 32]']]

    labels=[['12_1_identity', '1 layer'], ['12_2_identity', '2 layers'],['12_3_identity', '3 layers']]
    name_start='loss_train'
    loss_list=[]


    for i in range(len(labels)):
        loss_list.append(np.load('results/disc_learning/fraud/NN_size_I/'+name_start+labels[i][0]+'.npy', allow_pickle=True))

    epoch=range(len(loss_list[0]))

    plt.figure()
    for j, i in enumerate(loss_list):
        plt.plot(epoch, i, label=labels[j][1])
        
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.tight_layout()
    plt.legend()
    #plt.legend(prop={'size': 8 }, loc="upper right", ncol=2)
    plt.savefig('results/disc_learning/assets/NN_sizes_I_50samp_fraud.pdf')
    plt.clf()

def plot_lr():
    labels=[['01','0.1'],['005','0.05'],['001','0.01']]

    #switch between two colors, different forms of lines on H1 and H2

    loss_list=[]
    name_start='loss_trainsig_12_2_lr'

    for i in range(len(labels)):
        loss_list.append(np.load('results/disc_learning/fraud/lr/'+name_start+labels[i][0]+'.npy', allow_pickle=True))

    epoch=range(len(loss_list[0]))
    
    plt.figure()
    for j, i in enumerate(loss_list):
        plt.plot(epoch, i, label=r'$\gamma=$'+labels[j][1])
        
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.tight_layout()
    #plt.legend()
    plt.legend(prop={'size': 8 }, loc="upper right")
    plt.savefig('results/disc_learning/assets/lr_12_2_w_sig_lr.pdf')
    plt.clf()

def plot_gen(name_start, labels, end):
    loss_list=[]

    for i in range(len(labels)):
        loss_list.append(np.load('results/disc_learning/fraud/optimizer/'+name_start+labels[i][0]+'.npy', allow_pickle=True))

    epoch=range(len(loss_list[0]))
    
    #plt.figure()
    plt.figure(figsize=[FIGWIDTH, FIGHEIGHT/2])

    #fig, ax = plt.subplots()
    for j, i in enumerate(loss_list):
        if j%2==0:
            pass
            #plt.plot(epoch, i, label=labels[j][1])
        else:
            plt.plot(epoch, i, label=labels[j][1])
        
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    #plt.yscale('log')
    #ax.yaxis.set_major_locator(tick.LogLocator(0.5))
    #plt.yticks(np.arange(min(loss_list[0]), max(loss_list[1]), 1.))
    plt.tight_layout()
    #plt.legend()
    plt.legend(prop={'size': 6.5})#, loc="lower left")
    plt.savefig('results/disc_learning/assets/'+name_start+end+'.pdf')
    plt.clf()


def plot_gen_sub(name_start, labels, end):
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




#plot_NN_sizes()
#plot_activation_functions()
#plot_bias()
#plot_lr()
#plot_finale_seeds(True, True)
#plot_lr_search()
#plot_fraud()
#plot_multiple_samples()
#plot_optim_search()
#plot_three_sub()
#genereal_plotter('results/disc_learning/mnist/loss_trainnetwork_24_3_4samples.npy', 'mnist_12_sample_24_3_lr001_nosig')
#plot_gen('loss_trainsig_12_2_lr', [['001_ams','AMSgrad', 0],['001_H3_ams','AMSgrad', 1],['001','RMSProp', 0], ['001_H3','RMSProp', 1]], 'optim_sub')
#plot_gen('loss_train12_2_sig_', [['HN','He N'], ['HU','He U'], ['XN','Xavier N'],['XU','Xavier U']], 'initialisation')
