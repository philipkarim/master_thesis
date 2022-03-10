
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style("darkgrid")
#x=np.load('results/arrays/learningrate0.507.npy', allow_pickle=True)

def plot_fraud():
    acc_train=np.load('results/fraud/acc_train_5050.npy', allow_pickle=True)
    acc_test=np.load('results/fraud/acc_test_5050.npy', allow_pickle=True)
    loss_train=np.load('results/fraud/loss_train_5050.npy', allow_pickle=True)
    loss_test=np.load('results/fraud/loss_test_5050.npy', allow_pickle=True)


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


plot_fraud()
