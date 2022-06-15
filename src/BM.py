from sklearn import linear_model, metrics
from sklearn.neural_network import BernoulliRBM
from sklearn.pipeline import Pipeline
from sklearn.base import clone
import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import binarize
from sklearn.metrics import accuracy_score, confusion_matrix, log_loss


import matplotlib.pyplot as plt
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
          #'figure.dpi' : 1000.0,
          #'text.latex.unicode': True,
          }
plt.rcParams.update(params)

def train_rbm(dataset, best_params=None, plot_acc_vs_epoch=0, name='', binarize_data=True, plot_acc=True, cm=True, roc=False, data_val=None):
    """
    Running classical boltzmann machine with binarized values
    """
    X_train=dataset[0]; y_train=dataset[1]; X_test=dataset[2];  y_test=dataset[3]

    #Binarizing the input
    #X_train=binarize(X_train, threshold=0.5)
    #X_test=binarize(X_test, threshold=0.5)

    # Models we will use
    logistic = linear_model.LogisticRegression()
    rbm = BernoulliRBM(random_state=0, verbose=True)

    model = Pipeline(steps=[('rbm', rbm), ('logistic', logistic)])

    ###############################################################################
    # Training
    if binarize_data:
        X_train=binarize(X_train, threshold=0.5)
        X_test=binarize(X_test, threshold=0.5)

    rbm.n_components = 4
    #rbm.n_iter = 30

    if best_params is not None:
        model.set_params(**best_params)
    else:
        #30/28x28
        if name=='mnist1':
            rbm.learning_rate = 0.05
            rbm.batch_size=1
            logistic.C = 100
        
        #4/digit 8x8x
        elif name=='mnist':
            #rbm.learning_rate = 0.1
            #rbm.batch_size=1
            #logistic.C = 50

            rbm.learning_rate = 0.5
            rbm.batch_size=5
            logistic.C = 500
            
        else:
            #Set best values
            rbm.learning_rate = 0.001
            rbm.batch_size=1
            logistic.C = 0.5

    if plot_acc_vs_epoch==0:
        # Training RBM-Logistic Pipeline
        model.fit(X_train, y_train)
        print()
        print("Logistic regression using RBM features:\n%s\n" % (
        metrics.classification_report(
            y_test,
            model.predict(X_test))))

        precision, recal, beta, temp= metrics.precision_recall_fscore_support(y_test,model.predict(X_test), average='macro')
        print(f'Acc: {accuracy_score(y_test, model.predict(X_test))}, Pre: {precision}, Rec: {recal}, F1: {beta}')


    elif not isinstance(plot_acc_vs_epoch, int):
        X_val=plot_acc_vs_epoch[0]
        y_val=plot_acc_vs_epoch[1]
        #Final testset computatations
        model.fit(X_train, y_train)
        #Remember to set optimal

        y_pred=model.predict(X_val)
        print(f'Accuracy on final testset: {accuracy_score(y_val, y_pred)}')

        print("Final testset:\n%s\n" % (
        metrics.classification_report(
            y_val, y_pred)))

        cf_matrix = confusion_matrix(y_val, y_pred, normalize='all')

        plt.figure()
        ax = sns.heatmap(cf_matrix/np.sum(cf_matrix), annot=True,
            fmt='.2%', cmap='Blues', cbar=False)

   
        ## Ticket labels - List must be in alphabetical order
        if name=='mnist':
            ax.set_ylabel('\nPredicted labels')
            ax.set_xlabel('True labels ')
            ax.xaxis.set_ticklabels(['0','1', '2', '3'])
            ax.yaxis.set_ticklabels(['0','1', '2', '3'])
        else:
            ax.set_xlabel('\nPredicted labels')
            ax.set_ylabel('True labels ')
            ax.xaxis.set_ticklabels(['False','True'])
            ax.yaxis.set_ticklabels(['False','True'])


        ## Display the visualization of the Confusion Matrix.
        plt.tight_layout()
        #plt.savefig('results/disc_learning/assets/rbm/CM'+name+'.pdf')
        plt.clf()

    else:
        train_acc=[]
        test_acc=[]
        for i in range(plot_acc_vs_epoch):
            rbm.n_iter = i
            model.fit(X_train, y_train)

            #print(model.predict_proba(X_train))

            if plot_acc is False:
                train_acc.append(log_loss(y_train, model.predict_proba(X_train)))
                test_acc.append(log_loss(y_test, model.predict_proba(X_test)))
                y_label='Loss'
            else:
                train_acc.append(accuracy_score(y_train, model.predict(X_train)))
                test_acc.append(accuracy_score(y_test, model.predict(X_test)))
                y_label='Accuracy'

        if plot_acc is False:
            best_index=test_acc.index(min(test_acc))
        else:
            best_index=test_acc.index(max(test_acc))
            
        print(f'Best index test: {best_index}')

        plt.figure()
        plt.plot(list(range(len(train_acc))), train_acc, label='Train')
        plt.plot(list(range(len(train_acc))), test_acc, label='Validation')
        plt.xlabel('Epoch')
        plt.ylabel(y_label)
        plt.tight_layout()
        plt.legend()
        plt.savefig('results/disc_learning/assets/rbm/rbm_acc_vs_epoch'+name+'.pdf')
        plt.clf()

        if cm:
            rbm.n_iter = best_index
            model.fit(X_train, y_train)

            y_pred=model.predict(X_test)
            print(f'Accuracy on final testset: {accuracy_score(y_test, y_pred)}')

            print("Final testset:\n%s\n" % (
            metrics.classification_report(
                y_test, y_pred)))

            cf_matrix = confusion_matrix(y_test, y_pred)#,  normalize='all')

            plt.figure()
            ax = sns.heatmap(0.01*cf_matrix/np.sum(cf_matrix), annot=True,
                fmt='.2%', cmap='Blues', cbar=False)

            ## Ticket labels - List must be in alphabetical order
            if name=='mnist':
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
            plt.savefig('results/disc_learning/assets/rbm/CM'+name+'.pdf')
            plt.clf()
        
        if roc:
            y_pred_proba = model.predict_proba(X_test)
            print(y_pred_proba[:10])

            if name=='mnist':
                for i in range(len(y_test)):
                    y_pred_proba[i]=y_pred_proba[i][y_test[i]]
    
                y_pred_proba=y_pred_proba[:,0]

            print(y_pred_proba[:10])
            
            fprRBM, tprRBM, _ = metrics.roc_curve(y_test,y_pred_proba)
            aucRBM = metrics.roc_auc_score(y_test,y_pred_proba)
            plt.plot(fprRBM, tprRBM,label="Bernoulli Restricted Boltzmann Machine, auc: " + str(aucRBM),color="gray",alpha=0.8)
            plt.plot([0,1], [0,1],color="red",alpha=0.8)
            plt.xlim([0.00,1.01])
            plt.ylim([0.00,1.01])
            plt.title("Validation ROC Curve - Bernoulli Restricted Boltzmann Machine")
            plt.xlabel("Specificity")
            plt.ylabel("Sensitivity")
            plt.legend(loc=4)

            plt.tight_layout()
            plt.savefig('results/disc_learning/assets/rbm/roc'+name+'.pdf')
            plt.clf()


    """
    plt.figure(figsize=(4.2, 4))
    for i, comp in enumerate(rbm.components_):
        plt.subplot(10, 10, i + 1)
        plt.imshow(comp.reshape((8, 8)), cmap=plt.cm.gray_r, interpolation="nearest")
        plt.xticks(())
        plt.yticks(())
    plt.suptitle("100 components extracted by RBM", fontsize=16)
    plt.subplots_adjust(0.08, 0.02, 0.92, 0.85, 0.08, 0.23)

    plt.show()
    """

def gridsearch_params(dataset, n_iterations, binarize_data=True):
    """
    Function to grid search parameters using rbm
    """
    X_train=dataset[0]; y_train=dataset[1]; X_test=dataset[2];  y_test=dataset[3]

    if binarize_data:
        X_train=binarize(X_train, threshold=0.5)
        X_test=binarize(X_test, threshold=0.5)
    
    logistic = linear_model.LogisticRegression()
    rbm = BernoulliRBM(random_state=0, verbose=True)

    model = Pipeline(steps=[('rbm', rbm), ('logistic', logistic)])

    rbm.n_iter = n_iterations
    rbm.n_components=4

    parameters = {'rbm__learning_rate': [0.001, 0.005, 0.01, 0.05, 0.1, 0.5],
                  'rbm__batch_size': [1,2,5,10],
                  'logistic__C': [0.5,1.0, 5.0, 10, 50.0, 100.0, 500.0]
                  }


    clf = GridSearchCV (model, parameters, n_jobs=-1, verbose=1)
    clf.fit(X_train,y_train)
    
    print("------------------------------------------")
    print (f'Grid best params: {clf.best_params_}')
    print("------------------------------------------")

    return clf.best_params_


def rbm_plot_scores(dataset, best_params=None, name='', binarize_input=True):
    """
    Running classical boltzmann machine with binarized values
    """
    X_train=dataset[0]; y_train=dataset[1]; X_test=dataset[2];  y_test=dataset[3]

    #Binarizing the input
    if binarize_input==True:
        X_train=binarize(X_train, threshold=0.5)
        X_test=binarize(X_test, threshold=0.5)
            
    # Models we will use
    logistic = linear_model.LogisticRegression()
    rbm = BernoulliRBM(random_state=0)

    model = Pipeline(steps=[('rbm', rbm), ('logistic', logistic)])

    test_accuracy=[]
    train_accuracy=[]

    if best_params is not None:
        model.set_params(**best_params)
    else:
        pass
        #Set best values
        #rbm.learning_rate = 0.01
        #rbm.n_iter = 50
        #rbm.n_components = 100

        #logistic.C = 5000.0

    h_nodes=[2, 5, 10, 15, 20, 30, 40,50,60, 75,100]
    
    #h_nodes=list(range(2, 106, 5))

    for i in h_nodes:
        rbm.n_components = i
        model.fit(X_train, y_train)
        score=accuracy_score(y_test, model.predict(X_test))
        test_accuracy.append(score)
        train_accuracy.append(accuracy_score(y_train, model.predict(X_train)))

    plt.figure()
    plt.plot(h_nodes, test_accuracy, label='Test')
    plt.plot(h_nodes, train_accuracy, label='Train')
    plt.xlabel('Hidden nodes')
    plt.ylabel('Accuracy')
    plt.tight_layout()
    plt.legend()
    plt.savefig('results/disc_learning/assets/rbm/rbm_h_vs_acc_'+name+'.pdf')
    plt.clf()





   
