from importlib.metadata import requires
from sklearn import linear_model, metrics
from sklearn.linear_model import Ridge, Lasso, LogisticRegression
from sklearn.neighbors import KNeighborsRegressor

import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, log_loss, precision_recall_fscore_support
from sklearn.utils import shuffle
import torch.optim as optim_torch
from torch.nn import CrossEntropyLoss, BCELoss

import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib
from utils import *
from NN_class import *

matplotlib.rcParams['mathtext.fontset'] = 'stix'
matplotlib.rcParams['font.family'] = 'STIXGeneral'

sns.set_style("darkgrid")

FIGWIDTH=4.71935 #From latex document
FIGHEIGHT=FIGWIDTH/1.61803398875

params = {'text.usetex' : True,
          'font.size' : 10,
          'font.family' : 'lmodern',
          'figure.figsize' : [FIGWIDTH, FIGHEIGHT],
          }
plt.rcParams.update(params)

class MlMethods():
    """
    Class: Contains different machine learning methods utilizing mostly scikit
    """
    def __init__(self, X_train, y_train, X_test, y_test):
        self.X_train = X_train
        self.y_train=y_train
        self.X_test=X_test
        self.y_test=y_test
    
    def computeScores(self, prediction, true_labels=None):
        """
        Computing the precision, recall and f1 score
        """
        if true_labels is not None:
            target=true_labels
        else:
            target=self.y_test

        precision, recal, beta, temp= precision_recall_fscore_support(target, prediction, average='weighted')

        print(f'Accuracy: {accuracy_score(target, prediction)}, Precision: {precision}, recal: {recal}, f1-score: {beta}')

    def ols_reg(self):
        """
        Linear regression: Ordinary linear regression
        """

        model = Ridge(alpha=0)
        model.fit(self.X_train, self.y_train)
        preds=model.predict(self.X_test)
        self.computeScores(preds)

    def ridge_reg(self):
        """
        Linear regression: Ridge
        """
        model = Ridge()
        model.fit(self.X_train, self.y_train)
        preds=model.predict(self.X_test)
        self.computeScores(preds)

    def lasso_reg(self):
        """
        Linear regression: Lasso
        """
        model = Lasso()
        model.fit(self.X_train, self.y_train)
        preds=model.predict(self.X_test)
        self.computeScores(preds)

    def k_nn(self):
        """
        K-Nearest Neighbors
        """
        model = KNeighborsRegressor(n_neighbors = 3)
        model.fit(self.X_train, self.y_train)
        preds=model.predict(self.X_test)
        self.computeScores(preds)

    def logistic_reg(self):
        """
        Logistic regression
        """
        model = LogisticRegression()
        model.fit(self.X_train, self.y_train)
        preds=model.predict(self.X_test)
        self.computeScores(preds)

    def neural_net(self, output_size, lr=0.01, network_coeff=NN_nodes(8,5, sig_last=True), n_epochs=50):
        """
        Feed forward neural network without the VarQBM
        """
        #lr=0.001
        net=Net(network_coeff, self.X_train[0], output_size)
        net.apply(init_weights_XN)
        optimizer = optim_torch.RMSprop(net.parameters(), lr=lr)

        loss_mean=[]
        loss_mean_test=[]
        H_coefficients=[]
        predictions_train=[]
        targets_train=[]
        predictions_test=[]
        targets_test=[]
        target_score=[]

        #Floating the network parameters
        net = net.float()
        if output_size==1:
            criterion = BCELoss()
        else:
            criterion = CrossEntropyLoss()



        for epoch in range(n_epochs):
            #print(f'Epoch: {epoch}/{n_epochs}')

            #Lists to save the predictions of the epoch
            pred_epoch=[]
            loss_list=[]
            targets=[]

            #Loops over each sample
            X_train, y_train = shuffle(self.X_train, self.y_train, random_state=0)
            
            for i,sample in enumerate(X_train):
                pred_samp=net(sample)

                #print(f'Prediction from network{pred_samp}')

                if output_size==1:
                    target_data=np.zeros(2)
                    p_QBM=np.zeros(2)
                    p_QBM[1]=pred_samp.item()
                    p_QBM[0]=1-p_QBM[1]
                    pred_epoch.append(0) if p_QBM[0]>0.5 else pred_epoch.append(1)

                elif output_size==4:
                    target_data=np.zeros(2)
                    p_QBM=np.array(pred_samp)

                    pred_epoch.append(np.where(p_QBM==p_QBM.max())[0][0])

                else:
                    sys.exit('Redefine output size of network')

                target_data[y_train[i]]=1
                targets.append(target_data)

                if pred_samp.item()<0:
                    pred_samp[0]=0+1e-8

                elif pred_samp.item()>1:
                    pred_samp[0]=1-1e-8


                #yhat = torch.Tensor([[0.4, 0.6]], requires_grad = True)
                #y = torch.Tensor([1]).to(torch.long)

                #print(yhat)
                #print(pred_samp)
                
                #loss = criterion(input=yhat, target=y)



                loss = criterion(pred_samp, torch.tensor([y_train[i]]).float())
                #loss = criterion(torch.tensor([0.4,0.6]).float(), torch.tensor([0,1]).float())


                #print(yhat)
                # tensor([[0.5000, 1.5000, 0.1000],
                #         [2.2000, 1.3000, 1.7000]])

                #print(y)
                # tensor([1, 2])

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                #loss=-np.sum(target_data*np.log(p_QBM))
                loss_list.append(loss.item())
                #output_coef.backward(torch.tensor(gradient_loss, dtype=torch.float64))
                #optimizer.step()

                #print(f'TRAIN: Loss: {loss}, p_QBM: {p_QBM}, target: {target_data}')

            loss_mean.append(np.mean(loss_list))
            predictions_train.append(pred_epoch)
            targets_train.append(targets)

            loss_list=[]
            pred_epoch=[]
            targets=[]
        
            with torch.no_grad():
                for i,sample in enumerate(self.X_test):
                    #Network parameters
                    pred_samp=net(sample)

                    if output_size==1:
                        target_data=np.zeros(2)
                        p_QBM=np.zeros(2)
                        p_QBM[1]=pred_samp
                        p_QBM[0]=1-p_QBM[1]
                        pred_epoch.append(0) if p_QBM[0]>0.5 else pred_epoch.append(1)

                    elif output_size==4:
                        target_data=np.zeros(2)
                        p_QBM=np.array(pred_samp)
                        pred_epoch.append(np.where(p_QBM==p_QBM.max())[0][0])

                    else:
                        sys.exit('Redefine output size of network')

                    target_data[self.y_test[i]]=1
                    targets.append(target_data)                    
                    
                    #loss = CrossEntropyLoss(pred_samp, self.y_test[i])
                    if pred_samp.item()<0:
                        pred_samp[0]=0+1e-8
                    elif pred_samp.item()>1:
                        pred_samp[0]=1-1e-8
                    loss = criterion(pred_samp, torch.tensor([y_train[i]]).float())

                    loss_list.append(loss)

                    
                #Computes the test scores regarding the test set:
                loss_mean_test.append(np.mean(loss_list))
                predictions_test.append(pred_epoch)
                targets_test.append(targets)
            print(f'TRAIN loss: {loss_mean[-1]}TEST: Loss: {loss_mean_test[-1]}')

        #Save scores    
        target_list=[]
        loss_mean_test=np.array(loss_mean_test)
        predictions_test=np.array(predictions_test)
        targets_test=np.array(targets_test)
        
        pred_net=predictions_test[np.where(loss_mean_test == np.amin(loss_mean_test))[0]][0]
        targ_net=targets_test[np.where(loss_mean_test == np.amin(loss_mean_test))[0]][0]
        
        if output_size==1:
            for i in targ_net:
                if i[0]==1:
                    target_list.append(0)
                else:
                    target_list.append(1)

        else:
            for i in targ_net:
                if i[0]==1:
                    target_list.append(0)
                elif i[1]==1:
                    target_list.append(1)
                elif i[2]==1:
                    target_list.append(2)
                else:
                    target_list.append(3)

        
        self.computeScores(pred_net, target_list)




        """
        if nickname is not None:
            path='results/disc_learning/'+folder
            dir_exist = os.path.exists('results/disc_learning/'+folder)

        if not dir_exist:
            # Create a new directory because it does not exist
            os.makedirs(path)

        np.save(path+'/loss_test'+nickname+'.npy', np.array(loss_mean_test))
        np.save(path+'/loss_train'+nickname+'.npy', np.array(loss_mean))
        np.save(path+'/predictions_train'+nickname+'.npy', np.array(predictions_train))
        np.save(path+'/predictions_test'+nickname+'.npy', np.array(predictions_test))
        np.save(path+'/targets_train'+nickname+'.npy', np.array(targets_train))
        np.save(path+'/targets_test'+nickname+'.npy', np.array(targets_test))
        np.save(path+'/H_coeff'+nickname+'.npy', np.array(H_coefficients))
        if task=='classification':
                np.save(path+'/targets_test_score'+nickname+'.npy', np.array(target_score))
        """



   
