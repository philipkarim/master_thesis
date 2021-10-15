import numpy as np

class optimize:
    def __init__(self, number_params, learning_rate=0.001, circuit=None):
        """
        This class is handling everything regarding optimizing the parameters 
        and loss

        Args:
            learning_rate:  Learning rate used in gradient descent
            circuit:        Quantum circuit that is used
        """
        self.number_params=number_params
        self.learning_rate=learning_rate
        self.circuit=circuit
        self.t=1 #or 1?    
        self.m = np.zeros(number_params)
        self.v = np.zeros(number_params)

    def cross_entropy_new(self, p_data,p_BM):
        """
        Loss function from article (2)
        """
        loss=0
        for i in range(len(p_data)):
            loss+=p_data[i]*np.log(p_BM[i])

        return -loss
    # gradient descent algorithm with adam
    def adam(self, x, g, beta1=0.9, beta2=0.999, eps=1e-8):
        """
        I guess something like this should work?
        
        based on the following article:
        https://machinelearningmastery.com/adam-optimization-from-scratch/
        """
        #Just using formulas from 
        # https://ruder.io/optimizing-gradient-descent/index.html#adam
        self.m = beta1 * self.m+ (1.0 - beta1) * g
        self.v = beta2 * self.v + (1.0 - beta2) * g**2
        mhat = self.m / (1.0 - beta1**(self.t))
        vhat = self.v / (1.0 - beta2**(self.t))
        x -= self.learning_rate*mhat / (np.sqrt(vhat) + eps)
            
        #Add 1 to the counter
        self.t+=1

        return x
    
    def gradient_descent_gradient_done(self, params, lr, gradient):
        """
        Gradient descent function.
        
        Args:
                params:     Variational parameters(list or array)
                lr:         Learning rate
                gradint:     True labels of the samples(list or array)
                samples:    Samples reprenting the true labels and 
                            predictions. (2D Matrix)
        """
        params-=lr*gradient
        return params
    
    def gradient_descent(self, params, predicted, target, samples):
        """
        Gradient descent function.
        
        Args:
                params:     Variational parameters(list or array)
                predicted:  List of predicted values(list or array)
                target:     True labels of the samples(list or array)
                samples:    Samples reprenting the true labels and 
                            predictions. (2D Matrix)
        """
        update=self.learning_rate *self.gradient_of_loss(params, predicted, target, samples)
        params-= update

        return params
    
    def cross_entropy_distribution(self, p_data, p_BM):
        """
        Computes loss by cross entropy between the visible nodes and the boltzmann distribution
        
        Args:
            p_data: (list?) distribution data to be modelled
            p_BM:    (list?)BM distributions computed by probability_function in varQBM corresponding to each visible node v
                    maybe create a list depending on the different v?
        
        Returns: (float) Loss as a scalar
        """
        loss=0
        for v in range(len(p_data)):
            loss-=p_data[v]*np.log(p_BM[v])

        return loss

    def cross_entropy(self, preds, targets, classes=2, epsilon=1e-12):
        """
        Computes cross entropy between the true labels and the predictions
        by using distributions. (Not used, binary cross entropy function beneath)
        
        Args:
            preds:   predictions as an array or list
            targets: true labels as an array or list  
        
        Returns: loss as a scalar
        """
        #Creates matrixes to use one hot encoded labels
        distribution_preds=np.zeros((len(preds), classes))
        distribution_target=np.zeros((len(targets), classes))

        #Just rewriting the predictions and labels
        for i in range(len(preds)):
            distribution_preds[i][0]=1-preds[i]
            distribution_preds[i][1]=preds[i]
            
            if targets[i]==0:
                distribution_target[i][0]=1
            elif targets[i]==1:
                distribution_target[i][1]=1

        distribution_preds = np.clip(distribution_preds, epsilon, 1. - epsilon)
        n_samples = len(preds)
        loss = -np.sum(distribution_target*np.log(distribution_preds+1e-9))/n_samples
        return loss

    def binary_cross_entropy(self, preds, targets):
        """
        Computes binary cross entropy between the true labels and the predictions
        
        Args:
            preds:   predictions as an array or list
            targets: true labels as an array or list  
        
        Returns: loss as a scalar
        """
        sum=0
        n_samples=len(preds)
        for index in range(n_samples):
            sum+=targets[index]*np.log(preds[index])+(1-targets[index])*np.log(1-preds[index])

        return -sum/n_samples


    def parameter_shift(self, sample, theta_array, theta_index):
        """
        Parameter shift funtion, that finds the derivative of a
        certain variational parameter, by evaluating the cirquit
        shiftet pi/2 to the left and right.

        Args:
                sample:     Sample that will be optimized with respect 
                            of(list or array)
                theta_array:Variational parameters(list or array)
                theta_index:Index of the parameter that we want the 
                            gradient of

        Returns: optimized variational value
        """
        #Just copying the parameter arrays
        theta_left_shift=theta_array.copy()
        theta_right_shift=theta_array.copy()
        
        #Since the cirquits are normalised the shift is 0.25 which represents pi/2
        theta_right_shift[theta_index]+=0.25
        theta_left_shift[theta_index]-=0.25

        #Predicting with the shifted parameters
        pred_right_shift=self.circuit.predict(np.array([sample]),theta_right_shift)
        pred_left_shift=self.circuit.predict(np.array([sample]),theta_left_shift)
        
        #Computes the gradients
        theta_grad=(pred_right_shift[0]-pred_left_shift[0])/2

        return theta_grad

    def gradient_of_loss(self, thetas, predicted, target, samples):
        """
        Finds the gradient used in gradient descent.

        Args:
                thetas:     Variational parameters(list or array)
                predicted:  List of predicted values(list or array)
                target:     True labels of the samples(list or array)
                samples:    Samples reprenting the true labels and 
                            predictions. (2D Matrix)

        Returns:            Gradient 
        """

        gradients=np.zeros(len(thetas))
        eps=1E-8
        #Looping through variational parameters
        for thet in range(len(thetas)):
            sum=0
            #Looping through samples
            for i in range(len(predicted)):
                grad_theta=self.parameter_shift(samples[i], thetas, thet)
                deno=(predicted[i]+eps)*(1-predicted[i]-eps)
                sum+=grad_theta*(predicted[i]-target[i])/deno

            gradients[thet]=sum
        
        return gradients