from sklearn import linear_model, metrics
from sklearn.neural_network import BernoulliRBM
from sklearn.pipeline import Pipeline
from sklearn.base import clone
import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import binarize



def train_rbm3(dataset, learning_rate):
    """
    Training a classical restricted boltzmann machine

    With images, the pixels should be 0 or 1 
    """

    X_train=dataset[0]; y_train=dataset[1]; X_test=dataset[2];  y_test=dataset[3]

    # Models we will use
    logistic = linear_model.LogisticRegression(solver="newton-cg", tol=1)
    rbm = BernoulliRBM(random_state=0, verbose=True)

    rbm_features_classifier = Pipeline(steps=[("rbm", rbm), ("logistic", logistic)])

    # #############################################################################
    # Training

    # Hyper-parameters. These were set by cross-validation,
    # using a GridSearchCV. Here we are not performing cross-validation to
    # save time.
    rbm.learning_rate = 0.06
    rbm.n_iter = 20
    # More components tend to give better prediction performance, but larger
    # fitting time
    rbm.n_components = 100
    logistic.C = 6000

    # Training RBM-Logistic Pipeline
    rbm_features_classifier.fit(X_train, y_train)

    # Training the Logistic regression classifier directly on the pixel
    raw_pixel_classifier = clone(logistic)
    raw_pixel_classifier.C = 100.0
    raw_pixel_classifier.fit(X_train, y_train)

    # #############################################################################
    # Evaluation

    y_pred = rbm_features_classifier.predict(X_test)
    print(
        "Logistic regression using RBM features:\n%s\n"
        % (metrics.classification_report(y_test, y_pred))
    )

    y_pred = raw_pixel_classifier.predict(X_test)
    print(
        "Logistic regression using raw pixel features:\n%s\n"
        % (metrics.classification_report(y_test, y_pred))
    )


def train_rbm2(dataset, lr):
    """
    Another rbm func
    """
    X_train=dataset[0]; y_train=dataset[1]; X_test=dataset[2];  y_test=dataset[3]

    logistic = linear_model.LogisticRegression(C=10)
    rbm = BernoulliRBM(n_components=180, learning_rate=0.01, batch_size=10, n_iter=50, verbose=True, random_state=None)
    clf = Pipeline(steps=[('rbm', rbm), ('clf', logistic)])
    #X_train, X_test, y_train, y_test = cross_validation.train_test_split( X, Y, test_size=0.2, random_state=0)
    clf.fit(X_train,y_train)
    y_pred = clf.predict(X_test)

    y_test=np.ravel(y_test)

    print (f'Score: {metrics.classification_report(y_test,y_pred)}')

    pipeline = Pipeline(steps=[('rbm', rbm), ('clf', logistic)])

    parameters = {'clf__C': [0.001, 0.01, 0.1, 1, 10, 100, 1000],
                  'rbm_n_components': [50, 100, 200, 500]
                  }
    clf = GridSearchCV (pipeline, parameters, n_jobs=-1, verbose=1)
    clf.fit(X_train,y_train)
    print (f'Grid search best: {clf.best_estimator_}')


def train_rbm(dataset, best_params=None):
    """
    Running classical boltzmann machine with binarized values
    """
    X_train=dataset[0]; y_train=dataset[1]; X_test=dataset[2];  y_test=dataset[3]

    #Binarizing the input
    X_train=binarize(X_train, threshold=0.5)
    X_test=binarize(X_test, threshold=0.5)

    # Models we will use
    logistic = linear_model.LogisticRegression()
    rbm = BernoulliRBM(random_state=0, verbose=True)

    model = Pipeline(steps=[('rbm', rbm), ('logistic', logistic)])

    ###############################################################################
    # Training

    # Hyper-parameters. These were set by cross-validation,
    # using a GridSearchCV. Here we are not performing cross-validation to
    # save time.

    if best_params is not None:
        model.set_params(**best_params)
    else:
        #Set best values
        rbm.learning_rate = 0.06
        rbm.n_iter = 6
        rbm.n_components = 100
        logistic.C = 6000.0

    # Training RBM-Logistic Pipeline
    print('-----')
    model.fit(X_train, y_train)
    print('-----')

    #model.gibbs()

    #Okay just loop over and run alot of iterations each time 



    ###############################################################################
    # Evaluation

    print()
    print("Logistic regression using RBM features:\n%s\n" % (
        metrics.classification_report(
            y_test,
            model.predict(X_test))))
    
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

    # Training Logistic regression
    logistic_model = linear_model.LogisticRegression(C=100.0)
    logistic_model.fit(X_train, y_train)

    print("Logistic regression using raw pixel features:\n%s\n" % (
        metrics.classification_report(
            y_test,
            logistic_model.predict(X_test))))
    

def gridsearch_params(dataset, n_iterations):
    """
    Function to grid search parameters using rbm
    """
    X_train=dataset[0]; y_train=dataset[1]; X_test=dataset[2];  y_test=dataset[3]

    logistic = linear_model.LogisticRegression()
    rbm = BernoulliRBM(random_state=0, verbose=True)

    model = Pipeline(steps=[('rbm', rbm), ('logistic', logistic)])

    rbm.n_iter = n_iterations

    parameters = {#'rbm__learning_rate': [0.001, 0.01, 0.1, 1],
                  'rbm__learning_rate': [0.001, 0.01],

                  #'rbm__n_components': [2, 10, 50, 100, 200],
                  #'rbm__n_iter': [20, 50,100],
                  #'rbm_batch_size': [1,5,10,20],
                  #'clf__C': [0.5,1.0,5.0,50.0]
                  'logistic__C': [0.5,2.0]

                  }


    clf = GridSearchCV (model, parameters, n_jobs=-1, verbose=1)
    clf.fit(X_train,y_train)
    print (f'Grid search best: {clf.best_estimator_}')
    print (f'Grid best params: {clf.best_params_}')

    print(clf.best_estimator_[0])
    print(type(clf.best_estimator_[1]))

    print('_----_________---__-_--_-_-_-__---_-_----_-__')
    #print(clf.get_params())

    model.set_params(**clf.best_params_)

    #bestParams = gsc.best_estimator_.get_params()

    #TODO: params=best params

    #TODO: Epochs/iterations as function of loss, what to plot.

    #TODO: How to run on validation set

    #TODO: Test on fraud, why are the predictions so bad?

    #TODO: What to check: iterations as function of loss train, test and validation


    return clf.best_params_
