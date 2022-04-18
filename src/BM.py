from sklearn import linear_model, metrics
from sklearn.neural_network import BernoulliRBM
from sklearn.pipeline import Pipeline
from sklearn.base import clone
import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn import datasets
from sklearn.model_selection import train_test_split
from scipy.ndimage import convolve


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


def train_rbm(dataset, lr):
    """
    mnist = datasets.fetch_mldata('MNIST original')
    X, Y = mnist.data, mnist.target
    X = np.asarray( X, 'float32')
    # Scaling between 0 and 1
    X = (X - np.min(X, 0)) / (np.max(X, 0) + 0.0001)  # 0-1 scaling
    # Convert to binary images
    X = X > 0.5
    print 'Input X shape', X.shape
    """

    # Load Data
    digits = datasets.load_digits(n_class=4)
    X = np.asarray(digits.data, 'float32')
    #X, Y = nudge_dataset(X, digits.target)
    Y=digits.target
    X = (X - np.min(X, 0)) / (np.max(X, 0) + 0.0001)  # 0-1 scaling

    X_train, X_test, Y_train, Y_test = train_test_split(X, Y,
                                                        test_size=0.2,
                                                        random_state=0)

    # Models we will use
    logistic = linear_model.LogisticRegression()
    rbm = BernoulliRBM(random_state=0, verbose=True)

    classifier = Pipeline(steps=[('rbm', rbm), ('logistic', logistic)])

    ###############################################################################
    # Training

    # Hyper-parameters. These were set by cross-validation,
    # using a GridSearchCV. Here we are not performing cross-validation to
    # save time.
    rbm.learning_rate = 0.06
    rbm.n_iter = 20
    # More components tend to give better prediction performance, but larger
    # fitting time
    rbm.n_components = 100
    logistic.C = 6000.0

    # Training RBM-Logistic Pipeline
    classifier.fit(X_train, Y_train)

    # Training Logistic regression
    logistic_classifier = linear_model.LogisticRegression(C=100.0)
    logistic_classifier.fit(X_train, Y_train)

    ###############################################################################
    # Evaluation

    print()
    print("Logistic regression using RBM features:\n%s\n" % (
        metrics.classification_report(
            Y_test,
            classifier.predict(X_test))))

    print("Logistic regression using raw pixel features:\n%s\n" % (
        metrics.classification_report(
            Y_test,
            logistic_classifier.predict(X_test))))
    

    classifier = Pipeline(steps=[('rbm', rbm), ('clf', logistic)])

    parameters = {'rbm__learning_rate': [0.001, 0.01, 0.1, 1, 10, 100, 1000],
                  'rbm__n_components': [50, 100, 200, 500],
                  #'rbm__n_iter': [20, 50,100],
                  'clf__C': [1.0,5.0]
                  }


    clf = GridSearchCV (classifier, parameters, n_jobs=-1, verbose=1)
    clf.fit(X_train,Y_train)
    print (f'Grid search best: {clf.best_estimator_}')
    #bestParams = gsc.best_estimator_.get_params()

    #TODO: params=best params

    #TODO: Epochs/iterations as function of loss, what to plot.

    #TODO: How to run on validation set

    #TODO: Test on fraud, why are the predictions so bad?
