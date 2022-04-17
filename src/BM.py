
def train_rbm(dataset, learning_rate):
    from sklearn import linear_model, metrics
    from sklearn.neural_network import BernoulliRBM
    from sklearn.pipeline import Pipeline
    from sklearn.base import clone

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

    Y_pred = rbm_features_classifier.predict(X_test)
    print(
        "Logistic regression using RBM features:\n%s\n"
        % (metrics.classification_report(y_test, Y_pred))
    )

    Y_pred = raw_pixel_classifier.predict(X_test)
    print(
        "Logistic regression using raw pixel features:\n%s\n"
        % (metrics.classification_report(y_test, Y_pred))
    )

