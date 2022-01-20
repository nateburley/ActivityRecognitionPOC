"""
Module containing functions pertaining to training models on watch data.

Author: Nate Burley
"""

import numpy as np
import pickle
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score


def train_extra_trees(clf: ExtraTreesClassifier, X_train: np.array, y_train: np.array, save: bool=True) -> ExtraTreesClassifier:
    """
    Function to train an optimal Extra Trees Classifier using a grid search.
    Helpful example: https://www.kaggle.com/eikedehling/extra-trees-tuning
    More info on hyperparameters here: https://machinelearningmastery.com/extra-trees-ensemble-with-python/
    """
    # Define our grid search
    gsc = GridSearchCV(
        estimator=clf,
        param_grid={
            'n_estimators': range(50, 150, 50),
            'max_features': ['sqrt', 'log2', None],
            # 'min_samples_leaf': range(1,51,10),
            # 'min_samples_split': range(2,53,10)
        },
        scoring='accuracy',
        verbose=2
    )

    # Train the classifier and find the optimal parameters
    grid_result = gsc.fit(X_train, y_train)

    # Display best parameters and score
    print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))

    # Display the results of training across all parameters
    for test_mean, param in zip(
            grid_result.cv_results_['mean_test_score'],
            # grid_result.cv_results_['mean_train_score'],
            grid_result.cv_results_['params']):
        print("Test : %f with: %r" % (test_mean, param))

    # Construct and fit optimal model
    model = ExtraTreesClassifier(**grid_result.best_params_)
    model.fit(X_train, y_train)

    # Save the optimal classifier!
    if save:
        outfile = open('best_extra_trees.pickle', 'wb')
        pickle.dump(model, outfile)
    
    # Return the optimal model
    return model


def evaluate_model(model: ExtraTreesClassifier, X_test: np.array, y_test: np.array) -> None:
    # Get our model's predictions on the test data
    y_test_pred = model.predict(X_test)

    # Generate a classification report
    print("CLASSIFICATION REPORT")
    print(classification_report(y_true=y_test, y_pred=y_test_pred))

    # Print confusion matrix
    print("\nCONFUSION MATRIX")
    cm = confusion_matrix(y_true=y_test, y_pred=y_test_pred)
    print(cm)

    # And print the global accuracy
    print("\nACCURACY SCORES")
    print(accuracy_score(y_true=y_test, y_pred=y_test_pred))
