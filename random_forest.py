from sklearn.ensemble import RandomForestClassifier
import numpy as np
import matplotlib.pyplot as plt

def train_random_forest(train_features, train_labels, n_trees):
    """
    Train a random forest given anumber of trees and training data.
    """
    rf = RandomForestClassifier(n_estimators=n_trees, random_state=0)
    rf.fit(train_features, train_labels)
    return rf

def classify_new_data(rf, test_features):
    """
    Returns the most likely class and confidence about that
    prediction for each lightcurve feature set.
    """
    predicted_probs = rf.predict_proba(test_features)
    max_probs = np.amax(predicted_probs, axis=1)
    predicted_classes = rf.predict(test_features)
    return predicted_classes, max_probs

def calc_accuracy(pred_classes, test_labels):
    """
    Calculates the accuracy of the random forest after predicting
    all classes.
    """
    num_total = len(pred_classes)
    num_correct = np.sum(np.where(pred_classes == test_labels, 1, 0))
    return num_correct/num_total

def tune_n_trees(train_features, train_labels, test_features, test_labels):
    ntrees_grid = np.arange(25, 525, 25)
    acc_arr = []
    for ntrees in ntrees_grid:
        rf = train_random_forest(train_features, train_labels, ntrees)
        acc = calc_accuracy(rf, test_features, test_labels)
        print("ntrees=%d, acc=%.04f" % (ntrees, acc))
        acc_arr.append(acc)
    plt.scatter(ntrees, acc_arr)
    plt.xlabel("Number of trees")
    plt.ylabel("Accuracy")
    plt.title("Optimizing ntrees")
    plt.savefig("ntrees_opt.png")
    plt.close()
    