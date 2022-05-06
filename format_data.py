import numpy as np
import torch
import csv
import sklearn
from sklearn.model_selection import train_test_split, KFold
from imblearn.over_sampling import SMOTE
#from sklearn.cross_validation import train_test_split

def import_features_and_labels(input_csv):
    """
    Import all features and labels, convert to label and features
    numpy arrays.
    """
    features = []
    labels = []
    with open(input_csv, newline='') as csvfile:
        csvreader = csv.reader(csvfile, delimiter=',', quotechar='|')
        next(csvreader, None)
        for row in csvreader:
            features.append(row[2:])
            labels.append(row[1])
    return np.array(features).astype(float), np.array(labels)

def divide_into_training_test_set(features, labels, test_fraction):
    """
    Divides dataset into set fraction of test samples and remaining as
    training data.
    """
    return train_test_split(features, labels, test_size=test_fraction, random_state=42)

def generate_K_fold(features, num_folds):
    """
    Generates set of K test sets and corresponding training sets
    """
    if num_folds == -1:
        kf = KFold(n_splits=len(features), shuffle=True) # cross-one out validation
    else:
        kf = KFold(n_splits=num_folds, shuffle=True)
    return kf.split(features)

def tally_each_class(labels):
    """
    Print number of samples with each class label.
    """
    tally_dict = {}
    for label in labels:
        if label not in tally_dict:
            tally_dict[label] = 1
        else:
            tally_dict[label] += 1
    for tally_label in tally_dict:
        print(tally_label,": ", str(tally_dict[tally_label]))
    print()

def generate_two_class_labels(labels):
    """
    For the binary classification problem.
    """
    labels_copy = np.copy(labels)
    labels_copy[labels_copy != "SN Ia"] = "other"
    return labels_copy

def oversample_minority_classes(features, labels):
    """
    Uses SMOTE to oversample data from rarer classes so
    classifiers are not biased toward SN-1a or SN-II
    """
    oversample = SMOTE()
    features_smote, labels_smote = oversample.fit_resample(features, labels)
    return features_smote, labels_smote

def normalize_features(features):
    """
    Normalize the features for feeding into the neural network.
    """
    print("Feature mean", features.mean(axis=0))
    print("Feature std", features.std(axis=0))
    return (features - features.mean(axis=0)) / features.std(axis=0)

def main():
    input_csv = "input_data.csv"
    features, labels = import_features_and_labels(input_csv)
    tally_each_class(labels)
    labels_twoclass = generate_two_class_labels(labels)
    tally_each_class(labels_twoclass)
    train_features, test_features, train_labels, test_labels = \
        divide_into_training_test_set(features, labels, 0.1)

