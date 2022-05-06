import numpy as np
from format_data import *
from random_forest import *
from plotting import *
from mlp import *

def main():
    input_csv = "input_data.csv"
    num_folds = 10
    n_trees = 100
    fn_purity = "cm_presmote_p.png"
    fn_completeness = "cm_presmote_c.png"
    #fn_purity = "cm_%df_%dt_p.png" % (num_folds, n_trees)
    #fn_completeness = "cm_%df_%dt_c.png" % (num_folds, n_trees)
    #fn_purity_07 = "cm_%df_%dt_p_07.png" % (num_folds, n_trees)
    #fn_completeness_07 = "cm_%df_%dt_c_07.png" % (num_folds, n_trees)
    
    features, labels = import_features_and_labels(input_csv)
    #labels = generate_two_class_labels(labels) # comment if not doing binary classification
    features = normalize_features(features)
    tally_each_class(labels) # original tallies
    kfold_idx = generate_K_fold(features, num_folds)
    
    true_classes = np.array([])
    predicted_classes = np.array([])
    prob_above_07 = np.array([], dtype=bool)
    
    true_classes_mlp = np.array([])
    predicted_classes_mlp = np.array([])
    prob_above_07_mlp = np.array([], dtype=bool)
    ct = 0
    
    #for MLP
    input_dim = len(features[0])
    output_dim = 5 # number of classes
    print(input_dim, output_dim)
    for train_index, test_index in kfold_idx:
        ct += 1
        print("Fold number %d" % ct)
        train_features = features[train_index]
        test_features = features[test_index]
        train_labels = labels[train_index]
        test_labels = labels[test_index]
        
        # apply SMOTE to training data
        #train_features, train_labels = oversample_minority_classes(train_features, train_labels)
        
        # Run random forest
        rf = train_random_forest(train_features, train_labels, n_trees)
        pred_classes, pred_probs = classify_new_data(rf, test_features)
        predicted_classes = np.append(predicted_classes, pred_classes)
        prob_above_07 = np.append(prob_above_07, pred_probs > 0.7)
        true_classes = np.append(true_classes, test_labels)
        
        # Run MLP
        #labels_to_classes = {"SN Ia": 0, "other": 1}
        #classes_to_labels = {0: "SN Ia", 1: "other"}
        labels_to_classes = {"SN Ia": 0, "SN II": 1, "SN IIn": 2, "SLSN-I": 3, "SLSN-II": 4}
        classes_to_labels = {0: "SN Ia", 1: "SN II", 2: "SN IIn", 3: "SLSN-I", 4: "SLSN-II"}
        train_classes = np.array([labels_to_classes[l] for l in train_labels]).astype(int)
        test_classes = np.array([labels_to_classes[l] for l in test_labels]).astype(int)
        #Convert to Torch DataSet objects
        train_data = create_dataset(train_features, train_classes)
        test_data = create_dataset(test_features, test_classes)
        
        # Train and evaluate multi-layer perceptron
        test_classes, pred_classes, pred_probs = run_mlp(train_data, test_data, input_dim, output_dim)
        test_labels = np.array([classes_to_labels[l] for l in test_classes])
        pred_labels = np.array([classes_to_labels[l] for l in pred_classes])
        predicted_classes_mlp = np.append(predicted_classes_mlp, pred_labels)
        prob_above_07_mlp = np.append(prob_above_07_mlp, pred_probs > 0.7)
        true_classes_mlp = np.append(true_classes_mlp, test_labels)
        
    print("Random forest Accuracy: %.04f" % calc_accuracy(predicted_classes, true_classes))
    plot_confusion_matrix(true_classes, predicted_classes, "rf_"+fn_purity, True)
    plot_confusion_matrix(true_classes, predicted_classes, "rf_"+fn_completeness, False)
    
    print("MLP Accuracy: %.04f" % calc_accuracy(predicted_classes_mlp, true_classes_mlp))
    plot_confusion_matrix(true_classes_mlp, predicted_classes_mlp, "mlp_"+fn_purity, True)
    plot_confusion_matrix(true_classes_mlp, predicted_classes_mlp, "mlp_"+fn_completeness, False)
    """
    plot_confusion_matrix(true_classes[prob_above_07], predicted_classes[prob_above_07], fn_purity_07, True)
    plot_confusion_matrix(true_classes[prob_above_07], predicted_classes[prob_above_07], fn_completeness_07, False)
    """
main()
        
        
        

