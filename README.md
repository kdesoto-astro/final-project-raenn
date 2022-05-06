# final-project-raenn
Final project for ASTRO 589 for Spring 2022. Uses the fitting parameters of SuperFit.jl as inputs to both a random forest and multi-layer perceptron (MLP). Classification purity, completeness, and accuracy are compared for the two classifiers.

File breakdown:
- import_features.py: Imports the median fitting parameters from the fit files, and maps the fits to classification labels. Converts to object types that can be read into sklearn and pytorch.
- random_forest.py: Implements the random forest on the fitting parameters and classifications. Also includes function to classify new inputs.
- mlp.py: Implements the multi-layer perceptron on the fitting parameters and classification labels. Also includes function to classify new inputs.
- plotting.py: Includes functions to create confusion matrices and loss and accuracy over epoch number. 
