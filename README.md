# mnist-with-path-signatures

The first notebook titled "signature_level_9_for_mnist_dataset" contains code to processes the MNIST dataset using path signatures as feature representations for handwritten digit images. Here's a brief summary:

1. Load MNIST Dataset:
Downloads and loads the MNIST dataset, containing grayscale images of handwritten digits (0-9).

2. Compute Path Signatures:
Extracts nonzero pixel coordinates from each image as a 2D path.
Computes level-9 path signatures using the iisignature library.

3. Save Path Signatures with Labels:
Saves the computed path signatures along with digit labels to CSV files for later use.

4. Load Saved Signatures:
Reads the previously saved CSV files and separates the labels from the computed signatures.

5. Standardization:
Applies StandardScaler to normalize the signature features.

6. Save Standardized Signatures:
Saves the standardized path signatures to new CSV files for further processing.

Key Features Are:
Uses path signatures (a mathematical feature extraction technique) for image representation.
Saves both raw and standardized path signatures for later use.
Ensures reproducibility by loading and transforming data systematically.



The second notebook titled "MLP_single_layer_of_100_neurons_with_path_signatures" contains code to train and evaluate an MLPClassifier (Multi-Layer Perceptron Neural Network) on the path signature features extracted from the MNIST dataset. Here's a summary:

1. Load Data
Reads the standardized path signatures (saved in CSV) for both training and testing.
Separates labels (first column) and features (remaining columns).
2. Train MLP Classifier
Initializes an MLPClassifier with:
One hidden layer containing 100 neurons.
ReLU activation function.
300 maximum iterations.
Trains the model on the training set.
Measures the training time.
3. Make Predictions
Uses the trained MLP to predict labels on the test set.
4. Evaluate the Model
Computes accuracy of predictions.
Displays a classification report (precision, recall, F1-score).
Generates a confusion matrix.
5. Visualize Confusion Matrix
Uses seaborn to create a heatmap of the confusion matrix.
Helps visualize model performance across different digit classes.

Key Features Are:
Uses path signatures as input features for MLP.
Achieves classification on MNIST without raw pixel data.
Evaluates and visualizes model performance effectively.
