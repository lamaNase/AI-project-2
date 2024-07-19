import numpy as np
import csv
import sys
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import confusion_matrix

TEST_SIZE = 0.2
K = 3

class NN:
    def __init__(self, trainingFeatures, trainingLabels):
        self.trainingFeatures = trainingFeatures
        self.trainingLabels = trainingLabels

    def predict(self, features, k):
        """
        Given a list of feature vectors of testing examples,
        return the predicted class labels (list of either 0s or 1s)
        using the k nearest neighbors.
        """
        predictions = []
        for feature in features:
            distances = np.linalg.norm(self.trainingFeatures - feature, axis=1)
            nearest_indices = np.argsort(distances)[:k]
            nearest_labels = [self.trainingLabels[i] for i in nearest_indices]
            prediction = 1 if np.mean(nearest_labels) >= 0.5 else 0
            predictions.append(prediction)
        return predictions


def load_data(filename):
    """
    Load spam data from a CSV file `filename` and convert it into a list of
    feature vectors and a list of target labels. Return a tuple (features, labels).

    Feature vectors should be a list of lists, where each list contains the
    57 features.

    Labels should be the corresponding list of labels, where each label
    is 1 if spam, and 0 otherwise.
    """
    features = []
    labels = []
    with open(filename, 'r') as file:
        csv_reader = csv.reader(file)
        for row in csv_reader:
            features.append([float(val) for val in row[:-1]])
            labels.append(int(row[-1]))
    return features, labels


def preprocess(features):
    """
    Normalize each feature by subtracting the mean value in each
    feature and dividing by the standard deviation.
    """
    features = np.array(features)
    mean = np.mean(features, axis=0)
    std = np.std(features, axis=0)
    return (features - mean) / std


def train_mlp_model(features, labels):
    """
    Given a list of feature lists and a list of labels, return a
    fitted MLP model trained on the data using the scikit-learn implementation.
    """
    model = MLPClassifier(hidden_layer_sizes=(10, 5), activation='logistic', random_state=1)
    model.fit(features, labels)
    return model


def evaluate(labels, predictions):
    """
    Given a list of actual labels and a list of predicted labels,
    return (accuracy, precision, recall, f1-score).

    Assume each label is either a 1 (positive) or 0 (negative).
    """
    tn, fp, fn, tp = confusion_matrix(labels, predictions).ravel()

    accuracy = (tp + tn) / len(labels)
    precision = tp / (tp + fp) if tp + fp > 0 else 0
    recall = tp / (tp + fn) if tp + fn > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0

    return accuracy, precision, recall, f1


def main():
    # Check command-line arguments
    if len(sys.argv) != 2:
        sys.exit("Usage: python template.py ./spambase.csv")

    # Load data from spreadsheet and split into train and test sets
    features, labels = load_data(sys.argv[1])
    features = preprocess(features)
    X_train, X_test, y_train, y_test = train_test_split(
        features, labels, test_size=TEST_SIZE)

    # Train a k-NN model and make predictions
    model_nn = NN(X_train, y_train)
    predictions = model_nn.predict(X_test, K)
    accuracy, precision, recall, f1 = evaluate(y_test, predictions)

    # Print results
    print("**** k-NN Results ****")
    print("Accuracy:", accuracy)
    print("Precision:", precision)
    print("Recall:", recall)
    print("F1-Score:", f1)
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, predictions))

    # Train an MLP model and make predictions
    model = train_mlp_model(X_train, y_train)
    predictions = model.predict(X_test)
    accuracy, precision, recall, f1 = evaluate(y_test, predictions)

    # Print results
    print("\n**** MLP Results ****")
    print("Accuracy:", accuracy)
    print("Precision:", precision)
    print("Recall:", recall)
    print("F1-Score:", f1)
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, predictions))


if __name__ == "__main__":
    main()
