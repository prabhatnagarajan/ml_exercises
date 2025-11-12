import pandas as pd
import download
import numpy as np
from pdb import set_trace
from collections import Counter
import matplotlib.pyplot as plt

def knn_classify(x_train, y_train, x_test, k=3):
    predictions = []
    for _, test_point in x_test.iterrows():
        # Compute distances from the test point to all training points
        distances = np.linalg.norm(x_train.values - test_point.values, axis=1)
        # Get the indices of the k nearest neighbors
        knn_indices = np.argsort(distances)[:k]
        # Get the labels of the k nearest neighbors
        knn_labels = y_train.iloc[knn_indices]
        # Majority vote
        most_common = Counter(knn_labels).most_common(1)[0][0]
        predictions.append(most_common)
    return pd.Series(predictions, index=x_test.index)

def accuracy(y_true, y_pred):
    return (y_true == y_pred).mean()

def plot_accuracies(accuracies, interval):
    plt.plot(range(1, len(accuracies) + 1), accuracies)
    plt.fill_between(range(1, len(accuracies) + 1),
                     np.array(accuracies) - interval,
                     np.array(accuracies) + interval,
                     color='b', alpha=0.2)
    plt.xlabel('k (Number of Neighbors)')
    plt.ylabel('Accuracy')
    plt.title('k-NN Classifier Accuracy vs k')
    plt.grid()
    plt.show()

def experiment():
    x_train, y_train, x_test, y_test = download.iris_train_test_split(random_state=np.random.randint(0, 10000))
    train_classes, train_counts = np.unique(y_train, return_counts=True)
    test_classes, test_counts = np.unique(y_test, return_counts=True)
    # Just to confirm a good dataset split
    if not (all(train_counts >= 10) and all(test_counts >= 6) and len(train_classes) == len(test_classes) == 3):
        return experiment()
    category_mapping = {cls: idx for idx, cls in enumerate(train_classes)}
    # rename the classes to be 0, 1, 2
    y_train = y_train.map(category_mapping)
    y_test = y_test.map(category_mapping)
    accuracies = []
    for k in range(1, len(x_train)):
        y_pred = knn_classify(x_train, y_train, x_test, k=k)
        acc = accuracy(y_test, y_pred)
        accuracies.append(acc)
        print(f"{k}-NN Classifier Accuracy: {acc * 100:.2f}%")
    return accuracies

if __name__ == "__main__":
    print("This script is being run directly.")
    num_experiments = 30
    experiment_accuracies = []
    for i in range(num_experiments):
        print(f"\nExperiment {i + 1}/{num_experiments}")
        accuracies = experiment()
        experiment_accuracies.append(accuracies)
    mean = np.mean(experiment_accuracies, axis=0)
    std_err = np.std(experiment_accuracies, axis=0) / np.sqrt(num_experiments)
    interval = std_err * 1.96  # 95% confidence interval
    plot_accuracies(mean, interval)