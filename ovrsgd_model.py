import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time

class SGD(object):
    def __init__(self, eta=0.001, n_iter=20, random_state=1):
        self.eta = eta
        self.n_iter = n_iter
        self.random_state = random_state
        self.w_ = None
        self.cost_ = []  # Track cost for each iteration
        self.accuracies_ = []  # Track accuracy for each iteration

    def fit(self, X, y):
        # Normalize input data
        X_normalized = self._normalize(X)
        
        rgen = np.random.RandomState(self.random_state)
        self.w_ = rgen.normal(loc=0.0, scale=0.01, size=1 + X_normalized.shape[1])
        
        for _ in range(self.n_iter):
            X_normalized, y = self._shuffle(X_normalized, y)
            cost = []
            correct = 0
            for xi, target in zip(X_normalized, y):
                cost.append(self._update_weights(xi, target))
                if self.predict(xi) == target:
                    correct += 1
            avg_cost = sum(cost) / len(y)
            self.cost_.append(avg_cost)
            accuracy = correct / len(y)  # Calculate accuracy for current iteration
            self.accuracies_.append(accuracy)
        return self
    
    def _normalize(self, X):
        mean = np.mean(X, axis=0)
        std = np.std(X, axis=0)
        return (X - mean) / std
    
    def _shuffle(self, X, y):
        r = np.random.permutation(len(y))
        return X[r], y[r]

    def _update_weights(self, xi, target):
        output = self.net_input(xi)
        error = (target - output)
        self.w_[1:] += self.eta * xi.dot(error)
        self.w_[0] += self.eta * error
        cost = 0.5 * error**2
        return cost
        
    def net_input(self, X):
        return np.dot(X, self.w_[1:]) + self.w_[0]

    def predict(self, X):
        return np.where(self.net_input(X) >= 0.0, 1, -1)
    
class OvRClassifierSGD:
    def __init__(self, binary_classifier, random_state=1):
        self.binary_classifier = binary_classifier
        self.random_state = random_state
        self.classifiers = {}
        self.errors_ = {}
        self.accuracies_ = {}

    def fit(self, X, y):
        unique_classes = np.unique(y)
        for cls in unique_classes:
            binary_y = np.where(y == cls, 1, -1)
            classifier = self.binary_classifier(random_state=self.random_state)
            classifier.fit(X, binary_y)
            self.classifiers[cls] = classifier
            self.errors_[cls] = classifier.cost_
            self.accuracies_[cls] = classifier.accuracies_

    def predict(self, X):
        scores = {}
        for cls, classifier in self.classifiers.items():
            scores[cls] = classifier.predict(X)

        predictions = []
        for i in range(X.shape[0]):
            instance_scores = [scores[cls][i] for cls in scores]
            predicted_class = max(scores, key=lambda cls: scores[cls][i])
            predictions.append(predicted_class)
        return np.array(predictions)

def load_iris():
    # Load Iris dataset
    iris_df = pd.read_csv('IrisDataset\\iris.data', header=None)
    X_iris = iris_df.iloc[:, [0, 3]].values
    iris_class_mapping = {'Iris-setosa':1, 'Iris-versicolor':2,'Iris-virginica':3}
    iris_y_encoded = iris_df.iloc[:, 4].map(iris_class_mapping)
    y_iris_ovr = np.array(iris_y_encoded)
    return X_iris, y_iris_ovr

def load_drybeans():
    # Load Dry Beans dataset
    dry_beans_df = pd.read_excel('DryBeansDataset\\Dry_Bean_Dataset.xlsx', skiprows=1)
    X_dry_beans = dry_beans_df.iloc[:, [0, 15]].values
    class_mapping = {'SEKER':1, 'BARBUNYA':2, 'BOMBAY':3, 'CALI':4, 'HOROZ':5, 'SIRA':6, 'DERMASON':7}
    y_encoded = dry_beans_df.iloc[:, 16].map(class_mapping)
    y_dry_beans_ovr = np.array(y_encoded)
    return X_dry_beans, y_dry_beans_ovr


classifiers = ['OVRSGD']
datasets = ['Iris', 'Drybeans']

def execute(classifier, dataset):
    print("Executing...")
    if (classifier=='OVRSGD'):
        if(dataset=='Iris'):
            # Load Iris Dataset
            X_iris, y_iris_ovr = load_iris()
            # Train and test OvRClassifierSGD with Iris dataset
            map = {1:'Iris-setosa', 2:'Iris-versicolor',3:'Iris-virginica'}
            start_time = time.time()
            ovr_sgd_iris = OvRClassifierSGD(SGD)
            ovr_sgd_iris.fit(X_iris, y_iris_ovr)
            end_time = time.time()
            required_time = end_time - start_time

            fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(12, 4))

            # Plot errors for Iris dataset
            for cls, errors in ovr_sgd_iris.errors_.items():
                ax[0].plot(range(1, len(errors) + 1), errors, marker='o', label=map[cls])
                ax[0].set_xlabel('Epochs')
                ax[0].set_ylabel('Cost')
                ax[0].set_title('OvR-SGD: Cost per Epoch (Iris Dataset)')
                ax[0].legend()

            # Plot accuracy for Iris dataset
            for cls, accuracies in ovr_sgd_iris.accuracies_.items():
                ax[1].plot(range(1, len(accuracies) + 1), accuracies, marker='o', label=map[cls])
                ax[1].set_xlabel('Epochs')
                ax[1].set_ylabel('Accuracy')
                ax[1].set_title('OvR-SGD: Accuracy per Epoch (Iris Dataset)')
                ax[1].legend()

            plt.tight_layout()
            plt.show()

            print("--> Errors:")
            for cls, errors in ovr_sgd_iris.errors_.items():   
                for i in range(0, len(errors)):
                    print(f"Class: {map[cls]}, Epoch: {i+1}, {errors[i]}")
            print("--> Accuracies:")
            for cls, accuracies in ovr_sgd_iris.accuracies_.items():   
                for i in range(0, len(errors)):
                    print(f"Class: {map[cls]}, Epoch: {i+1}, {accuracies[i]}")
            print("--> Time required:", required_time, 'seconds',"\n")

        elif(dataset=='Drybeans'):
            # Load Dry Beans Dataset
            X_dry_beans, y_dry_beans_ovr = load_drybeans()
            # Train and test OvRClassifierSGD with Dry Beans dataset
            map = {1:'SEKER', 2:'BARBUNYA', 3:'BOMBAY', 4:'CALI', 5:'HOROZ', 6:'SIRA', 7:'DERMASON'}
            start_time = time.time()
            ovr_sgd_dry_beans = OvRClassifierSGD(SGD)
            ovr_sgd_dry_beans.fit(X_dry_beans, y_dry_beans_ovr)
            end_time = time.time()
            required_time = end_time - start_time

            fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(12, 8))

            # Plot errors for Dry Beans dataset
            for cls, errors in ovr_sgd_dry_beans.errors_.items():
                ax[0].plot(range(1, len(errors) + 1), errors, marker='o', label=map[cls])
                ax[0].set_xlabel('Epochs')
                ax[0].set_ylabel('Cost')
                ax[0].set_title('OvR-SGD: Cost per Epoch (Dry Beans Dataset)')
                ax[0].legend(bbox_to_anchor=(0.065, 0.15), ncol=2)

            # Plot accuracy for Dry Beans dataset
            for cls, accuracies in ovr_sgd_dry_beans.accuracies_.items():
                ax[1].plot(range(1, len(accuracies) + 1), accuracies, marker='o', label=map[cls])
                ax[1].set_xlabel('Epochs')
                ax[1].set_ylabel('Accuracy')
                ax[1].set_title('OvR-SGD: Accuracy per Epoch (Dry Beans Dataset)')
                ax[1].legend(bbox_to_anchor=(0.55, 0.15), ncol=2)

            plt.tight_layout()
            plt.show()

            print("--> Errors:")
            for cls, errors in ovr_sgd_dry_beans.errors_.items():   
                for i in range(0, len(errors)):
                    print(f"Class: {map[cls]}, Epoch: {i+1}, {errors[i]}")
            print("--> Accuracies:")
            for cls, accuracies in ovr_sgd_dry_beans.accuracies_.items():   
                for i in range(0, len(errors)):
                    print(f"Class: {map[cls]}, Epoch: {i+1}, {accuracies[i]}")
            print("Time required:", required_time, 'seconds')

    ask_classifier()


def ask_dataset(classifier):
    print("--> Please input the data name: 'Iris' / 'Drybeans'")
    dataset = input()
    if dataset not in datasets:
        print("--> Wrong dataset name!")
        ask_dataset(classifier)
    else:
        execute(classifier, dataset)

def ask_classifier():
    print("--> Please input the classifier name: 'OVRSGD' or 'Stop' to terminate")
    classifier = input()
    if classifier=='Stop':
        print("--> Thank you!")
        return
    elif classifier not in classifiers:
        print("--> Wrong classifier name!")
        ask_classifier()
    else:
        ask_dataset(classifier)

ask_classifier()