import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time


class Perceptron (object):
    def __init__(self , eta=0.001, n_iter=20, random_state=1): 
        self.eta = eta
        self.n_iter=n_iter
        self.random_state = random_state
        self.w_ = None
        self.errors_ = []
        self.accuracies_ = []

    def fit(self, X, y):
        rgen = np.random.RandomState(self.random_state)
        self.w_ = rgen.normal(loc=0.0, scale=0.01, size=1 + X.shape[1])
        for _ in range(self.n_iter):
            errors = 0
            correct = 0  # Counter for correctly classified instances
            for xi, target in zip(X, y):
                update = self.eta * (target - self.predict(xi))
                self.w_[1:] += update * xi
                self.w_[0] += update
                errors += int(update != 0.0)  # count misclassified instances
                if update == 0:
                    correct += 1
            accuracy = correct / len(y)  # Calculate accuracy for current iteration
            self.accuracies_.append(accuracy)  # Store accuracy for each iteration
            self.errors_.append(errors)
        return self
        
    def net_input(self, X):
        return np.dot(X, self.w_[1:]) + self.w_[0] # w[0] not in dot product as x0=1
    
    def predict(self, X):
        return np.where(self.net_input(X) >= 0.0, 1, -1)
    


class Adaline(object):
    def __init__(self, eta=0.01, n_iter=20, random_state=1): 
        self.eta = eta
        self.n_iter = n_iter
        self.random_state = random_state
        self.w_ = None
        self.cost_ = []
        self.accuracies_ = []  # Track accuracy for each iteration

    def fit(self, X, y):
        rgen = np.random.RandomState(self.random_state)
        self.w_ = rgen.normal(loc=0.0, scale=0.01, size=1 + X.shape[1])
        for _ in range(self.n_iter):
            net_input = self.net_input(X)
            output = net_input
            errors = (y - output)
            self.w_[1:] += self.eta * X.T.dot(errors)
            self.w_[0] += self.eta * errors.sum()
            cost = (errors**2).sum() / 2.0
            self.cost_.append(cost)
            
            # Calculate accuracy for current iteration
            y_pred = self.predict(X)
            correct = np.sum(y_pred == y)
            accuracy = correct / len(y)
            self.accuracies_.append(accuracy)  # Store accuracy for each iteration
        return self
        
    def net_input(self, X):
        return np.dot(X, self.w_[1:]) + self.w_[0]

    def predict(self, X):
        return np.where(self.net_input(X) >= 0.0, 1, -1)
    

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

def load_iris():
    # Load Iris dataset
    iris_df = pd.read_csv('IrisDataset\\iris.data', header=None)
    X_iris = iris_df.iloc[:, [0, 2]].values
    y_iris = np.where(iris_df.iloc[:, 4].values == 'Iris-setosa', -1, 1)
    return X_iris, y_iris

def load_drybeans():
    # Load Dry Beans dataset
    dry_beans_df = pd.read_excel('DryBeansDataset\\Dry_Bean_Dataset.xlsx', skiprows=1)
    X_dry_beans = dry_beans_df.iloc[:, [0, 15]].values
    y_dry_beans = np.where(dry_beans_df.iloc[:, 16].values == 'SEKER', -1, 1)
    return X_dry_beans, y_dry_beans


classifiers = ['Perceptron','Adaline','SGD']
datasets = ['Iris', 'Drybeans']

def execute(classifier, dataset):
    print("Executing...")
    if (classifier=='Perceptron'):
        if(dataset=='Iris'):
            # Load Iris Dataset
            X_iris, y_iris = load_iris()
            # Train and test Perceptron with Iris dataset
            start_time = time.time()
            perceptron_iris = Perceptron(eta=0.0001, n_iter=20)
            perceptron_iris.fit(X_iris, y_iris)
            end_time = time.time()
            required_time = end_time - start_time

            fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(12, 4))

            # Plot errors for Iris dataset
            ax[0].plot(range(1, perceptron_iris.n_iter + 1), perceptron_iris.errors_, marker='o')
            ax[0].set_xlabel('Epochs')
            ax[0].set_ylabel('Number of Misclassifications')
            ax[0].set_title('Perceptron: Errors per Epoch (Iris Dataset)')

            # Plot accuracy for Iris dataset
            ax[1].plot(range(1, perceptron_iris.n_iter + 1), perceptron_iris.accuracies_, marker='o')
            ax[1].set_xlabel('Epochs')
            ax[1].set_ylabel('Accuracy')
            ax[1].set_title('Perceptron: Accuracy per Epoch (Iris Dataset)')
            plt.show()

            print("--> Errors:")
            for i, item in enumerate(perceptron_iris.errors_, start=1):
                print(f"Epoch {i}: {item}")
            print("--> Accuracies:")
            for i, item in enumerate(perceptron_iris.accuracies_, start=1):
                print(f"Epoch {i}: {item}")
            print("--> Time required:", required_time, 'seconds',"\n")

        elif(dataset=='Drybeans'):
            # Load Dry Beans Dataset
            X_dry_beans, y_dry_beans = load_drybeans()
            # Train and test Perceptron with Dry Beans dataset
            start_time = time.time()
            perceptron_dry_beans = Perceptron(eta=0.0001, n_iter=20)
            perceptron_dry_beans.fit(X_dry_beans, y_dry_beans)
            end_time = time.time()
            required_time = end_time - start_time

            fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(12, 4))

            # Plot errors for Iris dataset
            ax[0].plot(range(1, perceptron_dry_beans.n_iter + 1), perceptron_dry_beans.errors_, marker='o')
            ax[0].set_xlabel('Epochs')
            ax[0].set_ylabel('Number of Misclassifications')
            ax[0].set_title('Perceptron: Errors per Epoch (Dry Beans Dataset)')

            # Plot accuracy for Iris dataset
            ax[1].plot(range(1, perceptron_dry_beans.n_iter + 1), perceptron_dry_beans.accuracies_, marker='o')
            ax[1].set_xlabel('Epochs')
            ax[1].set_ylabel('Accuracy')
            ax[1].set_title('Perceptron: Accuracy per Epoch (Dry Beans Dataset)')
            plt.show()

            print("--> Errors:")
            for i, item in enumerate(perceptron_dry_beans.errors_, start=1):
                print(f"Epoch {i}: {item}")
            print("--> Accuracies:")
            for i, item in enumerate(perceptron_dry_beans.accuracies_, start=1):
                print(f"Epoch {i}: {item}")
            print("--> Time required:", required_time, 'seconds',"\n")
    
    elif (classifier=='Adaline'):
        if(dataset=='Iris'):
            # Load Iris Dataset
            X_iris, y_iris = load_iris()
            # Train and test Perceptron with Iris dataset
            start_time = time.time()
            adaline_iris = Adaline(eta=0.0001, n_iter=20)
            adaline_iris.fit(X_iris, y_iris)
            end_time = time.time()
            required_time = end_time - start_time

            fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(12, 4))

            # Plot errors for Iris dataset
            ax[0].plot(range(1, adaline_iris.n_iter + 1), adaline_iris.cost_, marker='o')
            ax[0].set_xlabel('Epochs')
            ax[0].set_ylabel('Cost')
            ax[0].set_title('Adaline: Cost per Epoch (Iris Dataset)')

            # Plot accuracy for Iris dataset
            ax[1].plot(range(1, adaline_iris.n_iter + 1), adaline_iris.accuracies_, marker='o')
            ax[1].set_xlabel('Epochs')
            ax[1].set_ylabel('Accuracy')
            ax[1].set_title('Adaline: Accuracy per Epoch (Iris Dataset)')
            plt.show()

            print("--> Cost:")
            for i, item in enumerate(adaline_iris.cost_, start=1):
                print(f"Epoch {i}: {item}")
            print("--> Accuracies:")
            for i, item in enumerate(adaline_iris.accuracies_, start=1):
                print(f"Epoch {i}: {item}")
            print("--> Time required:", required_time, 'seconds',"\n")

        elif(dataset=='Drybeans'):
            # Load Dry Beans Dataset
            X_dry_beans, y_dry_beans = load_drybeans()
            # Train and test Adaline with Dry Beans dataset
            start_time = time.time()
            adaline_dry_beans = Adaline(eta=0.00000000000001, n_iter=20)
            adaline_dry_beans.fit(X_dry_beans, y_dry_beans)
            end_time = time.time()
            required_time = end_time - start_time

            fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(12, 4))

            # Plot errors for Dry Beans dataset
            ax[0].plot(range(1, adaline_dry_beans.n_iter + 1), adaline_dry_beans.cost_, marker='o')
            ax[0].set_xlabel('Epochs')
            ax[0].set_ylabel('Cost')
            ax[0].set_title('Adaline: Cost per Epoch (Dry Beans Dataset)')

            # Plot accuracy for Dry Beans dataset
            ax[1].plot(range(1, adaline_dry_beans.n_iter + 1), adaline_dry_beans.accuracies_, marker='o')
            ax[1].set_xlabel('Epochs')
            ax[1].set_ylabel('Accuracy')
            ax[1].set_title('Adaline: Accuracy per Epoch (Dry Beans Dataset)')
            plt.show()

            print("--> Cost:")
            for i, item in enumerate(adaline_dry_beans.cost_, start=1):
                print(f"Epoch {i}: {item}")
            print("--> Accuracies:")
            for i, item in enumerate(adaline_dry_beans.accuracies_, start=1):
                print(f"Epoch {i}: {item}")
            print("--> Time required:", required_time, 'seconds',"\n")

    elif (classifier=='SGD'):
        if(dataset=='Iris'):
            # Load Iris Dataset
            X_iris, y_iris = load_iris()
            # Train and test SGD with Iris dataset
            start_time = time.time()
            sgd_iris = SGD(eta=0.0001, n_iter=20)
            sgd_iris.fit(X_iris, y_iris)
            end_time = time.time()
            required_time = end_time - start_time

            fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(12, 4))

            # Plot errors for Iris dataset
            ax[0].plot(range(1, sgd_iris.n_iter + 1), sgd_iris.cost_, marker='o')
            ax[0].set_xlabel('Epochs')
            ax[0].set_ylabel('Cost')
            ax[0].set_title('SGD: Cost per Epoch (Iris Dataset)')

            # Plot accuracy for Iris dataset
            ax[1].plot(range(1, sgd_iris.n_iter + 1), sgd_iris.accuracies_, marker='o')
            ax[1].set_xlabel('Epochs')
            ax[1].set_ylabel('Accuracy')
            ax[1].set_title('SGD: Accuracy per Epoch (Iris Dataset)')
            plt.show()

            print("--> Cost:")
            for i, item in enumerate(sgd_iris.cost_, start=1):
                print(f"Epoch {i}: {item}")
            print("--> Accuracies:")
            for i, item in enumerate(sgd_iris.accuracies_, start=1):
                print(f"Epoch {i}: {item}")
            print("--> Time required:", required_time, 'seconds',"\n")

        elif(dataset=='Drybeans'):
            # Load Dry Beans Dataset
            X_dry_beans, y_dry_beans = load_drybeans()
            # Train and test SGD with Dry Beans dataset
            start_time = time.time()
            sgd_dry_beans = SGD(eta=0.0001, n_iter=20)
            sgd_dry_beans.fit(X_dry_beans, y_dry_beans)
            end_time = time.time()
            required_time = end_time - start_time

            fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(12, 4))

            # Plot errors for Iris dataset
            ax[0].plot(range(1, sgd_dry_beans.n_iter + 1), sgd_dry_beans.cost_, marker='o')
            ax[0].set_xlabel('Epochs')
            ax[0].set_ylabel('Cost')
            ax[0].set_title('SGD: Cost per Epoch (Iris Dataset)')

            # Plot accuracy for Iris dataset
            ax[1].plot(range(1, sgd_dry_beans.n_iter + 1), sgd_dry_beans.accuracies_, marker='o')
            ax[1].set_xlabel('Epochs')
            ax[1].set_ylabel('Accuracy')
            ax[1].set_title('SGD: Accuracy per Epoch (Iris Dataset)')
            plt.show()

            print("--> Cost:")
            for i, item in enumerate(sgd_dry_beans.cost_, start=1):
                print(f"Epoch {i}: {item}")
            print("--> Accuracies:")
            for i, item in enumerate(sgd_dry_beans.accuracies_, start=1):
                print(f"Epoch {i}: {item}")
            print("--> Time required:", required_time, 'seconds',"\n")

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
    print("--> Please input the classifier name: 'Perceptron' / 'Adaline' / 'SGD' or 'Stop' to terminate")
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