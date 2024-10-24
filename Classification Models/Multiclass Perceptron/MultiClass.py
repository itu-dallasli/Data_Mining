############################################################################################
#               Implementation of MultiClass Perceptron.                                   #
############################################################################################


import numpy as np
from numpy.linalg import norm
import matplotlib.pyplot as plt
from collections import defaultdict
from sklearn.model_selection import train_test_split
import time



def visualizer(loss, accuracy, n_epochs):
    """
    Returns the plot of Training/Validation Loss and Accuracy.
    :param loss: A list defaultdict with 2 keys "train" and "val".
    :param accuracy: A list defaultdict with 2 keys "train" and "val".
    :param n_epochs: Number of Epochs during training.
    :return:
    """
    fig, axs = plt.subplots(1, 2, figsize=(20, 10))
    x = np.arange(0, n_epochs, 1)
    axs[0].plot(x, loss['train'], 'b')
    axs[0].plot(x, loss['val'], 'r')
    axs[1].plot(x, accuracy['train'], 'b')
    axs[1].plot(x, accuracy['val'], 'r')

    axs[0].set_xlabel("Epochs")
    axs[0].set_ylabel("Loss value")
    axs[1].set_xlabel("Epochs")
    axs[1].set_ylabel("Accuracy value (in %)")

    axs[0].legend(['Training loss', 'Validation loss'])
    axs[1].legend(['Training accuracy', 'Validation accuracy'])


class OneVsAll:
    def __init__(self, x_train, y_train, x_test, y_test, alpha, beta, mb, n_class, F, n_epochs, info):
        """
        This is an implementation from scratch of Multi Class Perceptron using One vs All strategy,
        and Momentum with SGD optimizer.

        :param x_train: Vectorized training data.
        :param y_train: Label training vector.
        :param x_test: Vectorized testing data.
        :param y_test: Label test vector.
        :param alpha: The learning rate.
        :param beta: Momentum parameter.
        :param mb: Mini-batch size.
        :param n_class: Number of classes.
        :param F: Number of features.
        :param n_epochs: Number of Epochs.
        :param info: 1 to show training loss & accuracy over epochs, 0 otherwise.
        """
        self.x_train = x_train
        self.y_train = y_train
        self.x_test = x_test
        self.y_test = y_test
        self.alpha = alpha
        self.beta = beta
        self.mb = mb 
        self.n_class = n_class
        self.F = F 
        self.n_epochs = n_epochs
        self.info = info

    def relabel(self, label):
        """
        This function takes a class, and relabels the training label vector into a binary class,
        it's used to apply One vs All strategy.

        :param label: The class to relabel.
        :return: A new binary label vector.
        """

        y = self.y_train.tolist()
        n = len(y)
        y_new = [1 if y[i] == label else 0 for i in range(n)]

        return np.array(y_new).reshape(-1, 1)

    def momentum(self, y_relab):
        """
        This function is an implementation of the momentum with SGD optimization algorithm, and it's
        used to find the optimal weight vector of the perceptron algorithm.
        :param y_relab: A binary label vector.
        :return: A weight vector, and history of loss/accuracy over epochs.
        """


        # Initialize weights and velocity vectors
        W = np.zeros((self.F + 1, 1))
        V = np.zeros((self.F + 1, 1))

        # Store loss & accuracy values for plotting
        loss = defaultdict(list)
        accuracy = defaultdict(list)

        # Split into training and validation sets
        x_train, x_val, y_train, y_val = train_test_split(self.x_train, y_relab, test_size=0.1, random_state=42)
        n_train = len(x_train)
        n_val = len(x_val)

        for _ in range(self.n_epochs):

            start = time.time()
            train_loss = 0
    
            for i in range(0, n_train - self.mb + 1, self.mb):
                x_batch = x_train[i:i + self.mb]   
                y_batch = y_train[i:i + self.mb]         
                x_batch = np.hstack((np.ones((len(x_batch),1)),x_batch)) 
  
                y_hat = 1 / (1 + np.exp(np.dot(x_batch,W)))
                y_hat = y_hat 
                pure_error = y_hat - y_batch

                #### CROSS ENTROPY FOR CALCULATING ERROR: I wanted to do this way but I don't think it is too accurate. ###
                batch_loss = -np.mean(y_batch * np.log(y_hat + 1e-10) + (1 - y_batch) * np.log(1 - y_hat + 1e-10)) # Calculate the cross-entropy loss
                train_loss += batch_loss * (self.mb / n_train)  # Scaling the batch loss by the batch size and accumulate


                grad = x_batch.T @ pure_error / self.mb
                
                
                V = self.beta * V - self.alpha * grad # Updating velocities and the weights
                W += V


            
            x_train_bias = np.hstack((np.ones((len(x_train), 1)), x_train)) # Adding a bias term to the feature matrix of the training data.
            y_hat = 1 / (1 + np.exp(np.dot(x_train_bias, W)))
            y_hat_ed = (y_hat > 0.5).astype(int)  # This line replaces the two lines in the code for converting to binary labels. Obtained from open source.
            train_acc = 100 * np.mean(y_hat_ed == y_train) # Accuracy calculation

            # Bunch of computations 
            x_val_bias = np.hstack((np.ones((len(x_val), 1)), x_val))
            y_hat = 1 / (1 + np.exp(np.dot(x_val_bias,W)))
            y_hat_ed = y_hat.copy()
            y_hat_ed[y_hat > 0.5] = 1
            y_hat_ed[y_hat <= 0.5] = 0
            val_acc = 100*(np.sum(y_hat_ed == y_val)/n_val)
            val_loss = np.mean((y_hat - y_val))           
            
            end = time.time()
            duration = round(end - start, 2)

            if self.info: print("Epoch: {} | Duration: {}s | Train loss: {} |"" Train accuracy: {}% | Validation loss: {} | "
                                "Validation accuracy: {}%".format(_, duration,
                                round(train_loss, 5), train_acc, round(val_loss, 5), val_acc))

            loss['train'].append(train_loss)
            loss['val'].append(val_loss)
            accuracy['train'].append(train_acc)
            accuracy['val'].append(val_acc)

        return W, loss, accuracy

    def train(self):
        """
        This function trains the model using One-vs-All strategy, and returns a weight
        matrix, to be used during inference.
        :return: A weight matrix of size (F+1, n_class), where F is the number of features,
        and n_class is the number of classes to predict.
        """


        weights = []
        loss, accuracy = 0, 0

        for i in range(1, self.n_class + 1):
            y_relab = self.relabel(i) 
            W, loss, accuracy = self.momentum(y_relab)
            weights.append(W)

        weights = np.array(weights)
        weights = np.squeeze(weights, axis=-1) 
        
        return weights, loss, accuracy

    def test(self, weights):
        """
        This function is used to test the model over new testing data samples, using
        the weights matrix obtained after training.
        :param weights: A weight matrix of size (F+1, n_class), where F is the number of features,
        and n_class is the number of classes to predict.
        :return:
        """

        x_test_bias = np.hstack((np.ones((len(self.x_test), 1)), self.x_test))
        y_hat = 1 / (1 + np.exp(np.dot(x_test_bias,weights.T))) 
        test_acc = 100*np.mean(np.argmax(y_hat, axis= 1) == self.y_test) 

##############################################################################
        print("-" * 50 + "\n Test accuracy is {}%".format(test_acc))

