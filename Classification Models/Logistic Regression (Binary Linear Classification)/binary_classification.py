import numpy as np


class LogisticRegression:
    
    def __init__(self, x_train, y_train, x_test, y_test):
        """
        Constructor assumes a x_train matrix in which each column contains an instance.
        Vector y_train contains one integer for each instance, indicating the instance's label. 
        
        Constructor initializes the weights W and B, alpha, and a one-vector Y containing the labels
        of the training set. Here we assume there are 10 labels in the dataset. 
        """
        self._x_train = x_train
        self._y_train = y_train
        self._x_test = x_test
        self._y_test = y_test
        self._m = x_train.shape[1]
        
        self._W = np.random.randn(10, 784) * 0.01
        self._B = np.zeros((10, 1))
        self._Y = np.zeros((10, self._m))
        self._alpha = 0.05

        for index, value in enumerate(self._y_train):
            self._Y[value][index] = 1
            
    def sigmoid(self, Z):
        """
            Args:
            - Z (numpy.ndarray): The input array.

            Returns:
            - numpy.ndarray: An array of the same shape as Z, where each element is the sigmoid of the corresponding element in Z.
            
        """
        
        ##############################################################################
        #Computes the sigmoid value for all values in vector Z
        # TODO: # Write the computation of the sigmoid function for a given matrix Z                                                        #
        ##############################################################################
        # Replace "pass" statement with your code
        
        return 1 / (1 + np.exp(-Z))
        
        ##############################################################################
        #                             END OF YOUR CODE                               #
        ##############################################################################
        return sigmoid_val

    def derivative_sigmoid(self, A):
        """
            Args:
            - A (numpy.ndarray): The input array.

            Returns:
            - numpy.ndarray: An array of the same shape as A, where each element is the derivative of the sigmoid function
            for the corresponding element in A.
        """
 
        ##############################################################################
        #Computes the derivative of the sigmoid for all values in vector A
        # TODO: # Write the derivative of the sigmoid function for a given value A produced by the sigmoid                                                       #
        ##############################################################################
        # Replace "pass" statement with your code
        
        deriv_sigmoid_val = A * (1 - A)
        ##############################################################################
        #                             END OF YOUR CODE                               #
        ##############################################################################
        return deriv_sigmoid_val

    def h_theta(self, X):
        """
            Args:
            - X (numpy.ndarray): The input feature matrix.

            Returns:
            - numpy.ndarray: A column vector of predicted values obtained
        """

        ##############################################################################
        #Computes the value of the hypothesis according to the logistic regression rule
        # TODO: # Write the computation of the hypothesis for a given matrix X                                                       #
        ##############################################################################
        # Replace "pass" statement with your code
        
        h_theta = self.sigmoid(np.dot(self._W, X) + self._B)
        
        ##############################################################################
        #                             END OF YOUR CODE                               #
        ##############################################################################

        return h_theta
    
    def return_weights_of_digit(self, digit):
        """
            Args:
            - digit (int): The digit for which the weights are to be returned.

            Returns:
            - numpy.ndarray: A row vector of weights from the weights matrix corresponding to the given digit.
        """

        ##############################################################################
        # TODO: # Returns the weights of the model for a given digit                                                      #
        ##############################################################################
        # Replace "pass" statement with your code
        
        weights_of_digits = self._W[digit]
        
        ##############################################################################
        #                             END OF YOUR CODE                               #
        ##############################################################################

        return weights_of_digits
    
    def train_mse_loss(self, iterations):
        """
        Performs a number of iterations of gradient descend equals to the parameter passed as input.
        
        Returns a list with the percentage of instances classified correctly in the training and in the test sets.
        """
        classified_correctly_train_list = []
        classified_correctly_test_list = []
        
        for i in range(iterations):
 
            ##############################################################################
            # TODO: #Please write your answers for A, pure_error, W, B , classified_correctly, 
            #         percentage_classified_correctly,test_correct parts      

            # Write the following four lines of code for computing the value produced by the model (A)
            # The pure error for all training instances (pure_error)
            # And adjust the matrices self._W and self._B according to the gradient descent rule
            ####                                      
            ##############################################################################
            
            A = self.h_theta(self._x_train)
            pure_error = A - self._Y
            self._W = self._W - self._alpha * np.dot(pure_error, self._x_train.T) / self._m
            self._B =  self._B - self._alpha * np.sum(pure_error, axis=1, keepdims=True) / self._m

            if i % 100 == 0:
                classified_correctly = np.sum(np.argmax(A, axis=0) == self._y_train)
                percentage_classified_correctly = (classified_correctly / self._m) * 100
                classified_correctly_train_list.append(percentage_classified_correctly)
                
                Y_hat_test = self.h_theta(self._x_test)
                test_correct = np.sum([1 if pred == true_label else 0 for pred, true_label in zip(np.argmax(Y_hat_test, axis=0), self._y_test)])
                classified_correctly_test_list.append((test_correct)/len(self._y_test) * 100)
                
                print('Accuracy train data: %.2f' % percentage_classified_correctly)
        return classified_correctly_train_list, classified_correctly_test_list


    def train_cross_entropy_loss(self, iterations):
        """
        Performs a number of iterations of gradient descend equals to the parameter passed as input.
        
        Returns a list with the percentage of instances classified correctly in the training and in the test sets.
        """

        classified_correctly_train_list_ce = []
        classified_correctly_test_list_ce = []
        
        for i in range(iterations):
            # Write the following four lines of code for computing the value produced by the model (A)
            # The pure error for all training instances (pure_error)
            # And adjust the matrices self._W and self._B according to the gradient descent rule

            ##############################################################################
            # TODO: #Please write your answers for A, pure_error, W, B , classified_correctly, 
            #         percentage_classified_correctly_ce,test_correct parts      

            # Write the following four lines of code for computing the value produced by the model (A)
            # The pure error for all training instances (pure_error)
            # And adjust the matrices self._W and self._B according to the gradient descent rule
            ####                                      
            ##############################################################################
            
            A = self.h_theta(self._x_train)
            pure_error = A - self._Y
            self._W = self._W - self._alpha * np.dot(pure_error, self._x_train.T) / self._m
            self._B =  self._B - self._alpha * np.sum(pure_error, axis=1, keepdims=True) / self._m

 

            if i % 100 == 0:
                classified_correctly = np.sum(np.argmax(A, axis=0) == self._y_train)
                percentage_classified_correctly_ce = (classified_correctly / self._m) * 100
                classified_correctly_train_list_ce.append(percentage_classified_correctly_ce)
                
                Y_hat_test = self.h_theta(self._x_test)
                test_correct = np.sum([1 if pred == true_label else 0 for pred, true_label in zip(np.argmax(Y_hat_test, axis=0), self._y_test)])
                classified_correctly_test_list_ce.append((test_correct)/len(self._y_test) * 100)
                
                print('Accuracy train data: %.2f' % percentage_classified_correctly_ce)
        return classified_correctly_train_list_ce, classified_correctly_test_list_ce




