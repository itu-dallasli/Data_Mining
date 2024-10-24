import numpy as np

def rel_error(x, y):
    return np.max(np.abs(x - y) / (np.maximum(1e-8, np.abs(x) + np.abs(y))))

class minMaxScaler():
    def __init__(self, x, min_limit=0, max_limit=1):
        self.x = x
        self.min = np.min(x, axis=0)
        self.max = np.max(x, axis=0)
        self.min_limit = min_limit
        self.max_limit = max_limit
  
    def transform(self, x ):
        x_scaled = x
        ##############################################################################
        # TODO: Implement min max scaler function that scale a data range into       #
        # min_limit and max_limit                                                    #
        # https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.   #
        # MinMaxScaler.html                                                          #
        ##############################################################################
        # Replace "pass" statement with your code
        x_Std = (x_scaled - self.min) / (self.max - self.min)
        x_scaled = x_Std * (self.max_limit - self.min_limit) + self.min_limit
        
    
        
        ##############################################################################
        #                             END OF YOUR CODE                               #
        ##############################################################################
        return x_scaled
    
    def inverse_transform(self, x_scaled):
        x = x_scaled
        
        ##############################################################################
        # TODO: Implement inverse min max scaler that scale a data range back into   #
        # previous range.                                                            #
        ##############################################################################
        # Replace "pass" statement with your code
        x_nrm = (x_scaled/(self.max_limit - self.min_limit)) - self.min_limit 
        x = (x_nrm * (self.max - self.min)) + self.min
        ##############################################################################
        #                             END OF YOUR CODE                               #
        ##############################################################################        
        return x
    
        ##############################################################################


def leastSquares(x, y):
    w = np.zeros((x.shape[1]+1, 1))
    ##############################################################################
    # TODO: Implement least square theorem.                                      #
    #                                                                            #
    # You may not use any built in function which directly calculate             #
    # Least squares except matrix operation in numpy.                            #
    ##############################################################################
    # Replace "pass" statement with your code
 
    ones_column = np.ones((x.shape[0] , 1))
    X = np.hstack((ones_column, x))
    np.delete(w, (0), axis=0) # Unnecessary column deleted bcs of the w definition.
    w = np.matmul(np.linalg.inv(np.matmul(X.transpose(), X)), np.matmul(X.transpose(), y))

    ##############################################################################
    #                             END OF YOUR CODE                               #
    ##############################################################################
    return w

class gradientDescent():
    def __init__(self, x, y, w, lr, num_iters):
        self.x = np.c_[np.ones((x.shape[0], 1)), x]
        self.y = y
        self.lr = lr
        self.num_iters = num_iters
        self.epsilon = 1e-4
        self.w = w.copy()
        self.weight_history = [self.w]
        self.cost_history = [np.sum(np.square(self.predict(self.x)-self.y))/self.x.shape[0]]
        
    def gradient(self):
        gradient = np.zeros_like(self.w)
        ##############################################################################
        # TODO: Calculate gradient of gradient descent algorithm.                    #
        #                                                                            #
        ##############################################################################
        #  Replace "pass" statement with your code

        m = self.x.shape[0]
        
        predictions = np.dot(self.x, self.w)
        errors = predictions - self.y

        for j in range(len(self.w)):
            feature_j = self.x[:, j]
            gradient[j] = (1/m) * np.dot(errors.T, feature_j)
        
        ##############################################################################
        #                             END OF YOUR CODE                               #
        ##############################################################################
        return gradient
    
    
    def fit(self,lr=None, n_iterations=None):
        k = 0
        if n_iterations is None:
            n_iterations = self.num_iters
        if lr != None and lr != "diminishing":
            self.lr = lr
        ##############################################################################
        # TODO: Implement gradient descent algorithm.                                #
        #                                                                            #
        # You may not use any built in function which directly calculate             #
        # gradient.                                                                  #
        # Steps of gradient descent algorithm:                                       #
        #   1. Calculate gradient of cost function. (Call gradient() function)       #
        #   2. Update w with gradient.                                               #
        #   3. Log weight and cost for plotting in weight_history and cost_history.  #
        #   4. Repeat 1-3 until the cost change converges to epsilon or achieves     #
        # n_iterations.                                                              #
        #                                                                            #
        # !!WARNING-1: Use Mean Square Error between predicted and actual y values   #
        # for cost calculation.                                                      #
        #                                                                            #
        # !!WARNING-2: Be careful about lr can takes "diminishing". It will be used  #
        # for further test in the jupyter-notebook. If it takes lr decrease with     #
        # iteration as (lr =(1/k+1), here k is iteration).                           #
        ##############################################################################
        while k < n_iterations:
          grad = self.gradient()
          if lr == "diminishing":
              self.lr = 1 / (k + 1)
          
          self.w = self.w - (self.lr * grad)
          
          ms_error = np.sum(np.square(self.predict(self.x) - self.y)) / self.x.shape[0]
          self.cost_history.append(ms_error)
          self.weight_history.append(self.w)

          if k > 0:
            if abs(self.cost_history[-1] - self.cost_history[-2]) < self.epsilon:
                break
        
          k = k + 1
        ##############################################################################
        #                             END OF YOUR CODE                               #
        ##############################################################################
        return self.w, k
    
    def predict(self, x):
        
        y_pred = np.zeros_like(self.y)

        y_pred = x.dot(self.w)

        return y_pred
