import numpy as np 





class Linear_Regression: 
    def __init__(self): 
        self.w = None
        self.b = None
        self.w_b = None 

    def fit(self, X_train, y_train):
        # X_train is a N x D numpy array  
        # y_train is a M x 1 numpy array
        # self.weight is a N x M numpy array
        # self.bias is a M x 1 numpy array
        [N, D] = X_train.shape
        M = y_train.shape[0]

        X_T_X = np.dot(X_train.T, X_train)
        X_T_y = np.dot(X_train.T, y_train)
        inv_X_T_X = np.linalg.inv(X_T_X) 

        self.w_b = np.dot(inv_X_T_X, X_T_y)
        
        self.b = self.w_b[-1]
        self.w = np.delete(self.w_b, -1, axis= 0)
        

    def predict(self, X_test): 
        return np.dot(self.W.T, X_test) + self.bias 