import numpy as np

class KalmanFilter:
    def __init__(self, dim_x, dim_z, dim_u):
        # Initialize state vector
        self.x = np.zeros((dim_x, 1))
        # State covariance matrix
        self.P = np.eye(dim_x) * 1000.0 # change value to play around with certainity
        # Process noise covariance matrix
        Q_custom = 0.1 * np.eye(dim_x)
        #Q_custom[3, 3] = 0.001  # lower process noise for obstacle speed
        #Q_custom[9, 9] = 0.001  # lower for ego speed
        self.Q = Q_custom
        # Measurement matrix (dim_z = number of measurement variables)
        self.H = np.eye(dim_z)
        # Measurement noise covariance matrix
        R_custom = 5.0 * np.eye(dim_z)
        #R_custom[3, 3] = 0.02   # adjust measurement noise for obstacle speed
        #R_custom[9, 9] = 0.02   # adjust for ego speed
        self.R = R_custom
        # Control matrix (dim_u = number of control variables)
        self.B = np.zeros((dim_x, dim_u))
        # State transition matrix
        self.F = np.eye(dim_x)
        
    def predict(self, u=None):
        # Predict the next state with control input if available
        if u is not None:
            u = np.reshape(u, (self.B.shape[1], 1))
            self.x = np.dot(self.F, self.x) + np.dot(self.B, u)
        else:
            self.x = np.dot(self.F, self.x)
        
        # Predict the error covariance
        self.P = np.dot(np.dot(self.F, self.P), self.F.T) + self.Q

    def update(self, z):
        z = np.reshape(z, (self.H.shape[0], 1))  # Ensure z is a column vector
        y = z - np.dot(self.H, self.x)  # Measurement residual
        S = np.dot(np.dot(self.H, self.P), self.H.T) + self.R  # Residual covariance
        K = np.dot(np.dot(self.P, self.H.T), np.linalg.inv(S))  # Kalman gain

        # Update state and covariance matrix
        self.x = self.x + np.dot(K, y)
        I = np.eye(self.x.shape[0])
        self.P = (I - np.dot(K, self.H)) @ self.P
