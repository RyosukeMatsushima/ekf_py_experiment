import numpy as np

class EKF:

    def __init__(self):

        # states: [position_x, position_y, velocity_x, velocity_y, yaw]
        self.x = np.array([[0.0, 0.0, 0.0, 0.0, 0.0]]).T
        self.P = np.array([[1000, 0.0, 0.0, 0.0, 0.0],
                           [0.0, 1000, 0.0, 0.0, 0.0],
                           [0.0, 0.0, 1000, 0.0, 0.0],
                           [0.0, 0.0, 0.0, 1000, 0.0],
                           [0.0, 0.0, 0.0, 0.0, 1000]])
        self.rho_v = 1.0
        self.C = np.array([[1.0, 0.0, 0.0, 0.0, 0.0],
                           [0.0, 1.0, 0.0, 0.0, 0.0]])
        self.R = np.array([[1.0], [1.0]])

        self.last_update_t = 0.0

        return

    # prediction step: calculate x_p, c_p
    # return predict states
    def predict(self, u, dt):

        A, B_u, B = self.get_liner_model(self.x, u, dt)

        x_predict = A @ self.x + B_u @ u
        P_predict = A @ self.P @ A.T + self.rho_v * B @ B.T

        return x_predict, P_predict

    # update with observed values
    def update(self, y, u, t):

        dt = t - self.last_update_t
        self.last_update_t = t
        x_p, P_p = self.predict(u, dt)

        denominator = self.C @ P_p @ self.C.T + self.R

        G = P_p @ self.C.T @ np.linalg.inv(denominator)
        self.x = x_p + G @ (y - self.C @ x_p)
        return self.x

    # return liner model matrix A, B_u, B
    def get_liner_model(self, x, u, dt):

        a_x = u[0][0]
        a_y = u[1][0]
        yaw = x[4][0]

        A = np.array([[0.0, 0.0, 1.0, 0.0, 0.0],
                      [0.0, 0.0, 0.0, 1.0, 0.0],
                      [0.0, 0.0, 0.0, 0.0, -a_x * np.sin(yaw) - a_y * np.cos(yaw)],
                      [0.0, 0.0, 0.0, 0.0, a_x * np.cos(yaw) - a_y * np.sin(yaw)],
                      [0.0, 0.0, 0.0, 0.0, 0.0]])

        B_u = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 1.0]])

        B = np.array([[0.1], [0.1], [0.01], [0.01], [0.02]])

        return A, B_u, B
