import numpy as np

class EKF:

    def __init__(self):

        # states: [position_x, position_y, velocity_x, velocity_y, yaw]
        self.x = np.array([[0.0, 0.0, 0.0, 0.0, 0.0]]).T
        self.P = np.array([[0.01, 0.0, 0.0, 0.0, 0.0],
                           [0.0, 0.01, 0.0, 0.0, 0.0],
                           [0.0, 0.0, 0.01, 0.0, 0.0],
                           [0.0, 0.0, 0.0, 0.01, 0.0],
                           [0.0, 0.0, 0.0, 0.0, 0.000000001]])
        self.Q = np.array([[0.01, 0.0, 0.0],
                           [0.0, 0.01, 0.0],
                           [0.0, 0.0, 0.0000001]])
        self.C = np.array([[1.0, 0.0, 0.0, 0.0, 0.0],
                           [0.0, 1.0, 0.0, 0.0, 0.0]])
        self.R = np.array([[0.00001], [0.00001]])

        self.last_update_t = 0.0

        return

    # prediction step: calculate x_p, c_p
    # return predict states
    def predict(self, u, dt):

        a_x = u[0][0]
        a_y = u[1][0]
        yaw_rate = u[2][0]
        yaw = self.x[4][0]

        x_dot = np.array([[self.x[2][0],
                           self.x[3][0],
                           a_x * np.cos(yaw) - a_y * np.sin(yaw),
                           a_x * np.sin(yaw) + a_y * np.cos(yaw),
                           yaw_rate]]).T
        x_predict = self.x + x_dot * dt

        A, B = self.get_liner_model(a_x, a_y, yaw_rate, yaw, dt)
        P_predict = A @ self.P @ A.T + B @ self.Q @ B.T

        return x_predict, P_predict

    # update with observed values
    def update(self, y, u, t):

        dt = t - self.last_update_t
        self.last_update_t = t
        x_p, P_p = self.predict(u, dt)

        denominator = self.C @ P_p @ self.C.T + self.R

        G = P_p @ self.C.T @ np.linalg.inv(denominator)
        self.x = x_p + G @ (y - self.C @ x_p)

        self.P = (np.eye(len(self.x)) - G @ self.C) @ P_p

        return self.x

    # return liner model matrix A, B_u, B
    def get_liner_model(self, a_x, a_y, yaw_rate, yaw, dt):

        A = np.array([[0.0, 0.0, 1.0, 0.0, 0.0],
                      [0.0, 0.0, 0.0, 1.0, 0.0],
                      [0.0, 0.0, 0.0, 0.0, -a_x * np.sin(yaw) - a_y * np.cos(yaw)],
                      [0.0, 0.0, 0.0, 0.0, a_x * np.cos(yaw) - a_y * np.sin(yaw)],
                      [0.0, 0.0, 0.0, 0.0, 0.0]])
        A = np.eye(len(self.x)) + A * dt

        B = np.array([[0.0, 0.0, 0.0],
                      [0.0, 0.0, 0.0],
                      [np.cos(yaw), -np.sin(yaw), 0.0],
                      [np.sin(yaw), np.cos(yaw), 0.0],
                      [0.0, 0.0, 1.0]])
        B = B * dt

        return A, B

    def yaw_correction(self, yaw):
        return yaw - int(yaw/np.pi) * np.pi

class OnlyIntegral(EKF):

    def update(self, y, u, t):
        dt = t - self.last_update_t
        self.last_update_t = t
        x_p, P_p = self.predict(u, dt)
        self.x = x_p

        return self.x
