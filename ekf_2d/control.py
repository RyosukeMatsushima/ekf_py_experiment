import numpy as np

class Control:
    def __init__(self):
        self.x_pid = PID(0.4, 0.0, 0.5)
        self.y_pid = PID(0.4, 0.0, 0.5)
        self.yaw_pid = PID(0.001, 0.0, 0.002)

        self.last_time = 0.0
        self.last_on_the_way_time = 0.0

    def is_reached(self, x, y, yaw):
        return  abs(x - self.x_pid.target) < 0.001 \
                and abs(y - self.y_pid.target) < 0.001 \
                and abs(yaw - self.yaw_pid.target) < 0.001

    def input(self, x, y, yaw, time):
        dt = time - self.last_time #TODO: check first time is zero
        self.last_time = time

        if not self.is_reached(x, y, yaw):
            self.last_on_the_way_time = time

        reached = time - self.last_on_the_way_time > 1.0
        if reached:
            self.update_target()
            print('reached')

        x_input = self.x_pid.input(x, dt)
        y_input = self.y_pid.input(y, dt)

        return (np.cos(yaw) * x_input + np.sin(yaw) * y_input,
                -np.sin(yaw) * x_input + np.cos(yaw) * y_input,
                self.yaw_pid.input(yaw, dt))

    def update_target(self):
        self.x_pid.set_target(np.random.randn() * 10)
        self.y_pid.set_target(np.random.randn() * 10)
        self.yaw_pid.set_target(np.random.randn() * np.pi)


class PID:
    def __init__(self, gain_p, gain_i, gain_d):
        self.gain_p = gain_p
        self.gain_i = gain_i
        self.gain_d = gain_d

        self.previous = 0.0
        self.fresh()

    def set_target(self, value):
        self.fresh()
        self.target = value

    def input(self, value, dt):
        diff = self.target - value
        self.integral += diff * dt

        d_diff = - (value - self.previous) / dt
        self.previous = value

        return self.gain_p * diff + self.gain_i * self.integral + self.gain_d * d_diff

    def fresh(self):
        self.target = 0.0
        self.integral = 0.0


