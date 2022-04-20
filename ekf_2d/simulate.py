import numpy as np
import sys
import pathlib

from state_logger import StateLogger
from control import Control

current_dir = pathlib.Path(__file__).resolve().parent
sys.path.append( str(current_dir) + '/../' )
from physics_simulator.rigid_body_2d.rigid_body_2d import RigidBody2D
from ekf import EKF
from ekf import OnlyIntegral

ekf = EKF()
onlyIntegral = OnlyIntegral()

trueStateLogger = StateLogger('true_state.csv', ('time',
                                                 'X',
                                                 'Y',
                                                 'X_dot',
                                                 'Y_dot',
                                                 'yaw',
                                                 'yaw_dot'))

ekfStateLogger = StateLogger('ekf_state.csv', ('time',
                                                'X',
                                                'Y',
                                                'X_dot',
                                                'Y_dot',
                                                'yaw'))

integralStateLogger = StateLogger('integral_state.csv', ('time',
                                                         'X',
                                                         'Y',
                                                         'X_dot',
                                                         'Y_dot',
                                                         'yaw'))

targetStateLogger = StateLogger('target_state.csv', ('time',
                                                     'X',
                                                     'Y',
                                                     'yaw'))

X = 0.0
X_dot = 0.0
Y = 0.0
Y_dot = 0.0
yaw = 0.0
yaw_dot = 0.0
init_state = (X, X_dot, Y, Y_dot, yaw, yaw_dot)
rigidBody2D = RigidBody2D(init_state)

controller = Control()

time = 0.
dt = 10**(-2)
max_step = 1000 * 10**(2) + 1

# def add_data(df):
for s in range(0, max_step):
    time = s * dt
    if s % int(1 / dt * 10) == 0:
        print(time)

    # generate sensor_data
    imu_data = rigidBody2D.get_sensor_data()
    imu_data = np.array([ [imu_data['accel'][0] + 0.1 * np.random.randn()],
                          [imu_data['accel'][1] + 0.1 * np.random.randn()],
                          [imu_data['angle_rate'] + 0.1 * np.random.randn()] ])
    pos_data = np.array([ [rigidBody2D.state[0] + 0.01 * np.random.randn()],
                          [rigidBody2D.state[2] + 0.01 * np.random.randn()] ])

    # update ekf
    estimated_state = ekf.update( pos_data, imu_data, time )

    # about only integral estimation
    only_integ_state = onlyIntegral.update( pos_data, imu_data, time )

    if s % 10 == 0:
        trueStateLogger.add_data([time,
                                  rigidBody2D.state[0],
                                  rigidBody2D.state[2],
                                  rigidBody2D.state[1],
                                  rigidBody2D.state[3],
                                  rigidBody2D.state[4],
                                  rigidBody2D.state[5]])

        ekfStateLogger.add_data([time,
                                 estimated_state[0][0],
                                 estimated_state[1][0],
                                 estimated_state[2][0],
                                 estimated_state[3][0],
                                 estimated_state[4][0]])

        integralStateLogger.add_data([time,
                                 only_integ_state[0][0],
                                 only_integ_state[1][0],
                                 only_integ_state[2][0],
                                 only_integ_state[3][0],
                                 only_integ_state[4][0]])

        targetStateLogger.add_data([time,
                                    controller.x_pid.target,
                                    controller.y_pid.target,
                                    controller.yaw_pid.target])


    rigidBody2D.step(dt)

    if s > 0.0:
        rigidBody2D.input = controller.input(rigidBody2D.state[0],
                                             rigidBody2D.state[2],
                                             rigidBody2D.state[4],
                                             time)

trueStateLogger.finish()
ekfStateLogger.finish()
integralStateLogger.finish()
targetStateLogger.finish()
