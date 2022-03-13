import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys
import pathlib

current_dir = pathlib.Path(__file__).resolve().parent
sys.path.append( str(current_dir) + '/../' )
from physics_simulator.rigid_body_2d.rigid_body_2d import RigidBody2D
from ekf import EKF

ekf = EKF()

X = 0.0
X_dot = 1.0
Y = 0.0
Y_dot = 4.0
yaw = 0.0
yaw_dot = 1.0
init_state = (X, X_dot, Y, Y_dot, yaw, yaw_dot)
rigidBody2D = RigidBody2D(init_state)
rigidBody2D.input = (1.0, 0.1, 0.00002)

time = 0.
dt = 10**(-2)
max_step = 10 * 10**(2) + 1

df = pd.DataFrame(columns=['time',
                           'X',
                           'X_dot',
                           'Y',
                           'Y_dot',
                           'yaw',
                           'yaw_dot'])

ekf_df = pd.DataFrame(columns=['time',
                               'X',
                               'Y',
                               'X_dot',
                               'Y_dot',
                               'yaw'])

# def add_data(df):
for s in range(0, max_step):
    time = s * dt
    tmp_data = tuple([time]) + rigidBody2D.state
    print(time)
    print('tmp_data')
    print(tmp_data)
    tmp_se = pd.Series(tmp_data, index=df.columns)
    df = df.append(tmp_se, ignore_index=True)

    # generate sensor_data
    imu_data = rigidBody2D.get_sensor_data()
    imu_data = np.array([ [imu_data['accel'][0]],
                          [imu_data['accel'][1]],
                          [imu_data['angle_rate']] ])

    pos_data = np.array([ [rigidBody2D.state[0]],
                          [rigidBody2D.state[2]] ])

    # update ekf
    estimated_state = ekf.update( pos_data, imu_data, time )
    tmp_data = tuple([time]) + tuple( val[0] for val in estimated_state )
    print('tmp_data')
    print(tmp_data)
    tmp_se = pd.Series(tmp_data, index=ekf_df.columns)
    ekf_df = ekf_df.append(tmp_se, ignore_index=True)

    rigidBody2D.step(dt)


ekf_df.plot(x='X', y='Y')
df.plot(x='X', y='Y')

ekf_df.plot(x='time', y='X')
df.plot(x='time', y='X')
ekf_df.plot(x='time', y='Y')
df.plot(x='time', y='Y')

ekf_df.plot(x='time', y='yaw')
df.plot(x='time', y='yaw')
plt.show()

