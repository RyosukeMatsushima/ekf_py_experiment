import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys
import pathlib

current_dir = pathlib.Path(__file__).resolve().parent
sys.path.append( str(current_dir) + '/../' )
from physics_simulator.rigid_body_2d.rigid_body_2d import RigidBody2D
from ekf import EKF
from ekf import OnlyIntegral

ekf = EKF()
onlyIntegral = OnlyIntegral()

X = 0.0
X_dot = 0.0
Y = 0.0
Y_dot = 0.0
yaw = 0.0
yaw_dot = 0.0
init_state = (X, X_dot, Y, Y_dot, yaw, yaw_dot)
rigidBody2D = RigidBody2D(init_state)
#rigidBody2D.input = (0.001, 0.001, 0.00002)
rigidBody2D.input = (1.0, 1.0, 0.0)
rigidBody2D.input = (0.001, 0.001, 0.0)

time = 0.
dt = 10**(-2)
max_step = 30 * 10**(2) + 1

state_df = pd.DataFrame(columns=['time',
                                 'X',
                                 'X_dot',
                                 'Y',
                                 'Y_dot',
                                 'yaw',
                                 'yaw_dot',
                                 'est_X',
                                 'est_Y',
                                 'est_X_dot',
                                 'est_Y_dot',
                                 'est_yaw',
                                 'X_err',
                                 'Y_err',
                                 'yaw_err',
                                 'only_integ_est_X',
                                 'only_integ_est_Y',
                                 'only_integ_est_X_dot',
                                 'only_integ_est_Y_dot',
                                 'only_integ_est_yaw',
                                 'only_integ_X_err',
                                 'only_integ_Y_err',
                                 'only_integ_yaw_err'])

# def add_data(df):
for s in range(0, max_step):
    time = s * dt
    print(time)

    # generate sensor_data
    imu_data = rigidBody2D.get_sensor_data()
    imu_data = np.array([ [imu_data['accel'][0] + 0.1 * np.random.randn()],
                          [imu_data['accel'][1] + 0.1 * np.random.randn()],
                          [imu_data['angle_rate'] + 0.000001 * np.random.randn()] ])

    pos_data = np.array([ [rigidBody2D.state[0] + 0.01 * np.random.randn()],
                          [rigidBody2D.state[2] + 0.01 * np.random.randn()] ])

    # update ekf
    estimated_state = ekf.update( pos_data, imu_data, time )

    # about only integral estimation
    only_integ_state = onlyIntegral.update( pos_data, imu_data, time )

    tmp_data = tuple([time]) \
             + rigidBody2D.state \
             + tuple( val[0] for val in estimated_state ) \
             + (rigidBody2D.state[0] - estimated_state[0], \
                rigidBody2D.state[2] - estimated_state[1], \
                rigidBody2D.state[4] - estimated_state[4]) \
             + tuple( val[0] for val in only_integ_state ) \
             + (rigidBody2D.state[0] - only_integ_state[0], \
                rigidBody2D.state[2] - only_integ_state[1], \
                rigidBody2D.state[4] - only_integ_state[4])
    tmp_se = pd.Series(tmp_data, index=state_df.columns)
    state_df = state_df.append(tmp_se, ignore_index=True)
 
    rigidBody2D.step(dt)

def show_plt(title, df, labels):
    fig = plt.figure(figsize=(6, 4))
    ax = fig.add_subplot(111)
    ax.set_title(title)

    for label in labels:
        ax.plot(df[label[0]], df[label[1]], label=label[2])

    plt.legend()

show_plt('X Y', state_df, [['X', 'Y', 'true'],
                           ['est_X', 'est_Y', 'est'],
                           ['only_integ_est_X', 'only_integ_est_Y', 'only_integ_est']])
show_plt('X', state_df, [['time', 'X', 'true'],
                         ['time', 'est_X', 'est'],
                         ['time', 'only_integ_est_X', 'only_integ_est']])
show_plt('Y', state_df, [['time', 'Y', 'true'],
                         ['time', 'est_Y', 'est'],
                         ['time', 'only_integ_est_Y', 'only_integ_est']])
show_plt('yaw', state_df, [['time', 'yaw', 'true'],
                           ['time', 'est_yaw', 'est'],
                           ['time', 'only_integ_est_yaw', 'only_integ_est']])

show_plt('X_err', state_df, [['time', 'X_err', 'X_err'],
                             ['time', 'only_integ_X_err', 'only_integ_X_err']])
show_plt('Y_err', state_df, [['time', 'Y_err', 'Y_err'],
                             ['time', 'only_integ_Y_err', 'only_integ_Y_err']])
show_plt('yaw_err', state_df, [['time', 'yaw_err', 'yaw_err'],
                               ['time', 'only_integ_yaw_err', 'only_integ_yaw_err']])

plt.show()
