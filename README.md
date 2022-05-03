# ekf_python_experiment
Python implementation to understand EKF operation.

## Overview
In this repository, EKF is implemented on simple robot models.

The data flow is here.

![data flow](docs/data_flow.png)

## Setup
Clone the repository and submodule.
```bash
git clone git@github.com:RyosukeMatsushima/ekf_py_experiment.git --recursive
```

## 2D robot model
[ekf_2d](ekf_2d) is about EKF with 2d robot model like this figure.

![2d robot model](docs/2d_robot_model.png)

The state to estimate is

$$
x = 
\begin{bmatrix}
X\\
Y\\
\dot{X}\\
\dot{Y}\\
\theta\\
\end{bmatrix}.
$$

IMU sensor data to predict are acceleration $[a_x, a_y]$ and angular velocity $\omega$ on robot coodinate.

Position data to update state is $[y_X, y_Y]$.

All input data includes sensor noise.

### Run
1. Run simulation.
```bash
python3 ekf_2d/simulate.py <simulate time[s]>
```
ex.
```bash
python3 ekf_2d/simulate.py 100
```

2. View result.
```bash
python3 ekf_2d/results_analyzer.py
```

## 3D robot model
TODO
