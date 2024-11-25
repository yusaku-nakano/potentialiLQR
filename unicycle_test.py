import numpy as np
import matplotlib.pyplot as plt

import ilqr
from ilqr import split_agents, plot_solve

dt = 0.05
N = 50

x = np.array([-10, 10, 10, 0], dtype=float)
x_goal = np.zeros((4, 1), dtype=float).T

dynamics = ilqr.UnicycleDynamics4D(dt)

Q = np.diag([1.0, 1, 0, 0])
Qf = 1000 * np.eye(Q.shape[0])
R = np.eye(2)
cost = ilqr.TrackingCost(x_goal, Q, R, Qf)

prob = ilqr.ilqrProblem(dynamics, cost)
ilqrx = ilqr.iLQR(prob, N)
#us_init = np.random.randn(N, ilqrx.nU)*0.0001
us_init = np.zeros((N, ilqrx.nU))
X, _, J_trace, J = ilqrx.run_ilqr(x, us_init)

plt.clf()
plot_solve(X, J, x_goal)
plt.show()