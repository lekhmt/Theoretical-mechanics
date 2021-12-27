import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint

def odesys(y, t, g, m, J, R, c):
    dy = np.zeros(4)
    dy[0] = y[2]
    dy[1] = y[3]

    a11 = J + m * R ** 2 * np.sin(y[1]) ** 2
    a12 = 0
    a21 = 0
    a22 = R

    b1 = -c * y[0] - m * R ** 2 * y[2] * y[3] * np.sin(2 * y[1])
    b2 = -g * np.sin(2 * y[1]) - R * y[2] ** 2 * np.sin(y[1])

    dy[2] = (b1 * a22 - b2 * a12) / (a11 * a22 - a12 * a21)
    dy[3] = (b2 * a11 - b1 * a21) / (a11 * a22 - a12 * a21)
    return dy

tn = np.linspace(0, 10, 2000)
xn = np.zeros_like(tn)
yn = np.zeros_like(tn)
zn = np.zeros_like(tn)

g = 9.81
m = 1
J = 0.5
R = 0.5
c = 2

phi0 = 0
theta0 = 0
dphi0 = 0
dtheta0 = 0.1
y0 = [phi0, theta0, dphi0, dtheta0]

Y = odeint(odesys, y0, tn, (g, m, J, R, c))

phi = Y[:, 0]
theta = Y[:, 1]

fig = plt.figure()

ax1 = fig.add_subplot(2, 2, 1)
ax1.plot(Y[:, 0])
ax1.set_title("$\\varphi$")

ax2 = fig.add_subplot(2, 2, 2)
ax2.plot(Y[:, 1])
ax2.set_title("$\\theta$")

ax3 = fig.add_subplot(2, 2, 3)
ax3.plot(Y[:, 2])
ax3.set_title("$\\varphi'$")

ax4 = fig.add_subplot(2, 2, 4)
ax4.plot(Y[:, 3])
ax4.set_title("$\\theta'$")

plt.show()
