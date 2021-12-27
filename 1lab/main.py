import numpy as np
import sympy as sp
import math
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

v0 = 25
R = 10

t = sp.Symbol('t')
x = v0 * t - R * sp.cos((v0 * t) / R - np.pi / 2)
y = R + R * sp.sin((v0 * t) / R - np.pi / 2)
vx = sp.diff(x, t)
vy = sp.diff(y, t)
wx = sp.diff(vx, t)
wy = sp.diff(vy, t)
cx = v0 * t

print("x(t) = ", x)
print("y(t) = ", y)
print("vx(t) = ", vx)
print("vy(t) = ", vy)
print("wx(t) = ", wx)
print("wy(t) = ", wy)


tn = np.linspace(0, 20, 1000)
xn = np.zeros_like(tn)
yn = np.zeros_like(tn)
vxn = np.zeros_like(tn)
vyn = np.zeros_like(tn)
wxn = np.zeros_like(tn)
wyn = np.zeros_like(tn)
cxn = np.zeros_like(tn)

for i in range(len(tn)):
    xn[i] = sp.Subs(x, t, tn[i])
    yn[i] = sp.Subs(y, t, tn[i])
    vxn[i] = sp.Subs(vx, t, tn[i])
    vyn[i] = sp.Subs(vy, t, tn[i])
    wxn[i] = sp.Subs(wx, t, tn[i])
    wyn[i] = sp.Subs(wy, t, tn[i])
    cxn[i] = sp.Subs(cx, t, tn[i])

fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
ax.axis('equal')
ax.plot(xn, yn, linestyle="--", color="gray")
ax.axhline(y=0, color='gray')
ax.axvline(x=0, color='gray')


def update(i):
    # точка
    point.set_data(xn[i], yn[i])
    # окружность
    global circle
    circle.remove()
    circle = plt.Circle((cxn[i], R), R, edgecolor="#ff924a", facecolor="#fad1b6")
    ax.add_patch(circle)
    # линия вектора скорости
    velocity.set_data([xn[i], xn[i] + vxn[i]], [yn[i], yn[i] + vyn[i]])
    # координаты конца вектора скорости ">"
    varrow_x, varrow_y = Rot2D(varrow_x0, varrow_y0, math.atan2(vyn[i], vxn[i]))
    varrow.set_data(varrow_x + xn[i] + vxn[i], varrow_y + yn[i] + vyn[i])
    # линия вектора ускорения
    acceleration.set_data([xn[i], xn[i] + wxn[i]], [yn[i], yn[i] + wyn[i]])
    # координаты конца вектора ускорения ">"
    warrow_x, warrow_y = Rot2D(warrow_x0, warrow_y0, math.atan2(wyn[i], wxn[i]))
    warrow.set_data(warrow_x + xn[i] + wxn[i], warrow_y + yn[i] + wyn[i])
    # центр окружности
    circle_center.set_data(cxn[i], R)
    # всё про кривизну
    curvature = (vxn[i] ** 2 + vyn[i] ** 2) / math.sqrt(wxn[i] ** 2 + wyn[i] ** 2)
    k = curvature / math.sqrt(vxn[i] ** 2 + vyn[i] ** 2)
    curvature_line.set_data([xn[i], xn[i] + vyn[i] * k], [yn[i], yn[i] - vxn[i] * k])
    curvature_center.set_data(xn[i] + vyn[i] * k, yn[i] - vxn[i] * k)
    global curvature_circle
    curvature_circle.remove()
    curvature_circle = plt.Circle((xn[i] + vyn[i] * k, yn[i] - vxn[i] * k), curvature, alpha=0.1)
    ax.add_patch(curvature_circle)

    return point, velocity, varrow, circle, acceleration, warrow, circle_center, curvature_line, curvature_center, curvature_circle


def Rot2D(X, Y, Alpha):
    RX = X * np.cos(Alpha) - Y * np.sin(Alpha)
    RY = X * np.sin(Alpha) + Y * np.cos(Alpha)
    return RX, RY


# точка
point = ax.plot(xn[0], yn[0], marker=".", color="black")[0]
# окружность
circle = plt.Circle((0, R), R, edgecolor="#ff924a", facecolor="#fad1b6")
ax.add_patch(circle)
# линия вектора скорости
velocity = ax.plot([xn[0], xn[0] + vxn[0]], [yn[0], yn[0] + vyn[0]], color="#2ec4f2", label="$v$")[0]
# координаты конца вектора скорости ">"
varrow_x0 = np.array([-0.2 * R, 0, -0.2 * R])
varrow_y0 = np.array([0.1 * R, 0, -0.1 * R])
varrow_x, varrow_y = Rot2D(varrow_x0, varrow_y0, math.atan2(vyn[0], vxn[0]))
varrow = ax.plot(varrow_x + xn[0] + vxn[0], varrow_y + yn[0] + vyn[0], color="#2ec4f2")[0]
# линия усоркения
acceleration = ax.plot([xn[0], xn[0] + wxn[0]], [yn[0], yn[0] + wyn[0]], color="#bd0404", label="$w$")[0]
# координаты конца вектора ускорения ">"
warrow_x0 = np.array([-0.2 * R, 0, -0.2 * R])
warrow_y0 = np.array([0.1 * R, 0, -0.1 * R])
warrow_x, warrow_y = Rot2D(warrow_x0, warrow_y0, math.atan2(wyn[0], wxn[0]))
warrow = ax.plot(warrow_x + xn[0] + wxn[0], warrow_y + yn[0] + wyn[0], color="#bd0404")[0]
# центр окружности
circle_center = ax.plot(xn[0], R, marker=".", color="#ff924a")[0]
# всё про кривизну
curvature = (vxn[0] ** 2 + vyn[0] ** 2) / math.sqrt(wxn[0] ** 2 + wyn[0] ** 2)
k = curvature / math.sqrt(vxn[0] ** 2 + vyn[0] ** 2)
curvature_line = ax.plot([xn[0], xn[0] + vyn[0] * k], [yn[0], yn[0] - vyn[0] * k], linestyle="--", color="#157c9e", label="$\\rho$")[0]
curvature_center = ax.plot(xn[0] + vyn[0] * k, yn[0] - vyn[0] * k, marker='.', color="#157c9e")[0]
curvature_circle = plt.Circle((xn[0] + wxn[0] * k, yn[0] + wyn[0] * k), curvature, alpha=0.1)
ax.add_patch(curvature_circle)

a = FuncAnimation(fig, update, frames=len(tn), interval=10)

ax.legend()
plt.show()
