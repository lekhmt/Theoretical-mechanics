import numpy as np
import sympy as sp
import math
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.patches import Circle
import mpl_toolkits.mplot3d.art3d as art3d

R = 5
k_phi = 0
k_theta = 0.4

t = sp.Symbol('t')
phi = k_phi * np.pi * t
theta = k_theta * np.pi * t
x = R * sp.sin(theta) * sp.cos(phi)
y = R * sp.sin(theta) * sp.sin(phi)
z = - R * sp.cos(theta)
vx = sp.diff(x, t)
vy = sp.diff(y, t)
vz = sp.diff(z, t)
wx = sp.diff(vx, t)
wy = sp.diff(vy, t)
wz = sp.diff(vz, t)

print("x(t) = ", x)
print("y(t) = ", y)
print("z(t) = ", z)
print("vx(t) = ", vx)
print("vy(t) = ", vy)
print("vz(t) = ", vz)
print("wx(t) = ", wx)
print("wy(t) = ", wy)
print("wz(t) = ", wz)

tn = np.linspace(0, 20, 500)
xn = np.zeros_like(tn)
yn = np.zeros_like(tn)
zn = np.zeros_like(tn)
vxn = np.zeros_like(tn)
vyn = np.zeros_like(tn)
vzn = np.zeros_like(tn)
wxn = np.zeros_like(tn)
wyn = np.zeros_like(tn)
wzn = np.zeros_like(tn)

for i in range(len(tn)):
    xn[i] = sp.Subs(x, t, tn[i])
    yn[i] = sp.Subs(y, t, tn[i])
    zn[i] = sp.Subs(z, t, tn[i])
    vxn[i] = sp.Subs(vx, t, tn[i])
    vyn[i] = sp.Subs(vy, t, tn[i])
    vzn[i] = sp.Subs(vz, t, tn[i])
    wxn[i] = sp.Subs(wx, t, tn[i])
    wyn[i] = sp.Subs(wy, t, tn[i])
    wzn[i] = sp.Subs(wz, t, tn[i])

fig = plt.figure()
ax = fig.add_subplot(projection="3d")
ax.set(xlim=[-8, 8], ylim=[-8, 8], zlim=[-8, 8])


def plot_vector(fig, orig, v, color='blue'):
    ax = fig.gca(projection='3d')
    orig = np.array(orig)
    v = np.array(v)
    ax.quiver(orig[0], orig[1], orig[2], v[0], v[1], v[2], color=color)
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.set_zlim(0, 10)
    ax = fig.gca(projection='3d')
    return fig


def rotation_matrix(d):
    sin_angle = np.linalg.norm(d)
    if sin_angle == 0: return np.identity(3)
    d /= sin_angle
    eye = np.eye(3)
    ddt = np.outer(d, d)
    skew = np.array([[0, d[2], -d[1]],
                     [-d[2], 0, d[0]],
                     [d[1], -d[0], 0]], dtype=np.float64)
    M = ddt + np.sqrt(1 - sin_angle ** 2) * (eye - ddt) + sin_angle * skew
    return M


def pathpatch_2d_to_3d(pathpatch, z, normal):
    if type(normal) is str:  # Translate strings to normal vectors
        index = "xyz".index(normal)
        normal = np.roll((1.0, 0, 0), index)

    normal /= np.linalg.norm(normal)  # Make sure the vector is normalised
    path = pathpatch.get_path()  # Get the path and the associated transform
    trans = pathpatch.get_patch_transform()

    path = trans.transform_path(path)  # Apply the transform

    pathpatch.__class__ = art3d.PathPatch3D  # Change the class
    pathpatch._code3d = path.codes  # Copy the codes
    pathpatch._facecolor3d = pathpatch.get_facecolor  # Get the face color

    verts = path.vertices  # Get the vertices in 2D

    d = np.cross(normal, (0, 0, 1))  # Obtain the rotation vector
    M = rotation_matrix(d)  # Get the rotation matrix

    pathpatch._segment3d = np.array([np.dot(M, (x, y, 0)) + (0, 0, z) for x, y in verts])


def pathpatch_translate(pathpatch, delta):
    pathpatch._segment3d += delta


def plot_plane(ax, point, normal, size=10, color='y'):
    p = Circle((0, 0), size, facecolor=color, alpha=.2)
    ax.add_patch(p)
    pathpatch_2d_to_3d(p, z=0, normal=normal)
    pathpatch_translate(p, (point[0], point[1], point[2]))


def update(i):
    point.set_data_3d(xn[i], yn[i], zn[i])
    velocity.set_data_3d(xn[i] + vxn[i], yn[i] + vyn[i], zn[i] + vzn[i])
    vv.set_data_3d([xn[i], xn[i] + vxn[i]], [yn[i], yn[i] + vyn[i]], [zn[i], zn[i] + vzn[i]])
    acceleration.set_data_3d(xn[i] + wxn[i], yn[i] + wyn[i], zn[i] + wzn[i])
    av.set_data_3d([xn[i], xn[i] + wxn[i]], [yn[i], yn[i] + wyn[i]], [zn[i], zn[i] + wzn[i]])
    global c
    c.remove()
    c = Circle((0, 0), R)
    c.set_alpha(0.4)
    ax.add_patch(c)
    pathpatch_2d_to_3d(c, z=0, normal=[yn[i], -xn[i], 0])
    return point


point = ax.plot(xn[0], yn[0], zn[0], marker=".", color="black")[0]
c = Circle((0, 0), R)
velocity = ax.plot(xn[0] + vxn[0], yn[0] + vyn[0], zn[0] + vzn[0], marker=".", color="blue")[0]
vv = ax.plot([xn[0], xn[0] + vxn[0]], [yn[0], yn[0] + vyn[0]], [zn[0], zn[0] + vzn[0]], color="blue")[0]
acceleration = ax.plot(xn[0] + wxn[0], yn[0] + wyn[0], zn[0] + wzn[0], marker=".", color="black")[0]
av = ax.plot([xn[0], xn[0] + wxn[0]], [yn[0], yn[0] + wyn[0]], [zn[0], zn[0] + wzn[0]], color="red")[0]
c.set_alpha(0.4)
ax.add_patch(c)
pathpatch_2d_to_3d(c, z=0, normal=[0, yn[0], 0])

a = FuncAnimation(fig, update, frames=len(tn), interval=10)
plt.show()
