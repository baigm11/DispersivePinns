import torch
import numpy as np
from scipy import optimize
from scipy import interpolate
import matplotlib.pyplot as plt

import math

k = 0.6
p1 = 1.5
p2 = 1.

c1 = 2 * k * k * k / (1 - k * k * p1 * p1)
c2 = 2 * k * k * k / (1 - k * k * p2 * p2)

w1 = - p1 * c1
w2 = - p2 * c2

A12 = ((p1 - p2) / (p1 + p2)) ** 2

a1 = 1 + k * p1
a2 = 1 + k * p2

b1 = 1 - k * p1
b2 = 1 - k * p2

v12 = 4 * k * k * k * (p1 - p2) * (p1 - p2) / ((1 - k * k * p1 * p1) * (1 - k * k * p2 * p2))

b12 = 8 * k ** 6 * (p1 - p2) ** 2 * (1 - k ** 4 * p1 * p1 * p2 * p2) / (
            (1 - k * k * p1 * p1) ** 2 * (1 - k * k * p2 * p2) ** 2)

alpha1_ = -2.


def theta1(y, t=0, alpha1=alpha1_):
    return p1 * (y - c1 * t - alpha1)


alpha2_ = 2.


def theta2(y, t=0, alpha2=alpha2_):
    return p2 * (y - c2 * t - alpha2)


def x_y(y, t=0, alpha=0):
    return y / k \
           + math.log((a1 * a2 + b1 * a2 * math.exp(theta1(y, t)) + b2 * a1 * math.exp(
        theta2(y, t)) + b1 * b2 * A12 * math.exp(theta1(y, t) + theta2(y, t)))
                      / ((b1 * b2 + a1 * b2 * math.exp(theta1(y, t)) + a2 * b1 * math.exp(
        theta2(y, t)) + a1 * a2 * A12 * math.exp(theta1(y, t) + theta2(y, t))))) \
           + alpha


def y_x(x, t=0, alpha=0):
    return optimize.bisect(lambda y: x_y(y, t) - x, -100., 100.)


def f_theta(theta1, theta2):
    return 1 + math.exp(theta1) + math.exp(theta2) + A12 * math.exp(theta1 + theta2)


def r_theta(theta1, theta2):
    return k + (2 / f_theta(theta1, theta2) ** 2) * (
                c1 * p1 * p1 * math.exp(theta1) + c2 * p2 * p2 * math.exp(theta2) + v12 * math.exp(theta1 + theta2) \
                + A12 * (c1 * p1 * p1 * math.exp(theta1 + 2 * theta2) + c2 * p2 * p2 * math.exp(2 * theta1 + theta2)))


def u_theta(theta1, theta2):
    return (2 / (k * r_theta(theta1, theta2) * f_theta(theta1, theta2) * f_theta(theta1, theta2))) \
           * (w1 * w1 * math.exp(theta1) + w2 * w2 * math.exp(theta2) + b12 * math.exp(theta1 + theta2) \
              + A12 * (w1 * w1 * math.exp(theta1 + 2 * theta2) + w2 * w2 * math.exp(2 * theta1 + theta2)))


def u_y(y, t=0):
    return u_theta(theta1(y, t), theta2(y, t))


def u_x(x, t=0):
    return u_y(y_x(x, t), t)









