import numpy as np
from scipy import optimize
from scipy import interpolate
import matplotlib.pyplot as plt

import math

k = 0.6
p = 1

c = 2*k*k*k / (1 - k*k*p*p)
c_tilte = c / k

def Theta(theta_):
    return theta_ / k + p * math.log(((1+k*p) + (1-k*p)*math.exp(theta_)) / ((1-k*p) + (1+k*p)*math.exp(theta_)))

def u0(theta_):
    return 2*k*c*p*p / ((1+k*k*p*p) +(1-k*k*p*p)*np.cosh(theta_))

def theta(Theta_):
    return optimize.bisect(lambda x : Theta(x) - Theta_, -100., 100.)

def Theta_x(x, t=0, x0=0):
    return p * (x - c_tilte * t + x0)


x = np.linspace(-20., 20., num=10000)
u = []
for i in range(len(x)):
    u.append(u0(theta(Theta_x(x[i]))))

# interpolate
f_inter = interpolate.interp1d(x, u)

#extrapolate
def f_extra(x):
    if x <=20. and x >= -20.:
        return f_inter(x)
    else:
        return 0.

def u0_x(x, x0=20.):
    return f_extra(x - x0)

# L = 0.
def u(x, t, x0=20.):
    return u0_x(x - c_tilte * t, x0)

print(f_extra(-20), f_extra(-21))