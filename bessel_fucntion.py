#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct 28 15:21:47 2023

@author: anushikhajoshi
"""

import numpy as np
import matplotlib.pyplot as plt

def bessel(m, x, N=1000):
    a, b = 0, np.pi
    h = (b - a) / N
    
    integral = 0.0
    for i in range(N+1):
        t = a + i * h
        if i == 0 or i == N:
            weight = 1
        elif i % 2 == 1:
            weight = 4
        else:
            weight = 2
        integral += weight * np.cos(m*t - x*np.sin(t))
    
    integral *= h / 3.0
    integral /= np.pi
    
    return integral


xvalues = np.linspace(0, 10, 20)

#bessel function
j0 = [bessel(0, x) for x in xvalues]
j1 = [bessel(1, x) for x in xvalues]
j2 = [bessel(2, x) for x in xvalues]

#plotting
plt.figure(figsize=(10, 6))
plt.plot(xvalues, j0, label="J0")
plt.plot(xvalues, j1, label="J1")
plt.plot(xvalues, j2, label="J2")
plt.xlabel('x')
plt.ylabel('Jm(x)')
plt.legend()
plt.title('bessel functions')
plt.grid(True)
plt.show()


#using the import jv
from scipy.special import jv

j0scipy = jv(0, xvalues)
j1scipy = jv(1, xvalues)
j2scipy = jv(2, xvalues)

#plotting the difference
plt.figure(figsize=(12, 7))
plt.plot(xvalues, np.abs(j0-j0scipy), label="|J0 - J0scipy|")
plt.plot(xvalues, np.abs(j1-j1scipy), label="|J1 - J1scipy|")
plt.plot(xvalues, np.abs(j2-j2scipy), label="|J2 - J2scipy|")
plt.xlabel('x')
plt.ylabel('Difference')
plt.legend()
plt.title('bessel functions Vs. scipy.special.jv')
plt.grid(True)
plt.show()


#This folllowing program then density plot of the intensity of the circular diffraction
#pattern of a point light source with λ = 500nm, in a square region of the focal
#plane of a telescope. The equation is in the form of a bessel function.

lambda1 = 500e-9 
k = 2 * np.pi / lambda1

rvalues = np.linspace(0, 1e-6, 400) 
X, Y = np.meshgrid(rvalues, rvalues)

#radial coordinates
R = np.sqrt(X**2 + Y**2)

#calculating intensity
numerator = jv(1, k * R)
denominator = k * R

# wherever the denominator is zero or the numerator is close enough to zero, 
# we set the intensity value to 0.5. Otherwise, compute the formula.
condition = (denominator != 0) & (np.abs(numerator) > 1e-10)
I = np.where(condition, (numerator / denominator)**2, 0.5)

plt.figure(figsize=(12, 7))
plt.imshow(I, vmax=0.01, extent=(0, 1e-6, 0, 1e-6), origin='upper')
plt.colorbar()
plt.title('density plot, I of the circular diffraction pattern')
plt.xlabel('r (m)')
plt.ylabel('r (m)')
plt.show()