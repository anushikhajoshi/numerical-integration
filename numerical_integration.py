#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct 28 01:45:36 2023

@author: anushikhajoshi
"""
#Evaluating the Dawson fucntion D(x) at a given point (x=4), comparing the value obtained when using the Trapezoidal vs. Simpson’s rule using N = 8 slices
import numpy as np
from scipy.special import dawsn
import timeit

#defining the dawson function
def f(t):
    return np.exp(t**2)

#trapezoidal rule
def trapezoidal_rule(a, b, N):
    h = (b - a) / N
    x = np.linspace(a, b, N+1)
    y = f(x)
    integral = h * (0.5*y[0] + 0.5*y[-1] + np.sum(y[1:-1]))
    return integral

#simpson's rule
def simpsons_rule(a, b, N):
    h = (b - a) / N
    x = np.linspace(a, b, N+1)
    y = f(x)
    integral = h/3 * (y[0] + y[-1] + 4*np.sum(y[1:-1:2]) + 2*np.sum(y[2:-2:2]))
    return integral

def main():
    a = 0
    b = 4
    N = 8

    #calculating the integral
    trapezoidal = trapezoidal_rule(a, b, N) * np.exp(-b**2)
    simpson = simpsons_rule(a, b, N) * np.exp(-b**2)
    true_value = dawsn(4) #trating scipy value as the true value
    
    print(f"Trapezoidal Rule Result (N=8): {trapezoidal}")
    print(f"Simpson's Rule Result (N=8): {simpson}")
    print(f"True Value from scipy: {true_value}")

    # this part is aimed towards answering the question of "how many slices do we need to approximate the integral with an error of O(10−9)"
    # no. of slices
    accuracy = 1e-9
    N = 2

    while abs(trapezoidal_rule(a, b, N) * np.exp(-b**2) - true_value) > accuracy:
        N *= 2
    print(f"no of required slices for Trapezoidal Rule: {N}")

    N = 2
    while abs(simpsons_rule(a, b, N) * np.exp(-b**2) - true_value) > accuracy:
        N *= 2
    print(f"no of required slices for simpson's Rule: {N}")
    
    #timing the computation
    number_of_runs = 1000
    trap_time = timeit.timeit(lambda: trapezoidal_rule(a, b, N), number=number_of_runs) / number_of_runs
    simp_time = timeit.timeit(lambda: simpsons_rule(a, b, N), number=number_of_runs) / number_of_runs
    
    print(f"time taken for trapezoidal Rule (average over {number_of_runs} runs): {trap_time} s")
    print(f"time taken for simpson's Rule (average over {number_of_runs} runs): {simp_time} s")
    print(f"Difference: {trap_time - simp_time} s")

if __name__ == '__main__':
    main()

#practical estimation of errors

a = 0
b = 4

#trapezoidal rule

## N1 = 32
t_i1 = trapezoidal_rule(a, b, 32) * np.exp(-b**2)

## N2 = 64
t_i2 = trapezoidal_rule(a, b, 64) * np.exp(-b**2)

## Error

eps = (1/3)*(t_i2 - t_i1)

print(f"Practical estimation of errors of N2 = 64 (N1 = 32) using trapezoidal rule : {eps}")

# Simpson rule

## N1 = 32 
s_i1 = simpsons_rule(a, b, 32) * np.exp(-b**2)

## N2 = 64
s_i2 = simpsons_rule(a, b, 64) * np.exp(-b**2)

eps = (1/15)*(s_i2 - s_i1)

print(f"Practical estimation of errors of N2 = 64 (N1 = 32) using simpsons rule : {eps}")




