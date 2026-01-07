#-----------------------------------------------------------------
# PS1: Fixed-Point, Bisection, and Newton-Raphson (Newton) Root Finding Algorithms
#-----------------------------------------------------------------

# import packages
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from math import exp, cos, sin, acos, log, sqrt, asin

#-----------------------------------------------------------------
# Define functions: f1, f2, and f3
#-----------------------------------------------------------------
def f1(x):
    f1val = 3 * cos(x) - sin(3 * x)
    return f1val
    
def f2(x):
    f2val = sin(x) ** 3 - 4 * cos(2 * x)
    return f2val

def f3(x):
    f3val = (cos(x ** 2)) ** 2 - (sin(x / 2))
    return f3val

#-----------------------------------------------------------------
# Plot each function
#-----------------------------------------------------------------
# Function 1
print("Plot of Function f1(x) = 3 * cos(x) - sin(3 * x).")

# Define the gridspace and function
x = np.linspace(-10,10)
y = 3 * np.cos(x) - np.sin(3 * x)

# Show the function crossing the x-axis explicitly
plt.axhline(y=0, color = 'gray', linestyle = '--')

# Plot the function
plt.plot(x,y)
plt.show()


# Function 2
print("Plot of Function f2(x) = (sin(x) ** 3) - 4 * cos(2 * x).")
x = np.linspace(-10,10)
y = np.sin(x) ** 3 - 4 * np.cos(2 * x)

# Show the function crossing the x-axis explicitly
plt.axhline(y=0, color = 'gray', linestyle = '--')

# Plot the function
plt.plot(x,y)
plt.show()


# Function 3
print("Plot of Function f3(x) = (cos(x ** 2)) ** 2 - (sin(x/2)).")
x = np.linspace(-10,10)
y = (np.cos(x ** 2)) ** 2 - (np.sin(x / 2))

# Show the function crossing the x-axis explicitly
plt.axhline(y=0, color = 'gray', linestyle = '--')

# Plot the function
plt.plot(x,y)
plt.show()

#-----------------------------------------------------------------
# Auxiliary functions for fixed-point iteration
#-----------------------------------------------------------------
def g1(x):
    g1val = acos(sin(3 * x) / 3)
    return g1val

def g2(x):
    g2val = (acos( (1/4) * (sin(x) ** 3) )) / 2
    return g2val

def g3(x):
    g3val = (acos( (sin(x / 2)) ** (1/2) )) ** (1/2)
    return g3val

#-----------------------------------------------------------------
# First derivatives of f1, f2, and f3 for Newton method    
#-----------------------------------------------------------------
def df1dx(x):
    df1dxval = - 3 * sin(x) - 3 * cos(3 * x)
    return df1dxval
    
def df2dx(x):
    df2dxval = 3 * sin(x) * cos(x) + 8 * sin(2 * x)
    return df2dxval
    
def df3dx(x):
    df3dxval = - 4 * x * sin(x ** 2) * cos(x ** 2) - (1/2) * cos(x / 2)
    return df3dxval

#-----------------------------------------------------------------
# Define functions implementing the three algorithms.
#-----------------------------------------------------------------
# Fixed point iteration.
# g is auxiliary function for f    
# x is starting value
def fixedpoint(g,x,tol,maxit,verbose=False):
    if verbose:
        print("Iteration  Est. Root")
        print("{:4d}     {:12.8f} ".format(0, x))
            
    for it in range(1,maxit):
        gx = g(x)
        if verbose:
            print("{:4d}     {:12.8f} ".format(it, gx))
        if abs(x-gx)<=tol:
            return gx
        x=gx
    print("Maximum iterations exhausted.")
    return None
        
# Bisection method
# a,b are starting values such that a<x<b
# where b is a root.
def bisection(f,a,b,tol,maxit,verbose=False):
    if(a >= b):
        print("Error: a must be less than b.")
        return None
    if verbose:
        print("Iteration  Est. Root")
    for it in range(maxit+1):
        x=(a+b)/2
        if verbose:
            print("{:4d}     {:12.8f} ".format(it, x))
        if abs(f(x)) <= tol:
            return x
        if f(a)*f(x)<0:
            b = x
        else:
            a = x
        lastx = x
    print("Maximum iterations exhausted.")
    return None
        
# Newton iteration
# df is the first derivative function
# x is the starting value
def newton(f,df,x,tol,maxit,verbose=False):
    if verbose:
        print("Iteration  Est. Root")
    for it in range(maxit+1):
        if verbose:
            print("{:4d}     {:12.8f} ".format(it, x))
        if abs(f(x)) <= tol:
            return x
        lastx = x
        x = x - f(x)/df(x)
    print("Maximum iterations exhausted.")
    return None

# End of function definitions    


#-----------------------------------------------------------------
# Set Parameters
#-----------------------------------------------------------------
tol=1e-6
maxit=100
verbose=True

#-----------------------------------------------------------------
# EXECUTION FOR f1
#-----------------------------------------------------------------
# Starting starting values for f1
x01=2
a01=1
b01=2

print("\nFixed Point Method")
print("Seeking root of f1(x) = 3 * cos(x) - sin(3 * x).")
print("Starting value: x01 = ",x01,"")
rootf1_fixedpoint = fixedpoint(g1,x01,tol,maxit,verbose)
if rootf1_fixedpoint:
    print("Computed root of f1 is {:8.6f}".format(rootf1_fixedpoint))
else:
    print("Fixed point for f1 failed to converge.")

print("\nBisection method")
print("Seeking root of f1(x) = 3 * cos(x) - sin(3 * x).")
print("Starting values: a01 =", a01,", b01 =",b01, "")
rootf1_bisection = bisection(f1,a01,b01,tol,maxit,verbose)
if rootf1_bisection:
    print("Computed root of f1 is {:8.6f}".format(rootf1_bisection))
else:
    print("Bisection for f1 failed to converge.")

print("\nNewton's method")
print("Seeking root of f1(x) = 3 * cos(x) - sin(3 * x).")
print("Starting value: x01 = ", x01, "")
rootf1_newton = newton(f1,df1dx,x01,tol,maxit,verbose)
if rootf1_newton:
    print("Computed root of f1 is {:8.6f}".format(rootf1_newton))
else:
    print("Newton for f1 failed to converge.")
# END OF EXECUTION FOR f1

#-----------------------------------------------------------------
# EXECUTION FOR f2
#-----------------------------------------------------------------
# Starting starting values for f2
x01=1
a01=0.5
b01=1

print("\nFixed Point Method")
print("Seeking root of f2(x) = sin(x) ** 3 - 4 * cos(2 * x).")
print("Starting value: x01 = ",x01,"")
rootf2_fixedpoint = fixedpoint(g2,x01,tol,maxit,verbose)
if rootf2_fixedpoint:
    print("Computed root of f2 is {:8.6f}".format(rootf2_fixedpoint))
else:
    print("Fixed point for f2 failed to converge.")

print("\nBisection method")
print("Seeking root of f2(x) = sin(x) ** 3 - 4 * cos(2 * x).")
print("Starting values: a01 =", a01,", b01 =",b01, "")
rootf2_bisection = bisection(f2,a01,b01,tol,maxit,verbose)
if rootf2_bisection:
    print("Computed root of f2 is {:8.6f}".format(rootf2_bisection))
else:
    print("Bisection for f2 failed to converge.")

print("\nNewton's method")
print("Seeking root of f2(x) = sin(x) ** 3 - 4 * cos(2 * x).")
print("Starting value: x01 = ", x01, "")
rootf2_newton = newton(f2,df2dx,x01,tol,maxit,verbose)
if rootf2_newton:
    print("Computed root of f2 is {:8.6f}".format(rootf2_newton))
else:
    print("Newton for f2 failed to converge.")
# END OF EXECUTION FOR f2

#-----------------------------------------------------------------
# EXECUTION FOR f3
#-----------------------------------------------------------------
# Starting starting values for f3
x01=1
a01=0.1
b01=1

print("\nFixed Point Method")
print("Seeking root of f3(x) = (cos(x ** 2)) ** 2 - (sin(x / 2)).")
print("Starting value: x01 = ",x01,"")
rootf3_fixedpoint = fixedpoint(g3,x01,tol,maxit,verbose)
if rootf3_fixedpoint:
    print("Computed root of f3 is {:8.6f}".format(rootf3_fixedpoint))
else:
    print("Fixed point for f3 failed to converge.")

print("\nBisection method")
print("Seeking root of f3(x) = (cos(x ** 2)) ** 2 - (sin(x / 2)).")
print("Starting values: a01 =", a01,", b01 =",b01, "")
rootf3_bisection = bisection(f3,a01,b01,tol,maxit,verbose)
if rootf3_bisection:
    print("Computed root of f3 is {:8.6f}".format(rootf3_bisection))
else:
    print("Bisection for f3 failed to converge.")

print("\nNewton's method")
print("Seeking root of f3(x) = (cos(x ** 2)) ** 2 - (sin(x / 2)).")
print("Starting value: x01 = ", x01, "")
rootf3_newton = newton(f3,df3dx,x01,tol,maxit,verbose)
if rootf3_newton:
    print("Computed root of f3 is {:8.6f}".format(rootf3_newton))
else:
    print("Newton for f3 failed to converge.")
# END OF EXECUTION FOR f3


