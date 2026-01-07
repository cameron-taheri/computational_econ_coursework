#========================================================
# import packages
#========================================================
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# header
print('Cameron Taheri')
print('Problem Set 4')

print("-" * 70)

# import maim function
sys.path.append('Users/camerontaheri/python/comp_econ/PS5/maim.py')
import maim

# set print formatting
np.set_printoptions(suppress=True,linewidth=200)

# set verbosity and maxlags
verbosity = 0
maxlag = 2

#========================================================
# define structural coefficient matrix (h)
#========================================================
# indexing
# one(-2) i(-2) pi(-2) rho(-2) y(-2) .... one(1) i(1) pi(1) rho(1) y(1)

h = np.array ([
[0, 0,	0,	0,	0,	-1,	0,	0,	0,	0,	1,	0,	0,	0,	0,	0,	0,	0,	0,	0],
[0,	0,	0,	0,	0,	0,	0,	-0.5,	0,	-0.5,	-0.011,	1,	-1,	0,	0,	0,	0,	0,	0,	0],
[0,	0,	0,	0,	0,	0,	0,	-0.5,	0,	0,	0,	0,	1,	0,	-0.1,	0,	0,	-0.5,	0,	0],
[0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	-1,	0,	48.619048,	0,	0,	0,	1,	-47.619048,	0],
[0,	0,	0,	0,	0.415,	0,	0,	0,	0.798,	-1.254,	-0.016758,	0,	0,	0,	1,	0,	0,	0,	0,	0]
])

#========================================================
# compute backward-looking reduced form matrix (b)
#========================================================
returncode,nshifts,nbigroots,roots,b,s = maim.maim(h,maxlag,verbosity=0)

print("-- Checking Output --")
if returncode != 0:
    print("Error: return code is not 0.")
else:
    print(f"Return Code: {returncode}")

if nshifts != 3:
    print("Error: nshifts is not 3.")
else:
    print(f"Num Shifts: {nshifts}")

if nbigroots != 2:
    print("Error: Num eigenvalues greater than 1 is not 2.")
else:
    print(f"Num eigenvalues (nbigroots): {nbigroots}")

if np.shape(b) != (5,10):
    print("Error: Backward-looking reduced form is not 5 by 10")
else:
    print(f"Backward-looking reduced form dimension: {np.shape(b)}")

print("\n-- Backward-Looking Reduced Form Matrix -- ")
print(b)

#========================================================
# simulate model
#========================================================
# steady state values
one_ss  = 1.0
i_ss    = 0.041
pi_ss   = 0.02
rho_ss  = 0.021
y_ss    = 0.0

# real interest rate (rho) shock
shock = 0.01
rho_shock = rho_ss + shock

# create 10 x 1 vector (array) of steady state variables (5 variables for 2 periods)
X_ss = np.array([
    [one_ss],             # one(-2)
    [i_ss],               # i(-2)
    [pi_ss],              # pi(-2)
    [rho_ss],             # rho(-2)
    [y_ss],               # y(-2)
    [one_ss],             # one(-1)
    [i_ss],               # i(-1)
    [pi_ss],              # pi(-1)
    [rho_shock],          # rho(-1)
    [y_ss]                # y(-1)
    
])

# simulate 20 periods forward and store results in a T + 2 x 5 matrix
T = 20
sim_output = np.zeros((T + 2, 5))    # adding 2 accounts for the two periods of historical data

# initialize state vector with steady states and shock to rho (assume to be historical observations)
X_historical = X_ss

# add historical values to the output matrix
sim_output[0, :] = X_historical[:5, :].T    # first row will be historical values for period -2
sim_output[1, :] = X_historical[5:, :].T    # second row will be historical values for period -1

# iterate over 20 periods
for t in range(0,T):
    # compute the 5 macro variables; variable one will always equal 1
    #   while the other 4 variables (i, pi, rho, y) will update 
    #   at X_t+1 using the backward-looking reduced form (b) and X_t
    
    X_current = b @ X_historical
    
    # for each row in simulated output matrix (T x 5) store values from X_current
    sim_output[t+2, :] = X_current.T
    
    # shift the t-1 values into t-2
    X_historical[0:5, :] = X_historical[5:, :]
    
    # update historical vector with new t-1 values
    X_historical[5:, :] = X_current

print(sim_output)

#========================================================
# create dataframe (DF)
#========================================================
# set an index for the DF
time_index = np.arange(0, T + 2)

# create the DF and label columns
df_sim_output = pd.DataFrame(
    sim_output[:,1:],
    index = time_index,
    columns= [
        'ST Nominal Interest Rate',
        'Inflation Rate',
        'LT Real Interest Rate',
        'Output Gap'
        ]
    )

# check the DF
print(df_sim_output.head())

# set up a figure with 4 subplots arranged vertically
fig, axes = plt.subplots(
    nrows=2, 
    ncols=2, 
    figsize=(8, 8), 
    sharex=True
)

# flatten the 2 x 2 axes array into a one-dimensional array and create an index variable
axes = axes.flatten()
time = df_sim_output.index

# loop through the columns and plot each one in its own subplot
for i, col in enumerate(df_sim_output.columns):
    # set the axis equal to the current column being plotted
    ax = axes[i]
    
    # plot the column against the time index
    ax.plot(time, df_sim_output[col], 
            linewidth= 2, 
            color = 'red',
            label=col
            )

    ax.set_title(col)
    
    ax.grid(True)
    
# Adjust layout to prevent overlap
plt.tight_layout()
plt.show()