
#!/usr/bin/python3

# import necessary packages
import numpy as np
from numpy.linalg import eig, norm, qr
from numpy.random import rand
from numpy import argsort, diag, eye, iscomplex, set_printoptions
from sys import exit


# header
print('Cameron Taheri')
print('Problem Set 3')

# set print formatting
printprec = 4
set_printoptions(precision = printprec, linewidth = 150, floatmode = 'fixed', sign=' ')

# define simultaneous iteration function
def simit(A, Q, R, tol = 1e-8, maxit = 100, verbose = False):
    if(verbose):
        print(' iter                     est eigenvalues                     estroc')

    for it in range(maxit + 1):
        AQ = A @ Q
        nerr = norm(AQ - Q @ R, ord = 2)

        if(verbose):
            if(it > 0):
                estroc = nerr / lastnerr
                print('{:4d} {:9.5f}  {:9.5f}  {:9.5f}  {:9.5f} {:9.5f}   {:9.5f}'.format(it, R[0,0], R[1,1], R[2,2], R[3,3], R[4,4],  estroc))
            else:
                print('{:4d} {:9.5f}  {:9.5f}  {:9.5f}  {:9.5f} {:9.5f}'.format(it, R[0,0], R[1,1], R[2,2], R[3,3], R[4,4]))
                
        Q,R = qr(AQ)

        lastnerr = nerr

        if(nerr <= tol):
            if(verbose):
                print('Converged at iteration {}'.format(it))
            return (it, Q, R)

    print('Solution not found after {} iterations.'.format(maxit))
    return (maxit, None, None)


# define Q_0 and R_0 as identity matrices
Q0, R0 = eye(5,5), eye(5,5)

print("Matrix Q0:")
print(Q0)

print("\nMatrix R0:")
print(R0)


# set iteration parameters
verbose = True
maxit = 200
tol = 1e-8


# repeatedly generate nonsymmetric 5x5 matrix N until all eigenvalues are real
# use rand when generating N --> use np.eig() to compute N's eigenvalues --> check if eigenvalues are real --> repeat until all eigenvalues are real

for it in range(maxit):
    N = rand(5,5) + 0.5 * eye(5,5)
    np_evals, np_evecs = np.linalg.eig(N)   # extract eigenvalues and eigenvectors
    
    check_complex = np.issubdtype(np_evals.dtype, np.complexfloating)   # create a boolean variable that checks if eigenvalues are complex
    
    if check_complex != True:
        print(f"\nAfter {it + 1} iteration(s), the below square, nonsymmetric matrix...\n {N} \n \n has real eigenvalues...\n {np_evals}.")
        break
    


# sort the eigenvalues generated from numpy's eig command in order of greatest to least modulus
sorted_indices = np.argsort(np.abs(np_evals))[::-1]
sorted_np_evals = np_evals[sorted_indices]

# check
print(sorted_np_evals)


# compute eigenvalue ratios with slicing
# numerator will be values with indices     1 2 3 4
# denomenator will be values with indices   0 1 2 3
numerator = sorted_np_evals[1:]
denomenator = sorted_np_evals[:-1]

# store the ratios in an array variable
np_eval_ratios = numerator / denomenator
np_ROC = np.abs(np_eval_ratios[np.argmax(abs(np_eval_ratios))]) # store the ROC in a variable

print(np_eval_ratios)
print(f"\nRate of convergence (largest of eval_ratios) is {np.round(np_ROC,4)}.")


# pass the nonsymmetric, real matrix with real eigenvalues and initial guesses for Q and R through the simit function
it, Q, R = simit(N, Q0, R0, tol, maxit, verbose)

# extract eigenvalues from the diagonal of matrix R (from QR Decomposition)
simit_evals = np.diag(R)

# show the numpy and simit eigenvalues
print(sorted_np_evals)
print(simit_evals)


# compare the numpy and simit eigenvalues
norm_diff = np.linalg.norm(np.abs(sorted_np_evals) - np.abs(simit_evals))
print(norm_diff)


# using eigendecomp in reverse to construct a symmetric matrix S

# S = Q @ R @ Q_inv

nice_evals = np.array([5, 4, 3, 2, 1])

# create a diagonal matrix (symmetric) with the desired eigenvalues (diagonals are symmetric by definition; S = S.T)



############ BFM ##################################################################################################
# S = np.diag(nice_evals)

# The S matrix that you have generated is symmetric but not
# random.  That is the reason that simit converges almost immediately.

randmat = rand(5,5)
Q,_ = qr(randmat)
D = np.diag(nice_evals)
S = Q @ D @ Q.T
###################################################################################################################

print(f"Matrix S:\n{S}")    # check it is symmetric

# perform a QR Decomposition on S

#########  BFM #####################################################################################################
# This is unnecessary and has no effect.  The Q, R that are generated are not used below
Q, R = qr(S)
###################################################################################################################

# extract eigenvalues and eigenvectors
S_evals, S_evecs = np.linalg.eig(S)

# sort eigenvalues of S
sorted_indices = np.argsort(np.abs(S_evals))[::-1]
sorted_S_evals = S_evals[sorted_indices]
print(f"\nSorted eigenvalues of matrix S from numpy eig command:\n{sorted_S_evals}")   # check they are 54321

# compute eigenvalue ratios with slicing
numerator = sorted_S_evals[1:]
denomenator = sorted_S_evals[:-1]

# store the ratios in an array variable
S_eval_ratios = numerator / denomenator
S_ROC = np.abs(S_eval_ratios[np.argmax(abs(S_eval_ratios))]) # store the ROC in a variable

print(f"\nComputed eigenvalue ratios:\n{S_eval_ratios}")
print(f"\nRate of convergence (largest of eval_ratios) is {np.round(S_ROC,4)}.")

print("\nBegin Simultaneous Iteration on S using Q0 and R0 (defined as identity matrices)")
print("-" * 100)
it, Q, R = simit(S, Q0, R0, tol, maxit, verbose)
print("-" * 100)

# extract eigenvalues from the simit function
simit_evals = np.diag(R)

print(f"\nSimit yields sorted eigenvalues of:\n{simit_evals}")
print(f"\nNumpy yields sorted eigenvalues of:\n{sorted_S_evals}")

# compute norm diff of eigenvalues from numpy (reverse QR decomp) and simit
norm_diff = np.linalg.norm(np.abs(sorted_S_evals) - np.abs(simit_evals))
print(f"\nNorm Diff between numpy evals and simit evals for matrix S:{norm_diff}")

###  BFM  ##########################################################################################
# print(f"\nMatrix Q from reverse eigendecomp seems to (numerically) be the identity matrix:\n{Q}")
# That is because the S you generated was diagonal.
####################################################################################################

print(f"\nMatrix R is a diagonal matrix with the desired eigenvalues:\n{R}")


