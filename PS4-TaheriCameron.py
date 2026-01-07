from numpy.linalg import det,norm,qr,svd
from numpy import array,diag,dot,eye,pi,sin,cos,set_printoptions,sign,sqrt
from numpy.random import rand
import matplotlib.pyplot as plt
import numpy as np

# header
print('Cameron Taheri')
print('Problem Set 4')

print("-" * 70)
print("--- Part 1 ---")
print("-" * 70)

# set print formatting
printprec = 4
set_printoptions(precision = printprec, linewidth = 150, floatmode = 'fixed', sign=' ')

# define simultaneous iteration function
def simit(A, Q, R, tol = 1e-8, maxit = 100, verbose = False):
    if(verbose):
        print(' iter                     est eigenvalues                     estroc')

    for it in range(maxit + 1):
        AQ = A @ Q
        nerr = norm(AQ - Q @ R, ord=2)

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

# simitsvd computes SVD of a matrix
# signature: itV, U, D, V = simitsvd(A,Q,R,tol=1e-8,maxit=100,verbose=False)
# where itV --> # of iterations to get V matrix

def simitsvd(A, Q, R, tol=1e-8, maxit=100, verbose=False):
    
    print("--- Beginning outer simitsvd ---")
        
    S1 = A @ A.T
    
    print("\n--- Beginning nested simit for U (AA\')---")
    itU, U, R1 = simit(S1, Q, R, tol, maxit, verbose)
    
    print("\n--- Beginning nested simit for V (A\'A)---")
    S2 = A.T @ A
    itV, V, R2 = simit(S2, Q, R, tol, maxit, verbose)
    
    D = sqrt(diag(R1))
    
    if U is None or V is None:
        print("Solution not found.\n")
        return (maxit, None, None, None)
    else:
        return (itV, U, D, V)    

# generate a random, nonsymmetric 5x5 matrix A using SVD decomp in reverse
A_U = rand(5,5) + 0.5 * eye(5,5)
U, R_U = qr(A_U)

A_V = rand(5,5) + 0.5 * eye(5,5)
V, R_V = qr(A_V)

# set singular values
singular_vals = [5, 4, 3, 2, 1]
D = diag(singular_vals)

# use SDV decomp to generate A
A = U @ D @ V.T

print(f"Nonsymmetric 5x5 matrix A, derived from SVD decomp in reverse:\n{A}")
print("-" * 70)

# use simitsvd to compute the SVD of A
tol = 1e-8; maxit = 100; verbose = True # set parameters

# initialize R and Q for simit nested in simitsvd
R0 = eye(5,5)
randmat = rand(5,5)+0.5 * eye(5,5)
Q0,_ = qr(randmat)

# run simitsvd on random, nonsymmetric 5x5 matrix A
itV, U, D, V = simitsvd(A, Q0, R0, tol, maxit, verbose)

# use numpy svd to compute u, d, and v
u, d, v = svd(A)
d = array(d)        # convert matrix D to an array (vector)
v = v.T             # numpy returns V', so we must transpose V' to get V

# compute norm diffs for U and V
norm_diff_U = norm(abs(U) - abs(u))
norm_diff_V = norm(abs(V) - abs(v))
norm_diff_D = norm(abs(D) - abs(d))

print("\n--- Comparison of simitsvd and numpy ---")

# compare U matrices
print(f"\nMatrix U (from simitsvd):\n{U}")
print(f"\nMatrix u (from numpy):\n{u}")
print(f"\nNorm diff for U matrices:\n{norm_diff_U}")

print("-" * 20)

# compare V matrices
print(f"\nMatrix V (from simitsvd):\n{V}")
print(f"\nMatrix v (from numpy):\n{v}")
print(f"\nNorm diff for V matrices:\n{norm_diff_V}")

print("-" * 20)

# compare D matrices
print(f"\nMatrix D (from simitsvd):\n{D}")
print(f"\nMatrix d (from numpy):\n{d}")
print(f"\nNorm diff for D matrices:\n{norm_diff_D}")

print("-" * 70)
print("--- Part 2 ---")
print("-" * 70)

# generate a random, nonsymmetric 5x5 matrix A using SVD decomp in reverse
A_U = rand(2,2) + 0.5 * eye(2,2)
U, R_U = qr(A_U)

A_V = rand(2,2) + 0.5 * eye(2,2)
V, R_V = qr(A_V)

# set singular values
singular_vals = [2, 0.5]
D = diag(singular_vals)

# use SDV decomp to generate A
A = U @ D @ V.T

# flip the determinant by flipping sign of one column of V (or U, but I chose V here)
V_negative = V.copy()
V_negative[:, 1] = -V_negative[:, 1]
A_negative = U @ D @ V_negative.T

if det(A) > 0:
    A_positive = A
else:
    A_negative = A
    A_positive = A_negative

# store matrices in a list
A_matrices = {
    "A (Positive Det)": A_positive, 
    "A (Negative Det)": A_negative
    }

print(f"\nNonsymmetric 2x2 matrix A, derived from SVD decomp in reverse:\n{A}")
print(f"\nNonsymmetric 2x2 matrix A_negative, derived from SVD decomp in reverse:\n{A_negative}")

# create a unit circle - I'll use sin and cos to do this - and map to ellipse
theta = np.linspace(0, 2 * np.pi, 200)
x = cos(theta)
y = sin(theta)
circle = array([x,y])   # convert unit circle array into a matrix

for name, matrix in A_matrices.items():
    
    plt.figure(figsize=(12, 6))
    
    # update U and V
    U, D, V_T = svd(matrix)
    D = array(D)
    V = V.T
    
    ellipse = matrix @ circle    # mapping to ellipse

    # plot the circle (orange) and ellipse (green)
    plt.plot(circle[0, :], circle[1, :], 
             color='orange', label='Unit Circle (Pre-image)'
             )
    plt.plot(ellipse[0, :], ellipse[1, :], 
             color='green', label='Ellipse (Image)'
             )

    # draw arrow from 0,1 and from 1,0
    start_x1, start_y1 = 0, 1
    end_x1, end_y1 = matrix[:, 1]
    diff_x1 = end_x1 - start_x1; diff_y1 = end_y1 - start_y1

    start_x2, start_y2 = 1, 0
    end_x2, end_y2 = matrix[:, 0]
    diff_x2 = end_x2 - start_x2; diff_y2 = end_y2 - start_y2

    # use matplotlib arrow
    plt.arrow(start_x1, start_y1, diff_x1, diff_y1,     # plot 0,1
            color='black', linewidth=0.8, 
            head_width=0.03, head_length=0.03, 
            length_includes_head=True
            )

    plt.arrow(start_x2, start_y2, diff_x2, diff_y2,     # plot 1,0
            color='black', linewidth=0.8, 
            head_width=0.03, head_length=0.03, 
            length_includes_head=True
            )

    # major and minor axes of the ellipse (image)
    ellip_major_direction = U[:, 0]   # grab the first column of U
    ellip_minor_direction = U[:, 1]   # grab the second column of U

    major_length = D[0]      # grab the first entry of D
    minor_length = D[1]      # grab the second entry of D

    # compute end point for major
    ellip_major_end = major_length * ellip_major_direction
    ellip_major_x = [-ellip_major_end[0], ellip_major_end[0]]
    ellip_major_y = [-ellip_major_end[1], ellip_major_end[1]]

    # compute end point for minor
    ellip_minor_end = minor_length * ellip_minor_direction
    ellip_minor_x = [-ellip_minor_end[0], ellip_minor_end[0]]
    ellip_minor_y = [-ellip_minor_end[1], ellip_minor_end[1]]

    # plot major and minor
    plt.plot(ellip_major_x, ellip_major_y, 
             color='red', linestyle='-', 
             linewidth=1.5, label='Ellipse Major Axis'
             )
   
    plt.plot(ellip_minor_x, ellip_minor_y, 
             color='blue', linestyle='-', 
             linewidth=1.5, label='Ellipse Minor Axis'
             )

    # major and minor axes of the circel (preimage)
    circ_major_end = V[:, 0]   # grab the first column of U
    circ_minor_end = V[:, 1]   # grab the second column of U

    circ_major_x = [-circ_major_end[0], circ_major_end[0]]
    circ_major_y = [-circ_major_end[1], circ_major_end[1]]

    circ_minor_x = [-circ_minor_end[0], circ_minor_end[0]]
    circ_minor_y = [-circ_minor_end[1], circ_minor_end[1]]

    # plot major and minor
    plt.plot(circ_major_x, circ_major_y, 
             color='red', linestyle='dashed', 
             linewidth=1.5, label='Preimage Major Axis'
             )
    plt.plot(circ_minor_x, circ_minor_y, 
             color='blue', linestyle='dashed', 
             linewidth=1.5, label='Preimage Minor Axis'
             )

    # add x- and y-axis through the origin
    plt.axhline(y=0, color='gray', linestyle='--', linewidth=0.5)
    plt.axvline(x=0, color='gray', linestyle='--', linewidth=0.5)
    
    # make it a square plot
    plt.xlim(-2.5, 2.5); plt.ylim(-2.5, 2.5)
    plt.axis('equal')               # make sure plotting is uniform
    plt.legend(loc='upper left')    # add a legend
    
    plt.title(f"Showing Plot of matrix {name}")
    
    plt.show()