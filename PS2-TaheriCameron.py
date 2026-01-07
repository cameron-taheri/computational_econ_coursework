# import necessary packages
import numpy as np

# set print formatting
printprec = 4
np.set_printoptions(precision=printprec,linewidth=150,floatmode='fixed',sign=' ')

# header
print("Cameron Taheri")
print("Problem Set 2")

# -----------------------------------------------------------------
# Functions
# -----------------------------------------------------------------
def powerit(A, alpha0, x0, tol=1e-4, maxit=100, verbose=True):
    x = x0
    alpha = alpha0
    
    if verbose:
        print("Iter   Dom Eigenval                 Dom Eigenvect")
    
    for it in range(maxit):
        u = x / np.linalg.norm(x)      # normalize vector (Euclidean) to ensure calculations are not too large
        
        if verbose:
            print('{:4d}  {:10.4f}   {} '.format(it,  alpha, u.T))
        
        x = A @ u
        
        if np.linalg.norm(x - alpha * u) <= tol:
            return (it, alpha, u)

        # For pedagogical purposes, using np.vdot to obtain the (complex) dot product
        #   (although dominant eigenvalue for real matrix can never be complex)
        alpha = np.vdot(u,x)
        
    print('Power iteration did not converge after {} iterations.'.format(it))
    return maxit,None,None

# define a function that creates a random symmetric matrix (ensure dimension is an integer)
def randsymmat(n):
    if type(n) == int:
        A = np.random.rand(n,n)
        A_symm = A @ A.T          # definition of a symmetric matrix
    
    else:
        print("Error: dimension must be integer.")
    
    return A_symm

# define a function that creates a random n-dim vector (ensure dimension is an integer)
def randvect(n):
    if type(n) == int:
        x = np.random.rand(n)
    
    else:
        print("Error: dimension must be integer.")
    
    return x

# -----------------------------------------------------------------
# Powerit Algorithm
# -----------------------------------------------------------------
# set initial values
alpha0 = 1
verbose = True
tol = 1e-08
maxit = 100

# set dimension of A & x
dim = 5

# create for loop that iterates 3 times for dim = 5, 6, & 7
for i in range(3):
    print(f"MATRIX SIZE = {dim}")
    
    # use randsymmat to generate random symmetric nxn matrix A, where n = dim
    A = randsymmat(dim)
    print(f"A = \n {A}")
    
    # -----------------------------------------------------------------
    # RANDOM EIGENVECTOR
    # -----------------------------------------------------------------
    # use randvect to generate random n-dim vector x0, where n = dim
    x = randvect(dim)
    print(f"x0 = \n {x.reshape(-1,1)}")
    
    # call powerit; pass symmetric matrix, initial eigenvalue (alpha ~ lambda) & eigenvector, tolerance, & max iterations
    # dom_eignvec_powerit will store the dominant eigenvector returned by powerit (u in the powerit function)
    it, alpha, dom_eigenvec_powerit = powerit(A, alpha0, x, tol, maxit)
    
    # print the results of the powerit function with randomly generated starting vector x
    print("\npowerit results:")
    print(f"alphapowit = {alpha}")
    print(f"xpowit = \n {dom_eigenvec_powerit.reshape(-1,1)}")
    
    # call the np.linalg.eig function to find all eigenpairs of the randomly generated symmetric square matrix A
    eigenvalues, eigenvectors = np.linalg.eig(A)
    
    # sort the results in reverse order (greatest to least) in terms of absolute value
    sorted_indices = np.argsort(np.abs(eigenvalues))[::-1]      # grab the index values when sorted in reverse order (will be a list variable)
    sorted_eigenvalues = eigenvalues[sorted_indices]            # sort the eigenvalues using the reverse order index
    sorted_eigenvectors = eigenvectors[: , sorted_indices]      # sort the eigenvectors using the reverse order index
    
    # grab the index of the largest modulus of the array
    dom_eigenval_numpy = sorted_eigenvalues[0]
    
    # use the index of the dominant eigenvalue to find the corresponding dominant eigenvector
    dom_eigenvec_numpy = sorted_eigenvectors[:, 0]
    
    # take the norm difference of the dominant powerit eigenvector and the dominant numpy eigenvector
    norm_difference = np.linalg.norm(np.abs(dom_eigenvec_powerit) - np.abs(dom_eigenvec_numpy))
    
    # show results of computation from np.linalg.norm
    np.set_printoptions(precision=4, suppress=True, linewidth=150, floatmode='fixed')
    print("\nnp.linalg.eig results:")
    
    print("Eigenvalues")
    print(sorted_eigenvalues)
    
    print("Eigenvectors")
    print(sorted_eigenvectors)
    
    print(f"\nNorm difference of computed dominant eigenvectors = {norm_difference}")
    
    print("-" * 100)

    # -----------------------------------------------------------------
    # NUMPY SECOND-DOMINANT EIGENVECTOR
    # -----------------------------------------------------------------
    # grab the index of the SECOND largest modulus of the eigenvalue array
    # because eigenvalues are sorted in ascending, we can grab dom_index - 1 to get the second-dom eigenvalue
    sec_dom_eigenvec_numpy = sorted_eigenvectors[:, 1]
    
    # use the second-dominant eigenvector calculated by numpy as the initial guess for x
    x = sec_dom_eigenvec_numpy
    print(f"x0 = \n {x.reshape(-1,1)}")
    
    # call powerit on the second dominant eigenvector of A, computed by numpy
    it, alpha, dom_eigenvec_powerit = powerit(A, alpha0, x, tol, maxit)
    
    # print the results
    print("\npowerit results:")
    print(f"alphapowit = {alpha}")
    print(f"xpowit = \n {dom_eigenvec_powerit.reshape(-1,1)}")
    
    # show results of computation from np.linalg.norm
    np.set_printoptions(precision=4, suppress=True, linewidth=150, floatmode='fixed')
    print("\nnp.linalg.eig results:")
    
    print("Eigenvalues")
    print(sorted_eigenvalues)
    
    print("Eigenvectors")
    print(sorted_eigenvectors)
    
    # take the norm difference of the dominant powerit eigenvector and the dominant numpy eigenvector
    norm_difference = np.linalg.norm(np.abs(dom_eigenvec_powerit) - np.abs(sec_dom_eigenvec_numpy))
    
    print(f"\nNorm difference of computed second dominant eigenvectors = {norm_difference}")
    
    #print(f"\nNorm difference of computed dominant eigenvectors = {np.linalg.norm(x - alpha*u)}")
    print("=" * 100)
    
    # add one to the dimension
    dim+=1