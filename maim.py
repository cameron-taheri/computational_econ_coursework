#!/usr/bin/python3

# Insert the following somewhere when debugging
#   import pdb; pdb.set_trace()

import sys
from scipy import linalg
from numpy import array,diag,dot,eye,pi,sin,cos,set_printoptions,sign,sqrt
import numpy as np
import matplotlib.pyplot as plt

sys.path.append('Users/camerontaheri/python/comp_econ/PS5/maim.py')
import maim

np.set_printoptions(suppress=True,linewidth=200)

def maim(h,maxlag,verbosity=0,uprbnd=1.000001,ussmethod="schur",checkh=False):

    ## returncode,nshifts,nbigroots,roots,b,s =
    ##  maim(h,maxlag,verbosity=0,uprbnd=1.000001,ussmethod="schur",checkh=False)

    ## Computes the backward-looking reduced form b if it exists and
    ## the observable structure matrix x given a forward- and
    ## backward-looking structural coefficient matrix h and the number
    ## of lags maxlag.
    ##
    ## This code implements the Anderson-Moore algorithm.  See:
    ## Anderson, G. and Moore, G. "A Linear Algebraic Procedure for
    ## Solving Linear Perfect Foresight Models."  Economics Letters,
    ## 17, 1985.  It also implements some related methods for solving
    ## such models.
    ##  
    ## This code classifies an eigenvalue as explosive if its modulus
    ## is greater than uprbnd.  By default, uprbnd=1+10^-6.  An
    ## infinite generalized eigenvalue is considered explosive.
    ##
    ## Verbosity is an integer between 0 and 10.
    ##
    ## Unstable subspace methods are:
    ## ['eig','geig','geigexa','gschur','gschurexa','schur'].
    ## All of the schur methods are real Schur decompositions.
    ## schur is the default.
    ##
    ## Returns the following in a list: returncode, the number of
    ## shiftrights, the number of big roots, the roots, the reduced
    ## for b, and the observable structure matrix s.
    ##
    ## Brian Madigan
    ## Georgetown University
    ## Revised April 16, 2024
    ##

    ## returncodes
    ##  -4 to -1: problem with structural coefficient matrix h
    ##   0: unique stable reduced form
    ##   1: too many orthogonality conditions
    ##   2: too few orthogonality conditions
    ##   3: qtheta is singular 
    ##   4: zero row encountered in h
    ##   5: SVD of htheta failed
    ##   6: Eigen, generalized Eigen, Schur, or QZ decomposition of STM failed
    ##   7: SVD of q failed
    ##   8: 
    ##   9: 
    ##  10: error in computing eigenvalues after Schur decomposition
    ##  11: no such invariant subspace ussmethod
    ##  12: too many shiftrights
    ##  13: warning: qtheta is badly conditioned
    ##  14: qtheta is singular

    if checkh:
        hok = checkhOk(h,maxlag)
        if hok < 0:
            return hok,None,None,None,None,None

    if ussmethod not in ['eig','geig','geigexa','gschur','gschurexa','schur']:
        print('Invalid ussmethod: ', ussmethod)
        returncode = 11
        return returncode,None,None,None,None,None

    if ussmethod in ['eig','schur']:
        numericshifts=True
    else:
        numericshifts=False

    if(verbosity > 2):
        print('structural coefficient matrix h =\n', h)
        print('ussmethod is', ussmethod, '.')

    #  Make a copy of h, because shiftrights alter h but original h is
    #   needed to compute the observable structure.
    hcopy = np.copy(h)
    (neq,hcols) = h.shape

    maxlead = int(hcols/neq) - maxlag - 1
    growsneeded = neq*maxlead
    guprbnd = uprbnd

    if(verbosity > 0):
        print('Need %d orthogonality conditions.' % growsneeded)

    # If necessary for ussmethod, compute shiftrights
    if ussmethod in ['eig','geigexa','gschurexa','schur']:
        returncode,h,nexashifts,gexa,nnumshifts,gnum = shiftrights(h,maxlead,verbosity,numericshifts)
        nshifts = nexashifts+nnumshifts
        gshifts=np.concatenate((gexa[:nexashifts],gnum[:nnumshifts]))
        if(returncode == 1 or returncode == 12):
            if(verbosity>0):
                print("Too many orthogonality conditions.")
                return returncode,nshifts,None,None,None,None
        if(returncode == 4):
            if(verbosity>0):
                print("Zero row encountered in h.")
                return returncode,nshifts,None,None,None,None
        if(returncode == 5):
            if(verbosity>0):
                print("Singular value decomposition of h-theta failed.")
                return returncode,nshifts,None,None,None,None
        if(verbosity > 0):
            print('nexashifts=',nexashifts)
            print('nnumshifts=',nnumshifts)
            print('nshifts=',nshifts)
            
    else:
        nshifts = 0
            
    returncode,roots,nbigroots,gunstable = unstablesubspace(h,verbosity, ussmethod, uprbnd)
    
    if(not any(np.iscomplex(roots))):
        roots=np.real(roots)

    if (returncode==6):
        if(verbosity>0):
            print("Eigen, generalized Eigen, Schur, or QZ decomposition of STM failed.")
        return returncode,nshifts,None,None,None,None
    elif (nshifts+nbigroots>growsneeded):
        if(verbosity>0):
            print("Too many orthogonality conditions.")
        returncode=1
        return returncode,nshifts,nbigroots,roots,None
    elif (nshifts+nbigroots<growsneeded):
        if(verbosity>0):
            print("Too few orthogonality conditions.")
            print("growsneeded =", growsneeded)
            print("nexashifts =", nexashifts)
            print("nnumshifts =", nnumshifts)
            print("nbigroots =", nbigroots)
        returncode=2
        return returncode,nshifts,nbigroots,roots,None,None

    # Assemble orthogonality conditions into q
    if ussmethod in ['eig','geigexa','gschurexa','schur']:
        q = np.concatenate((gshifts, gunstable),axis=0)
    else:
        q = gunstable

    returncode, b, bbar = reducedform(q,neq,maxlag,verbosity)
    
    # Compute the observable structure. Need to use a copy of the
    # structural h.
    if returncode==0:
        s = obstruct(hcopy,bbar)
        return returncode,nshifts,nbigroots,roots,b,s
    else:
        return returncode,None,None,None,None,None


def shiftrights(h,maxlead,verbosity,numericshifts=True):
    (neq,hcols) = h.shape
    hblocks=int(hcols/neq)
    maxgrows = int(neq*maxlead)
    gcols=int(hcols-neq)
    gexa=np.zeros((maxgrows,gcols))
    qrow=0
    nnumshifts = 0
    neqzeros = np.zeros((1,neq))[0];
    returncode = 0
    
    # Exact shiftrights
    hrow=0
    while (hrow < neq):
        hthetarow = h[hrow,range(neq*(hblocks-1),neq*hblocks)]
        if(np.array_equal(hthetarow,neqzeros)):
            nextqrow=h[hrow,range(neq*(hblocks-1))]
            if qrow > maxgrows:
                if verbosity > 0:
                    print("Too many shiftrights.")
                returncode=12
                return returncode,h,nexashifts,gexa,nnumshifts,gnum
            gexa[qrow]=nextqrow
            h[hrow,range(neq,neq*hblocks)]=nextqrow
            h[hrow,range(neq)]=0
            qrow = qrow+1
            if verbosity > 0:
                print('h =\n',h)
        else:
            hrow=hrow+1
    nexashifts=qrow

    gexa=gexa[range(nexashifts),:]           

    if not numericshifts:
        return returncode,h,nexashifts,gexa,nnumshifts,np.zeros((0,gcols))

    # Numeric shiftrights
    gnum = np.zeros((maxgrows-nexashifts,gcols))
    gnumrow = 0

    while True:
        htheta = h[:,range(gcols,hcols)]
            
        # Numpy defines svd as U*diag(D)*V=A rather than the more
        # conventional U*diag(D)*V'
        try:
            [U,D,V] = np.linalg.svd(htheta)
        except np.linalg.LinAlgError:
            if(verbosity>0):
                print('Error in singular value decomposition of htheta after %d shiftrights.' % nnumshifts)
                returncode=5
            return returncode,h,nexashifts,gexa,nnumshifts,gnum

        # Multiply through by U' no matter the rank of htheta
        h=U.transpose()@h
        rankhtheta = mrank(D)
        if verbosity > 0:
            print('rank(htheta) = ', rankhtheta)

        if (neq - rankhtheta) > (maxgrows - nexashifts - nnumshifts):
            nnumshifts = nnumshifts + neq - rankhtheta
            if verbosity > 0:
                print("Too many shiftrights.")
            returncode=12
            return returncode,h,nexashifts,gexa,nnumshifts,gnum

        # If htheta is of full rank, solve through h
        if (rankhtheta == neq):
            DINV=np.diag(1/D)
            h = -V.transpose()@DINV@h
            break
        else:
            for hrow in range(rankhtheta,neq):
                nextqrow=h[hrow,range(neq*(hblocks-1))]
                gnum[gnumrow]=nextqrow
                h[hrow,range(neq,neq*hblocks)]=nextqrow
                h[hrow,range(neq)]=0
                gnumrow = gnumrow+1
                nnumshifts=nnumshifts+1

        if verbosity > 0:
            print('h =\n',h)
                
    gnum=gnum[:nnumshifts]
    return returncode,h,nexashifts,gexa,nnumshifts,gnum

#  Compute the unstable left subspace against which solution vectors
#  will be orthogonalized.  Returns returncode, the roots, the number
#  of bigroots, and the vectors corresponding to the big roots in
#  columns of a matrix.
# 
def unstablesubspace(h,verbosity,ussmethod,uprbnd):
    (neq,hcols) = h.shape
    gcols = hcols  - neq
    A = np.zeros((gcols,gcols))

    A[0:gcols-neq,neq:gcols]   =   np.identity(gcols-neq)
    A[gcols-neq:gcols,0:gcols]  =  h[:,0:hcols-neq]
    AT=A.transpose()
    if(verbosity > 5):
        print('A =\n', A)
        print('AT =\n', AT)

    if ussmethod in ['geig','geigexa','gschur','gschurexa']:
        B = np.zeros((gcols,gcols))
        B[0:gcols-neq,0:gcols-neq]   =   np.identity(gcols-neq)
        B[gcols-neq:gcols,gcols-neq:gcols]  =  -h[:,gcols:hcols]
        BT=B.transpose()
        if(verbosity > 5):
            print('B =\n', B)
            print('BT =\n', B)

    if ussmethod=='eig':
        try:
            roots, evects = linalg.eig(AT,left=False,right=True)
            bigrootsidx = abs(roots) > uprbnd
            bigroots = roots[bigrootsidx]
            nbigroots = bigroots.size
            bigvects=evects[:,bigrootsidx]
            print('bigvects',bigvects)
            
            # Convert complex conjugate vectors to real vectors
            # (float64) spanning same space.
            if bigvects.dtype=='complex128':
                bigroot = 0
                while bigroot < nbigroots-1:
                    if ((bigroots[bigroot].imag != 0) and
                        (abs(bigroots[bigroot]-abs(bigroots[bigroot+1])<1e-14))):
                        bigvects[:,bigroot] = np.real(bigvects[:,bigroot])
                        bigroot = bigroot+1
                        bigvects[:,bigroot] = np.imag(bigvects[:,bigroot])
                    bigroot = bigroot+1
                bigvects = np.real(bigvects)
            gunstable = bigvects.transpose()

        except np.linalg.LinAlgError:
            if(verbosity>0):
                print('Error in eigenvalue decomposition of state transition matrix.')
            returncode=6
            return returncode, None, None, None

    elif ussmethod=='schur':
        try:
            T, Z, nbigroots = linalg.schur(AT,output='real',sort=lambda r, i: np.sqrt(r**2+i**2) > uprbnd)
            gunstable=Z[:,range(nbigroots)].transpose()
        except np.linalg.LinAlgError:
            if(verbosity>0):
                print('Error in Schur decomposition of state transition matrix.')
            returncode=6
            return returncode, None, None, None

        try:
            roots = linalg.eigvals(T)
            if(verbosity>0):
                print("nbigroots =", nbigroots)
                print("roots = \n", roots)
                print("absroots = \n", abs(roots))
        except np.linalg.LinAlgError:
            if(verbosity>0):
                print('Error in computing eigenvalues of Schur T.')
            returncode=6
            return returncode, None, None, None

    elif ussmethod in ['gschur','gschurexa']:
        try:
            AA,BB,alpha,beta,Q,Z = linalg.ordqz(AT, BT, output='real', sort=select_roots_gschur)
        except np.linalg.LinAlgError:
            if(verbosity>0):
                print('Error in generalized Schur (QZ) decomposition of state transition matrix.')
            returncode=6
            return returncode, None, None, None

        roots=np.array([ab[0]/ab[1] if abs(ab[1])>1e-12 else np.inf for ab in zip(alpha,beta)])
        bigindex = abs(roots) > uprbnd
        bigroots = roots[bigindex]
        nbigroots = bigroots.size
        bigvects=Z[:,range(nbigroots)]
        gunstable=(AT@bigvects).transpose()
        if(verbosity>0):
            print("nbigroots =", nbigroots)
            print("roots = \n", roots)

    elif ussmethod in ['geig','geigexa']:
        try:
            roots,Z = linalg.eig(AT, BT)
        except np.linalg.LinAlgError:
            if(verbosity>0):
                print('Error in generalized Schur (QZ) decomposition of state transition matrix.')
            returncode=6
            return returncode, None, None, None

        bigindex = abs(roots) > uprbnd
        bigroots = roots[bigindex]
        nbigroots = bigroots.size
        bigvects=Z[:,bigindex]

        # Convert complex conjugate vectors to real vectors
        # (float64) spanning same space.
        if bigvects.dtype=='complex128':
            bigroot = 0
            while bigroot < nbigroots-1:
                if ((bigroots[bigroot].imag != 0) and
                    (abs(bigroots[bigroot]-abs(bigroots[bigroot+1])<1e-14))):
                    bigvects[:,bigroot] = np.real(bigvects[:,bigroot])
                    bigroot = bigroot+1
                    bigvects[:,bigroot] = np.imag(bigvects[:,bigroot])
                bigroot = bigroot+1
            bigvects = np.real(bigvects)

        gunstable=(AT@bigvects).transpose()

    returncode=0
    return returncode, roots, nbigroots, gunstable

def reducedform(q,neq,maxlag,verbosity):
    returncode = 0
    grows,gcols = q.shape
    
    # Solve for reduced form
    qtheta = q[:,range(gcols-grows,gcols)]
    qminus = q[:,range(gcols-grows)]

    try:
        bbar = -np.linalg.solve(qtheta,qminus)
    except np.linalg.LinAlgError:
        if(verbosity>0):
            print('qtheta is singular.')
        returncode=14
        return returncode, None, None
    except np.linalg.LinAlgWarning:
        if(verbosity>0):
            print('qtheta is badly conditioned.')
            returncode=13

    if verbosity>0:
        print('q =\n', q)
        print('qtheta =\n', qtheta)
        print('cond(qtheta)=', np.linalg.cond(qtheta))

    b = bbar[:neq,:neq*maxlag]
    return returncode, b, bbar

def obstruct(h,bbar):
    ## Compute time-t expectations observable structure matrix s

    neq,hcols = h.shape
    bbarrows,bbarcols = bbar.shape
    maxlag = int(bbarcols/neq)
    scols = neq*(maxlag+1)
    s = np.zeros((neq,scols))

    ## minus is the indexes of the lags and contemporaneous blocks of h.
    minus = list(range(neq*(maxlag+1)))

    ## plus includes only the lead blocks of h
    plus  = list(range(neq*(maxlag+1),hcols))
    hplus=h[:,plus]
    s[:,list(range(neq,neq*(maxlag+1)))] = hplus@bbar

    hminus=h[:,minus]
    s = s + hminus
    return s





def simmaim(nperiods,initconds,b,verbosity=0,shocks=None,s=None):
    ## Computes a deterministic simulation or a stochastic simulation
    ## of a forward-looking model given the reduced form matrix b or
    ## the observable structure matrix s.  Both are computed using
    ## maim, an implementation of the Anderson-Moore algorithm.
    ##
    ## initconds is np.array((maxlag,neq))
    ## 
    ## If a stochastic simulation is desired, shocks and s must be
    ## provided.  shocks is np.array((nperiods,neq)).  If shocks is
    ## not provided, the simulation will be deterministic using b.
    ##
    ## If successful, returns returncode=0 and the simulation values
    ## in simx, which has maxlag+nperiods rows and neq columns.
    ## If the singular value decomposition fails, returns 1, None.
    ## 

    bAndsok = checkbAndsOk(b,s)
    if bAndsok != 0:
        return -1

    neq,bcols = b.shape
    if not s is None:
        scols = s.shape[1]
    
    maxlag = bcols//neq

    if shocks is None:
        stochsim = False
    else:
        stochsim = True

    if stochsim: # Stochastic sim requested
        (nshockrows,nshockcols) = shocks.shape
        if (nshockrows != nperiods):
            print('Number of shock rows (%d) is not equal to number of simulation periods (%d)' % (nshockrows,nperiods))
            return -1
        if(s is None):
            print('Observable structure matrix s must be supplied for stochastic simulation.')
            return -1

        ## s0 is the rightmost (contemporaneous) block of s
        s0 = s[:,neq*maxlag:neq*(maxlag+1)]
        ## sminus is the first maxlag blocks of s
        sminus = s[:,:(neq*maxlag)]

        try:
            lu,piv = linalg.lu_factor(s0)
        except linalg.LinAlgError:
            if(verbosity>0):
                print('Error in inverting s0.')
            returncode=1
            return returncode, None
  
    sim=np.zeros((maxlag+nperiods,neq))
    sim[:maxlag,:]=initconds
    
    for period in range(nperiods):
        xtm1=np.reshape(sim[period:period+maxlag],(neq*maxlag,1) )
        
        if(stochsim):
            shocksperiod=np.asmatrix(shocks[period,:]).transpose()
            prod=sminus@xtm1
            xt=linalg.lu_solve((lu,piv),shocksperiod-prod)
        else:
            xt=b@xtm1

        sim[period+maxlag,:]=xt.transpose()

    returncode = 0
    return returncode, sim


# Global variable
guprbnd = 1.000001

# This function adapted from scipy.linalg source code.  Required
#  calling signature necessitates use of global variable for upper
#  bound.
def select_roots_gschur(x, y):
    out = np.empty_like(x, dtype=bool)
    xzero = (x == 0)
    yzero = (y == 0)
    out[xzero & yzero] = False
    out[~xzero & yzero] = True
    out[~yzero] = (abs(x[~yzero]/y[~yzero]) >= guprbnd)
    return out

def mrank(s):
    # s must be a vector of singular values sorted from largest to
    # smallest.
    # Uses Matlab definition of tol
    n=s.shape[0]
    tol=n*np.spacing(s[0])
    return sum(s>tol)

def checkhOk(h,maxlag):
    if(not isinstance(h, np.ndarray)):
        print('h must be a neq by neq*(maxlag+1+maxlead) numpy array.')
        return -1
    
    neq,hcols = h.shape
    
    if hcols % neq != 0:
        print('Number of columns in h must be an integer multiple of the number of rows of h.')
        return -2
        
    if hcols < (maxlag + 1 + 1)*neq:
        print('h must be a neq*(maxlag+1+maxlead) numpy array.')
        return -3
        
    ## Test whether structural model is statically determinate,
    ## meaning that given values for lags and leads, unique values for
    ## period t could be computed.  In other words, h0 is nonsingular.
    h0 = h[:,neq*maxlag:neq*(maxlag+1)]

    try:
        d = np.linalg.svd(h0,compute_uv=False)
    except np.linalg.LinAlgError:
        if(verbosity>0):
            print('Error in singular value decomposition of structural h0.')
        return -4
    if(mrank(d) < neq):
        print('Model is not statically determinate (rank(h0) < neq).')
        return -5
        
    return 0

def checkbAndsOk(b,s):
    msgb = 'b must be a neq by neq*maxlag numpy array.'
    if(not isinstance(b, np.ndarray)):
        print(msgb)
        return -1
    else:
        neqb,bcols=b.shape
        if(bcols%neqb != 0):
            print(msgb)
            return -1
        else:
            maxlag=bcols/neqb
        if bcols != neqb*maxlag:
            print(msgb)
            return -1
        
        if not s is None:
            msgs = 's must be a neq by neq*(maxlag+1) numpy array.'
            if(not isinstance(s, np.ndarray)):
                print(msgs)
                return -2
            else:
                neqs,scols=s.shape
                if neqs != neqb:
                    print('Number of rows in b and s are not equal.')
                    return -2
                if scols != neqs*(maxlag+1):
                    print(msgs)
                    return -2
    return 0

def main():
    # A test function and a single test case is provided below.
    # After successfully computing the backward looking reduced form b
    # and the observable structure matrix, a simple stochastic sim is
    # performed.
    
     
    verbosity=10
    uprbnd = 1.
    ussmethod = "schur"
    checkh = True
    
    def runtestmaim(h,maxlag,verbosity,uprbnd,ussmethod,checkh):
        print('h =\n', h)
        neq,hcols=h.shape
        ret,nshifts,nbigroots,roots,b,s = maim(h,maxlag,verbosity,uprbnd,ussmethod,checkh)
        print("returncode=",ret)
        print("nshifts=",nshifts)
        print("nbigroots=",nbigroots)
        print("roots",roots)
        print("b =\n",b)
        print("s =\n",s,"\n")

        initconds=np.zeros((maxlag,neq))
        initconds[0,0]=1
        nperiods=10
        shocks=None
        returncode, sim=simmaim(nperiods,initconds,b,verbosity,shocks,s)
        print("deterministic sim")
        print(sim)

        shocks=.1*np.random.normal(size=nperiods*neq)
        shocks=np.reshape(shocks,(nperiods,neq))

        shocks=np.zeros((nperiods,neq))
        shocks[0,]  =[.1, .2]
        shocks[1,]=[-.1, -.2]
        shocks[2,]=[.1, .2]

        
        returncode, sim=simmaim(nperiods,initconds,b,verbosity,shocks,s)
        print("stochastic sim")
        print(sim)

        return returncode, sim
    
    h = np.array([[-.1,-.1,-.1,-.1,1,-.1,-.1,-.2,-.1,-.1],[-.1,0,0,-.1,0,1,-.1,-.1,0,0]])
    print("h =\n", h)
    maxlag=2
    returncode, sim = runtestmaim(h,maxlag,verbosity,uprbnd,ussmethod,checkh)
    print("sim = \n", sim)

if __name__ == "__main__":
    main()

verbosity = 0
maxlag = 2

#========================================================
# define h (structural coefficient) matrix
#========================================================

# indexing
#one(-2) i(-2) pi(-2) rho(-2) y(-2) .... one(1) i(1) pi(1) rho(1) y(1)

h = np.array ([
[0,	0,	0,	0,	0,	-1,	0,	0,	0,	0,	1,	0,	0,	0,	0,	0,	0,	0,	0,	0],
[0,	0,	0,	0,	0,	0,	0,	-0.5,	0,	-0.5,	-0.011,	1,	-1,	0,	0,	0,	0,	0,	0,	0],
[0,	0,	0,	0,	0,	0,	0,	-0.5,	0,	0,	0,	0,	1,	0,	-0.1,	0,	0,	-0.5,	0,	0],
[0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	-1,	0,	48.619048,	0,	0,	0,	1,	-47.619048,	0],
[0,	0,	0,	0,	0.415,	0,	0,	0,	0.798,	-1.254,	-0.016758,	0,	0,	0,	1,	0,	0,	0,	0,	0]
])

#========================================================
# compute reduced form
#========================================================
returncode,nshifts,nbigroots,roots,b,s = maim(h,maxlag,verbosity=0,uprbnd=1.000001,ussmethod="schur",checkh=False)

print("\nBackward-Looking Reduced Form Matrix:\n")
print(b)
print(np.shape(b))

#========================================================
# simulate model
#========================================================

# steady state values
one_ss = 1
i_ss = .041
pi_ss = .02
rho_ss = .021
y_ss = 0

# create array of steady state variables


'''
5x10 backwards-looking reduced form matrix b

[[-0.0000 -0.0000 -0.0000 -0.0000 -0.0000  1.0000 -0.0000 -0.0000 -0.0000 -0.0000]
 [-0.0000 -0.0000 -0.0000 -0.0000 -0.2538  0.0244 -0.0000  1.3421 -0.4880  1.0640]
 [ 0.0000  0.0000  0.0000  0.0000 -0.2538  0.0134  0.0000  0.8421 -0.4880  0.5640]
 [-0.0000 -0.0000 -0.0000 -0.0000 -0.0476  0.0219 -0.0000  0.0511 -0.0915  0.1050]
 [-0.0000 -0.0000 -0.0000 -0.0000 -0.4150  0.0168 -0.0000 -0.0000 -0.7980  1.2540]]

'''
