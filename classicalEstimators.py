'''
Estimators for Entropy and Mutual Information. 

This module contains the following estimators:
    1. The classical KL estimator for entropy (EntropyKL)
    2. The classical KSG estimator for Mutual Information (MutualInformationKSG)

The other estimators in this file should not really be used.
They are all test estimators.
The LP estimators required Gaussian Integral estimations which was only
available in Julia and hence the work published in Physical Review article
was performed in Julia.

'''

import scipy.spatial as ss
from scipy.special import digamma,gamma
import numpy.random as nr
import numpy as np
import numpy.linalg as npla
from math import e, log, exp, pi
import random
from math import e as naturalbase

#########################################
#  Classical KL and Kraskov estimators  #
#########################################


def EntropyKL(x, k=3, base=naturalbase, jiggleIntensity=1e-10):
    '''
    Calculate entropy of X
    - X is the random variable and x are its realisations
    - x is a matrix of size (N,d)
    - where N is the sample size
    - and x \in R^d
    '''
    assert k <= np.size(x,0) - 1, "Set k smaller than number of samples in ensemble - 1"
    d = np.size(x,1)
    N = np.size(x,0)
    x = x + jiggleIntensity*nr.random(np.shape(x))
    tree = ss.cKDTree(x)
    q = tree.query(x,k+1,p=float('inf')) ## k+1 because the point itself is included as a neighbour
    epsiVec = q[0][:,k]
    const = digamma(N)-digamma(k) + d*log(2)
    # ent = (const + d*np.mean(map(log,epsiVec)))/log(base)
    ent = (const + d*np.mean(np.log(epsiVec)))/log(base)
    return ent

def MutualInformationKSG(x,y,k=3,base=naturalbase, jiggleIntensity=1e-10):
    """ 
    Original Kraskov MI estimator I(X;Y)
    """
    assert np.size(x,0)==np.size(y,0), "The number of samples in the ensemble should be the same"
    assert k <= np.size(x,0) - 1, "Set k smaller than number of samples in ensemble - 1"

    dx = np.size(x,1)
    dy = np.size(y,1)
    N = np.size(x,0)

    x = x + jiggleIntensity*nr.random(np.shape(x))
    y = y + jiggleIntensity*nr.random(np.shape(y))

    points = np.hstack((x,y)) # The joint array

    tree = ss.cKDTree(points)
    q = tree.query(points,k+1,p=float('inf')) ## k+1 because the point itself is included as a neighbour
    epsiVec = q[0][:,k]

    a,b =  AvgDigammaDvec(x,epsiVec), AvgDigammaDvec(y,epsiVec)
    mi = digamma(N) + digamma(k) - a - b
    return mi/log(base)

def MutualInformationKSG_2(x,y,k=3,base=naturalbase, jiggleIntensity=1e-10):
    """ 
    Original Kraskov MI estimator I(X;Y)
    """
    assert np.size(x,0)==np.size(y,0), "The number of samples in the ensemble should be the same"
    assert k <= np.size(x,0) - 1, "Set k smaller than number of samples in ensemble - 1"

    dx = np.size(x,1)
    dy = np.size(y,1)
    N = np.size(x,0)

    x = x + jiggleIntensity*nr.random(np.shape(x))
    y = y + jiggleIntensity*nr.random(np.shape(y))

    points = np.hstack((x,y)) # The joint array

    tree = ss.cKDTree(points)
    q = tree.query(points,k+1,p=float('inf')) ## k+1 because the point itself is included as a neighbour
    epsiVec = q[0][:,k]

    epsiVec_x = np.zeros(N)
    epsiVec_y = np.zeros(N)

    for i in range(N):
        for j in range(1,k+1):
            l_epsiVec_x =  np.max(np.abs(x[i,:] - x[q[1][i,j],:]))
            l_epsiVec_y =  np.max(np.abs(y[i,:] - y[q[1][i,j],:]))
            if l_epsiVec_x > epsiVec_x[i]:
                epsiVec_x[i] = l_epsiVec_x
            if l_epsiVec_y > epsiVec_y[i]:
                epsiVec_y[i] = l_epsiVec_y

    # a,b =  AvgDigammaDvec_2(x,epsiVec_x, epsiVec), AvgDigammaDvec_2(y,epsiVec_y, epsiVec)
    a,b =  AvgDigammaDvec(x,epsiVec_x), AvgDigammaDvec(y,epsiVec_y)
    mi = digamma(N) + digamma(k) - a - b - 1.0/k
    return mi/log(base)

def MutualInformationKSG_lnc(x,y,k=3, alpha = 0.25, low_fac=0.5, base=naturalbase, jiggleIntensity=1e-10):
    """ 
    LNC correction by npeet group
    """
    assert np.size(x,0)==np.size(y,0), "The number of samples in the ensemble should be the same"
    assert k <= np.size(x,0) - 1, "Set k smaller than number of samples in ensemble - 1"

    dx = np.size(x,1)
    dy = np.size(y,1)
    dz = dx + dy
    N = np.size(x,0)

    assert k >= dx + dy, "Set k larger or equal to dimX + dimY"

    x = x + jiggleIntensity*nr.random(np.shape(x))
    y = y + jiggleIntensity*nr.random(np.shape(y))

    points = np.hstack((x,y)) # The joint array
    z = points

    tree = ss.cKDTree(points)
    q = tree.query(points,k+1,p=float('inf')) ## k+1 because the point itself is included as a neighbour
    epsiVec = q[0][:,k]

    epsiVec_x = np.zeros(N)
    epsiVec_y = np.zeros(N)


    for i in range(N):
        for j in range(1,k+1):
            l_epsiVec_x =  np.max(np.abs(x[i,:] - x[q[1][i,j],:]))
            l_epsiVec_y =  np.max(np.abs(y[i,:] - y[q[1][i,j],:]))
            if l_epsiVec_x > epsiVec_x[i]:
                epsiVec_x[i] = l_epsiVec_x
            if l_epsiVec_y > epsiVec_y[i]:
                epsiVec_y[i] = l_epsiVec_y

    sum_log_V = 0.0
    sum_log_RV = 0.0

    for i in range(N):

        #Compute the volumes in the joint space
        log_V = dx*log(2.0*epsiVec_x[i]) + dy*log(2.0*epsiVec_y[i])

        neighbour_indices = q[1][i,1:k+1]
        zp = z[neighbour_indices,:] - z[i,:]

        # Compute SVD of zp
        u,s,v= np.linalg.svd(zp.T,full_matrices=False)

        proj = np.dot(np.diag(s),v)
        log_RV = np.sum(np.log(2.0*np.max(np.abs(proj),axis=1)))
        # log_RV = np.sum(np.log(2.0*s))

        if log_RV - log_V > log(alpha):
            log_RV = log_V
        elif log_RV - log_V < dz * log(low_fac):
            log_RV = dz * log(low_fac) + log_V


        sum_log_V += log_V
        sum_log_RV += log_RV

    mean_log_V = sum_log_V/N
    mean_log_RV = sum_log_RV/N

    lnc_correction = mean_log_RV - mean_log_V
    a,b =  AvgDigammaDvec(x,epsiVec_x), AvgDigammaDvec(y,epsiVec_y)
    mi = digamma(N) + digamma(k) - a - b - 1.0/k  - lnc_correction
    return mi/log(base)

def MutualInformationKSG_marginal(x,y,k=3,base=naturalbase, jiggleIntensity=1e-10):
    """ 
    Original Kraskov MI estimator I(X;Y)
    Basically Singh
    """
    assert np.size(x,0)==np.size(y,0), "The number of samples in the ensemble should be the same"
    assert k <= np.size(x,0) - 1, "Set k smaller than number of samples in ensemble - 1"

    dx = np.size(x,1)
    dy = np.size(y,1)
    N = np.size(x,0)

    x = x + jiggleIntensity*nr.random(np.shape(x))
    y = y + jiggleIntensity*nr.random(np.shape(y))

    z = np.hstack((x,y)) # The joint array

    tree = ss.cKDTree(z)
    dd,ii = tree.query(z,k+1,p=float('inf')) ## k+1 because the point itself is included as a neighbour
    epsiVec = dd[:,k]

    a,b =  AvgStripCounting(x,epsiVec,k), AvgStripCounting(y,epsiVec,k)
    # mi = digamma(N) + digamma(k) + a + b
    mi = log(N) + log(k) - a - b
    return mi/log(base)


###############################
#  Boundary Layer estimators  #
###############################

def EntropyBL(x, k=3, p=10, base=naturalbase, jiggleIntensity=1e-10):
    '''
    Calculate entropy of X
    - X is the random variable and x are its realisations
    - x is a matrix of size (N,d)
    - where N is the sample size
    - and x \in R^d
    - Here on the boundary layer (BL) geometric correction of the volume is used
    - where (2*eps)^d in classical KL is replaced by (4/3 * eps)^d for the boundary points
    '''
    assert k <= np.size(x,0) - 1, "Set k smaller than number of samples in ensemble - 1"
    assert p>k, "Set p greater than k"
    d = np.size(x,1)
    N = np.size(x,0)
    x = x + jiggleIntensity*nr.random(np.shape(x))

    # Compute tree for the p neighbours because p>k
    tree = ss.cKDTree(x)
    dd,ii = tree.query(x,p+1,p=float('inf')) ## k+1 because the point itself is included as a neighbour
    epsiVec = dd[:,k]

    alpha = computeAlpha(x,ii)
    
    const = digamma(N)-digamma(k) + d*log(2.0)*(1.0-alpha) + d*log(4.0/3)*alpha
    ent = (const + d*np.mean(map(log,epsiVec)))/log(base)
    return ent


def MutualInformationBL(x, y, k=3, p=10, base=naturalbase, jiggleIntensity=1e-10):
    '''
    Calculate MI between X and Y
    - X is the random variable and x are its realisations
    - Y is the random variable and y are its realisations
    - x is a matrix of size (N,dx)
    - y is a matrix of size (N,dy)
    - where N is the sample size
    - Here on the boundary layer (BL) geometric correction of the volume is used
    - where (2*eps)^d in classical KL is replaced by (4/3 * eps)^d for the boundary points
    '''

    assert np.size(x,0)==np.size(y,0), "The number of samples in the ensemble should be the same"
    assert k <= np.size(x,0) - 1, "Set k smaller than number of samples in ensemble - 1"

    dx = np.size(x,1)
    dy = np.size(y,1)
    d = dx + dy
    N = np.size(x,0)

    x = x + jiggleIntensity*nr.random(np.shape(x))
    y = y + jiggleIntensity*nr.random(np.shape(y))

    z = np.hstack((x,y)) # The joint array

    tree_z = ss.cKDTree(z)
    dd,ii = tree_z.query(z,p+1,p=float('inf')) ## k+1 because the point itself is included as a neighbour
    epsiVec = dd[:,k]

    tree_x = ss.cKDTree(x)
    ddx,iix = tree_x.query(x,p+1,p=float('inf')) ## k+1 because the point itself is included as a neighbour

    tree_y = ss.cKDTree(y)
    ddy,iiy = tree_y.query(y,p+1,p=float('inf')) ## k+1 because the point itself is included as a neighbour

    alpha = computeAlpha(z,ii)
    alphax = computeAlpha(x,iix)
    alphay = computeAlpha(y,iiy)

    correctionx = dx*(1.0-alphax)*log(2) + dx*alphax*log(4.0/3)
    correctiony = dy*(1.0-alphay)*log(2) + dy*alphay*log(4.0/3)
    correction  = d*(1.0-alpha)*log(2)   + d*alpha*log(4.0/3)

    a,b =  AvgDigammaDvec(x,epsiVec), AvgDigammaDvec(y,epsiVec)
    mi = digamma(N) + digamma(k) - a - b
    mi = mi + correctionx + correctiony - correction

    return mi

#########################
#  LP method where p=k  #
#########################

def EntropyLP(x, k=3, base=naturalbase, jiggleIntensity=1e-10):
    '''
    Calculate entropy of X
    - X is the random variable and x are its realisations
    - x is a matrix of size (N,d)
    - where N is the sample size
    - and x \in R^d

    - k is the number of nearest neighbours to be used
    - Here k is also used to determine the empirical variance to compute tr(B)
    '''
    assert k <= np.size(x,0) - 1, "Set k smaller than number of samples in ensemble - 1"
    d = np.size(x,1)
    N = np.size(x,0)
    x = x + jiggleIntensity*nr.random(np.shape(x))

    tree = ss.cKDTree(x)
    dd, ii = tree.query(x,k+1,p=float('inf')) ## k+1 because the point itself is included as a neighbour
    epsiVec = dd[:,k] # distances to the k nearest neighbours
    kNNindices = ii[:,1:] # indices of the k nearest neighbours

    constClassical = digamma(N)-digamma(k) + d*log(2)
    
    correction = computeCorrection(x,epsiVec,kNNindices)

    constCorrection = np.mean(map(log,correction))
    
    ent = (constClassical + constCorrection + d*np.mean(map(log,epsiVec)))
    ent = ent/log(base)
    return ent

################################################
#  LPP method where p is provided by the user  #
################################################


def EntropyLPP(x, k=3, p=10, base=naturalbase, jiggleIntensity=1e-10):
    '''
    Calculate entropy of X
    - X is the random variable and x are its realisations
    - x is a matrix of size (N,d)
    - where N is the sample size
    - and x \in R^d

    - k is the number of nearest neighbours to be used
    - Here p is also used to determine the empirical variance to compute tr(B)
    '''
    assert k <= np.size(x,0) - 1, "Set k smaller than number of samples in ensemble - 1"
    d = np.size(x,1)
    N = np.size(x,0)
    x = x + jiggleIntensity*nr.random(np.shape(x))

    tree = ss.cKDTree(x)
    dd, ii = tree.query(x,k+1,p=float('inf')) ## k+1 because the point itself is included as a neighbour
    epsiVec = dd[:,k] # distances to the k nearest neighbours
    kNNindices = ii[:,1:] # indices of the k nearest neighbours

    constClassical = digamma(N)-digamma(k) + d*log(2)
  

    ddp, iip = tree.query(x,p+1,p=float('inf')) ## k+1 because the point itself is included as a neighbour
    epsiVecp = ddp[:,p] # distances to the k nearest neighbours
    kNNindicesp = iip[:,1:] # indices of the k nearest neighbours

    correction = computeCorrection(x,epsiVecp,kNNindicesp)

    constCorrection = np.mean(map(log,correction))
    
    ent = (constClassical + constCorrection + d*np.mean(map(log,epsiVec)))
    ent = ent/log(base)
    return ent

#######################
#  Utility functions  #
#######################

def computeAlpha(x,ii):
    pNNindices = ii[:,1:] # indices of the k nearest neighbours
    # mark Boundary points
    ## Find barycenter
    barx = np.mean(x,0)
    numBLPoints = 0
    for i,point in enumerate(x):
        vecToCenter = barx - point #pointing towards the centre
        # the neighbours of this point
        neighbourIndices = pNNindices[i,:]
        neighbours = x[neighbourIndices,:]
        vecsToNeighbours = neighbours - point
        dotp = np.dot(vecsToNeighbours,vecToCenter)
        if np.all(dotp>=0):
            numBLPoints += 1
    N = np.size(x,0)
    alpha = float(numBLPoints)/N
    return alpha
    
def computeCorrection(x, epsiVec, kNNindices):
    d = np.size(x,1)
    N = np.size(x,0)
    correctionVec = np.zeros(N)
    for i in range(N):
            xpoint = x[i,:]
            epsi = epsiVec[i]
            neighbourIndices = kNNindices[i,:]
            neighbours = x[neighbourIndices,:]
            empiricalTrace = 0.0
            k = np.size(neighbours,0)
            for j in range(k):
                    currNeighbour = neighbours[j,:]
                    diffNeighbour = currNeighbour - xpoint
                    empiricalTrace = empiricalTrace + np.sum(diffNeighbour**2)
            empiricalTrace = empiricalTrace/k
            y = (2*epsi)**2
            num = (d-1)*y**2/144.0 + y**2/80.0 - epsi**2*d*y/72.0
            den = (d-1)*y**2/144.0 + y**2/80.0 - epsi**2*empiricalTrace/6.0
            correction = num/den
            correctionVec[i] = correction

    return correctionVec

def AvgDigammaDvec(points,epsiVec):
    N = np.size(points,0)
    tree = ss.cKDTree(points)
    digammaSum = 0.0
    for i in range(N):
        dist = epsiVec[i]
        #subtlety, we don't include the boundary point, 
        #but we are implicitly adding 1 to kraskov def bc center point is included
        num_points = len(tree.query_ball_point(points[i,:],dist-1e-15,p=float('inf'))) 
        digammaSum += digamma(num_points)
    avg = digammaSum/N
    return avg

def AvgStripCounting(points,epsiVec,k):
    N = np.size(points,0)
    tree = ss.cKDTree(points)
    alpha = 0.0
    for i in range(N):
        dist = epsiVec[i]
        #subtlety, we don't include the boundary point, 
        #but we are implicitly adding 1 to kraskov def bc center point is included
        num_points = len(tree.query_ball_point(points[i,:],dist-1e-15,p=float('inf'))) 
        alpha += log(num_points)
    avg = alpha/float(N)
    return avg




