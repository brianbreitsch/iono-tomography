#
#
# author: Brian Breitsch
# email: brianbw@colostate.edu
#

import numpy as np

def art(p, A, x0, relax=0.1, iters=1, only_positive=False):
    """Performs algebraic reconstruction (using ART) of image`x` given projection 
    data `p` and a projection matrix `A` which satisfies
    :math:`\vec{p} = A\vec{s}`
    
    Parameters
    ----------
    p : array_like, dtype=float
        Array containing set of projection data.
    A : array_like, dtype=float
        Projection matrix for p.
    x0 :  array_like, dtype=float
        Initial guess for image vector.
    relax : relaxation parameter
    iters : int, optional
        The number of iterations to perform during reconstruction.
    only_positive : boolean, optional
        Constrain reconstruction so that only positive values are allowed in
        the image.
        
    Returns
    -------
    s : array_like, dtype=float
        The reconstructed image.
    """
    x = x0
    SA2 = np.linalg.norm(A, axis=1)**2
    ind = np.nonzero(SA2)[0]
    for it in range(iters):
        for i in range(len(ind)):
            update = relax * (p[ind[i]] - A[ind[i],:].dot(x)) / SA2[ind[i]]
            x = x + update * A[ind[i],:]
            if only_positive:
                x = np.maximum(x,0)
    return x
# old implementation
#
#x = x0
#n_rows = A.shape[0]
#for _ in range(iters):
#    for i in range(n_rows):
#        normA2 = np.linalg.norm(A[i,:])**2
#        if normA2 == 0.:
#            continue
#        update = relax * ((p[i] - np.sum(A[i,:] * x)) / normA2) * (A[i,:])
#        x = x + update
#        if only_positive:
#            x = np.maximum(x, 0.)
#return x


def mart(p, A, x0, relax=0.1, iters=1, only_positive=False):
    """Performs multiplicitive algebraic reconstruction (MART) of image`x` given
    projection data `p` and a projection matrix `A` which satisfies
    :math:`\vec{p} = A\vec{s}`
    
    Parameters
    ----------
    p : array_like, dtype=float
        Array containing set of projection data.
    A : array_like, dtype=float
        Projection matrix for p.
    x0 : array_like, dtype=float
        Initial guess for image vector.
    relax : relaxation parameter
    iters : int, optional
        The number of iterations to perform during reconstruction.
    only_positive : boolean, optional
        Constrain reconstruction so that only positive values are allowed in
        the image.
        
    Returns
    -------
    s : array_like, dtype=float
        The reconstructed image.

    TODO: implement
    Notes: must assume positivity of image--which makes sense for ionosphere tomography
    """
    assert(False)
    x = x0
    SA2 = np.linalg.norm(A, axis=1)**2
    ind = np.nonzero(SA2)[0]
    for it in range(iters):
        for i in range(len(ind)):
            update = relax * (p[ind[i]] - A[ind[i],:].dot(x)) / SA2[ind[i]]
            x = x + update * A[ind[i],:]
            if only_positive:
                x = np.maximum(x,0)
    return x

    

def sart(p, A, x0, relax=0.1, iters=1):
    '''
    Performs Simultaneous Algebraic Rconstruction Technique (SART) to reconstruct the image
    `x` given projections `p` and their relation to the image defined by
    projection matrix `A`.
    
    see: Paper of A.H.Anderson and A.C.Kak " Simultaneous Algebraic Rconstruction Technique (SART)"
    
    parameters
    ----------
    p : n_lines array : the projections
    A : n_pixels-by-n_lines array : the projection matrix
    x0 : n_pixels array : the initial guess for the reconstructed image
    
    returns
    -------
    x : n_pixels array : the reconstructed image
    '''
    x = x0.copy()
    sumR = np.sum(A,axis=1)
    indL = sumR!=0# we remove the lines which don't intersect any point of the grid
    Abis = A[indL,:]
    n_rows = Abis.shape[0]
    sumC = np.sum(Abis,axis=0)
    indA = sumC!=0# we remove the points which are not crossed by any line
    Atris = Abis[:,indA]
    sumR = np.sum(Atris,axis=1)
    sumC = np.sum(Atris,axis=0)
    for _ in range(iters):
        x[indA] = x[indA] + relax * np.sum((Atris*((p[indL] - Atris.dot(x[indA]))/(sumR))[:,None]),axis=0) / (sumC)
    return x
  

def ODT(p, projmtx, basis, only_positives = False, reshaped = True):
    '''
    Performs Orthogonal Decomposition Technique (ODT) for finding the image
    `x` given projections `p`, projection matrix `A` and a list of basis functions.
    IMPORTANT: this algorithm works better with a projection matrix made using the voxels/lines
    overlaps than with a projection matrix made wih centers/lines impact parameters.
    
    parameters
    ----------
    p : n_lines-by-one array : the projections
    A : n_lines-by-n_pixels array : the projection matrix
    basis : a list of basis functions of the ionosphere (3D arrays)
    
    returns
    -------
    x : n_pixels (reshaped) array : the reconstructed image
    coeffs : n_basis array : the coefficients
    '''
    
    
    
    n = len(basis)
    ionoshape = basis[0].shape
    
    
    #Step1: Projection						  
    basis = np.array([phi.flatten() for phi in basis])
    
    projbasis = (basis).dot(projmtx.T)
    
    
    #Step2: Gram-Schmidt orthogonalisation process
    N = len(projbasis[0])
    GSbasis = np.zeros((n,N))
    
    GSbasis[0] = projbasis[0]
    GSbasis[0] = GSbasis[0]/np.linalg.norm(GSbasis[0])
    
    
    for i in range(1,n):
        v = projbasis[i]
        j = 0
        for e in GSbasis:
            if np.dot(e,e)==0:
                continue
            product = np.dot(e, v) / np.dot(e, e)
            pv = product * e
            v = v - pv
        
        v = v/np.linalg.norm(v)
        GSbasis[i] = v
    
    
    #Step3: Decomposition
    C = np.linalg.inv(GSbasis.dot(projbasis.T))
    
    projcoeffs = (GSbasis).dot(p)
    if only_positives:
        ind_neg = [projcoeffs<0.]
        projcoeffs[ind_neg] = 0.
        
    coeffs = C.dot(projcoeffs)#We use the matrix C to get back the real coefficients
    
    #Step4: Reconstruction
    Decomposed = [basis[i]*coeffs[i] for i in range(n)]
    x = np.sum(Decomposed,axis=0)
    if reshaped:
        x = x.reshape((ionoshape))
    
    return x , coeffs
    #TODO: if possible, make a relevant only_positive mode

  
def art_function_based(p, A, x0, basis, relax=0.1, iters=1):
    '''
    Performs Algebraic Rconstruction Technique (ART) for finding the image
    `x` given projections `p` and their relation to the image defined by
    projection matrix `A`. This time we are going to use a basis of ionosphere functions.
    
    see: http://en.wikipedia.org/wiki/Algebraic_Reconstruction_Technique
    
    parameters
    ----------
    p : the projections
    A : the projection matrix
    x0 : the initial guess for the reconstructed image
    basis : nfunctions-by-3D array : the matrix of basis functions
    
    returns
    -------
    img : the reconstructed image
    x : the coefficients
    '''
    basis_fl = np.array([phi.flatten() for phi in basis])
    x = x0.flatten()
    x = (basis_fl).dot(x)
    a = A.dot(basis_fl.T)
    n_rows = A.shape[0]
    for _ in range(iters):
        for i in range(n_rows):
            norma = np.linalg.norm(a[i,:])
            if norma == 0.:
                continue
            x = x + relax * (p[i] - np.sum(a[i,:] * x)) / (norma**2) * (a[i,:])
            x[x<0] = 0.
    img = (basis_fl.T).dot(x)
    return img, x
  
  
  