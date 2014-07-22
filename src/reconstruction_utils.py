#
#
# author: Brian Breitsch
# email: brianbw@colostate.edu
#

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

    

