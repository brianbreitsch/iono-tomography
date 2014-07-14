import numpy as np
from numpy import pi, cos, sin, arctan2, ones, zeros, sqrt, \
                  matrix, array, asarray, reshape, size
from numpy.linalg import inv, norm

def impact_paramter(u, v):
    '''Computes the impact parameter, i.e. closest point to the origin, for a given line.
    Creates matrix :math: `\bf A = \begin{bmatrix}\cos\theta & u_x - v_x \\ \sin\theta & 
    u_y - v_y \end{bmatrix}`. Then `\begin{bmatrix} \tau \\ \alpha \end{bmatrix} = 
    \bf A^{-1} u`.
    ----------
    u : 2-tuple of the first point defining the line
    v : 2-tuple of the second point defining the line
        
    Returns
    -------
    a : the impact parameter of the line
    '''
    dx, dy = v[0] - u[0], v[1] - u[1]
    theta = pi / 2 + arctan2(dy, dx)
    A = matrix([[cos(theta), -dx],[sin(theta), -dy]])
    t = asarray(A.I * reshape(u, (2,1))).reshape((2,))
    return t[0] * cos(theta), t[0] * sin(theta)

def proj2cart(theta, t, d):
    '''Computes the cartesian coordinates of a point that lies displacement d along a line
    described by theta and t
    ----------
    theta : angle of the normal vector to line
    t : orthogonal displacement of line from origin
    d : displacement of point along the line (positive direction is to right of normal)
    Returns
    -------
    (x,y) : tuple containing cartesian coordinates of the point
    '''
    x = t * cos(theta) + d * sin(theta)
    y = t * sin(theta) + d * cos(theta)
    return (x,y)

def intersect_lines_lc(line1, line2):
    '''Computes the 2D cartesian coordinates of the point of intersection between two
    lines described by line coordinates in the 2-tuples line1 and line2
    ----------
    line  : 2-tuple containing
        theta : angle of the normal vector that describes the first line
        t : orthogonal displacement of first line from origin
    -------
    (x,y) : tuple containing cartesian coordinates of the intersection
    '''
    theta1, t1 = line1
    theta2, t2 = line2
    assert(not theta1 == theta2)
    u1 = t1 * array([cos(theta1), sin(theta1)])
    u2 = t2 * array([cos(theta2), sin(theta2)])
    A = matrix([[sin(theta1), -sin(theta2)], [-cos(theta1), cos(theta2)]])
    tau = asarray(A.I * reshape((u2 - u1), (2,1))).reshape((2,))
    return u1 + tau[0] * array([sin(theta1), -cos(theta1)])

def polar_grid_projection(line, lats, alts):
    '''Computes the overlap of a line with each cell of a polar grid. 
    Grid elements are latitude (expressed in line coordinates) and
    altitude (expressed as distance from origin). A projection
    ----------
    line : 2-tuple containing line coordinates for projection line
    lats : a list of 2-tuples containing the line coordinates for 
        the latitudinal lines
    alts : a list of altitudes for the grid
    -------
    G : len(alts) by len(lats) ndarray containing values of 
        projection line overlap
    pnts : cartesian coordinates of the intersection points
    '''
    pnts = []
    
    M, N = len(alts), len(lats)
    G = zeros((M,N))
    i, j = 0, 0
    theta, t = line
    # first we need to find `u`, the starting point on the projection line
    u = array([t*cos(theta), t*sin(theta)])
    while j < M and i < N:
        #print('{0:.3f} {1:.3f} {2} {3}'.format(u[0],u[1],i,j))
        pnts.append(u)
        v = get_intersect(line, lats[i])
        if norm(v) < alts[j]:
            # then we intersect latitudinally
            G[j,i] = norm(v - u)
            i += 1
            u = v
        else:
            # then we intersect vertically
            front = u[1] * cos(theta) - u[0] * sin(theta)
            xcos, ysin = u[0] * cos(theta), u[1] * sin(theta)
            rad = -xcos**2 - ysin**2 - 2 * xcos * ysin + alts[j]**2
            if rad < 0: j += 1; continue
            tau1, tau2 = front + sqrt(rad), front - sqrt(rad)
            tau = tau1 if tau1 > 0 else tau2
            v = u + tau * array([sin(theta), -cos(theta)])
            G[j,i] = norm(v - u)
            j += 1
            u = v
    pnts.append(u)
    return G, pnts
