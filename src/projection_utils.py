#
# projection_utils.py
#
# author: Brian Breitsch
#
#

import numpy as np
from numpy.linalg import inv, norm
import imp
coordinate_utils = imp.load_source('coordinate_utils', '../src/coordinate_utils.py')

def line_plane_intersection(p0, u, v0, n):
    '''
    Calculates the intersection of a line with a plane in 3
    dimensions.
    
    see: http://geomalgorithms.com/a05-_intersect-1.html
    ----------
    p0 : N-by-3 ndarray : point through which the line passes
    u  : N-by-3 ndarray : unit vector of line direction
    v0 : N-by-3 ndarray : point through which the plane passes
    n  : N-by-3 ndarray : normal vector of plane
    Returns
    -------
    N-by-3 array : intersection points of lines with planes
    '''
    w = p0 - v0
    s = - np.sum(n * w) / np.sum(n * u)
    return p0 + s * u


def line_quadroid_intersection(line, corners):
    '''
    Computes the coordinates and indices of intersection of a line
    through with a quadroid defined by corners.
    ----------
    line : 2-by-3 ndarray : line to project through the grid mesh
    corners : 2-by-2-by-2-by-3 ndarray : grid mesh
    Returns
    -------
    pnt1, pnt2 : points where the line intersects the quadroid, None
        if no intersection occurs
    '''
    
    ind_x = np.array([0,1,0,1,0,1,0,1])
    ind_y = np.array([0,0,1,1,0,0,1,1])
    ind_z = np.array([0,0,0,0,1,1,1,1])
    
    faces = np.array([(0,1,2,3),
                      (0,2,4,6),
                      (0,1,4,5),
                      (1,3,5,7),
                      (2,3,6,7),
                      (4,5,6,7),
                     ])
    
    p0, u = line
    pnts = []
    #eps = 0.01 * np.linalg.norm(corners[-1,-1,-1,:] - corners[0,0,0,:])
    eps = 1e-8
    
    for face in faces:
        v = corners[ind_x[face], ind_y[face], ind_z[face], :]
        v0, v1, v2, v3 = v[0,:], v[1,:], v[2,:], v[3,:]
        n = np.cross(v2 - v0, v1 - v0)
        n = n / np.linalg.norm(n)
        #if np.linalg.norm(np.cross(u, n)) > np.linalg.norm(n) - eps:
        #    print('parallel')
        #    continue
        # this was not a good check--instead, should normalize vectors, then check angle
        if abs(np.inner(u, n)) < eps:
            #print('parallel')
            continue
        pnt = line_plane_intersection(p0, u, v0, n)
        # determine if intersection lies within face
        norm = np.linalg.norm(np.cross(v2 - v0, v1 - v0)) + np.linalg.norm(np.cross(v2 - v3, v1 - v3))
        area =  np.linalg.norm(np.cross(pnt - v0, v1 - v0)) + \
                np.linalg.norm(np.cross(pnt - v0, v2 - v0)) + \
                np.linalg.norm(np.cross(pnt - v3, v1 - v0)) + \
                np.linalg.norm(np.cross(pnt - v3, v2 - v0))
        if area - norm < eps:
            pnts.append(pnt)
    if len(pnts) == 2:
        return pnts[0], pnts[1]
    elif len(pnts) == 1: # check, remove later
        print('1 pnt')
    elif len(pnts) > 2:
        print('mas pnts')
    return None


def grid_mesh_from_centers(xs, ys, zs):
    '''
    Creates a grid mesh, i.e. points that define the corners of cells
    whose centers lie at the intersections of the surfaces contained
    in xs, ys, and zs.
    ----------
    xs : vector of scalars : horizontal grid boundaries
    ys : vector of scalars : inward grid boundaries
    zs : vector of scalars : vertical grid boundaries
    Returns
    -------
    L-by-M-by-N-by-3 ndarray : the geodetic grid mesh
    '''
    nx, ny, nz = len(xs), len(ys), len(zs)
    mesh = np.zeros((nx + 1, ny + 1, nz + 1, 3))
    
    def meshify_1d(vals):
        arr = np.zeros(len(vals) + 1)
        arr[1:-1] = vals[:-1] + 0.5 * (vals[1:] - vals[:-1])
        arr[0] = vals[0] - 0.5 * (vals[1] - vals[0])
        arr[-1] = vals[-1] + 0.5 * (vals[-1] - vals[-2])
        return arr
    
    if nx == 1:
        mesh[0,:,:,0] = xs[0] - 1
        mesh[1,:,:,0] = xs[0] + 1
    else:
        mesh[:,:,:,0] = meshify_1d(xs)[:,None,None]
        
    if ny == 1:
        mesh[:,0,:,1] = ys[0] - 1
        mesh[:,1,:,1] = ys[0] + 1
    else:
        mesh[:,:,:,1] = meshify_1d(ys)[None,:,None]
        
    if nz == 1:
        mesh[:,:,0,2] = zs[0] - 1
        mesh[:,:,1,2] = zs[0] + 1
    else:
        mesh[:,:,:,2] = meshify_1d(zs)[None,None,:]
   
    return mesh


def grid_projection_matrix_from_mesh(mesh, lines):
    '''
    Creates a projection matrix given the mesh which defines the grid
    boundaries and a set of lines which traverse the grid.
    ----------
    mesh : LxNxMx3 ndarray : grid cell boundary mesh
    lines : P-by-2-by-3 ndarray : lines to project through the grid

    Returns
    -------
    P-by-N ndarray : the projection matrix
    '''
    n_x, n_y, n_z, _ = mesh.shape
    print(mesh.shape)
    n_lines = len(lines)
    n_x, n_y, n_z = n_x - 1, n_y - 1, n_z - 1 
    N = n_x * n_y * n_z
    print(N)
    
    projmtx = np.zeros((n_lines, N))
    points = []
    
    for l, line in enumerate(lines):
        for k in range(n_z):
            for j in range(n_y):
                for i in range(n_x):
                    corners = mesh[i:i+2,j:j+2,k:k+2]
                    pnts = line_quadroid_intersection(line, corners)
                    if not pnts:
                        continue
                    for p in pnts:
                        points.append(p)
                    projmtx[l, i + j * n_x + k * n_x * n_y] = np.linalg.norm(pnts[1] - pnts[0])
    return projmtx, np.array(points)


def geodetic_projection_matrix(lats, lons, alts, lines):
    '''
    Creates a projection matrix given the latitudes, longitudes, 
    and altitudes that define a given geodetic grid, and a set of
    lines which traverse the grid.
    ----------
    lats : vector of scalars : latitudinal grid boundaries
    lons : vector of scalars : longitudinal grid boundaries
    alts : vector of scalars : altitude grid boundaries
    lines : P-by-2-by-3 ndarray : lines to project through the grid
    Returns
    -------
    P-by-N ndarray : the projection matrix
    '''
    n_lats, n_lons, n_alts, n_lines = len(lats), len(lons), len(alts), len(lines)
    N = n_lats * n_lons * n_alts
    
    # get grid mesh--i.e. set of points defining grid regions
    mesh = grid_mesh_from_centers(lats, lons, alts)
    shape = mesh.shape
    mesh = coordinate_utils.geo2ecef(mesh.reshape((shape[0] * shape[1] * shape[2], 3))).reshape(shape)
    
    #projmtx = np.zeros((n_lines, N))
    #points = []
    #
    #for l, line in enumerate(lines):
    #    for k in range(n_alts):
    #        for j in range(n_lons):
    #            for i in range(n_lats):
    #                corners = mesh[i:i+2,j:j+2,k:k+2]
    #                pnts = line_quadroid_intersection(line, corners)
    #                if not pnts:
    #                    continue
    #                for p in pnts: # TODO get rid of
    #                    points.append(p)
    #                projmtx[l, i + j * n_lats + k * n_lats * n_lons] = np.linalg.norm(pnts[1] - pnts[0])
    #return projmtx, np.array(points)
    return grid_projection_matrix_from_mesh(mesh, lines)


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


def line_coords_to_cartesian(theta, t, d):
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


