"""
projection_utils.py

@author Brian Breitsch
@email brianbw@colostate.edu

Notes:
  The functions that a program would usually call in order to create a projection matrix
exist towards the end of the file. The geodetic_projection_matrix and projection_matrix functions
call the projection_matrix_from_centers and projection_matrix_from_mesh functions. The first
function performs impact parameter calculation for each ray with each image voxel center, whereas
the second performs a overlap algorithm that assumes quadroid volumes for each voxel and uses a three
dimensional mesh.
"""

import numpy as np
from numpy.linalg import inv, norm
import imp
coordinate_utils = imp.load_source('coordinate_utils', '../src/coordinate_utils.py')


def impact_parameter(x0, lines):
    """Computes the impact parameter, i.e. closest point to the origin, for each line in
    lines.

    Parameters
    ----------
    x0 : (3,1) ndarray
        the point of interest/origin
    lines : (N,2,3) ndarray
        points which define the lines
        
    Returns
    -------
    outputs : a
    the impact parameters of the lines
    """
    v = x0[None,:] - lines[:,0,:]
    s = np.sum(v * lines[:,1,:], axis=1)
    return lines[:,0,:] + s[:,None] * lines[:,1,:]


def line_plane_intersection(p0, u, v0, n):
    """Calculates the intersection of a line with a plane in 3 dimensions.

    Parameters
    ----------
    p0 : (N,3) ndarray
        point through which the line passes
    u  : (N,3) ndarray
        unit vector of line direction
    v0 : (N,3) ndarray
        point through which the plane passes
    n  : (N,3) ndarray
        normal vector of plane
    
    Returns
    -------
    output : (N,3) ndarray
        intersection points of lines with planes

    Notes
    -----
    see: http://geomalgorithms.com/a05-_intersect-1.html
    """
    w = p0 - v0
    s = - np.sum(n * w) / np.sum(n * u)
    return p0 + s * u


def line_quadroid_intersection(line, corners):
    """Computes the coordinates and indices of intersection of a line
    through with a quadroid defined by corners.

    Parameters
    ----------
    line : (2,3) ndarray
        line to project through the grid mesh
    corners : (2,2,2,3) ndarray
        grid mesh

    Returns
    -------
    pnt1, pnt2 : (2,3) ndarrays
        points where the line intersects the quadroid, None if no intersection occurs
    """
    
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
        #print('1 pnt')
        pass
    elif len(pnts) > 2:
        #print('mas pnts')
        pass
    return None


def grid_mesh_from_center_planes(xs, ys, zs):
    """Creates a grid mesh, i.e. points that define the corners of cells
    whose centers lie at the intersections of the surfaces which pass through
    points located xs, ys, and zs along respective axes and which lie normal 
    to respective axes.

    Parameters
    ----------
    xs : (X,) ndarray
        horizontal grid boundaries
    ys : (Y,) ndarray
        inward grid boundaries
    zs : (Z,) ndarray
        vertical grid boundaries

    Returns
    -------
    (L,M,N,3) ndarray
        the geodetic grid mesh
    """
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


def grid_centers(xs, ys, zs):
    """Creates (N,3) ndarray of grid centers"""
    nx, ny, nz = len(xs), len(ys), len(zs)
    centers = np.zeros((nx, ny, nz, 3))
    centers[:,:,:,0] = xs[:,None,None]
    centers[:,:,:,1] = ys[None,:,None]
    centers[:,:,:,2] = zs[None,None,:]
    return centers.reshape((nx*ny*nz,3))


def projection_matrix_from_3d_mesh(mesh, lines):
    """Creates a projection matrix given the mesh which defines the grid
    boundaries and a set of lines which traverse the grid.

    Parameters
    ----------
    mesh : (L,N,M,3) ndarray
        grid cell boundary mesh
    lines : (P,2,3) ndarray
        lines to project through the grid

    Returns
    -------
    (P,N) ndarray
        the projection matrix
    """
    n_x, n_y, n_z, _ = mesh.shape
    # print(mesh.shape)
    n_lines = lines.shape[0]
    n_x, n_y, n_z = n_x - 1, n_y - 1, n_z - 1 
    N = n_x * n_y * n_z
    # print(N)
    
    projmtx = np.zeros((n_lines, N))
    points = []
    
    for l in range(n_lines):
        line = lines[l,:,:]
        for k in range(n_z):
            for j in range(n_y):
                for i in range(n_x):
                    corners = mesh[i:i+2,j:j+2,k:k+2]
                    pnts = line_quadroid_intersection(line, corners)
                    if not pnts:
                        continue
                    for p in pnts:
                        points.append(p)
                    projmtx[l, i*n_z*n_y + j*n_z + k] = np.linalg.norm(pnts[1] - pnts[0])
                    #projmtx[l, i + j * n_x + k * n_x * n_y] = np.linalg.norm(pnts[1] - pnts[0])
    return projmtx, np.array(points).squeeze()


def projection_matrix_from_centers(centers, lines):
    """Creates a projection matrix by computing impact parameter given the
    points that define the centers of image region voxels and a set of lines
    which traverse the grid.

    Note: projection matrix should be conditioned afterwards. TODO maybe in this function?

    Parameters
    ----------
    centers : (N,3) ndarray
        arrays which define the coordinates of the center of image region voxels
    lines : (P,2,3) ndarray
        lines to project through the image region

    Returns
    -------
    outputs : (P,N) ndarray
        the projection matrix
    """

    N, three = centers.shape
    assert(three == 3)

    L, two, three = lines.shape
    assert(three == 3)
    assert(two == 2)

    projmtx = np.zeros((L, N))
    points = []
    for l in range(L):
        for n in range(N):
            a = impact_parameter(centers[n,:], lines[l:l+1,:,:])
            projmtx[l,n] = np.linalg.norm(a - centers[n,:])
            points.append(a)

    # condition the projection matrix so that the closest (smallest) impact parameters become the highest matrix values
    diff = np.linalg.norm(centers[:,None,:] - centers[None,:,:], axis=2)
    ind = np.arange(0, diff.size, diff.shape[0]+1)
    shape = diff.shape
    diff = np.delete(diff.flatten(), ind).reshape(shape[0], shape[1]-1)
    diff = np.sort(diff)
    radius = np.mean(diff[:,:4], axis=1) / 2.

    projmtx = radius[None,:] - projmtx
    projmtx[projmtx < 0.] = 0.

    return projmtx, np.array(points).squeeze()
    # TODO could vectorize entire process with:
    # v = x0[None,:,:] - lines[:,0,:]
    # return np.inner(v, lines[None,:,1,:], axis=-1)
    # for n in range(N):
    #    projmtx[:,n] = impact_parameter(centers[n,:], 


def projection_matrix(xs, ys, zs, lines, from_mesh=False):
    """Creates a projection matrix given the points that define the centers of 
    image region voxels along each axis and a set of lines which traverse the grid.

    Parameters
    ----------
    xs, ys, zs : (N,1) ndarray
        arrays which define the coordinates of the center of image region voxels along each axis
    lines : (P,2,3) ndarray
        lines to project through the image region

    Returns
    -------
    outputs : (P,N) ndarray
        the projection matrix
    """
    if from_mesh:
        mesh = grid_mesh_from_center_planes(xs, ys, zs)
        return projection_matrix_from_3d_mesh(mesh, lines)
    else:
        centers = grid_centers(xs, ys, zs)
        return projection_matrix_from_centers(centers, lines)


def geodetic_projection_matrix(lats, lons, alts, lines, from_mesh=False):
    """Creates a projection matrix given the latitudes, longitudes, 
    and altitudes that define a given geodetic grid, and a set of
    lines which traverse the grid.

    Parameters
    ----------
    lats : (X,) ndarray
        latitudinal grid boundaries
    lons : (Y,) ndarray
        longitudinal grid boundaries
    alts : (Z,) ndarray
        altitude grid boundaries
    lines : (P,2,3) ndarray
        lines to project through the grid
    from_mesh : bool
        if True, uses lat, lon, alt to create a mesh from which quadroid intersections
        are then used to determine cell overlap. Otherwise, does cell impact parameter
        algorithm in `projection_matrix_from_centers`

    Returns
    -------
    (P,N) ndarray
        the projection matrix
    """
    n_lats, n_lons, n_alts, n_lines = len(lats), len(lons), len(alts), len(lines)
    N = n_lats * n_lons * n_alts
    
    # get grid mesh--i.e. set of points defining grid regions
    if from_mesh:
        mesh = grid_mesh_from_center_planes(lats, lons, alts)
        shape = mesh.shape
        mesh = coordinate_utils.geo2ecef(mesh.reshape((shape[0] * shape[1] * shape[2], 3))).reshape(shape)
        return projection_matrix_from_3d_mesh(mesh, lines)
    else:
        centers = grid_centers(lats, lons, alts)
        centers = coordinate_utils.geo2ecef(centers)
        return projection_matrix_from_centers(centers, lines)
