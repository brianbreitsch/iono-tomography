import numpy as np
from numpy import pi, cos, sin, arcsin, arctan2, ones, zeros, sqrt, \
                  matrix, array, asarray, reshape, size, maximum, \
                  absolute, rad2deg, deg2rad
from numpy.linalg import inv, norm


def ecef2geo( x0 ):
    '''
    Converts ECEF coordinates to geodetic coordinates, 
    x0 is an ndarray of N ECEF coordinates with shape (N,3).
    
    >>> import numpy as np
    >>> geo = np.array([27.174167, 78.042222, 0])
    >>> ecef = geo2ecef(np.deg2rad(geo))
    >>> new_geo = ecef2geo(ecef)
    array([[             nan],
    	   [  7.08019709e+01],
           [ -6.37805436e+06]])
    >>> # [1176.45, 5554.887, 2895.397] << should be this 
    >>> ecef2geo(np.array([
    	[27.174167, 78.042222, 0],
    	[39.5075, -84.746667, 0]])).reshaps((3,2))
    array([[             nan,              nan],
           [  7.08019709e+01,  -6.50058423e+01],
           [ -6.37805436e+06,  -6.37804350e+06]])
    [1176.45, 5554.887, 2895.397]
    [451.176, -4906.978, 4035.946]
    '''

    # we = 7292115*1**-11 # Earth angular velocity (rad/s)
    # c = 299792458    # Speed of light in vacuum (m/s)
    rf = 298.257223563 # Reciprocal flattening (1/f)
    a = 6378137.       # Earth semi-major axis (m)
    b = a - a / rf     # Earth semi-minor axis derived from f = (a - b) / a
    x = x0[:,0]; y = x0[:,1]; z = x0[:,2];

    # We must iteratively derive N
    lat = arctan2(z, sqrt(x**2 + y**2))
    h = z / sin(lat)
    d_h = 1.; d_lat = 1.

    while (d_h > 1e-10) and (d_lat > 1e-10):
    
        N = a**2 / (sqrt(a**2 * cos(lat)**2 + b**2 * sin(lat)**2))
        N1 = N * (b / a)**2
    
        temp_h = sqrt(x**2 + y**2) / cos(lat) - N
        temp_lat = arctan2(z / (N1 + h), sqrt(x**2 + y**2) / (N + h))
        d_h = np.max(np.abs(h - temp_h))
        d_lat = np.max(np.abs(lat - temp_lat))

        h = temp_h
        lat = temp_lat

    lon = arctan2(y,x)

    lat = rad2deg(lat)
    lon = rad2deg(lon)

    geo = np.column_stack((lat, lon, h))
    return geo


def ecef2enu( x0, xs ):
    '''
    Converts satellite ECEF coordinates to user-relative ENU coordinates.
    '''

    # get the lat and lon of the user position
    geo = ecef2geo(x0)
    lat = deg2rad(geo[:,0]); lon = deg2rad(geo[:,1])

    # create the rotation matrix
    Rl = array([[-sin(lon),               cos(lon),              0],
          [-sin(lat) * cos(lon), -sin(lat) * sin(lon), cos(lat)],
          [cos(lat) * cos(lon),  cos(lat) * sin(lon), sin(lat)]])
    # NOTE: This matrix is not fixed to do multiple user locations yet...

    dx = xs - x0
    enu = Rl.dot(dx.T)
    return enu.reshape(3)


def ecef2sky( x0, xS ):
    '''
    Converts user and satellite ecef coordinates to azimuth and elevation 
    from user on Earth.
    '''

    enu = ecef2enu(x0, xS)
    e = enu[0]; n = enu[1]; u = enu[2]
    az = arctan2(e[0],n[0])
    el = arcsin(u[0]/sqrt(e[0]**2 + n[0]**2 + u[0]**2))

    sky = np.column_stack((rad2deg(az), rad2deg(el)))
    return sky


def geo2ecef( geo ):
    '''
    Converts geodetic coordinates to ECEF coordinates
    
    see: TODO website
    ----------
    geo : N-by-3 ndarray : geodetic coordinates to convert
    Returns
    -------
    N-by-3 array : ecef coordinates
    '''
    a = 6378137. # Earth semi-major axis (m)
    rf = 298.257223563 # Reciprocal flattening (1/f)
    b = a * (rf - 1) / rf # Earth semi-minor axis derived from f = (a - b) / a
    lat = deg2rad(geo[:,0]); lon = deg2rad(geo[:,1]); h = geo[:,2]

    N = a**2 / (sqrt(a**2 * cos(lat)**2 + b**2 * sin(lat)**2))
    N1 = N * (b / a)**2

    x = (N + h) * cos(lat) * cos(lon)
    y = (N + h) * cos(lat) * sin(lon)
    z = (N1 + h) * sin(lat)

    x0 = np.column_stack((x, y, z))
    return x0


def geo2sky( geo0, geoS ):
    '''
    Converts user and satellite geodetic coordinates to azimuth and elevation 
    from user on Earth.
    '''
    x0 = geo2ecef(geo0)
    xS = geo2ecef(geoS)

    sky = ecef2sky(x0, xS)
    return sky


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


def line2cart(theta, t, d):
    '''Computes the cartesian coordinates of a point that lies displacement d along a line
    described by line coordinates :math: `(\theta,t)`.
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


def intersect_lines(line1, line2):
    '''
    Computes the 2D cartesian coordinates of the point of intersection between two
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



# TODO this function does not belong in this file. it's also probs buggy
def polar_grid_cell_overlap(line, lats, alts):
    '''
    Computes the overlap of a line with each cell of a polar grid. 
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
