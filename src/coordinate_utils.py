"""

coordinate_utils.py

@author Brian Breitsch
@email brianbw@colostate.edu

"""

import numpy as np

def ecef2geo( x0 ):
    """Converts ECEF coordinates to geodetic coordinates, 

    Parameters
    ----------
    x0 : an ndarray of N ECEF coordinates with shape (N,3).

    Returns
    -------
    output : (N,3) ndarray
        geodetic coordinates

    Notes
    -----
   
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
    """

    # we = 7292115*1**-11 # Earth angular velocity (rad/s)
    # c = 299792458    # Speed of light in vacuum (m/s)
    rf = 298.257223563 # Reciprocal flattening (1/f)
    a = 6378137.       # Earth semi-major axis (m)
    b = a - a / rf     # Earth semi-minor axis derived from f = (a - b) / a
    x = x0[:,0]; y = x0[:,1]; z = x0[:,2];

    # We must iteratively derive N
    lat = np.arctan2(z, np.sqrt(x**2 + y**2))
    h = z / np.sin(lat)
    d_h = 1.; d_lat = 1.

    while (d_h > 1e-10) and (d_lat > 1e-10):
    
        N = a**2 / (np.sqrt(a**2 * np.cos(lat)**2 + b**2 * np.sin(lat)**2))
        N1 = N * (b / a)**2
    
        temp_h = np.sqrt(x**2 + y**2) / np.cos(lat) - N
        temp_lat = np.arctan2(z / (N1 + h), np.sqrt(x**2 + y**2) / (N + h))
        d_h = np.max(np.abs(h - temp_h))
        d_lat = np.max(np.abs(lat - temp_lat))

        h = temp_h
        lat = temp_lat

    lon = np.arctan2(y,x)

    lat = np.rad2deg(lat)
    lon = np.rad2deg(lon)

    geo = np.column_stack((lat, lon, h))
    return geo


def ecef2enu( x0, xs ):
    """Converts satellite ECEF coordinates to user-relative ENU coordinates.

    Parameters
    ----------
    x0 : ndarray of shape (3,)
        observer coordinate
    xs : ndarray of shape(N,3)
        object coordinates

    Returns
    -------
    output : ndarray of shape(N,3)
        The east-north-up coordinates

    Notes
    -----
    """

    # get the lat and lon of the user position
    geo = ecef2geo(x0)
    lat = np.deg2rad(geo[:,0]); lon = np.deg2rad(geo[:,1])

    # create the rotation matrix
    Rl = array([[-np.sin(lon),               np.cos(lon),              0],
          [-np.sin(lat) * np.cos(lon), -np.sin(lat) * np.sin(lon), np.cos(lat)],
          [np.cos(lat) * np.cos(lon),  np.cos(lat) * np.sin(lon), np.sin(lat)]])
    # NOTE: This matrix is not fixed to do multiple user locations yet...

    dx = xs - x0
    enu = Rl.dot(dx.T)
    return enu.reshape(3)


def ecef2sky( x0, xS ):
    """Converts user and satellite ecef coordinates to azimuth and elevation
    from user on Earth.

    Parameters
    ----------
    x0 : ndarray of shape (3,)
        observer coordinate
    xs : ndarray of shape(N,3)
        object coordinates

    Returns
    -------
    output : ndarray of shape(N,3)
        The objects' sky coordinatescoordinates

    Notes
    -----
    """

    enu = ecef2enu(x0, xS)
    e = enu[0]; n = enu[1]; u = enu[2]
    az = np.arctan2(e[0],n[0])
    el = np.arcsin(u[0]/np.sqrt(e[0]**2 + n[0]**2 + u[0]**2))

    sky = np.column_stack((np.rad2deg(az), np.rad2deg(el)))
    return sky


def geo2ecef( geo ):
    """Converts geodetic coordinates to ECEF coordinates

    Parameters
    ----------
    geo : ndarray of shape (N,3)
        geodetic coordinates

    Returns
    -------
    output : ndarray of shape(N,3)
        ECEF coordinates

    Notes
    -----
    """
    a = 6378137. # Earth semi-major axis (m)
    rf = 298.257223563 # Reciprocal flattening (1/f)
    b = a * (rf - 1) / rf # Earth semi-minor axis derived from f = (a - b) / a
    lat = np.deg2rad(geo[:,0]); lon = np.deg2rad(geo[:,1]); h = geo[:,2]

    N = a**2 / (np.sqrt(a**2 * np.cos(lat)**2 + b**2 * np.sin(lat)**2))
    N1 = N * (b / a)**2

    x = (N + h) * np.cos(lat) * np.cos(lon)
    y = (N + h) * np.cos(lat) * np.sin(lon)
    z = (N1 + h) * np.sin(lat)

    x0 = np.column_stack((x, y, z))
    return x0


def geo2sky( geo0, geoS ):
    """Converts observer and object geodetic coordinates to azimuth and elevation 
    from observer on Earth.

    Parameters
    ----------
    geo0 : ndarray of shape (N,3)
        geodetic coordinates of observer
    geoS : ndarray of shape (N,3)
        geodetic coordinates of object

    Returns
    -------
    output : ndarray of shape(N,3)
        sky coordinates

    Notes
    -----
    """
    x0 = geo2ecef(geo0)
    xS = geo2ecef(geoS)

    sky = ecef2sky(x0, xS)
    return sky


