import requests
from BeautifulSoup import BeautifulSoup
import numpy as np
import scipy as sp
from scipy import interpolate

import imp
projection_utils = imp.load_source('projection_utils', '../src/projection_utils.py')
coordinate_utils = imp.load_source('coordinate_utils', '../src/coordinate_utils.py')


class IRIFetcher:
    """This class outlines objects that can make url queries to an online version
    of the International Reference Ionosphere (IRI) model. Given a set of
    latitudes and longitudes, an IRIFetcher object will generate a
    3-dimentional numpy grid.

    Notes
    -----

    """
    
    default_params = {
                'model' : 'iri_2012',  'year' : 2000,         'month' : 1,
                'day' : 1,             'time_flag' : 0,       'hour' : 1,
                'geo_flag' : 0,        'latitude' : 0,        'longitude' : 0,
                'height' : 100,        'profile' : 1,         'start' : 60,
                'stop' : 1500,         'step' : 40,           'format' : 0,
                'sun_n' : '',          'ion_n' : '',          'radio_f' : '',
                'radio_f81' : '',      'htec_max' : '',       'ne_top' : 0,
                'imap' : 0,            'ffof2' : 0,           'ib0' : 0,
                'probab' : 0,          'ffoE' : 0,            'dreg' : 0,
                'tset' : 0,            'icomp' : 0,           'nmf2' : 0,
                'hmf2' : 0,            'user_nme' : 0,        'user_hme' : 0,
                'format' : 0,          'linestyle' : 'solid', 'charsize' : '',
                'symbol' : 2,          'symsize' : '',        'yscale' : 'Linear',
                'xscale' : 'Linear',   'imagex' : 640,        'imagey' : 480,
            }
    
    default_outputs = set([6, 17])
    
    default_url = 'http://omniweb.gsfc.nasa.gov/cgi/vitmo/vitmo_model.cgi'
    
    def __init__(self, params={}, outputs=[], url=None):
        self.params = IRIFetcher.default_params
        self.params.update(params)
        self.outputs = IRIFetcher.default_outputs
        self.outputs.update(outputs)
        self.url = IRIFetcher.default_url if url is None else url
        
    def fetch(self):
        pairs = []
        for key, val in self.params.items():
            pairs.append((key, val))
        for val in self.outputs:
            pairs.append(('vars', val))
        resp = requests.post(self.url, pairs)
        parsed = BeautifulSoup(resp.text)
        pre = parsed.find('pre').text
        lines = pre.split('\n')
        data = []
        for line in lines[6:]:
            data.append([float(f) for f in line.split()])
        return data

    def create_tec_image(self, lats=[], lons=[]):
        
        if not np.any(lats):
            lats = [self.params['latitude']]
        if not np.any(lons):
            lons = [self.params['longitude']]
        assert(-90 <= np.min(lats) <= np.max(lats) <= 90)
        assert(0 <= np.min(lons) <= np.max(lons) <= 360)
        
        data0, data1 = [], []
        for lat in lats:
            for lon in lons:
                self.params['latitude'], self.params['longitude'] = lat, lon
                data1.append( self.fetch() )
            data0.append(data1)
            data1 = []
        return np.array(data0)


def gaussian_blob_from_centers(centers, pos=(0.,0.,0.), sig=(1.,1.,1.)):
    """Given an ndarray of shape (N,3), which contains the center points of voxels,
    returns an ndarray of shape (N,) whose values correspond to a gaussian function
    evaluated at each center point.

    Parameters
    ----------
    centers : ndarray of shape (N,3)
        3d centers of voxels
    Returns
    -------
    output : ndarray of shape (N,)
        The blob values

    Notes
    -----
    """
    N, three = centers.shape
    blob = np.zeros((N,))
    pos = np.array(pos)
    sig = np.array(sig)
    for i in range(N):
        blob[i] = np.exp(-np.sum((centers[i,:] - pos)**2 / sig**2 / 2.))
    if np.max(blob) != 0:
        blob = blob / np.max(blob)
    return blob


def gaussian_blob(xs, ys, zs=None, pos=(0.,0.,0.), sig=(1.,1.,1.)):
    """Given ndarrays that describe voxels centers along each axis, 
    returns an ndarray of shape (L,M,N) whose values correspond to 
    a gaussian function evaluated at each center point.

    If geodetic is true, treats xs, ys, zs as lats, lons, alts respectively and
    treats pos and sig as geodetic coordinates.
    """
    if not np.any(zs):
        zs = np.zeros(1)
        assert(len(pos) == 2 and len(sig) == 2)
        pos += (0.,)
        sig += (1.,)
    assert(len(pos) == 3 and len(sig) == 3)
    nx, ny, nz = len(xs), len(ys), len(zs)
    centers = projection_utils.grid_centers(xs, ys, zs)
    centers = centers.reshape((nx * ny * nz, 3))
    centers = projection_utils.grid_centers(xs, ys, zs)
    blob = gaussian_blob_from_centers(centers, pos, sig)
    shape = (nx, ny, nz)
    return blob.reshape(shape)
