import requests
from BeautifulSoup import BeautifulSoup
import numpy as np
from numpy import pi
import scipy as sp
from scipy import interpolate

class IRIFetcher:
    '''This class outlines objects that can make url queries to an online version
    of the International Reference Ionosphere (IRI) model. Given a set of
    latitudes and longitudes, an IRIFetcher object will generate a
    3-dimentional numpy grid.
    '''
    
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

def gaussian_blob(xs, ys, zs=None, pos=(0.,0.,0.), sig=(1.,1.,1.)):
    n_xs, n_ys, n_zs = len(xs), len(ys), len(zs)
    x2vox = 1. if n_xs < 2 else n_xs / (xs[-1] - xs[0])
    y2vox = 1. if n_ys < 2 else n_ys / (ys[-1] - ys[0])
    z2vox = 1. if n_zs < 2 else n_zs / (zs[-1] - zs[0])
    
    i_m = (pos[0] - xs[0]) * x2vox
    j_m = (pos[1] - ys[0]) * y2vox
    k_m = 0. if len(pos) < 3 else (pos[2] - zs[0]) * z2vox

    sig_i = sig[0] * x2vox
    sig_j = sig[1] * y2vox
    sig_k = 1. if len(pos) < 3 else sig[2] * z2vox

    i, j, k = np.arange(n_xs), np.arange(n_ys), np.arange(n_zs)
    ii, jj, kk = np.meshgrid(i,j,k)

    if len(pos) < 3:
        blob = np.exp(-((ii - i_m)**2 / sig_i + (jj - j_m)**2 / sig_j)).T
    else:
        blob = np.exp(-((ii - i_m)**2 / sig_i + (jj - j_m)**2 / sig_j + (kk - k_m)**2 / sig_k)).T
    
    return blob


def gaussian_blob_from_mesh(mesh, pos=(0.,0.,0.), sig=(1.,1.,1.)):
    '''
    Given an ndarray mesh of shape (L,M,N,3), returns an ndarray
    of shape (L,M,N) whose values correspond to a gaussian function
    evaluated at each point in the mesh.
    '''
    shape = mesh.shape
    N, three = shape[0] * shape[1] * shape[2], shape[3]
    mesh = mesh.reshape((N,3))
    assert(three == 3)
    blob = np.zeros((N,3))
    pos = np.array(pos)
    sig = np.array(sig)
    for i in range(N):
        blob[i,:] = 1. / (np.sqrt(2 * np.pi) * sig) * np.exp(-(mesh[i,:] - pos)**2 / sig**2 / 2.)
    blob = blob.reshape(shape)
    blob = np.product(blob, axis=-1)
    blob = blob / np.max(blob)
    return blob
