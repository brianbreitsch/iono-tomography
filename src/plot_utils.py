#
# plot_utils.py
#
# author: Brian Breitsch
#
# This file contains functions useful for plotting verification of the
# ionosphere tomography routines.
#
import numpy as np
from coordinate_utils import geo2ecef
from projection_utils import grid_mesh_from_centers

def plot_line(ax, p0, u, tau):
    '''
    Plots a line given an origin `p0`, a unit direction vector
    `u`, and `tau` which defines where the endpoints lie for the
    line segment we wish to draw.
    '''
    tau = np.array([0, tau])
    line = p0 + (u[None,:] * tau[:,None]).squeeze()
    ax.plot(line[:,0], line[:,1], line[:,2], 'g')


def plot_projmtx(ax, lats, lons, alts, projmtx, intersections=None, lines=None, line_tau=2e6, plot_mesh=False):
    '''
    Plots a visual representation of a projection matrix and its values. This is
    useful for verification of projection matrix's correctness. The function 
    parameters all coorespond to the inputs and outputs to the function
    `geodetic_grid_projection_matrix`.
    '''
    n_lines, N = projmtx.shape
    
    if len(intersections) > 0:
        ax.scatter3D(intersections[:,0], intersections[:,1], intersections[:,2])

    colors = np.sum(projmtx, axis=0)
    colors = colors.reshape((N, 1)).repeat(4, axis=1) / np.max(colors)
    colors[:,0] = 0.
    colors[:,3] = .75

    x, y, z = np.meshgrid(lats, lons, alts)
    pnts = np.concatenate((x[:,:,:,None], y[:,:,:,None], z[:,:,:,None]), axis=3)
    pnts = pnts.swapaxes(0,1)
    pnts = pnts.reshape((N, 3), order='f')
    pnts = geo2ecef(pnts)

    ax.scatter3D(pnts[:,0], pnts[:,1], pnts[:,2], c=colors, s=70)

    if plot_mesh:
        mesh = grid_mesh_from_centers(lats, lons, alts)
        shape = mesh.shape
        mesh = geo2ecef(mesh.reshape((shape[0] * shape[1] * shape[2], 3))).reshape(shape)
        ax.scatter3D(mesh[:,:,:,0].flatten(), mesh[:,:,:,1].flatten(), mesh[:,:,:,2].flatten(), c=(1,0,1,.2), s=5)

    if lines:
        for line in lines:
            plot_line(ax, line[0], line[1], line_tau)

    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])

