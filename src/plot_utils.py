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
from projection_utils import grid_mesh_from_centers, grid_centers

def plot_line(ax, p0, u, tau):
    """Plots a line given an origin `p0`, a unit direction vector
    `u`, and `tau` which defines where the endpoints lie for the
    line segment we wish to draw.

    Parameters
    ----------
    ax : 
        figure axes object
    p0 : (3,) ndarray
        line origin
    u : (3,) ndarray
        unit vector of the line direction
    tau : real scalar
        length of line to draw

    Returns
    -------

    Notes
    -----
    """
    tau = np.array([0, tau])
    line = p0 + (u[None,:] * tau[:,None]).squeeze()
    ax.plot(line[:,0], line[:,1], line[:,2], 'g')


def plot_projmtx(ax, xs, ys, zs, projmtx, intersections=None, lines=None, line_tau=2e6, plot_mesh=False, geodetic=False):
    """Plots a visual representation of a projection matrix and its values. This is
    useful for verification of projection matrix's correctness. This function assumes
    a grid generated using axes centers.

    Parameters
    ----------
    ax : 
        figure axes object
    xs : (L,) ndarray
        x values of grid centers
    ys : (M,) ndarray
        y values of grid centers
    zs : (N,) ndarray
        z values of grid centers
    projmtx : (P,Q) ndarray
        The projection matrix to visualize.
        The visualization Behavior is to sum along each column (voxel), normalize, and use
        as component value for plotting color.
    intersections : (R,3) ndarray, optional
        Points where lines intersected grid boundaries. Include this if you want to plot them.
    lines : (P,2,3) ndarray, optional
        Array of pairs of points that define lines. Second dimension is line origin;
        last dimension is line unit direction vector.
    line_tau : real scalar, optional
        Length of the lines, if lines are specified.
    plot_mesh : bool, optional
        Whether or not to plot the mesh that surrounds the grid centers.
    geodetic : bool, optional
        When true, performs necessary geodeteic to ECEF coordinate conversion, treating
        xs, ys, zs as latitude, longitude, altitude respectively

    Returns
    -------
    Notes
    -----
    """
    n_lines, N = projmtx.shape
    
    if np.any(intersections):
        if len(intersections) > 0:
            ax.scatter3D(intersections[:,0], intersections[:,1], intersections[:,2])

    colors = np.sum(projmtx, axis=0)
    colors = colors.reshape((N, 1)).repeat(4, axis=1) / np.max(colors)
    colors[:,0] = 0.
    colors[:,3] = .75

    x, y, z = np.meshgrid(xs, ys, zs)
    pnts = np.concatenate((x[:,:,:,None], y[:,:,:,None], z[:,:,:,None]), axis=3)
    pnts = pnts.swapaxes(0,1)
    pnts = pnts.reshape((N, 3), order='f')

    centers = projection_utils.grid_centers(xs, ys, zs)
    centers = centers.reshape((N, 3), order='f')
    if geodetic:
        centers = geo2ecef(centers)
    ax.scatter3D(centers[:,0], centers[:,1], centers[:,2], c=colors, s=70)

    if plot_mesh:
        mesh = grid_mesh_from_centers(xs, ys, zs)
        if geodetic:
            shape = mesh.shape
            mesh = geo2ecef(mesh.reshape((shape[0] * shape[1] * shape[2], 3))).reshape(shape)
        ax.scatter3D(mesh[:,:,:,0].flatten(), mesh[:,:,:,1].flatten(), mesh[:,:,:,2].flatten(), c=(1,0,1,.2), s=5)
    
    if np.any(lines):
        # may request different number of lines to plot than indicated by projmtx
        n_lines = lines.shape[0]
        for l in range(n_lines):
            plot_line(ax, lines[l,0,:], lines[l,1,:], line_tau)

