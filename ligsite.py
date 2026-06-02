# -*- coding: utf-8 -*-
"""
function definitions for LigSite and cavity detection
"""
import itertools
import math

import numpy as np
from numpy.lib.stride_tricks import as_strided

from .pdb_structure import PDBAtom, PDBStructure

# 32-bit precision provides speed-up in HP annotation
FP_DTYPE = np.float32  

def setup_grid(coords, cushion, d_grid, init, dtype):
    """
    Setup and initialize grid using atom coordinates
    """
    # min and max in cartesian coordinates
    min_coords = coords.min(axis=0)
    max_coords = coords.max(axis=0)
    # min and max in grid coordinates
    min_grid = np.floor((min_coords - cushion) / d_grid)
    max_grid = np.ceil((max_coords + cushion) / d_grid)

    origin = min_grid * d_grid
    extent = (max_grid - min_grid + 1).astype(int)

    return Grid(origin=origin, extent=extent, d=d_grid, init=init, dtype=dtype)


def mask_grid(
    grid, coords, radii, radius_factor, probe_radius, softness, protein_flag, soft_flag
):
    """
    Mask grid using atom coordinates and radii
    """
    # atom coordinates in grid units
    grid_coords = (coords - grid.origin) / grid.d
    # atom radii in grid units
    grid_radii = (radii * radius_factor + probe_radius) / grid.d
    r1sq = grid_radii**2  # squared outer, soft radius
    r2sq = (grid_radii - softness) ** 2  # squared inner, hard radius

    # origin and space-diagonal coordinates of the sub-grids around the atoms
    sg_start = np.clip(
        np.floor(grid_coords - grid_radii.reshape(-1, 1)).astype(int),
        (0, 0, 0),
        grid.extent,
    )
    sg_end = np.clip(
        np.ceil(grid_coords + grid_radii.reshape(-1, 1)).astype(int) + 1,
        (0, 0, 0),
        grid.extent,
    )

    for i in range(grid_coords.shape[0]):  # loop over all atom coordinates
        x_start, y_start, z_start = sg_start[i]
        x_end, y_end, z_end = sg_end[i]
        sub_grid = grid.get_subgrid(x_start, y_start, z_start, x_end, y_end, z_end)

        x, y, z = np.ogrid[x_start:x_end, y_start:y_end, z_start:z_end]
        dist = (
            (x - grid_coords[i, 0]) ** 2
            + (y - grid_coords[i, 1]) ** 2
            + (z - grid_coords[i, 2]) ** 2
        )

        # grid points within the inner, hard radius
        mask_hard = dist < r2sq[i]
        sub_grid[mask_hard] = protein_flag

        # grid points between inner and outer radius
        mask_soft = (dist < r1sq[i]) & (sub_grid != protein_flag)
        sub_grid[mask_soft] = soft_flag


def do_ligsite(masked_grid, protein_flag):
    """
    Run ligsite algorithm.
    Calculate LigSite scores for a masked grid
    """
    grid = masked_grid.get_grid()
    nx = masked_grid.nx
    ny = masked_grid.ny
    nz = masked_grid.nz
    offset = masked_grid.offset

    # analyze along the x-direction
    for iy in range(ny):
        for iz in range(nz):
            line = grid[:, iy, iz]
            _analyze_line(line, protein_flag)

    # analyze along the y-direction
    for ix in range(nx):
        for iz in range(nz):
            line = grid[ix, :, iz]
            _analyze_line(line, protein_flag)

    # analyze along the z-direction
    for ix in range(nx):
        for iy in range(ny):
            line = grid[ix, iy, :]
            _analyze_line(line, protein_flag)

    # space-diagonal 1 (1,1,1)
    off = offset(1, 1, 1)
    for ix in range(nx):
        for iy in range(ny):
            length = min(nx - ix, ny - iy, nz)
            line = as_strided(grid[ix, iy, :], shape=(length,), strides=(off,))
            _analyze_line(line, protein_flag)

    for ix in range(nx):
        for iz in range(1, nz):
            length = min(nx - ix, ny, nz - iz)
            line = as_strided(grid[ix, :, iz], shape=(length,), strides=(off,))
            _analyze_line(line, protein_flag)

    for iy in range(1, ny):
        for iz in range(1, nz):
            length = min(nx, ny - iy, nz - iz)
            line = as_strided(grid[:, iy, iz], shape=(length,), strides=(off,))
            _analyze_line(line, protein_flag)

    # space-diagonal 2 (-1,1,1)
    off = offset(-1, 1, 1)
    for ix in range(nx):
        for iy in range(ny):
            length = min(ix + 1, ny - iy, nz)
            line = as_strided(grid[ix, iy, :], shape=(length,), strides=(off,))
            _analyze_line(line, protein_flag)

    for ix in range(nx):
        for iz in range(1, nz):
            length = min(ix + 1, ny, nz - iz)
            line = as_strided(grid[ix, :, iz], shape=(length,), strides=(off,))
            _analyze_line(line, protein_flag)

    for iy in range(1, ny):
        for iz in range(1, nz):
            length = min(nx, ny - iy, nz - iz)
            line = as_strided(grid[::-1, iy, iz], shape=(length,), strides=(off,))
            _analyze_line(line, protein_flag)

    # space-diagonal 3 (1,-1,1)
    off = offset(1, -1, 1)
    for ix in range(nx):
        for iy in range(ny):
            length = min(nx - ix, iy + 1, nz)
            line = as_strided(grid[ix, iy, :], shape=(length,), strides=(off,))
            _analyze_line(line, protein_flag)

    for ix in range(nx):
        for iz in range(1, nz):
            length = min(nx - ix, ny, nz - iz)
            line = as_strided(grid[ix, ::-1, iz], shape=(length,), strides=(off,))
            _analyze_line(line, protein_flag)

    for iy in range(ny - 1):
        for iz in range(1, nz):
            length = min(nx, iy + 1, nz - iz)
            line = as_strided(grid[:, iy, iz], shape=(length,), strides=(off,))
            _analyze_line(line, protein_flag)

    # space-diagonal 4 (-1,-1,1) equiv. to (1,1,-1)
    off = offset(-1, -1, 1)
    for ix in range(nx):
        for iy in range(ny):
            length = min(ix + 1, iy + 1, nz)
            line = as_strided(grid[ix, iy, :], shape=(length,), strides=(off,))
            _analyze_line(line, protein_flag)

    for ix in range(nx):
        for iz in range(1, nz):
            length = min(ix + 1, ny, nz - iz)
            line = as_strided(grid[ix, ::-1, iz], shape=(length,), strides=(off,))
            _analyze_line(line, protein_flag)

    for iy in range(ny - 1):
        for iz in range(1, nz):
            length = min(nx, iy + 1, nz - iz)
            line = as_strided(grid[::-1, iy, iz], shape=(length,), strides=(off,))
            _analyze_line(line, protein_flag)


def _analyze_line(line, protein_flag):
    """
    Analyze line in grid.
    Analyze a line from the masked grid and increment
    all ligsite scores for points between two protein grid points

    assumes "line" to be an integer or float numpy array

    protein_flag ... value to denote protein grid points
    """
    # indices of the protein grid points
    ind = np.argwhere(line == protein_flag).flatten()
    if ind.shape[0] > 0:  # if there are any protein points in the line
        start = int(ind[0]) + 1
        end = int(ind[-1])
        if (end - start) > 0:
            # part of the line between the first and the last protein grid point
            tmp = line[start:end]
            # increment all non-protein grid points in that part
            tmp[tmp > protein_flag] += 1


def find_cavities(ligsite_grid, cutoff, gap, min_size, max_size, radius=1.4, vol_res=3.):
    """
    Find cavities in a LigSite grid using sets
    """
    cavities = []  # list of cavity objects to be returned

    d_grid = ligsite_grid.d  # get grid spacing
    origin = ligsite_grid.origin  # get origin of the grid in cartesian coordinates
    grid = ligsite_grid.get_grid()  # get underlying grid array

    # array of indices grid points above threshold
    indices = np.argwhere(grid >= cutoff)

    # set of those indices
    ind_set = {tuple(row) for row in indices}

    # get relative indices around (but not including) the origin
    environment = np.array(
        [
            xyz
            for xyz in itertools.product(range(-1 - gap, 2 + gap), repeat=3)
            if xyz != (0, 0, 0)
        ]
    ).reshape(1, -1, 3)

    while ind_set:
        points = []  # indices of a new cavity

        seeds = [
            ind_set.pop(),
        ]
        while seeds:
            # add the point(s) to the growing cavity
            points.extend(seeds)

            # create a set of neighbours
            neighbour_coords = (
                np.array(seeds).reshape(-1, 1, 3) + environment
            ).reshape(-1, 3)
            neighbours = {tuple(row) for row in neighbour_coords}

            # set of neighbours that qualify as cavity points
            in_cavity = ind_set.intersection(neighbours)

            # remove them from the set of indices
            ind_set.difference_update(in_cavity)

            # these points are the 'seeds' for the next iteration
            seeds = list(in_cavity)

        # now 'points' contains a complete cavity
        if min_size <= len(points) <= max_size:
            cavities.append(_construct_cavity(points, grid, d_grid, origin, radius, vol_res))

    # sort cavities according to their size in descending order and generate PDBStructure objects
    cavities.sort(key=lambda x: x.size, reverse=True)
    _gen_cav_pdb(cavities)

    return cavities


def find_cavities_dist(ligsite_grid, cutoff, max_dist, min_size, max_size, radius=1.4, vol_res=3.):
    """
    Find cavities in a LigSite grid, using a distance cutoff
    """
    cavities = []  # list of cavity objects to be returned

    d_grid = ligsite_grid.d  # get grid spacing
    origin = ligsite_grid.origin  # get origin of the grid in cartesian coordinates
    grid = ligsite_grid.get_grid()  # get underlying grid array

    # array of indices with LigSite scores above cutoff
    indices = np.argwhere(grid >= cutoff)
    n_ind = indices.shape[0]
    # flag for cavity search, True ... grid point not yet analyzed
    flag = np.ones(n_ind, dtype=bool)

    # pairwise squared distances in grid coordinates
    norm = (indices * indices).sum(axis=1).reshape(1, -1)
    dist = (norm + norm.T) - 2 * np.dot(indices, indices.T)

    # squared distance cutoff in grid coordinates
    sq_max = max_dist**2 / d_grid**2

    for i in range(n_ind):  # step through the array by lines
        if flag[i]:  # if we have not yet dealt with this grid point
            seeds = [
                i,
            ]
            flag[i] = False
            points = []

            while seeds:
                tmp_list = []
                for j in seeds:
                    ind_j = tuple(indices[j])
                    points.append(ind_j)
                    neigh = np.argwhere((dist[j] <= sq_max) & flag)[:, 0]
                    flag[neigh] = False
                    tmp_list.extend(neigh)
                seeds = tmp_list

            # now 'points' contains a complete cavity
            if min_size <= len(points) <= max_size:
                cavities.append(_construct_cavity(points, grid, d_grid, origin, radius, vol_res))

    # sort cavities according to their size in descending order and generate PDBStructure objects
    cavities.sort(key=lambda x: x.size, reverse=True)
    _gen_cav_pdb(cavities)

    return cavities


def _construct_cavity(points, grid, d_grid, origin, radius=1.4, vol_res=3.):
    """
    construct a cavity from the assembled grid points
    """
    points = np.array(points)  # grid indices
    coords = points.astype(FP_DTYPE) * d_grid + origin  # 3D coordinates
    # create new cavity object
    cav = Cavity(points, coords, d_grid)
    # estimate cavity volume
    cav.volume = estimate_volume(coords, radius, d_grid / vol_res)
    # add LigSite scores to annotations dictionary
    cav.annotations["LIG"] = grid[points[:, 0], points[:, 1], points[:, 2]]

    return cav


def _gen_cav_pdb(cavities):
    """
    generate PDBStructure objects for the cavities
    """
    n_at = 1
    for n_cav, cav in enumerate(cavities):
        cav.pdb = PDBStructure()
        if n_cav >= 9999:  # just in case that there are more than 9999 cavities
            n_cav -= 9999

        for n_point, xyz in enumerate(cav.coords):
            # generate a dummy atom
            atom = PDBAtom(
                f"HETATM{n_at:5d}  XP  CAV X   1       3.500  53.900  22.400  1.00  7.00"
            )
            # set the actual parameters
            atom.x, atom.y, atom.z = xyz
            atom.residue_number = n_cav + 1
            # store the LigSite score in the B-factor column
            atom.bfactor = cav.annotations["LIG"][n_point]
            # set the element to "X" (necessary for reading with Yasara)
            atom.element = "X"
            atom.keep_old_bfactor = atom.bfactor
            atom.add_prop_dic("LIG", atom.bfactor)
            cav.pdb.atom.append(atom)
            n_at += 1
            if n_at > 99999:
                n_at -= 99999


class Cavity:
    """
    definition of a cavity object
    """

    def __init__(self, points, coords, d_grid):
        self.size = len(points)
        self.volume = 0.
        self.d_grid = d_grid
        self.points = points  # cavity points in grid coordinates
        self.coords = coords  # cartesian cavity coordinates

        self.annotations = dict()  # cavity point annotations
        self.pdb = None  # cavity as a PDBStructure object

        self.shaped = False  # indicate, whether the cavity was shaped

    def __str__(self):
        return self.get_pdbstr()

    def get_pdbstr(self, annotation=None):
        header = self._get_pdb_header()
        if annotation is None:  # use the standard B-factor column
            return header + self.pdb.get_pdbstr()
        else:  # replace the B-factor column with annotation value
            return header + self.pdb.get_pdbstr("prop_dic", annotation)

    def _get_pdb_header(self):
        header = [
            "REMARK",
            f"REMARK number of grid_points:{self.size:7d}",
            f"REMARK approximate cavity volume:{self.volume:7.0f} Angs.**3",
        ]
        return "\n".join(header) + "\n"

    def write(self, file_obj, file_format="pdb", annotation=None):
        if file_format == "pdb":
            file_obj.write(self.get_pdbstr(annotation))
        elif file_format == "csv":
            scores = [
                self.annotations[key] for key in annotation if key in self.annotations
            ]
            for i, (x, y, z) in enumerate(self.coords):
                data = [x, y, z]
                data.extend([score[i] for score in scores])
                line = ", ".join([f"{item:f}" for item in data])
                file_obj.write(f"{line}\n")


    def shape(self, shaping_obj, d_max):
        """
        Shape a cavity based on points in 'shaping_obj' and a maximum distance of 'd_max'.
        The function modifies the original cavity!

        :param d_max: maximum distance of a cavity point from any point in 'shaping_obj' to be retained
        :param shaping_obj: Nx3 numpy array, coordinates of 'shaping_obj'
        :return: number of retained cavity points
        """
        flags = crop_pointcloud(self.coords, shaping_obj.astype(FP_DTYPE), d_max)

        num_retained = np.sum(flags)  # number of retained points
        self.size = num_retained

        self.points = self.points[flags]
        self.coords = self.coords[flags]

        for key, value in self.annotations.items():
            self.annotations[key] = self.annotations[key][flags]

        atom = self.pdb.atom
        self.pdb.atom = [atom[i] for i, flag in enumerate(flags) if flag]

        self.shaped = True

        return num_retained


def calc_dist(cloud_1, cloud_2, do_sqrt=False):
    """
    calculate the distance matrix for two point clouds
    :param cloud_1: N x k numpy array
    :param cloud_2: M x k numpy array
           (k ... dimension of the clouds)
    :param do_sqrt: if True return the square root
    :return: N x M numpy array with euclidean distances
    """
    norm_1 = (cloud_1 * cloud_1).sum(axis=1).reshape(-1, 1)
    norm_2 = (cloud_2 * cloud_2).sum(axis=1).reshape(1, -1)
    dist = (norm_1 + norm_2) - 2.0 * np.dot(cloud_1, cloud_2.T)

    if do_sqrt:
        return np.sqrt(dist, out=dist)
    else:
        return dist


def crop_pointcloud(cloud_1, cloud_2, d_max=2.0):
    """
    crop one point cloud based on the points of another point cloud,
    all points in 'big', which are closer than 'd_max' to any point in 'small',
    are retained

    :param cloud_1: N x k numpy array
    :param cloud_2: M x k numpy array
                  (k ... dimension of the clouds)
    :param d_max: maximum distance for a point in 'cloud_1' to be retained
    :return: boolean array with shape (N,)
             True ... point is retained
             False ... point is deleted
    """
    diff = calc_dist(cloud_1, cloud_2)  # calculate all pairwise distances
    min_dist = diff.min(axis=1)  # get the minimum distance

    return min_dist <= d_max**2


def estimate_volume(points, radius: float, d_grid: float) -> float:
    """
    estimate the volume of a point cloud
    point ... 3D coordinates of the cloud
    radius ... radius of the sphere placed at each point
    d_grid ... grid spacing

    The object is placed on a regular grid; voxels within the spheres placed at the points are
    counted; the estimated volume is n_voxels * d_grid**3
    """
    radii = np.ones(points.shape[0], dtype=FP_DTYPE) * radius
    grid = setup_grid(points, radius + d_grid, d_grid, 0, np.int8)
    mask_grid(grid, points, radii, 1.0, 0.0, 0.0, 1, 0)
    return np.sum(grid.get_grid()) * d_grid ** 3


class Grid:
    """
    Class for a grid object.

    Extension (# of grid points) given by a tuple of
    length 3. Default initialization with value "0".

    USAGE: Grid(origin, extend, init=0)
              extend: tuple:(x,y,z)  grid extension number of grid points
                              in (x , y , z)
              init:        :initialization value of grid point (standard=0)
    """

    def __init__(
        self, origin=(0.0, 0.0, 0.0), extent=(50, 50, 50), d=0.7, init=0, dtype=np.int32
    ):
        self.nx, self.ny, self.nz = extent
        self.extent = extent
        self.origin = np.array(origin, dtype=FP_DTYPE)
        self.d = d
        self._grid = np.zeros(extent, dtype=dtype)
        if init != 0:
            self._grid = init

    def get_subgrid(self, x0, y0, z0, x1, y1, z1):
        """
        Return a view of a sub-grid of the internal np.array
        :param x0: start x-index
        :param y0: start y-index
        :param z0: start z-index
        :param x1: end x-index
        :param y1: end y-index
        :param z1: end z-index
        :return: reference to the sub-grid
        """
        return self._grid[x0:x1, y0:y1, z0:z1]

    def get_grid(self):
        """
        Return a view of the complete grid object
        :return: reference to the internal np.array
        """
        return self._grid

    def get_value(self, indices):
        """
        Return value of grid point
        :param indices: tuple of indices (i, j, k); cannot be a np.array
        :return: value at grid point
        """
        return self._grid[indices]

    def set_value(self, indices, value):
        """
        Set value of a grid point
        :param indices: tuple of indices (i, j, k); cannot be a np.array
        :param value: new value
        """
        self._grid[indices] = value

    def is_valid_index(self, index):
        """
        Check whether 'index' contains valid indices of the grid.
        :param index: tuple of indices (i, j, k); cannot be an np.array
        :return: True/False
        """
        x, y, z = index
        return 0 <= x < self.nx and 0 <= y < self.ny and 0 <= z < self.nz

    def coordinates(self, index):
        """
        Return the cartesian coordinates of a grid point
        :param index: indices (i, j, k), tuple, list, np.array
        :return: np.array with cartesian coordinates
        """
        return self.origin + np.array(index, dtype=FP_DTYPE) * self.d

    def offset(self, dx, dy, dz):
        """
        Return the offset in bytes
        :param dx: increment in x
        :param dy: increment in y
        :param dz: increment in z
        :return: offset, to be used for as_strided
        """
        return ((dx * self.ny + dy) * self.nz + dz) * self._grid.itemsize
