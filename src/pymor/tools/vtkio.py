# This file is part of the pyMOR project (http://www.pymor.org).
# Copyright 2013-2016 pyMOR developers and contributors. All rights reserved.
# License: BSD 2-Clause License (http://opensource.org/licenses/BSD-2-Clause)

import numpy as np
import pyvtk
import pprint

from pymor.core.config import config
from pymor.grids import referenceelements
from pymor.grids.constructions import flatten_grid

if config.HAVE_PYVTK:
    from evtk.hl import _addDataToFile, _appendDataToFile
    from evtk.vtk import VtkGroup, VtkFile, VtkUnstructuredGrid, VtkTriangle, VtkQuad


def _write_vtu_series(grid, coordinates, connectivity, data, filename_base, last_step, is_cell_data):
    steps = last_step + 1 if last_step is not None else len(data)
    fn_tpl = "{}_{:08d}"

    ref = grid.reference_element
    if ref is referenceelements.triangle:
        vtk_grid = pyvtk.UnstructuredGrid(coordinates, triangle=connectivity)
    elif ref is referenceelements.square:
        vtk_grid = pyvtk.UnstructuredGrid(coordinates, quad=connectivity)
    else:
        raise NotImplementedError("vtk output only available for grids with triangle or rectangle reference elments")

    group = VtkGroup(filename_base)
    for i in range(steps):
        fn = fn_tpl.format(filename_base, i)
        raw_data = data[i, :]
        print('{}_s{}_raw{}'.format(fn, i, raw_data.shape))
        pprint.pprint(raw_data)

        if is_cell_data:
            vtk_data = pyvtk.CellData(pyvtk.Scalars(raw_data, name='CellData'))
        else:
            vtk_data = pyvtk.PointData(pyvtk.Scalars(raw_data, name='PointData'))

        vtk = pyvtk.VtkData(vtk_grid, 'example')
        vtk.tofile(fn)
        #vtk.tofile(fn,'binary')

        group.addFile(filepath=fn, sim_time=i)
    group.save()
    return (fn_tpl.format(filename_base, i) for i in range(steps))


def write_vtk(grid, data, filename_base, codim=2, binary_vtk=True, last_step=None):
    """Output grid-associated data in (legacy) vtk format

    Parameters
    ----------
    grid
        a |Grid| with triangular or rectilinear reference element

    data
        VectorArrayInterface instance with either cell (ie one datapoint per codim 0 entity)
        or vertex (ie one datapoint per codim 2 entity) data in each array element

    filename_base
        common component for output files in timeseries

    last_step
        if set must be <= len(data) to restrict output of timeseries

    :return list of filenames written
    """
    if not config.HAVE_PYVTK:
        raise ImportError('could not import pyevtk')
    if grid.dim != 2:
        raise NotImplementedError
    if codim not in (0, 2):
        raise NotImplementedError

    subentities, coordinates, entity_map = flatten_grid(grid)
    x, y, z = coordinates[:, 0].copy(), coordinates[:, 1].copy(), np.zeros(coordinates[:, 1].size)
    return _write_vtu_series(grid, coordinates=(x, y, z), connectivity=subentities, data=data.data,
                      filename_base=filename_base, last_step=last_step, is_cell_data=(codim == 0))

