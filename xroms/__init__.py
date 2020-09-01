try:
    from ._version import __version__
except ImportError:
    __version__ = "unknown"

from .xroms import (roms_dataset,
                    open_netcdf,
                    open_zarr,
                    hgrad,
                    relative_vorticity,
                    ertel)
from .roms_seawater import density, buoyancy
from .utilities import to_rho, to_psi, xisoslice, sel2d, argsel2d
# from .interp import setup, ll2xe, calc_zslices, interp
import xroms.interp
