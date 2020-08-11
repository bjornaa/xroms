import numpy as np
import xarray as xr
import xroms
import pytest

# Make a cartesian grid
imax = 8
jmax = 6
xi, eta = np.meshgrid(np.arange(imax), np.arange(jmax))
# Perform a conform (linear) mapping
lon_rho = xi + eta
lat_rho = xi - eta

# Make a small ROMS-type grid
A = xr.Dataset(
    data_vars={
        "lon_rho": (("eta_rho", "xi_rho"), lon_rho),
        "lat_rho": (("eta_rho", "xi_rho"), lat_rho),
    }
)


def test_xy2ll():
    x = np.array([2.5, 3.14, 5.0])
    y = np.array([4.3, 4.2, 4.09])
    lon, lat = xroms.xy2ll(A, x, y)
    assert np.allclose(lon, x + y)
    assert np.allclose(lat, x - y)


def test_ll2xy():
    lon = np.array([0.6, 1.23, 2.0, 8.4])
    lat = np.array([-0.5, 0.5, 1.0, 3.14])
    x, y = xroms.ll2xy(A, lon, lat)
    assert np.allclose(x, 0.5 * (lon + lat))
    assert np.allclose(y, 0.5 * (lon - lat))
