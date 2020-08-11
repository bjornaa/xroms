import numpy as np
from scipy.interpolate import griddata
import xarray as xr


def xy2ll(A, x, y):
    """Convert from grid coordinates to longitude/latitude"""

    xb, yb = np.broadcast_arrays(x, y)
    x_da = xr.DataArray(xb)
    y_da = xr.DataArray(yb)
    lon = A["lon_rho"].interp(xi_rho=x_da, eta_rho=y_da).values
    lat = A["lat_rho"].interp(xi_rho=x_da, eta_rho=y_da).values

    # Return scalars if both x and y are scalars
    if np.isscalar(x) and np.isscalar(y):
        lon = float(lon)
        lat = float(lat)

    return lon, lat

def ll2xy(A, lon, lat):
    gLon = A["lon_rho"].data.ravel()
    gLat = A["lat_rho"].data.ravel()
    X0 = A["xi_rho"].data
    Y0 = A["eta_rho"].data
    gX, gY = np.meshgrid(X0, Y0)
    gX = gX.ravel()
    gY = gY.ravel()
    x = griddata((gLon, gLat), gX, (lon, lat), "linear")
    y = griddata((gLon, gLat), gY, (lon, lat), "linear")
    return x, y
