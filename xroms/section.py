import numpy as np
import xarray as xr
from xgcm import Grid

# import xroms


def section(ds, X, Y):
    """Create a ROMS section Dataset with xgcm grid

    Inputs:
    ds: xroms Dataset
    X:  Seguence of xi_rho coordinates
    Y:  Sequence of eta_rho coordinates

    Outputs:
    dss:  ROMS section dataset
    sgrid: xgcm grid object

    The X and Y sequence should have the same length and gives the vertices
    (section cell boundaries) of the section. Most variables lives at the
    center points of the section cells.

    """

    # The center points
    X_c = xr.DataArray(0.5 * (X[:-1] + X[1:]), dims=["distance"])
    Y_c = xr.DataArray(0.5 * (Y[:-1] + Y[1:]), dims=["distance"])

    # Interpolate Dataset to the section
    # offset for xi_u and eta_v to handle the C-grid
    dss = ds.interp(xi_rho=X_c, eta_rho=Y_c, xi_u=X_c - 0.5, eta_v=Y_c - 0.5)

    # Remove coordinates and data variables at u, v and psi-points
    dss = dss.drop_vars(
        [
            v
            for v in list(dss.coords) + list(dss.data_vars)
            if v.endswith("_u") or v.endswith("_v") or v.endswith("_psi")
        ]
    )

    # Length of section segments
    dx = dss.dx * np.diff(X)
    dy = dss.dy * np.diff(Y)
    ddistance = np.sqrt(dx ** 2 + dy ** 2)
    dss["ddistance"] = ddistance

    # Normal and tangential component of current
    dss["u_normal"] = (-dss.u * dy + dss.v * dx) / ddistance
    dss["u_tangent"] = (dss.u * dx + dss.v * dy) / ddistance

    # Set cumulative distance along section as coordinates
    distance_b = np.concatenate(([0], np.cumsum(ddistance)))
    distance = 0.5 * (distance_b[:-1] + distance_b[1:])
    dss.coords["distance"] = distance
    dss.coords["distance_b"] = distance_b

    # Add center and boundary positions in grid coordinates
    dss["xi"] = X_c
    dss["eta"] = Y_c
    dss["xi_b"] = xr.DataArray(X, dims="distance_b")
    dss["eta_b"] = xr.DataArray(Y, dims="distance_b")

    # Make xgcm grid object
    dss["dArea"] = dss.ddistance * dss.dz
    sgrid = Grid(
        dss,
        coords={
            "L": {"center": "distance", "outer": "distance_b"},
            "Z": {"center": "s_rho", "outer": "s_w"},
        },
        metrics={("L",): ["ddistance"], ("Z",): ["dz"], ("L", "Z"): ["dArea"],},
        periodic=False,
    )

    return dss, sgrid
