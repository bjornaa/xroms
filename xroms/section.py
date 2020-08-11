import numpy as np
import xarray as xr
from xgcm import Grid
#import xroms

def section(ds, X, Y):
    """Create a section Dataset

    Arguments:
    ds: xroms Dataset
    X:  Seguence of xi_rho coordinates
    Y: Sequence of eta_rho coordinates

    The X and Y sequence should have the same length and gives the vertices
    (section cell boundaries) of the section. Most variables lives at the
    center points of the section cells,

    """

    # The center points
    X_c = xr.DataArray(0.5 * (X[:-1] + X[1:]), dims=["distance"])
    Y_c = xr.DataArray(0.5 * (Y[:-1] + Y[1:]), dims=["distance"])

    # Interpolate U and V to grid centers
    # U = grid.interp(A.u, 'X')
    # V = grid.interp(A.v, 'Y')

    # Interpolate Dataset to the section
    # offset for xi_u and eta_v to handle the C-grid
    dss = ds.interp(xi_rho=X_c, eta_rho=Y_c, xi_u=X_c - 0.5, eta_v=Y_c - 0.5)
    dss = dss.drop_vars(['xi_u', 'eta_v'])

    # Length of section segments
    dx = dss.dx * np.diff(X)
    dy = dss.dy * np.diff(Y)
    ddistance = np.sqrt(dx ** 2 + dy ** 2)
    dss["ddistance"] = ddistance

    # Normal component of current
    dss["u_normal"] = (-dss.u * dy + dss.v * dx) / ddistance
    dss["u_tangent"] = (dss.u * dx + dss.v * dy) / ddistance
    # T

    # Cumulative distance along section
    distance = np.concatenate(([0], np.cumsum(ddistance)))  # Unit = m
    distance_c = 0.5 * (distance[:-1] + distance[1:])

    coords = {
        "distance": (["distance",], distance_c),
        "distance_b": (["distance_b",], distance),
    }
    dss = dss.assign_coords(coords)

    # Add center and boundary positions in grid coordinates
    dss["xi_c"] = X_c
    dss["eta_c"] = Y_c
    dss["xi_b"] = xr.DataArray(X, dims="distance_b")
    dss["eta_b"] = xr.DataArray(Y, dims="distance_b")

    # Preliminary Grid
    sgrid0 = Grid(
        dss,
        coords={
            "L": {"center": "distance", "outer": "distance_b"},
            "Z": {"center": "s_rho", "outer": "s_w"},
        },
    )

    #dss["dz"] = sgrid0.diff(dss.z_w, "Z")
    dss["dArea"] = dss.ddistance * dss.dz

    metrics = {
        ("L",): ["ddistance"],
        ("Z",): ["dz"],
        ("L", "Z"): ["dArea"],
    }

    sgrid = Grid(
        dss,
        coords={
            "L": {"center": "distance", "outer": "distance_b"},
            "Z": {"center": "s_rho", "outer": "s_w"},
        },
        metrics=metrics,
        periodic=False,
    )

    return dss, sgrid
