import numpy as np
import xarray as xr
import xroms
import pytest


@pytest.fixture
def ds():
    """Make an idealized ROMS-like test dataset"""
    imax, jmax, kmax = 8, 6, 4
    xx, yy = np.meshgrid(np.arange(imax), np.arange(jmax))
    H = yy + 100
    s_w = np.linspace(-1, 0, kmax + 1)
    s_rho = 0.5 * (s_w[:-1] + s_w[1:])
    z_w = s_w[:, None, None] * H[None, :, :]
    z_rho = s_rho[:, None, None] * H[None, :, :]

    # Make xarray Dataset
    Z = ("s_rho",)
    Zw = ("s_w",)
    YX = ("eta_rho", "xi_rho")
    coords = {"z_rho": (Z + YX, z_rho), "z_w": (Zw + YX, z_w)}
    data_vars = {
        "h": (YX, H),
        "dx": (YX, np.ones((jmax, imax))),
        "dy": (YX, np.ones((jmax, imax))),
        "dz": (Z + YX, z_w[1:, :, :] - z_w[:-1, :, :]),
        "u": (
            ("s_rho", "eta_rho", "xi_u"),
            np.multiply.outer(np.ones(kmax), xx[:, :-1]),
        ),
        "v": (
            ("s_rho", "eta_v", "xi_rho"),
            np.multiply.outer(np.ones(kmax), yy[:-1, :]),
        ),
        "temp": (Z + YX, 20.0 + 0.1 * z_rho),
    }

    return xr.Dataset(data_vars=data_vars, coords=coords)


def test_idealized(ds):
    X = np.array([0.8, 1.4, 1.4, 3.0])
    Y = np.array([1.0, 1.0, 1.8, 2.4])
    dss, sgrid = xroms.section(ds, X, Y)

    # Geometry

    assert np.allclose(dss.xi, [1.1, 1.4, 2.2])
    assert np.allclose(dss.eta, [1.0, 1.4, 2.1])
    assert np.allclose(dss.ddistance, [0.6, 0.8, np.sqrt(2.92)])

    # Velocity

    # Manual bilinear interpolation
    U = ds.u[-1, :, :]
    V = ds.v[-1, :, :]
    U0 = 0.4 * U[0, 0] + 0.6 * U[0, 1]
    V0 = 0.45 * V[0, 1] + 0.45 * V[1, 1] + 0.05 * V[0, 2] + 0.05 * V[1, 2]

    ds0 = dss.isel(s_rho=-1)  # Surface values
    assert abs(ds0.u[0] - U0) < 1e-15
    assert abs(ds0.v[0] - V0) < 1e-15
    assert ds0.u_normal[0] == ds0.v[0]
    assert ds0.u_normal[1] == -ds0.u[1]
    assert abs(ds0.u_normal[2] - (-6 * ds0.u[2] + 16 * ds0.v[2]) / np.sqrt(292)) < 1e-12
