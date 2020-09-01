import numpy as np


def density(T, S, Z):
    """Return the density based on T, S, and Z. EOS based on ROMS Nonlinear/rho_eos.F

    Inputs:
    ------

    T       array-like, temperature
    S       array-like, salinity
    Z       array-like, depth. To specify a reference depth, use a constant

    Outputs:
    -------

    rho     array-like, density based on ROMS Nonlinear/rho_eos.F EOS
    """
    A00 = +19092.56
    A01 = +209.8925
    A02 = -3.041638
    A03 = -1.852732e-3
    A04 = -1.361629e-5
    B00 = +104.4077
    B01 = -6.500517
    B02 = +0.1553190
    B03 = +2.326469e-4
    D00 = -5.587545
    D01 = +0.7390729
    D02 = -1.909078e-2
    E00 = +4.721788e-1
    E01 = +1.028859e-2
    E02 = -2.512549e-4
    E03 = -5.939910e-7
    F00 = -1.571896e-2
    F01 = -2.598241e-4
    F02 = +7.267926e-6
    G00 = +2.042967e-3
    G01 = +1.045941e-5
    G02 = -5.782165e-10
    G03 = +1.296821e-7
    H00 = -2.595994e-7
    H01 = -1.248266e-9
    H02 = -3.508914e-9
    Q00 = +999.842594
    Q01 = +6.793952e-2
    Q02 = -9.095290e-3
    Q03 = +1.001685e-4
    Q04 = -1.120083e-6
    Q05 = +6.536332e-9
    U00 = +0.824493e0
    U01 = -4.08990e-3
    U02 = +7.64380e-5
    U03 = -8.24670e-7
    U04 = +5.38750e-9
    V00 = -5.72466e-3
    V01 = +1.02270e-4
    V02 = -1.65460e-6
    W00 = +4.8314e-4
    g = 9.81
    sqrtS = np.sqrt(S)
    den1 = (
        Q00
        + Q01 * T
        + Q02 * T ** 2
        + Q03 * T ** 3
        + Q04 * T ** 4
        + Q05 * T ** 5
        + U00 * S
        + U01 * S * T
        + U02 * S * T ** 2
        + U03 * S * T ** 3
        + U04 * S * T ** 4
        + V00 * S * sqrtS
        + V01 * S * sqrtS * T
        + V02 * S * sqrtS * T ** 2
        + W00 * S ** 2
    )
    K0 = (
        A00
        + A01 * T
        + A02 * T ** 2
        + A03 * T ** 3
        + A04 * T ** 4
        + B00 * S
        + B01 * S * T
        + B02 * S * T ** 2
        + B03 * S * T ** 3
        + D00 * S * sqrtS
        + D01 * S * sqrtS * T
        + D02 * S * sqrtS * T ** 2
    )
    K1 = (
        E00
        + E01 * T
        + E02 * T ** 2
        + E03 * T ** 3
        + F00 * S
        + F01 * S * T
        + F02 * S * T ** 2
        + G00 * S * sqrtS
    )
    K2 = G01 + G02 * T + G03 * T ** 2 + H00 * S + H01 * S * T + H02 * S * T ** 2
    bulk = K0 - K1 * Z + K2 * Z ** 2

    return (den1 * bulk) / (bulk + 0.1 * Z)


def buoyancy(T, S, Z, rho0=1000.0):
    """Return the buoyancy based on T, S, and Z. EOS based on ROMS Nonlinear/rho_eos.F

    Inputs:
    ------

    T       array-like, temperature
    S       array-like, salinity
    Z       array-like, depth. To specify a reference depth, use a constant

    Outputs:
    -------

    rho     array-like, buoyancy based on ROMS Nonlinear/rho_eos.F EOS
            rho = -g * rho / rho0

    Options:
    -------

    rho0    Constant. The reference density. Default rho0=1000.0
    """

    g = 9.81
    return -g * density(T, S, Z) / rho0


def stratification_frequency(ds):
    T = ds.temp
    S = ds.salt
    Zw = ds.z_w
    Zr = ds.z_rho

    sqrtS = np.sqrt(S)
    A00 = +19092.56
    A01 = +209.8925
    A02 = -3.041638
    A03 = -1.852732e-3
    A04 = -1.361629e-5
    B00 = +104.4077
    B01 = -6.500517
    B02 = +0.1553190
    B03 = +2.326469e-4
    D00 = -5.587545
    D01 = +0.7390729
    D02 = -1.909078e-2
    E00 = +4.721788e-1
    E01 = +1.028859e-2
    E02 = -2.512549e-4
    E03 = -5.939910e-7
    F00 = -1.571896e-2
    F01 = -2.598241e-4
    F02 = +7.267926e-6
    G00 = +2.042967e-3
    G01 = +1.045941e-5
    G02 = -5.782165e-10
    G03 = +1.296821e-7
    H00 = -2.595994e-7
    H01 = -1.248266e-9
    H02 = -3.508914e-9
    Q00 = +999.842594
    Q01 = +6.793952e-2
    Q02 = -9.095290e-3
    Q03 = +1.001685e-4
    Q04 = -1.120083e-6
    Q05 = +6.536332e-9
    U00 = +0.824493e0
    U01 = -4.08990e-3
    U02 = +7.64380e-5
    U03 = -8.24670e-7
    U04 = +5.38750e-9
    V00 = -5.72466e-3
    V01 = +1.02270e-4
    V02 = -1.65460e-6
    W00 = +4.8314e-4

    g = 9.81

    den1 = (
        Q00
        + Q01 * T
        + Q02 * T ** 2
        + Q03 * T ** 3
        + Q04 * T ** 4
        + Q05 * T ** 5
        + U00 * S
        + U01 * S * T
        + U02 * S * T ** 2
        + U03 * S * T ** 3
        + U04 * S * T ** 4
        + V00 * S * sqrtS
        + V01 * S * sqrtS * T
        + V02 * S * sqrtS * T ** 2
        + W00 * S ** 2
    )

    K0 = (
        A00
        + A01 * T
        + A02 * T ** 2
        + A03 * T ** 3
        + A04 * T ** 4
        + B00 * S
        + B01 * S * T
        + B02 * S * T ** 2
        + B03 * S * T ** 3
        + D00 * S * sqrtS
        + D01 * S * sqrtS * T
        + D02 * S * sqrtS * T ** 2
    )

    K1 = (
        E00
        + E01 * T
        + E02 * T ** 2
        + E03 * T ** 3
        + F00 * S
        + F01 * S * T
        + F02 * S * T ** 2
        + G00 * S * sqrtS
    )

    K2 = G01 + G02 * T + G03 * T ** 2 + H00 * S + H01 * S * T + H02 * S * T ** 2

    # below is some fugly coordinate wrangling to keep things in
    # the xarray universe. This could probably be cleaned up.
    Nm = len(ds.s_w) - 1
    Nmm = len(ds.s_w) - 2

    upw = {"s_w": slice(1, None)}
    dnw = {"s_w": slice(None, -1)}

    Zw_up = Zw.isel(**upw)
    Zw_up = Zw_up.rename({"s_w": "s_rho"})
    Zw_up.coords["s_rho"] = np.arange(Nm)
    Zw_dn = Zw.isel(**dnw)
    Zw_dn = Zw_dn.rename({"s_w": "s_rho"})
    Zw_dn.coords["s_rho"] = np.arange(Nm)

    K0.coords["s_rho"] = np.arange(Nm)
    K1.coords["s_rho"] = np.arange(Nm)
    K2.coords["s_rho"] = np.arange(Nm)
    den1.coords["s_rho"] = np.arange(Nm)

    bulk_up = K0 - Zw_up * (K1 - Zw_up * K2)
    bulk_dn = K0 - Zw_dn * (K1 - Zw_dn * K2)

    den_up = (den1 * bulk_up) / (bulk_up + 0.1 * Zw_up)
    den_dn = (den1 * bulk_dn) / (bulk_dn + 0.1 * Zw_dn)

    upr = {"s_rho": slice(1, None)}
    dnr = {"s_rho": slice(None, -1)}

    den_up = den1.isel(**upr)
    den_up = den_up.rename({"s_rho": "s_w"})
    den_up.coords["s_w"] = np.arange(Nmm)
    den_up = den_up.drop("z_rho")

    den_dn = den1.isel(**dnr)
    den_dn = den_dn.rename({"s_rho": "s_w"})
    den_dn.coords["s_w"] = np.arange(Nmm)
    den_dn = den_dn.drop("z_rho")

    Zr_up = Zr.isel(**upr)
    Zr_up = Zr_up.rename({"s_rho": "s_w"})
    Zr_up.coords["s_w"] = np.arange(Nmm)
    Zr_up = Zr_up.drop("z_rho")

    Zr_dn = Zr.isel(**dnr)
    Zr_dn = Zr_dn.rename({"s_rho": "s_w"})
    Zr_dn.coords["s_w"] = np.arange(Nmm)
    Zr_dn = Zr_dn.drop("z_rho")

    N2 = -g * (den_up - den_dn) / (0.5 * (den_up + den_dn) * (Zr_up - Zr_dn))

    # Put the rght vertical coordinates back, for plotting etc.
    N2.coords["z_w"] = ds.z_w.isel(s_w=slice(1, -1))
    N2.coords["s_w"] = ds.s_w.isel(s_w=slice(1, -1))

    return N2
