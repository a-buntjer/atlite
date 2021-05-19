# -*- coding: utf-8 -*-

# SPDX-FileCopyrightText: 2016-2019 The Atlite Authors
#
# SPDX-License-Identifier: GPL-3.0-or-later
import pvlib
from numpy import pi
import xarray as xr
from numpy import sin, cos, arcsin, arccos, arctan2, deg2rad

def SolarPosition(ds):
    ds=cutout.data
    t = ds.indexes['time']
    n = xr.DataArray(t.to_julian_date(), [t]) - 2451545.0
    hour = ds['time.hour']
    minute = ds['time.minute']

    if 'time' in ds.chunks:
        chunks = {'time': ds.chunks['time']}
        n = n.chunk(chunks)
        hour = hour.chunk(chunks)
        minute = minute.chunk(chunks)
  
    
    
    def _get_solarposition(t, y, x):
        solarpos = pvlib.solarposition.get_solarposition(t, y, x)
        return solarpos.azimuth, solarpos.apparent_zenith
            
        
    
    ds_solarpos = xr.apply_ufunc(_get_solarposition,
                   ds.time, ds.lat, ds.lon,
                   input_core_dims=[["time"], [], []],
                   output_core_dims=[["time"], ["time"]],
                   vectorize=True,
                   dask="parallelized")
    for ds_solar, name in zip(ds_solarpos, ['azimuth', 'apparent_zenith']):
        ds_solar.name = name
    
    ds_solarpos = xr.merge(ds_solarpos).transpose()   
        
    ds_airmass = xr.apply_ufunc(pvlib.atmosphere.get_relative_airmass,
                   ds_solarpos.apparent_zenith,
                   dask="parallelized")
    ds_airmass.name = 'airmass' 
    
    dni_extra = pvlib.irradiance.get_extra_radiation(t)
    
    
    surface_tilt=30
    surface_azimuth=180
    
    # poa_sky_diffuse = pvlib.irradiance.perez(
    #     surface_tilt,
    #     surface_azimuth,
    #     ds.influx_direct.isel(x=0,y=0),
    #     ds.influx_diffuse.isel(x=0,y=0),
    #     dni_extra,
    #     ds.apparent_zenith.isel(x=0,y=0),
    #     ds.azimuth.isel(x=0,y=0),
    #     ds.airmass.isel(x=0,y=0),
    # )
    
    def _get_poa_sky_diffuse(
            dni, dhi, apz, azi, airm, dni_extra, surface_tilt,
            surface_azimuth):
        return pvlib.irradiance.perez(
            surface_tilt, surface_azimuth, dni, dhi,
            dni_extra, apz, azi, airm)
    
    ds = ds.compute()
    ds_solarpos = ds_solarpos.compute()
    ds_airmass = ds_airmass.compute()
    ds_poa = xr.apply_ufunc(
        _get_poa_sky_diffuse,
        ds.influx_direct,
        ds.influx_diffuse,
        ds_solarpos.apparent_zenith,
        ds_solarpos.azimuth,
        ds_airmass,
        kwargs={"dni_extra": dni_extra,
                "surface_tilt": surface_tilt,
                "surface_azimuth": surface_azimuth,
                },
        input_core_dims=[["time"] for x in range(5)],
        output_core_dims=[["time"]],
        vectorize=True, dask="parallelized")
    
    poa_ground_diffuse = pvlib.irradiance.get_ground_diffuse(
        surface_tilt, tmy_data["GHI"], albedo=albedo
    )
    

def SolarPosition(ds):
    """
    Compute solar azimuth and altitude

    Solar altitude errors are up to 1.5 deg during sun-rise and set, but at
    0.05-0.1 deg during daytime.

    References
    ----------
    [1] Michalsky, J. J., The astronomical almanac’s algorithm for approximate
    solar position (1950–2050), Solar Energy, 40(3), 227–235 (1988).
    [2] Sproul, A. B., Derivation of the solar geometric relationships using
    vector analysis, Renewable Energy, 32(7), 1187–1205 (2007).
    [3] Kalogirou, Solar Energy Engineering (2009).

    More accurate algorithms would be
    ---------------------------------
    [4] I. Reda and A. Andreas, Solar position algorithm for solar
    radiation applications. Solar Energy, vol. 76, no. 5, pp. 577-589, 2004.
    [5] I. Reda and A. Andreas, Corrigendum to Solar position algorithm for
    solar radiation applications. Solar Energy, vol. 81, no. 6, p. 838, 2007.
    [6] Blanc, P., & Wald, L., The SG2 algorithm for a fast and accurate
    computation of the position of the sun for multi-decadal time period, Solar
    Energy, 86(10), 3072–3083 (2012).

    The unfortunately quite computationally intensive SPA algorithm [4,5] has
    been implemented using numba or plain numpy for a single location at
    https://github.com/pvlib/pvlib-python/blob/master/pvlib/spa.py.

    """

    # up to h and dec from [1]

    t = ds.indexes['time']
    n = xr.DataArray(t.to_julian_date(), [t]) - 2451545.0
    hour = ds['time.hour']
    minute = ds['time.minute']

    if 'time' in ds.chunks:
        chunks = {'time': ds.chunks['time']}
        n = n.chunk(chunks)
        hour = hour.chunk(chunks)
        minute = minute.chunk(chunks)

    L = 280.460 + 0.9856474 * n  # mean longitude (deg)
    g = deg2rad(357.528 + 0.9856003 * n)  # mean anomaly (rad)
    l = deg2rad(
        L +
        1.915 *
        sin(g) +
        0.020 *
        sin(2 * g))  # ecliptic long. (rad)
    ep = deg2rad(23.439 - 4e-7 * n)  # obliquity of the ecliptic (rad)

    ra = arctan2(cos(ep) * sin(l), cos(l))  # right ascencion (rad)
    lmst = (6.697375 + (hour + minute / 60.0) +
            0.0657098242 * n) * 15. + ds['lon']  # local mean sidereal time (deg)
    h = (deg2rad(lmst) - ra + pi) % (2 * pi) - \
        pi  # hour angle (rad)

    dec = arcsin(sin(ep) * sin(l))            # declination (rad)

    # alt and az from [2]
    lat = deg2rad(ds['lat'])
    # Clip before arcsin to prevent values < -1. from rounding errors; can
    # cause NaNs later
    alt = arcsin(
        (sin(lat) *
          sin(dec) +
          cos(lat) *
          cos(dec) *
          cos(h)).clip(min=-1., max=1.)).rename('altitude')

    az = arccos(
        ((sin(dec) *
          cos(lat) -
          cos(dec) *
          sin(lat) *
          cos(h)) /
          cos(alt)).clip(min=-1., max=1.))
    az = az.where(h <= 0, 2 * pi - az).rename('azimuth')

    if 'influx_toa' in ds:
        atmospheric_insolation = ds['influx_toa'].rename(
            'atmospheric insolation')
    else:
        # [3]
        atmospheric_insolation = (1366.1 * (1 + 0.033 * cos(g)) * sin(alt))\
                                  .rename('atmospheric insolation')

    vars = {da.name: da for da in [alt, az, atmospheric_insolation]}
    solar_position = xr.Dataset(vars)
    return solar_position
