# -*- coding: utf-8 -*-
"""
Created on Thu May 20 20:28:27 2021

@author: arneb
"""

import pvlib
import xarray as xr
from .. import wind as windm

def PVSystemPower(
        ds, surface_tilt=30, surface_azimuth=180,
        module_name = 'Silevo_Triex_U300_Black__2014_',
        inverter_name = 'ABB__PVI_3_6_OUTD_S_US_A__277V_'):
    
    t = ds.indexes['time']
    
    def _get_solarposition(t, y, x, h):
        solarpos = pvlib.solarposition.get_solarposition(
            t, y, x, h)
        return solarpos.azimuth, solarpos.apparent_zenith
             
    ds['azimuth'], ds['apparent_zenith'] = xr.apply_ufunc(_get_solarposition,
                   ds.time, ds.lat, ds.lon, ds.height,
                   input_core_dims=[['time'], [], [], []],
                   output_core_dims=[['time'], ['time']],
                   vectorize=True,
                   dask="parallelized")
    
    def _get_apparent_zenith(t, y, x):
        solarpos = pvlib.solarposition.get_solarposition(
            t, y, x)
        return solarpos.apparent_zenith
    
    ds['apparent_zenith'] = xr.apply_ufunc(_get_apparent_zenith,
                   ds.time, ds.lat, ds.lon,
                   input_core_dims=[['time'], [], []],
                   output_core_dims=[['time']],
                   vectorize=True,
                   dask="parallelized")
             
    ds['airmass_rel'] = xr.apply_ufunc(
        pvlib.atmosphere.get_relative_airmass,
        ds.apparent_zenith,
        dask="parallelized")
    ds['airmass_rel'] = ds['airmass_rel'].fillna(0.)
    #         dask_gufunc_kwargs=dict(allow_rechunk=True),
    ds['site_pressure'] = xr.apply_ufunc(
        pvlib.atmosphere.alt2pres,
        ds.height,
        dask="parallelized")
    
    
    ds['airmass_abs'] = xr.apply_ufunc(
        pvlib.atmosphere.get_absolute_airmass,
        ds.airmass_rel,
        dask="parallelized")
    
    ds['dni_extra'] = pvlib.irradiance.get_extra_radiation(t)
    ds['GHI'] = ds.influx_direct + ds.influx_diffuse
       
    ds['DNI'] = xr.apply_ufunc(
        pvlib.irradiance.dni,
        ds.GHI,
        ds.influx_diffuse,
        ds.apparent_zenith,
        dask="parallelized")
    
   
    def _get_poa_sky_diffuse(
            dhi, dni, apz, azi, airm, dni_extra, surface_tilt,
            surface_azimuth):
        return pvlib.irradiance.perez(
            surface_tilt, surface_azimuth, dhi, dni,
            dni_extra, apz, azi, airm)
    
    ds['poa_sky_diffuse'] = xr.apply_ufunc(
        _get_poa_sky_diffuse,
        ds.influx_diffuse,
        ds.DNI,
        ds.apparent_zenith,
        ds.azimuth,
        ds.airmass_rel,
        ds.dni_extra,
        kwargs={"surface_tilt": surface_tilt,
                "surface_azimuth": surface_azimuth,
                },
        input_core_dims=[["time"] for x in range(6)],
        output_core_dims=[["time"]],
        dask_gufunc_kwargs=dict(allow_rechunk=True),
        dask="parallelized")
        
    def _get_poa_ground_diffuse(ghi, albedo, surface_tilt):
        return pvlib.irradiance.get_ground_diffuse(surface_tilt, ghi, albedo)
    
    ds['albedo'] = ds.albedo.where(ds.albedo >= 0).fillna(0.)
    ds['poa_ground_diffuse'] = xr.apply_ufunc(
        _get_poa_ground_diffuse,
        ds.GHI,
        ds.albedo,
        kwargs={"surface_tilt": surface_tilt},
        input_core_dims=[["time"] for x in range(2)],
        output_core_dims=[["time"]],
        dask_gufunc_kwargs=dict(allow_rechunk=True),
        dask="parallelized")

    
    def _get_aoi(apz, azi, surface_azimuth, surface_tilt):
        return pvlib.irradiance.aoi(surface_tilt, surface_azimuth, apz, azi)
    
    ds['aoi'] = xr.apply_ufunc(
        _get_aoi,
        ds.apparent_zenith,
        ds.azimuth,
        kwargs={"surface_tilt": surface_tilt,
                "surface_azimuth": surface_azimuth},
        input_core_dims=[["time"] for x in range(2)],
        output_core_dims=[["time"]],
        dask_gufunc_kwargs=dict(allow_rechunk=True),
        dask="parallelized")
        
    def _get_poa_components(aoi, dni, poa_sky_diffuse, poa_ground_diffuse):
        poa_components = pvlib.irradiance.poa_components(
            aoi, dni, poa_sky_diffuse, poa_ground_diffuse)
        return (poa_components['poa_direct'],
                poa_components['poa_diffuse'],
                poa_components['poa_global'])
    
    ds['poa_direct'], ds['poa_diffuse'], ds['poa_global']  = xr.apply_ufunc(
        _get_poa_components,
        ds.aoi,
        ds.DNI,
        ds.poa_sky_diffuse,
        ds.poa_ground_diffuse,
        input_core_dims=[["time"] for x in range(4)],
        output_core_dims=[["time"] for x in range(3)],
        dask_gufunc_kwargs=dict(allow_rechunk=True),
        dask="parallelized")
    
    ds['wnd10m'] = windm.extrapolate_wind_speed(ds, to_height=10)
    
    temp_model = pvlib.temperature.TEMPERATURE_MODEL_PARAMETERS[
        "sapm"]["open_rack_glass_glass"]
    def _get_pvtemps(poa_global, temp, windspeed, a, b, deltaT):
        
        return pvlib.temperature.sapm_cell(
            poa_global, temp-273.15, windspeed, a, b, deltaT)
    
    ds['pv_temperature'] = xr.apply_ufunc(
        _get_pvtemps,
        ds.poa_global,
        ds.temperature,
        ds.wnd10m,
        kwargs=temp_model,
        input_core_dims=[["time"] for x in range(3)],
        output_core_dims=[["time"]],
        dask_gufunc_kwargs=dict(allow_rechunk=True),
        dask="parallelized")
    modules = pvlib.pvsystem.retrieve_sam(name="SandiaMod")
    module = modules[module_name]
    
    def _get_sapm_irr(poa_direct, poa_diffuse, air_m_abs, aoi, module):
        return  pvlib.pvsystem.sapm_effective_irradiance(
            poa_direct, poa_diffuse, air_m_abs, aoi, module)
    
    ds['sapm_irr'] = xr.apply_ufunc(
        _get_sapm_irr,
        ds.poa_direct,
        ds.poa_diffuse,
        ds.airmass_abs,
        ds.aoi,
        kwargs={'module': module},
        input_core_dims=[["time"] for x in range(4)],
        output_core_dims=[["time"]],
        dask_gufunc_kwargs=dict(allow_rechunk=True),
        dask="parallelized")
        
    def _get_sapm_out(sapm_irr, temp_cell, module):
        sapm_out = pvlib.pvsystem.sapm(
            sapm_irr, module=module, temp_cell=temp_cell)
        return sapm_out['v_mp'], sapm_out['p_mp']
    # calculate pv performance
    ds['v_mp'], ds['p_mp'] = xr.apply_ufunc(
        _get_sapm_out,
        ds.sapm_irr,
        ds.pv_temperature,
        kwargs={'module': module},
        input_core_dims=[["time"] for x in range(2)],
        output_core_dims=[["time"] for x in range(2)],
        dask_gufunc_kwargs=dict(allow_rechunk=True),
        dask="parallelized")
    
    peak_load = module.loc["Impo"] * module.loc["Vmpo"]
    
    
    def _get_inverter_load(v_mp, p_mp, inverter):
        return pvlib.inverter.sandia(
            inverter=inverter, v_dc=v_mp, p_dc=p_mp)
    
    inverters = pvlib.pvsystem.retrieve_sam("cecinverter")
    inverter = inverters[inverter_name]
    
    ds['inv_load'] = xr.apply_ufunc(
        _get_inverter_load,
        ds.v_mp,
        ds.p_mp,
        kwargs={'inverter': inverter},
        input_core_dims=[["time"] for x in range(2)],
        output_core_dims=[["time"]],
        dask_gufunc_kwargs=dict(allow_rechunk=True),
        dask="parallelized")
    
    ds['specific_load'] = ds['inv_load'] / peak_load
    
    # if 'snowfall' in ds:
    #     # Convert from cm/day to cm/h
    #     ds['snowfall'] = ds['snowfall']/24
    #     def _get_snowfall_coverage(t, snowfall, poa_global, temp, surface_tilt):
    #         return pvlib.snow.coverage_nrel(
    #             pd.Series(snowfall, index=t),
    #             pd.Series(poa_global, index=t),
    #             pd.Series(temp, index=t),
    #             surface_tilt)
                
        
    #     ds['snowfall_loss'] = xr.apply_ufunc(
    #         _get_snowfall_coverage,
    #         ds.time,
    #         ds.snowfall,
    #         ds.poa_global,
    #         ds.temperature,
    #         kwargs={"surface_tilt": surface_tilt},
    #         input_core_dims=[["time"] for x in range(4)],
    #         output_core_dims=[["time"]],
    #         dask_gufunc_kwargs=dict(allow_rechunk=True),
    #         vectorize=True,
    #         dask="parallelized")
        
                    
            
    ds = ds.transpose("time", "y", "x")
    ds = ds.unify_chunks()
    ds = ds.compute()
    ds = ds.fillna(0.)
    
    return ds.specific_load