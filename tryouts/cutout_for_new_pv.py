# -*- coding: utf-8 -*-
"""
Created on Wed May 19 16:50:58 2021

@author: arneb
"""


import atlite
import logging
import pvlib
import xarray as xr
from atlite import wind as windm
import yaml
import numpy as np
from operator import itemgetter
from atlite.resource import get_oedb_windturbineconfig
logging.basicConfig(level=logging.INFO)

cutout = atlite.Cutout(path="test_cutout_ger.nc",
                        module="era5",
                        x=slice(5.866, 15.1),
                        y=slice(47.2, 55.1),
                        time="2011-01"
                        )


cutout.prepare()
turbine = get_oedb_windturbineconfig(12)
# path = r'C:/Users/arneb/Documents/git_projects/atlite/atlite/resources/windturbine/Vestas_V112_3MW_offshore.yaml'
# with open(path) as f:
#     turbine_ct = yaml.safe_load(f)
# turbine=atlite.resource.get_windturbineconfig(path)
# turbine['c_t'] = np.array(turbine_ct['c_t'][:15])
wind=cutout.wind(turbine=turbine, k=0.075, x_d=5)

# V, POW, hub_height, P, c_t = itemgetter(
#         'V', 'POW', 'hub_height', 'P', 'c_t')(turbine)

# def _interpolate(da):
#       return np.interp(da, V, c_t)
    
# ct = xr.apply_ufunc(
#     _interpolate, wind,
#     input_core_dims=[[]],
#     output_core_dims=[[]],
#     output_dtypes=[wind.dtype],
#     dask="parallelized")

# def calculate_wake(wnd, ct, k, x, d):
#     # Jensen model
#     u_w = (1- (1- (1-ct)**(1/2))/(1+2*k*x/d)**2)*wnd
#     return u_w
    

# wnd_hub = calculate_wake(wind, ct, 0.075, 400, 100)
# ds=cutout.data
# surface_tilt=30
# surface_azimuth=180
# module_name = 'Canadian_Solar_CS5P_220M___2009_'
# inverter_name = 'ABB__MICRO_0_25_I_OUTD_US_208__208V_'