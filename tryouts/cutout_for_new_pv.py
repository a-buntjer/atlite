# -*- coding: utf-8 -*-
"""
Created on Wed May 19 16:50:58 2021

@author: arneb
"""

import atlite
import logging


logging.basicConfig(level=logging.INFO)

cutout = atlite.Cutout(path="test_cutout.nc",
                       module="era5",
                       x=slice(10., 11.),
                       y=slice(55., 56.),
                       time="2011-01"
                       )
cutout.prepare()
