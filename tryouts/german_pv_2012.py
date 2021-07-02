# -*- coding: utf-8 -*-
"""
Created on Fri May 21 00:04:53 2021

@author: arneb
"""

import atlite
import xarray as xr
import pandas as pd
import scipy.sparse as sp
import numpy as np

import pgeocode
from collections import OrderedDict

import matplotlib.pyplot as plt

import seaborn as sns
sns.set_style('whitegrid')

import requests
import os
import zipfile
from sklearn.metrics import mean_absolute_error, mean_squared_error
import plotly.graph_objects as go
from plotly.offline import plot

pd.options.plotting.backend = "plotly"

def download_file(url, local_filename):
    # variant of http://stackoverflow.com/a/16696317
    if not os.path.exists(local_filename):
        r = requests.get(url, stream=True)
        with open(local_filename, 'wb') as f:
            for chunk in r.iter_content(chunk_size=1024):
                if chunk:
                    f.write(chunk)
    return local_filename

opsd_fn = download_file('https://data.open-power-system-data.org/index.php?package=time_series&version=2019-06-05&action=customDownload&resource=3&filter%5B_contentfilter_cet_cest_timestamp%5D%5Bfrom%5D=2012-01-01&filter%5B_contentfilter_cet_cest_timestamp%5D%5Bto%5D=2013-05-01&filter%5BRegion%5D%5B%5D=DE&filter%5BVariable%5D%5B%5D=solar_generation_actual&filter%5BVariable%5D%5B%5D=wind_generation_actual&downloadCSV=Download+CSV',
                        'time_series_60min_singleindex_filtered.csv')

opsd = pd.read_csv(opsd_fn, parse_dates=True, index_col=0)

# we later use the (in current version) timezone unaware datetime64
# to work together with this format, we have to remove the timezone
# timezone information. We are working with UTC everywhere.

opsd.index = opsd.index.tz_convert(None)

# We are only interested in the 2012 data
opsd = opsd[("2011" < opsd.index) & (opsd.index < "2013")]

eeg_fn = download_file('http://www.energymap.info/download/eeg_anlagenregister_2015.08.utf8.csv.zip',
                        'eeg_anlagenregister_2015.08.utf8.csv.zip')

with zipfile.ZipFile(eeg_fn, "r") as zip_ref:
    zip_ref.extract("eeg_anlagenregister_2015.08.utf8.csv")
    
import cartopy.io.shapereader as shpreader
import geopandas as gpd
shp = shpreader.Reader(shpreader.natural_earth(resolution='10m', category='cultural', name='admin_0_countries'))
de_record = list(filter(lambda c: c.attributes['ISO_A2'] == 'DE', shp.records()))[0]
de = gpd.GeoSeries({**de_record.attributes, 'geometry':de_record.geometry})
x1, y1, x2, y2 = de['geometry'].bounds

cutout = atlite.Cutout('germany-2012',
                       module='era5',
                       x=slice(x1-.2,x2+.2), y=slice(y1-.2, y2+.2),
                       chunks={'time':100},
                       time="2012")
cutout.prepare()

def capacity_layout(cutout, typ, cap_range=None, until=None):
    """Aggregate selected capacities to the cutouts grid into a capacity layout.

    Parameters
    ----------
        cutout : atlite.cutout
            The cutout for which the capacity layout is contructed.
        typ : str
            Type of energy source, e.g. "Solarstrom" (PV), "Windenergie" (wind).
        cap_range : (optional) list-like
            Two entries, limiting the lower and upper range of capacities (in kW)
            to include. Left-inclusive, right-exclusive.
        until : str
            String representation of a datetime object understood by pandas.to_datetime()
            for limiting to installations existing until this datetime.

    """

    # Load locations of installed capacities and remove incomplete entries
    cols = OrderedDict((('installation_date', 0),
                        ('plz', 2), ('city', 3),
                        ('type', 6),
                        ('capacity', 8), ('level', 9),
                        ('lat', 19), ('lon', 20),
                        ('validation', 22)))
    database = pd.read_csv('eeg_anlagenregister_2015.08.utf8.csv',
                       sep=';', decimal=',', thousands='.',
                       comment='#', header=None,
                       usecols=list(cols.values()),
                       names=list(cols.keys()),
                       # German postal codes can start with '0' so we need to treat them as str
                       dtype={'plz':str},
                       parse_dates=['installation_date'],
                       na_values=('O04WF', 'keine'))

    database = database[(database['validation'] == 'OK') & (database['plz'].notna())]

    # Query postal codes <-> coordinates mapping
    de_nomi = pgeocode.Nominatim('de')
    plz_coords = de_nomi.query_postal_code(database['plz'].unique())
    plz_coords = plz_coords.set_index('postal_code')

    # Fill missing lat / lon using postal codes entries
    database.loc[database['lat'].isna(), 'lat'] = database['plz'].map(plz_coords['latitude'])
    database.loc[database['lon'].isna(), 'lon'] = database['plz'].map(plz_coords['longitude'])

    # Ignore all locations which have not be determined yet
    database = database[database['lat'].notna() & database['lon'].notna()]

    # Select data based on type (i.e. solar/PV, wind, ...)
    data = database[database['type'] == typ].copy()

    # Optional: Select based on installation day
    if until is not None:
        data = data[data['installation_date'] < pd.to_datetime(until)]

    # Optional: Only installations within this caprange (left inclusive, right exclusive)
    if cap_range is not None:
        data = data[(cap_range[0] <= data['capacity']) & (data['capacity'] < cap_range[1])]

    # Determine nearest cells from cutout
    cells = gpd.GeoDataFrame({'geometry': cutout.grid_cells,
                              'lon': cutout.grid_coordinates()[:,0],
                              'lat': cutout.grid_coordinates()[:,1]})

    nearest_cell = cutout.data.sel({'x': data.lon.values,
                                    'y': data.lat.values},
                                   'nearest').coords

    # Map capacities to closest cell coordinate
    data['lon'] = nearest_cell.get('lon').values
    data['lat'] = nearest_cell.get('lat').values

    new_data = data.merge(cells, how='inner')

    # Sum capacities for each grid cell (lat, lon)
    # then: restore lat lon as coumns
    # then: rename and reindex to match cutout coordinates
    new_data = new_data.groupby(['lat','lon']).sum()

    layout = new_data.reset_index().rename(columns={'lat':'y','lon':'x'})\
                        .set_index(['y','x']).capacity\
                        .to_xarray().reindex_like(cutout.data)

    layout = (layout/1e3).fillna(.0).rename('Installed Capacity [MW]')

    return layout

solar_layout = capacity_layout(cutout, 'Solarstrom', until="2012")

solar_layout.plot(cmap="inferno_r", size=8, aspect=1)
plt.title("Installed PV in Germany until 2012")
plt.tight_layout()

# ds=cutout.data
# surface_tilt=30
# surface_azimuth=180
# module_name = 'Canadian_Solar_CS5P_220M___2009_'
# inverter_name = 'ABB__MICRO_0_25_I_OUTD_US_208__208V_'
# pv = cutout.pv(panel="CSi", orientation={'slope': 30., 'azimuth': 0.}, layout=solar_layout)

pv = cutout.pv(
    surface_tilt=25, surface_azimuth=200,
    module_name = 'Canadian_Solar_CS5P_220M___2009_',
    inverter_name = 'ABB__MICRO_0_25_I_OUTD_US_208__208V_', 
    layout=solar_layout
    )

def compare_power_generation(
        renewable_technology, generation1, generation2, 
        surface_azimuth, surface_tilt, resample_freq="1d"):
    df_generation = pd.concat([generation1, generation2], axis=1).dropna()
    cols = df_generation.columns 
    mae = mean_absolute_error(df_generation[cols[0]],
                              df_generation[cols[1]])
    mse = mean_squared_error(df_generation[cols[0]],
                              df_generation[cols[1]])
    
    
    df_generation = df_generation.resample(resample_freq).mean()/1000
    fig = df_generation.plot(
        title=(
            f"Comparison of ENTSOE-E and model {renewable_technology} generation"),
        labels=dict(index="", value="Power in GW", variable="data"))
    fig.add_annotation(
        text=(f"slope angle: {surface_tilt}°"
              f"\nazimuth angle: {surface_azimuth}°"
              f"\nMSE: {mse:.2f}"
              f"\nMAE: {mae:.2f}"),
        xref="x domain", yref="y domain",
        x=0.5, y=-0.08, showarrow=False)
    plot(fig)

pv_model = pv.squeeze().to_series()
pv_model.name = 'Atlite'

compare_power_generation(
    "PV", pv_model,
    opsd['DE_solar_generation_actual'],
        surface_tilt=25, surface_azimuth=200,  resample_freq="1d")


compare = pd.DataFrame(
    dict(atlite=pv.squeeze().to_series(), opsd=opsd['DE_solar_generation_actual'] /1e3)) # in GW
compare.resample('1D').mean().plot(figsize=(8,5))
plt.ylabel("Feed-In [GW]")
plt.title('PV time-series Germany 2012')
plt.tight_layout()