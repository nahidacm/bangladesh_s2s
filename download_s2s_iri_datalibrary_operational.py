#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 17 14:21:29 2022

Download ECCC data from IRI data library operationally.

@author: bob
"""

import datetime
import os
import xarray as xr

from configparser import ConfigParser

# Read the basic paths from the config file
config = ConfigParser()
config.read('config_bd_s2s.ini')

output_dir = config['paths']['iri']

model = 'ECCC'
version = 'GEPS7'

variables = ['tasmax', 'tasmin', 'pr']


today = datetime.datetime.today()
lastweek = today - datetime.timedelta(7)
modeldatestr = today.strftime("%Y%m%d")


for var in variables:
    url_start = 'http://iridl.ldeo.columbia.edu/SOURCES/.Models/.SubX/.' 
    url_model = model + "/." + version + "/.forecast/." + var 
    y_ext = '/Y/(20N)/(27N)/RANGEEDGES'
    x_ext = '/X/(87E)/(93E)/RANGEEDGES'
    s_start = '/S/(0000%20{}%20{}%20{})'.format(lastweek.strftime('%d'),lastweek.strftime('%b'),lastweek.strftime('%Y'))
    s_end = '/(0000%20{}%20{}%20{})/RANGEEDGES'.format(today.strftime('%d'),today.strftime('%b'),today.strftime('%Y'))
    url_end = '/data.nc'
    
    fn_output = f"{output_dir}{model}_fc_{var}_downloaded_{modeldatestr}.nc"
    
    cmd = f'curl "{url_start}{url_model}{y_ext}{s_start}{s_end}{x_ext}{url_end}" > {fn_output}'
    
    os.system(cmd)
    
    # Open the dataset and drop the NaN values
    ds = xr.open_dataarray(fn_output)
    ds = ds.dropna(dim='S')
    unit = ds.units
    
    # Convert units
    if ds.units == 'Kelvin_scale':
        ds = ds - 273.15
        ds.attrs['units'] = 'degreec Celsius'
    elif ds.units == 'kg m-2 s-1':
        ds = ds * 24*3600
        ds.values[ds.values <= 0] = 0
        ds.attrs['units'] = 'mm'
    
    # Save one file for each forecast
    for forecast_nr in range(len(ds)):
        single_fc = ds[forecast_nr]
        datestr_long = ds[forecast_nr].S.values.astype(str)
        datestr = datestr_long[:4] + datestr_long[5:7] + datestr_long[8:10]
        
        # Change lead time dimension to datetimes
        single_fc.assign_coords(L=single_fc.L + ds.S.values[forecast_nr])
        
        single_fc.to_netcdf(f"{output_dir}{model}_fc_{var}_{datestr}.nc")
        
    # Remove downloaded file
    os.remove(fn_output)
        
