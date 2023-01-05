#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 17 14:21:29 2022

Download ECCC data from IRI data library operationally.

@author: bob
"""

import datetime
import os
import numpy as np
import xarray as xr

from configparser import ConfigParser

# Read the basic paths from the config file
config = ConfigParser()
config.read('config_bd_s2s.ini')

output_dir = config['paths']['iri']

model = 'ECCC'
version = 'GEPS7'

variables = ['tasmax', 'tasmin', 'pr']

fc_types = {'fc': 'forecast',
            'hc': 'hindcast'}

today = datetime.datetime.today()
lastweek = today - datetime.timedelta(7)
modeldatestr = today.strftime("%Y%m%d")

# Download the forecast
for var in variables:
    
    # Download both forecast and hindcast
    for fc_type in fc_types.keys():
        fc_type_full = fc_types[fc_type]
            
        fn_output = f"{output_dir}{model}_{fc_type}_{var}_downloaded_{modeldatestr}.nc"

        url_start = 'http://iridl.ldeo.columbia.edu/SOURCES/.Models/.SubX/.' 
        url_model = model + "/." + version + "/." + fc_type_full + "/." + var 
        y_ext = '/Y/(20N)/(27N)/RANGEEDGES'
        x_ext = '/X/(87E)/(93E)/RANGEEDGES'
        if fc_type == 'fc':
            s_start = '/S/(0000%20{}%20{}%20{})'.format(lastweek.strftime('%d'),lastweek.strftime('%b'),lastweek.strftime('%Y'))
            s_end = '/(0000%20{}%20{}%20{})/RANGEEDGES'.format(today.strftime('%d'),today.strftime('%b'),today.strftime('%Y'))
            fn_fc = fn_output            
        elif fc_type == 'hc':
            s_start = ''
            s_end = ''
            fn_hc = fn_output
        url_end = '/data.nc'
        
        cmd = f'curl "{url_start}{url_model}{y_ext}{s_start}{s_end}{x_ext}{url_end}" > {fn_output}'
        
        os.system(cmd)
    
    # Open the dataset and drop the NaN values
    ds_fc = xr.open_dataarray(fn_fc)
    ds_fc = ds_fc.dropna(dim='S')
    
    ds_hc = xr.open_dataarray(fn_hc)
    ds_hc = ds_hc.dropna(dim='S')   
    
    # Convert units
    def convert_units(ds):
        if ds.units == 'Kelvin_scale':
            ds = ds - 273.15
            ds.attrs['units'] = 'degreec Celsius'
        elif ds.units == 'kg m-2 s-1':
            ds = ds * 24*3600
            ds.values[ds.values <= 0] = 0
            ds.attrs['units'] = 'mm'
        return ds
    
    ds_hc = convert_units(ds_hc)
    ds_fc = convert_units(ds_fc)

    # Add extra coordinate 'month-day' to hindcasts
    ds_hc = ds_hc.assign_coords({'md': xr.DataArray([datetime.datetime.utcfromtimestamp(i.tolist()/1e9).strftime('%m-%d') for i in ds_hc.coords['S'].values], dims='S')})
    
    # Save one file for each forecast
    for forecast_nr in range(len(ds_fc)):
        single_fc = ds_fc[forecast_nr]
        datestr_long = ds_fc[forecast_nr].S.values.astype(str)
        datestr = datestr_long[:4] + datestr_long[5:7] + datestr_long[8:10]
        
        fc_date = datetime.datetime.utcfromtimestamp(single_fc.S.values.tolist()/1e9).strftime('%m-%d')
        
        # Change lead time dimension to datetimes
        single_fc = single_fc.assign_coords(L=single_fc.L + ds_fc.S.values[forecast_nr]-np.timedelta64(12,'h'))
        
        # Only take startdates that are in the forecast (month-day)
        single_hc = ds_hc.where(ds_hc.md==fc_date, drop=True).drop('md')
        
        # Create single time dimension
        single_hc = single_hc.transpose('S','L','M','Y','X')
        single_hc_data = single_hc.data.reshape((len(single_hc.S)*len(single_hc.L),)+np.shape(single_hc)[2:])
        
        # Create time array
        time = []
        for ss in single_hc.S.values:
            for ll in single_hc.L.values:
                time.append(ss+ll-np.timedelta64(12,'h'))
                
        # Set back to xarray
        single_hc_to_save = xr.DataArray(single_hc_data,
                                         coords=(time, single_hc.M.values, single_hc.Y.values, single_hc.X.values),
                                         dims=('L', 'M', 'Y', 'X'))
        
        # Save the data
        single_fc.to_netcdf(f"{output_dir}{model}_fc_{var}_{datestr}.nc")
        single_hc_to_save.to_netcdf(f"{output_dir}{model}_hc_{var}_{datestr}.nc")
        
    # Remove downloaded file
    os.remove(fn_fc)
    os.remove(fn_hc)
        
