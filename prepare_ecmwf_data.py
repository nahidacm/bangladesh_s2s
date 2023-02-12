#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan  3 09:55:09 2023

Prepare the data to daily regridded values

@author: bob
"""
import xarray as xr
import xcast as xc
import datetime
import os
import numpy as np

import warnings
warnings.filterwarnings('ignore')

from configparser import ConfigParser

# Read the basic paths from the config file
config = ConfigParser()
config.read('config_bd_s2s.ini')

# Set the directories from the config file
direc = config['paths']['s2s_dir'] 

input_dir_ec = direc + 'input_ecmwf/'
input_dir_obs = direc + 'input_bmd_gridded_data/'
output = direc + 'input_regrid/'

if not os.path.exists(output):
    os.makedirs(output)

# Take the modelrun of yesterday
modeldate = datetime.datetime.today() - datetime.timedelta(1)
modeldate = datetime.datetime(modeldate.year,
                              modeldate.month,
                              modeldate.day,
                              0,0)
modeldatestr = modeldate.strftime("%Y%m%d")

# List the variable names with a key and name in the ECMWF files
varnames = {'tmax': {'obs_filename': 'tmax',
                     'obs_ncname': 'temp',
                     'resample': 'max'},
            'tmin': {'obs_filename': 'tmin',
                     'obs_ncname': 'temp',
                     'resample': 'min'},
            'tp': {'obs_filename': 'rr',
                   'obs_ncname': 'precip',
                   'resample': 'sum'}}

# Make a forecast for each variable
for var in varnames.keys():
    print(f'Start forecasts for {var}.')
    
    # Load the ECMWF hindcast and forecast
    fn_ec_hc_02 = f"{input_dir_ec}ecmwf_hc_{var}_{modeldatestr}_02.nc"
    fn_ec_hc_04 = f"{input_dir_ec}ecmwf_hc_{var}_{modeldatestr}_04.nc"
    
    fn_ec_fc_02 = f"{input_dir_ec}ecmwf_fc_{var}_{modeldatestr}_02.nc"
    fn_ec_fc_04 = f"{input_dir_ec}ecmwf_fc_{var}_{modeldatestr}_04.nc"
    
    ec_02_fc = xr.open_dataarray(fn_ec_fc_02)
    ec_04_fc = xr.open_dataarray(fn_ec_fc_04)
    
    ec_02_hc = xr.open_dataarray(fn_ec_hc_02)
    ec_04_hc = xr.open_dataarray(fn_ec_hc_04)
    
    # Load the BMD gridded data
    obs_var_fn = varnames[var]['obs_filename']
    obs_var_nc = varnames[var]['obs_ncname']
    obs = xr.open_mfdataset(f'{input_dir_obs}merge_{obs_var_fn}_*')
    obs = obs[obs_var_nc]
    
    # Regrid the ECMWF data to the BMD gridded data
    print('Regrid ECMWF data to BMD grid')
    ec_02_fc_regrid = xc.regrid(ec_02_fc, obs.coords['Lon'].values, obs.coords['Lat'].values, x_sample_dim= 'member')
    ec_04_fc_regrid = xc.regrid(ec_04_fc, obs.coords['Lon'].values, obs.coords['Lat'].values, x_sample_dim= 'member')
    ec_02_hc_regrid = xc.regrid(ec_02_hc, obs.coords['Lon'].values, obs.coords['Lat'].values, x_sample_dim= 'member')
    ec_04_hc_regrid = xc.regrid(ec_04_hc, obs.coords['Lon'].values, obs.coords['Lat'].values, x_sample_dim= 'member')

    # Set variable name in xarray data array
    ec_02_fc_regrid.name = var
    ec_04_fc_regrid.name = var
    ec_02_hc_regrid.name = var
    ec_04_hc_regrid.name = var 
    
    # Merge the ECMWF 02 and 04 datasets
    ec_fc = xr.merge((ec_02_fc_regrid, ec_04_fc_regrid))[var]
    
    del ec_02_fc_regrid, ec_04_fc_regrid, ec_02_fc, ec_04_fc
    
    ec_hc = xr.merge((ec_02_hc_regrid, ec_04_hc_regrid))[var]
    
    # Remove the old variables
    del ec_02_hc_regrid, ec_04_hc_regrid, ec_02_hc, ec_04_hc
    
    # Generate daily values, use a time zone offset of 6 hours, so the daily
    # value is calculated from 6UTC-6UTC to match the Bangladesh day best
    resample = varnames[var]['resample']
    len_1yr = len(ec_fc)
    nr_years = int(len(ec_hc) / len_1yr)
    
    print('Resample data to daily values')
    if resample == 'max':
        ec_fc_daily = ec_fc.resample(time='24H', base=6).max('time')
        for yy in range(nr_years):
            ec_hc_daily_yr = ec_hc[len_1yr*yy:len_1yr*(yy+1)].resample(time='24H', base=6).max('time')
            if yy == 0:
                ec_hc_daily = ec_hc_daily_yr
            else:
                ec_hc_daily = xr.merge((ec_hc_daily, ec_hc_daily_yr))
    elif resample == 'min':
        ec_fc_daily = ec_fc.resample(time='24H', base=6).min('time')
        for yy in range(nr_years):
            ec_hc_daily_yr = ec_hc[len_1yr*yy:len_1yr*(yy+1)].resample(time='24H', base=6).min('time')
            if yy == 0:
                ec_hc_daily = ec_hc_daily_yr
            else:
                ec_hc_daily = xr.merge((ec_hc_daily, ec_hc_daily_yr))
    elif resample == 'sum':
        ec_fc_daily = ec_fc.resample(time='24H', base=6).sum('time')
        for yy in range(nr_years):
            ec_hc_daily_yr = ec_hc[len_1yr*yy:len_1yr*(yy+1)].resample(time='24H', base=6).sum('time')
            if yy == 0:
                ec_hc_daily = ec_hc_daily_yr
            else:
                ec_hc_daily = xr.merge((ec_hc_daily, ec_hc_daily_yr))
    else:
        raise(Exception(f'Unkown resample type {resample}'))
    
    del ec_fc, ec_hc
    print('Take overlapping time steps')
    # Set ec_hc_daily back to DataArray
    ec_hc_daily = ec_hc_daily.to_array()[0]
    
    # Change the ECMWF time array with 6 hours because of time difference
    ec_fc_daily = ec_fc_daily.assign_coords(time=ec_fc_daily.time - np.timedelta64(6, 'h'))
    ec_hc_daily = ec_hc_daily.assign_coords(time=ec_hc_daily.time - np.timedelta64(6, 'h'))
    
    # Start the observations at the same time as the hindcasts
    obs = obs[obs.time >= np.datetime64(ec_hc_daily.time.values[0])-np.timedelta64(2, 'D')]
    
    # Take the data from the observations to create the same time series
    obs = obs.assign_coords({'doy': xr.DataArray([datetime.datetime.utcfromtimestamp(i.tolist()/1e9).strftime('%m-%d') for i in obs.coords['time'].values], dims='time')})
    ec_fc_daily = ec_fc_daily.assign_coords({'doy': xr.DataArray([datetime.datetime.utcfromtimestamp(i.tolist()/1e9).strftime('%m-%d') for i in ec_fc_daily.coords['time'].values], dims='time')})
    ec_hc_daily = ec_hc_daily.assign_coords({'doy': xr.DataArray([datetime.datetime.utcfromtimestamp(i.tolist()/1e9).strftime('%m-%d') for i in ec_hc_daily.coords['time'].values], dims='time')})
    
    doy_fc = np.unique(ec_fc_daily.coords['doy'].values)
    days_to_take = [obs.doy[ii].values in doy_fc for ii in range(len(obs.doy))]
    
    obs = obs[days_to_take]
    
    # For now, only take 19 years, because 2022 data is not yet included from BMD gridded data
    obs = obs[:int(19*len(ec_hc_daily.time)/20)]
    ec_hc_daily = ec_hc_daily[:int(19*len(ec_hc_daily.time)/20)]
    
    print('Save the data')
    fn_hc = f"{output}ecmwf_hc_regrid_{var}_{modeldatestr}.nc"
    fn_fc = f"{output}ecmwf_fc_regrid_{var}_{modeldatestr}.nc"
    fn_obs = f"{output}obs_ecmwf_{var}_{modeldatestr}.nc"

    obs.to_netcdf(fn_obs)
    ec_fc_daily.to_netcdf(fn_fc)
    ec_hc_daily.to_netcdf(fn_hc)

    del obs, ec_fc_daily, ec_hc_daily
    