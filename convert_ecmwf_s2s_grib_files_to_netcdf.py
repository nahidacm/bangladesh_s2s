#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 16 13:41:39 2022

Convert the ECMWF GRIB files to netcdf.

The script should run twice a week on Monday and Thursday evening.

@author: bob
"""
import pygrib
import datetime
import numpy as np
import xarray as xr
import os
from configparser import ConfigParser

# Read the basic paths from the config file
config = ConfigParser()
config.read('config_bd_s2s.ini')

# Set the directories from the config file
input_dir = config['paths']['ecmwf_grib']
output_dir = config['paths']['s2s_dir'] + 'input_ecmwf/'
api_dir = config['paths']['wi_api']


# Define the filename starts for forecast and hindcast
fn_fc_start = 'A2F'
fn_hcst_start = 'A2H'

# Set the modeldate at today 0 UTC. The script should run at Monday and Thursday
# evening, as the new data is available from those moments.
modeldate = datetime.datetime.today()
modeldate = datetime.datetime(modeldate.year,
                              modeldate.month,
                              modeldate.day,
                              0,0)

# Set the modeldate as string for input and output data
modeldatestr = modeldate.strftime('%m%d')
output_modeldatestr = modeldate.strftime('%Y%m%d')

# Define the middle and end part of the input filenames
fn_middle = '0000'
fn_end = '____1'

# Set start and end day
start_day = 0
end_day = 40

# List the variable names with a key and name in the ECMWF files
varnames = {'tmax': 'Maximum temperature at 2 metres since last 6 hours',
            'tmin': 'Minimum temperature at 2 metres since last 6 hours',
            'tp': 'Total precipitation'}

# Loop over the forecasts and the hindcasts
for fc_type in ['fc', 'hc']:
    
    if fc_type == 'fc':
        fn_start = fn_fc_start
    elif fc_type == 'hc':
        fn_start = fn_hcst_start
    else:
        raise(ValueError(f'Unkown fc_type {fc_type}.'))

    # Loop over all variables
    for var_key in varnames.keys():
        
        # Select the variable name
        varname = varnames[var_key]
        
        # Create an empty array for the data and the ensemble number
        forecast = np.array([])
        dates = np.array([])
        
        # Loop over all forecast days (there is one file per day)
        for dd in range(start_day,end_day+1):
            
            # Set the validdate
            validdate = modeldate + datetime.timedelta(dd)
            validdatestr = validdate.strftime('%m%d')
            
            # Construct the filename from all different parts
            fn_fc = input_dir + fn_start + modeldatestr + fn_middle + validdatestr + fn_end
            
            # Create an array with the startsteps of this file
            if var_key in ['tmax', 'tmin']:
                if dd == 0:
                    # The first file contains 3 time steps
                    startsteps = dd*24+np.array([0,6,12])
                else:
                    startsteps = dd*24+np.array([-6,0,6,12])
            elif var_key == 'tp':        
                startsteps = dd*24+np.array([0,6,12,18])
            
            # Loop over all time steps during this day
            for step in startsteps:
                f_step_ensnr = np.array([])
                f_step_ens = np.array([])
                
                # Open the grib file
                grbs = pygrib.open(fn_fc)
                
                # Loop over all grib messages
                for grb in grbs:
                    
                    # Only take the data from this variable and timestep
                    if grb.parameterName == varname and grb.startStep == step:
                        
                        data, lats, lons = grb.data()
                        unit = grb.units
                        
                        try:
                            f_step_ens = np.dstack((f_step_ens,data))
                        except ValueError: #if "f_step_ens" is not yet defined because it is the first ensemble
                            f_step_ens = data
                        
                        # Store the ensemble member number to sort the ensembles later,
                        if fc_type == 'fc':
                            f_step_ensnr = np.append(f_step_ensnr,int(grb.perturbationNumber))
                        elif fc_type == 'hc':
                            f_step_ensnr = np.append(f_step_ensnr,str(grb.validDate.year)+'-{:02d}'.format(int(grb.perturbationNumber)))
                
                # First change dimensions of the array f_step_ens
                # Store the ens_numbers in axis=0 instead of axis=2
                f_step_ens = np.swapaxes(f_step_ens,0,2)
                f_step_ens = np.swapaxes(f_step_ens,1,2)
                
                # Forecasts are now stored in random sequence of ensembles.     
                # We now sort them according to ensemble number
                f_step_ens = np.array(f_step_ens)
                f_step_ensnr = np.array(f_step_ensnr)
        
                ens_inds = f_step_ensnr.argsort() #find indices of sorted ensembles
                f_step_ensnr = f_step_ensnr[ens_inds] #sort the ensembles_nrs array
                f_step_ens = f_step_ens[ens_inds,:,:] #sort the forecast array in ensembles           
                
                if fc_type == 'fc':
                    # Add a time-dimension at axis=0 to the f_step_ens before concatenating different timesteps
                    f_step_ens = np.expand_dims(f_step_ens,axis=0)

                elif fc_type == 'hc':
                    # Set the time from the year + validdate
                    f_step_ensnr = f_step_ensnr.reshape((20,11))
                    f_step_ens = f_step_ens.reshape((20,11)+f_step_ens.shape[1:])
                
                
                # Append the ensemble cube to the cube containing also a time dimension
                try:
                    forecast = np.concatenate((forecast,f_step_ens),axis=0)
                    if fc_type == 'fc':
                        dates = np.append(dates,modeldate + datetime.timedelta(step/24))
                    elif fc_type == 'hc':
                        # Add the hindcastdates from ensemble member and validdate
                        curr_date = modeldate + datetime.timedelta(step/24)
                        add_dates = np.array([datetime.datetime(int(f_step_ensnr[ii,0][:4]),modeldate.month,modeldate.day,modeldate.hour) for ii in range(20)])
                        add_dates = add_dates + datetime.timedelta(step/24)
                        
                        dates = np.append(dates,add_dates)

                except ValueError: #if "forecast" does not yet have dimensions because it is the first timestep
                    forecast = f_step_ens
                    if fc_type == 'fc':
                        dates =  modeldate + datetime.timedelta(step/24)
                    elif fc_type == 'hc':
                        # Add the hindcastdates from ensemble member and validdate
                        curr_date = modeldate + datetime.timedelta(step/24)
                        add_dates = np.array([datetime.datetime(int(f_step_ensnr[ii,0][:4]),modeldate.month,modeldate.day,modeldate.hour) for ii in range(20)])
                        add_dates = add_dates + datetime.timedelta(step/24)
                        
                        dates = add_dates
                
                if dd == 15 and step == startsteps[0]:
                    # Save the data after this step in a separate array
                    # From the next step the resolution will become 0.4 instead of 0.2
                    
                    if fc_type == 'hc':
                        # Sort the dates
                        dates_ind = dates.argsort()
                        forecast = forecast[dates_ind]
                        dates = dates[dates_ind]

                    # Convert units
                    if unit == 'K':
                        forecast = forecast - 273.15
                        unit = 'deg C'
                    elif unit == 'm':
                        forecast = 1000 * forecast
                        unit = 'mm'
                        
                    member = range(forecast.shape[1])
                    

                    latvec_02 = lats[:,0]
                    lonvec_02 = lons[0]
                    dates_02 = dates
                    xr02 = xr.DataArray(data=forecast,
                                        coords=(dates_02, member, latvec_02, lonvec_02),
                                        dims=('time', 'member', 'latitude', 'longitude'))

        if fc_type == 'hc':
            # Sort the dates
            dates_ind = dates.argsort()
            forecast = forecast[dates_ind]
            dates = dates[dates_ind]

        # Convert units
        if unit == 'K':
            forecast = forecast - 273.15
            unit = 'deg C'
        elif unit == 'm':
            forecast = 1000* forecast
            unit = 'mm'
                        
        member = range(forecast.shape[1])
            
        # Save the final time steps with the 0.4 degree resolution
        latvec_04 = lats[:,0]
        lonvec_04 = lons[0]
        dates_04 = dates
        xr04 = xr.DataArray(data=forecast,
                            coords=(dates_04, member, latvec_04, lonvec_04),
                            dims=('time', 'member', 'latitude', 'longitude'))
    
        # Save the data in netcdf
        xr02.to_netcdf(f'{output_dir}ecmwf_{fc_type}_{var_key}_{output_modeldatestr}_02.nc')
        xr04.to_netcdf(f'{output_dir}ecmwf_{fc_type}_{var_key}_{output_modeldatestr}_04.nc')
        
        # Save the data on the wi_api, so BMD can download it from there
        datasource = 'bangladesh_s2s'
        dataset02 = f'ecmwf_{fc_type}_{var_key}_02'
        dataset04 = f'ecmwf_{fc_type}_{var_key}_04'
        
        api_dir_02 = f'{api_dir}{datasource}/{dataset02}/netcdf/en/'
        fn_02 = f'{datasource}.{dataset02}.netcdf.en_{output_modeldatestr}0000.nc'
        api_dir_04 = f'{api_dir}{datasource}/{dataset04}/netcdf/en/'
        fn_04 = f'{datasource}.{dataset04}.netcdf.en_{output_modeldatestr}0000.nc'
        
        if not os.path.exists(api_dir_02):
            os.makedirs(api_dir_02)
        if not os.path.exists(api_dir_04):
            os.makedirs(api_dir_04)
        
        xr02.to_netcdf(api_dir_02 + fn_02)
        xr04.to_netcdf(api_dir_04 + fn_04)
        
        