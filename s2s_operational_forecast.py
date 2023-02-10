#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 28 11:13:24 2022

Generate operational forecast from ECMWF.

Read in the prepared netcdf files with forecast and hindcast.
Regrid the data to the BMD gridded data-grid

Calibrate the forecast and fit current forecast.

NOTE: currently the script only contains the ECMWF and ECCC forecast
TO DO: include multiple models in forecast

@author: bob
"""
import xarray as xr
import xcast as xc
import datetime
import os
import numpy as np
import matplotlib.pyplot as plt
import shapefile
import geopandas as gpd
import pandas as pd
import json
import s2s_library as s2s

import warnings
warnings.filterwarnings('ignore')

from configparser import ConfigParser

# Read the basic paths from the config file
config = ConfigParser()
config.read('config_bd_s2s.ini')

# Set the directories from the config file
direc = config['paths']['s2s_dir'] 

input_dir = direc + 'input_regrid/'
fig_dir_hc = direc + 'output_figures_hindcast/'
fig_dir_fc = direc + 'output_figures_forecast/'
output_dir = direc + 'output_forecast/'

# Make output directories if not existent
if not os.path.exists(fig_dir_hc):
    os.makedirs(fig_dir_hc)
if not os.path.exists(fig_dir_fc):
    os.makedirs(fig_dir_fc)
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
    
# Take the modelrun of yesterday
modeldate = datetime.datetime.today() - datetime.timedelta(1)
modeldate = datetime.datetime(modeldate.year,
                              modeldate.month,
                              modeldate.day,
                              0,0)
modeldatestr = modeldate.strftime("%Y%m%d")

# List the variable names with some properties
varnames = {'tmax': {'obs_filename': 'tmax',
                     'obs_ncname': 'temp',
                     'resample': 'mean',
                     'unit': 'degrees C'},
            'tmin': {'obs_filename': 'tmin',
                     'obs_ncname': 'temp',
                     'resample': 'mean',
                     'unit': 'degrees C'},
            'tp': {'obs_filename': 'rr',
                   'obs_ncname': 'precip',
                   'resample': 'sum',
                   'unit': 'mm'}}

# Make a dictionary with start and end times per period
start_end_times = {'week1': {'start': 1,
                             'end': 8,
                             'wknr': '1'},
                   'week2': {'start': 8,
                             'end': 15,
                             'wknr': '2'},
                   'week3+4': {'start': 15,
                               'end': 29,
                               'wknr': '34'}
                   }

# Make a dictionary with information on the spatial aggregation
shape_info = {0: {'name_column': 1,             # Level 0 is the country
                  'name_column_higher': 0,
                  'name_shapes': 'country'},
              1: {'name_column': 3,             # Level 1 are divisions
                  'name_column_higher': 2,
                  'name_shapes': 'division'},
              2: {'name_column': 6,             # Level 2 are districts
                  'name_column_higher': 4,
                  'name_shapes': 'district'},
              3: {'name_column': 9,             # Level 3 are upazillas
                  'name_column_higher': 7,
                  'name_shapes': 'upazilla'},
              4: {'name_column': 9,             # Level 4 are unions
                  'name_column_higher': 8,
                  'name_shapes': 'union'}}

# Set level for shape aggregation, choose from 0, 1, 2, 3 or 4
shp_level = 1

# Load shape information
shp_fn = f'gadm41_BGD_shp/gadm41_BGD_{shp_level}.shp'
shp_file = shapefile.Reader(shp_fn)
shape_mask_dir = direc + f'shape_mask/{shp_level}/'
gpd_data = gpd.read_file(shp_fn)

shape_name = shape_info[shp_level]['name_shapes']
name_col = shape_info[shp_level]['name_column']
name_col_higher = shape_info[shp_level]['name_column_higher']

# Generate a pandas table for the divisional values
shape_data = pd.DataFrame(index=range(len(gpd_data)), columns=[shape_name, 'tmax_week1', 'tmin_week1', 'tp_week1',
                                                               'tmax_week2', 'tmin_week2', 'tp_week2', 
                                                               'tmax_week3+4', 'tmin_week3+4', 'tp_week3+4'])
shape_data[shape_name] = gpd_data.iloc[:,name_col].values
shape_data = shape_data.set_index(shape_name)

# Create an empty array for data integrated to divisions
fc_shp = {}

# Make a forecast for each variable
for var in varnames.keys():
    print(f'Start forecasts for {var}.')

    resample = varnames[var]['resample']
    varunit = varnames[var]['unit']
    
    ######################################################
    ##                                                  ##
    ##      Check data availability and load data       ##
    ##      NOTE THAT other models will be added        ##
    ##                                                  ##
    ######################################################
    
    # Try to open ECMWF data
    try:
        print('Load the regridded data')
        fn_hc = f"{input_dir}ecmwf_hc_regrid_{var}_{modeldatestr}.nc"
        fn_fc = f"{input_dir}ecmwf_fc_regrid_{var}_{modeldatestr}.nc"
        fn_obs = f"{input_dir}obs_ecmwf_{var}_{modeldatestr}.nc"
    
        obs_ecm = xr.open_dataarray(fn_obs)
        ecm_fc_daily = xr.open_dataarray(fn_fc)
        ecm_hc_daily = xr.open_dataarray(fn_hc)
        
        ecmwf_available = True
    except:
        ecmwf_available = False
        obs_ecm = None
        ecm_fc_daily = None
        ecm_hc_daily = None
        
    # Try to open ECCC data
    try:
        fn_hc = f"{input_dir}eccc_hc_regrid_{var}_{modeldatestr}.nc"
        fn_fc = f"{input_dir}eccc_fc_regrid_{var}_{modeldatestr}.nc"
        fn_obs = f"{input_dir}obs_eccc_{var}_{modeldatestr}.nc"
    
        obs_ecc = xr.open_dataarray(fn_obs)
        ecc_fc_daily = xr.open_dataarray(fn_fc)
        ecc_hc_daily = xr.open_dataarray(fn_hc)
        
        eccc_available = True
    except:
        eccc_available = False
        obs_ecc = None
        ecc_fc_daily = None
        ecc_hc_daily = None
    
    # Combine models
    obs, fc_daily, hc_daily, models = s2s.combine_models(ecmwf_available, eccc_available, var,
                                                         obs_ecm, obs_ecc,
                                                         ecm_fc_daily, ecc_fc_daily,
                                                         ecm_hc_daily, ecc_hc_daily)
    
    # If there is no data available, continue
    if models == 'No data':
        print('No data available, continue')
        continue
    
    # Set mask to mask all data points outside of Bangladesh
    # Use the Tmax observations to load this mask
    if var == 'tmax':
        mask = np.isnan(obs[0].values)
    print(f'Make the forecast for {models}')
    
    # Loop over the different periods (week 1, week 2 and week 3+4)
    for period in start_end_times.keys():
        print(f'Generate forecast for {period}.')
        wknr = start_end_times[period]['wknr']
        
        # Resample data to only specific period
        obs_wk, fc_wk, hc_wk = s2s.resample_data(obs, fc_daily, hc_daily, var,
                                                 start_end_times, period, resample)
                
        # Add members to obs_wk to be able to use it in X-Cast
        obs_wk = obs_wk.expand_dims({"M":[0]})
        
        # Add one time step to fc_wk to be able to use it in X-Cast
        fc_wk = fc_wk.expand_dims({'time': [modeldate]})
        
        # Calculate BMD tercile categories
        bdohc = xc.RankedTerciles()
        bdohc.fit(obs_wk)
        bd_ohc_wk = bdohc.transform(obs_wk)

        ######################################################
        ##                                                  ##
        ##     Do short skill analysis of hindcast data     ##
        ##                                                  ##
        ######################################################
    
        print('Start with calibration on hindcasts')
        window = 3
           
        mlr_xval = []
        mc_xval = []
        
        i_test=0
        # Do a cross validation of the hindcast models
        for x_train, y_train, x_test, y_test in xc.CrossValidator(hc_wk, obs_wk, window=window, x_feature_dim='member'):
            
            ohc_train = xc.RankedTerciles()
            ohc_train.fit(y_train)
            ohc_y_train = ohc_train.transform(y_train)
            
            mlr = xc.rMultipleLinearRegression()
            mlr.fit(x_train, y_train, rechunk=False, x_feature_dim='member')
            mlr_preds = mlr.predict(x_test.rename({'member': 'M'}), rechunk=False, x_feature_dim='M')
            mlr_xval.append(mlr_preds.isel(time=window // 2))
    
            mc = xc.cMemberCount()
            mc.fit(x_train.rename({'member': 'M'}), ohc_y_train, x_feature_dim='M')
            mc_preds = mc.predict(x_test, x_feature_dim='member')
            mc_xval.append(mc_preds.isel(time=window // 2))
            
            i_test += 1
            print('Loop '+str(i_test)+' / '+str((len(obs_wk.time))))
        
        # Combine the cross validated output to a single dataset
        mlr_hcst = xr.concat(mlr_xval, 'time').mean('ND')
        mc_hcst = xr.concat(mc_xval, 'time')
            
        # Show skill of hindcasts
        print('Calculate and plot skill scores of hindcast')

        # Calculate and plot Pearson Correlation
        pearson = np.squeeze( xc.Pearson(mlr_hcst, obs_wk, x_feature_dim='M'), axis=[2,3])
        levels_pearson = np.linspace(-1,1,21)
        s2s.plot_skill_score(pearson, obs, levels_pearson, 'RdBu', 'both', 
                              f'Pearson correlation for {var} {period}',
                              fig_dir_hc,
                              f'hc_{var}_pearson_{period}_{modeldatestr}_{models}.png')
        
        # Calculate and plot Index of AGreement
        ioa = np.squeeze( xc.IndexOfAgreement(mlr_hcst, obs_wk, x_feature_dim='M'), axis=[2,3])
        levels_ioa = np.linspace(0,1,11)
        s2s.plot_skill_score(ioa, obs, levels_ioa, 'Blues', 'both',
                              f'IndexOfAgreement for {var} {period}',
                              fig_dir_hc,
                              f'hc_{var}_ioa_{period}_{modeldatestr}_{models}.png')        
        
        # Calculate and plot Generalized ROC
        groc = np.squeeze(xc.GeneralizedROC( mc_hcst, bd_ohc_wk, x_feature_dim='member'), axis = [2,3])
        levels_groc = np.linspace(0.5,1,11)
        cmapg = plt.get_cmap('autumn_r').copy()
        cmapg.set_under('lightgray')
        s2s.plot_skill_score(groc, obs, levels_groc, cmapg, 'min',
                              f'Generalized ROC for {var} {period}',
                              fig_dir_hc,
                              f'hc_{var}_groc_{period}_{modeldatestr}_{models}.png')  
        
        # Calculate and plot Rank Probability Skill Score
        
        # Get the climatological RPS for the skill score
        climatological_odds = xr.ones_like(mc_hcst) * 0.333
        
        skillscore_prec_wk = np.squeeze(xc.RankProbabilityScore( mc_hcst, bd_ohc_wk, x_feature_dim='member'), axis = [2,3])
        skillscore_climate_wk = np.squeeze(xc.RankProbabilityScore( climatological_odds, bd_ohc_wk, x_feature_dim='member'), axis=[2,3])
          
        rpss = 1 - skillscore_prec_wk / skillscore_climate_wk
        
        levels_rpss = np.linspace(0.,0.2,11)
        s2s.plot_skill_score(rpss, obs, levels_rpss, cmapg, 'both',
                              f'RPSS for {var} {period}',
                              fig_dir_hc,
                              f'hc_{var}_rpss_{period}_{modeldatestr}_{models}.png') 
        
        # Calculate and plot the Brier Skill Score for all 3 categories (AN/NN/BN)
        skillscore_prec_wk = np.squeeze(xc.BrierScore( mc_hcst, bd_ohc_wk, x_feature_dim='member'), axis = [3])
        skillscore_climate_wk = np.squeeze(xc.BrierScore( climatological_odds, bd_ohc_wk, x_feature_dim='member'), axis=[3])
     
        bss = 1 - skillscore_prec_wk / skillscore_climate_wk
        
        levels_bss = np.linspace(0.,1,11)
           
        for idx, cat in zip([0,1,2],['BN','NN','AN']):
            
            data_plot = bss[:,:,idx]
            
            s2s.plot_skill_score(data_plot, obs, levels_bss, cmapg, 'min',
                                  f'{cat} brier skill score for {var} {period}',
                                  fig_dir_hc,
                                  f'hc_{var}_bss_{cat}_{period}_{modeldatestr}_{models}.png') 
            

        ######################################################
        ##                                                  ##
        ##            Make operational forecast             ##
        ##                                                  ##
        ######################################################
        
        print('Generate operational forecast')
        # Use Multiple Linear Regression for deterministic foreast
        mlrfc = xc.rMultipleLinearRegression()
        mlrfc.fit(hc_wk.mean('member').expand_dims({'M':[0]}), obs_wk, rechunk=False, x_feature_dim='M')
        deterministic_forecast = mlrfc.predict(fc_wk.mean('member').expand_dims({'M':[0]}), rechunk=False, x_feature_dim='M').mean('ND')
        
        # Use Member Count for probabilistic forecast        
        mc = xc.cMemberCount()
        mc.fit(hc_wk.rename({'member':'M'}), obs_wk, x_feature_dim='M')
        probabilistic_forecast = mc.predict(fc_wk, x_feature_dim='member')        
        
        # Calculate the smoothed deterministic forecast
        deterministic_fc_smooth = xc.gaussian_smooth(deterministic_forecast, x_sample_dim='time', x_feature_dim='M',  kernel=3)
        deterministic_fc_smooth = deterministic_fc_smooth[0,0]
        deterministic_fc_smooth.values[mask] = np.nan
        deterministic_anomaly = deterministic_fc_smooth - obs_wk.mean('time').mean('M').rename({'Lat': 'latitude', 'Lon': 'longitude'})
        
        # Calculate the smoothed probabilistic forecast
        probabilistic_fc_smooth = xc.gaussian_smooth(probabilistic_forecast, x_sample_dim='time', x_feature_dim='member',  kernel=3)
        probabilistic_fc_smooth.values[:,:,mask] = np.nan
        
        # Plot the forecast
        s2s.plot_forecast(var, period, deterministic_fc_smooth,
                          deterministic_anomaly, probabilistic_fc_smooth,
                          fig_dir_fc, modeldatestr)
        
        # Save forecast as geotiff
        filename = f'det_fc_{var}_{period}_{modeldatestr}.tiff'
        s2s.save_forecast(var, varunit, deterministic_fc_smooth.data, 
                          deterministic_fc_smooth.latitude.values, 
                          deterministic_fc_smooth.longitude.values, 
                          output_dir, filename)
        
        # Load shapefile to integrate the gridded values to (divisions/districts/etc)
        shape_names, shapes = s2s.load_polygons(shp_file, name_col, name_col_higher, obs_wk.Lat.values, obs_wk.Lon.values, shape_mask_dir)
        
        print('Integrate the grid variables to polygons.')
        
        # Loop over all the shape names
        shapedirlist = os.listdir(shape_mask_dir)
        for (maskfile, ss) in zip(shapedirlist, range(len(shapedirlist))):
            
            # Load the shape mask
            district_name = maskfile[11:-4]
            mask_2d = np.load(shape_mask_dir + maskfile, allow_pickle=True)
            
            # Set the data and the integration method (now average is chosen)
            input_grid = deterministic_fc_smooth.values
            input_grid[np.isnan(input_grid)] = 0.
            method = 'average'
            
            # Integrate the gridded value to a single value for this polygon
            key = var+wknr+'_'+district_name.lower()[:4]
            value = np.round(s2s.integrate_grid_to_polygon(np.array([input_grid]), 
                                                           mask_2d, time_axis = 0, 
                                                           method = method),
                             decimals=1)[0]
            
            # Set very low precipitation values at 0 mm
            if var == 'tp' and value < 5:
                value = 0
            
            # Store value in json and dataframe
            fc_shp[key] = value
            shape_data.at[district_name, var+'_'+period] = value
            

# Save the divisional output in a json file for the bulletin
with open(output_dir+f'divisional_forecast_{modeldatestr}.json', 'w') as jsfile:
    json.dump(fc_shp, jsfile)

# Save the divisional output in a csv format to be used on BAMIS
shape_data.to_csv(output_dir+f'divisional_forecast_{modeldatestr}.csv')
