#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 28 11:13:24 2022

Generate operational forecast from ECMWF.

Read in the prepared netcdf files with forecast and hindcast.
Regrid the data to the BMD gridded data-grid

Calibrate the forecast and fit current forecast.

NOTE: currently the script only contains the ECMWF forecast
TO DO: include multiple models in forecast

@author: bob
"""
import xarray as xr
import xcast as xc
import datetime
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import cartopy.crs as ccrs
import cartopy.feature as cf
import cartopy.mpl.ticker as cticker

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
fig_dir_hc = direc + 'output_figures_hindcast/'
fig_dir_fc = direc + 'output_figures_forecast/'

if not os.path.exists(fig_dir_hc):
    os.makedirs(fig_dir_hc)
if not os.path.exists(fig_dir_fc):
    os.makedirs(fig_dir_fc)

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

start_end_times = {'week1': {'start': 1,
                             'end': 8},
                   'week2': {'start': 8,
                             'end': 15},
                   'week3+4': {'start': 15,
                             'end': 29}
                   }

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
    ec_hc = xr.merge((ec_02_hc_regrid, ec_04_hc_regrid))[var]
    
    # Remove the old variables
    del ec_02_fc_regrid, ec_04_fc_regrid, ec_02_hc_regrid, ec_04_hc_regrid
    
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
    
    # Set ec_hc_daily back to DataArray
    ec_hc_daily = ec_hc_daily.to_array()[0]
    
    # Start the observations at the same time as the hindcasts
    obs = obs[obs.time >= np.datetime64(ec_hc_daily.time.values[0])-np.timedelta64(2, 'D')]
    
    # Take the data from the observations to create the same time series
    obs = obs.assign_coords({'doy': xr.DataArray([datetime.datetime.utcfromtimestamp(i.tolist()/1e9).strftime('%-j') for i in obs.coords['time'].values], dims='time')})
    ec_fc_daily = ec_fc_daily.assign_coords({'doy': xr.DataArray([datetime.datetime.utcfromtimestamp(i.tolist()/1e9).strftime('%-j') for i in ec_fc_daily.coords['time'].values], dims='time')})
    ec_hc_daily = ec_hc_daily.assign_coords({'doy': xr.DataArray([datetime.datetime.utcfromtimestamp(i.tolist()/1e9).strftime('%-j') for i in ec_hc_daily.coords['time'].values], dims='time')})
    
    doy_fc = np.unique(ec_fc_daily.coords['doy'].values)
    days_to_take = [obs.doy[ii] in doy_fc for ii in range(len(obs.doy))]
    
    obs = obs[days_to_take]
    
    # For now, only take 19 years, because 2022 data is not yet included from BMD gridded data
    obs = obs[:int(19*len(ec_hc_daily.time)/20)]
    ec_hc_daily = ec_hc_daily[:int(19*len(ec_hc_daily.time)/20)]
    
    # Loop over the different periods
    for period in start_end_times.keys():
        print(f'Generate forecast for {period}.')
        
        # Select and agregate to period
        start = start_end_times[period]['start']
        end = start_end_times[period]['end']
        
        len_yr = len(ec_fc_daily)
        n_years = int(len(obs)/len(ec_fc_daily))
        
        # Resample data to only specific period
        if resample == 'max':
            ec_fc_wk = ec_fc_daily[start:end].max('time')
            for yy in range(n_years):
                obs_wk_yr = obs[len_yr*yy+start:len_yr*yy+end].max('time')
                ec_hc_wk_yr = ec_hc_daily[len_yr*yy+start:len_yr*yy+end].max('time')
                
                # Add variable name and expand dimensions
                obs_wk_yr.name = var
                ec_hc_wk_yr.name = var
                obs_wk_yr = obs_wk_yr.expand_dims({'time':obs.time.values[len_yr*yy+1:len_yr*yy+2]}).to_dataset()
                ec_hc_wk_yr = ec_hc_wk_yr.expand_dims({'time':obs.time.values[len_yr*yy+1:len_yr*yy+2]}).to_dataset()
                
                if yy == 0:
                    obs_wk = obs_wk_yr
                    ec_hc_wk = ec_hc_wk_yr
                else:
                    obs_wk = xr.merge((obs_wk, obs_wk_yr))
                    ec_hc_wk = xr.merge((ec_hc_wk, ec_hc_wk_yr))
        
        elif resample == 'min':
            ec_fc_wk = ec_fc_daily[start:end].min('time')
            for yy in range(n_years):
                obs_wk_yr = obs[len_yr*yy+start:len_yr*yy+end].min('time')
                ec_hc_wk_yr = ec_hc_daily[len_yr*yy+start:len_yr*yy+end].min('time')
                
                # Add variable name and expand dimensions
                obs_wk_yr.name = var
                ec_hc_wk_yr.name = var
                obs_wk_yr = obs_wk_yr.expand_dims({'time':obs.time.values[len_yr*yy+1:len_yr*yy+2]}).to_dataset()
                ec_hc_wk_yr = ec_hc_wk_yr.expand_dims({'time':obs.time.values[len_yr*yy+1:len_yr*yy+2]}).to_dataset()
                
                if yy == 0:
                    obs_wk = obs_wk_yr
                    ec_hc_wk = ec_hc_wk_yr
                else:
                    obs_wk = xr.merge((obs_wk, obs_wk_yr))
                    ec_hc_wk = xr.merge((ec_hc_wk, ec_hc_wk_yr))

        elif resample == 'sum':
            ec_fc_wk = ec_fc_daily[start:end].sum('time')
            for yy in range(n_years):
                obs_wk_yr = obs[len_yr*yy+start:len_yr*yy+end].sum('time')
                ec_hc_wk_yr = ec_hc_daily[len_yr*yy+start:len_yr*yy+end].sum('time')
                
                # Add variable name and expand dimensions
                obs_wk_yr.name = var
                ec_hc_wk_yr.name = var
                obs_wk_yr = obs_wk_yr.expand_dims({'time':obs.time.values[len_yr*yy+1:len_yr*yy+2]}).to_dataset()
                ec_hc_wk_yr = ec_hc_wk_yr.expand_dims({'time':obs.time.values[len_yr*yy+1:len_yr*yy+2]}).to_dataset()
                
                if yy == 0:
                    obs_wk = obs_wk_yr
                    ec_hc_wk = ec_hc_wk_yr
                else:
                    obs_wk = xr.merge((obs_wk, obs_wk_yr))
                    ec_hc_wk = xr.merge((ec_hc_wk, ec_hc_wk_yr))
        else:
            raise(Exception(f'Unkown resample type {resample}'))
    
        obs_wk = obs_wk.to_array()[0]
        ec_hc_wk = ec_hc_wk.to_array()[0]
        
        # Add members to obs_wk
        obs_wk = obs_wk.expand_dims({"M":[0]})
        
        # Add one time step to ec_fc_wk
        ec_fc_wk = ec_fc_wk.expand_dims({'time': [modeldate]})
        
        # Calculate BMD tercile categories
        bdohc = xc.RankedTerciles()
        bdohc.fit(obs_wk)
        bd_ohc_wk = bdohc.transform(obs_wk)

        print('Start machine learning algoritm on hindcasts')
        ND = 10
        hidden_layer_size = 30
        activation = 'relu'
        preprocessing = 'minmax'
        window = 3
           
        elm_xval = []
        poelm_xval = []
        
        i_test=0
        for x_train, y_train, x_test, y_test in xc.CrossValidator(ec_hc_wk, obs_wk, window=window, x_feature_dim='member'):
            
            ohc_train = xc.RankedTerciles()
            ohc_train.fit(y_train)
            ohc_y_train = ohc_train.transform(y_train)
            
            elm = xc.rExtremeLearningMachine(ND=ND, hidden_layer_size=hidden_layer_size, activation=activation, preprocessing=preprocessing)
            elm.fit(x_train, y_train, rechunk=False, x_feature_dim='member')
            elm_preds = elm.predict(x_test, rechunk=False, x_feature_dim='member')
            elm_xval.append(elm_preds.isel(time=window // 2))
    
            poelm = xc.cPOELM(ND=ND, hidden_layer_size=hidden_layer_size, activation=activation, preprocessing=preprocessing)
            poelm.fit(x_train, ohc_y_train, rechunk=False, x_feature_dim='member')
            poelm_preds = poelm.predict_proba(x_test, rechunk=False, x_feature_dim='member')
            poelm_xval.append(poelm_preds.isel(time=window // 2))
            
            print(i_test)
            i_test += 1
        
        elm_hcst = xr.concat(elm_xval, 'time').mean('ND')
        poelm_hcst = xr.concat(poelm_xval, 'time').mean('ND')
        
        # Define a function to plot skill scores
        def plot_skill_score(value, levels, cmap, extend, title, filename):
        
            plt.figure(figsize=(11,8.5))
        
            # Set the axes using the specified map projection
            ax=plt.axes(projection=ccrs.PlateCarree())
             
            # Make a filled contour plot
            cs=ax.contourf(obs['Lon'], obs['Lat'], value,
                           transform = ccrs.PlateCarree(),levels = levels, cmap=cmap, extend=extend)
            plt.colorbar(cs)
            
            # Add coastlines
            ax.coastlines()
            ax.add_feature(cf.BORDERS)
            
            # Define the xticks for longitude
            ax.set_xticks(np.arange(87,93,2), crs=ccrs.PlateCarree())
            lon_formatter = cticker.LongitudeFormatter()
            ax.xaxis.set_major_formatter(lon_formatter)
            
            # Define the yticks for latitude
            ax.set_yticks(np.arange(20,27,2), crs=ccrs.PlateCarree())
            lat_formatter = cticker.LatitudeFormatter()
            ax.yaxis.set_major_formatter(lat_formatter)
        
            plt.title(title)
            plt.savefig(fig_dir_hc + filename)
            
        # Show skill of hindcasts
        print('Calculate and plot skill scores')

        pearson = np.squeeze( xc.Pearson(elm_hcst, obs_wk, x_feature_dim='member'), axis=[2,3])
        levels_pearson = np.linspace(-1,1,21)
        plot_skill_score(pearson, levels_pearson, 'RdBu', 'both', 
                         f'Pearson correlation for {var} {period}',
                         f'hc_{var}_pearson_{period}_{modeldatestr}.png')
        
            
        ioa = np.squeeze( xc.IndexOfAgreement(elm_hcst, obs_wk, x_feature_dim='member'), axis=[2,3])
        levels_ioa = np.linspace(0,1,11)
        plot_skill_score(ioa, levels_ioa, 'Blues', 'both',
                         f'IndexOfAgreement for {var} {period}',
                         f'hc_{var}_ioa_{period}_{modeldatestr}.png')        
                
        groc = np.squeeze(xc.GeneralizedROC( poelm_hcst, bd_ohc_wk, x_feature_dim='member'), axis = [2,3])
        levels_groc = np.linspace(0.5,1,11)
        cmapg = plt.get_cmap('autumn_r').copy()
        cmapg.set_under('lightgray')
        plot_skill_score(groc, levels_groc, cmapg, 'min',
                         f'Generalized ROC for {var} {period}',
                         f'hc_{var}_groc_{period}_{modeldatestr}.png')  
                  
        climatological_odds = xr.ones_like(poelm_hcst) * 0.333
        
        skillscore_prec_wk = np.squeeze(xc.RankProbabilityScore( poelm_hcst, bd_ohc_wk, x_feature_dim='member'), axis = [2,3])
        skillscore_climate_wk = np.squeeze(xc.RankProbabilityScore( climatological_odds, bd_ohc_wk, x_feature_dim='member'), axis=[2,3])
          
        rpss = 1 - skillscore_prec_wk / skillscore_climate_wk
        
        levels_rpss = np.linspace(0.,0.2,11)
        plot_skill_score(rpss, levels_rpss, cmapg, 'both',
                         f'RPSS for {var} {period}',
                         f'hc_{var}_rpss_{period}_{modeldatestr}.png') 
        
        skillscore_prec_wk = np.squeeze(xc.BrierScore( poelm_hcst, bd_ohc_wk, x_feature_dim='member'), axis = [3])
        skillscore_climate_wk = np.squeeze(xc.BrierScore( climatological_odds, bd_ohc_wk, x_feature_dim='member'), axis=[3])
     
        bss = 1 - skillscore_prec_wk / skillscore_climate_wk
        
        levels_bss = np.linspace(0.,1,11)
           
        for idx, cat in zip([0,1,2],['BN','NN','AN']):
            
            data_plot = bss[:,:,idx]
            
            plot_skill_score(data_plot, levels_bss, cmapg, 'min',
                             f'{cat} brier skill score for {var} {period}',
                             f'hc_{var}_bss_{cat}_{period}_{modeldatestr}.png') 
            
        
        # Reduce ensemble size of forecast from 51 to 11 members to match the hindcast size
        # xcast requires the same size of ensemble for a fit and predict
        # reduction of ensemble members is done by taking 0-100 percentile with steps of 10
        ec_fc_reduced = ec_fc_wk.quantile(np.linspace(0,1,11), dim='member').rename({'quantile':'member'})
        
        # Make forecast
        elm = xc.rExtremeLearningMachine(ND=ND, hidden_layer_size=hidden_layer_size, activation=activation, preprocessing=preprocessing)
        elm.fit(ec_hc_wk, obs_wk, rechunk=False, x_feature_dim='member')
        deterministic_forecast = elm.predict(ec_fc_reduced, rechunk=False, x_feature_dim='member').mean('ND')
        
        poelm = xc.cPOELM(ND=ND, hidden_layer_size=hidden_layer_size, activation=activation, preprocessing=preprocessing)
        poelm.fit(ec_hc_wk, bd_ohc_wk, rechunk=False, x_feature_dim='member')
        probabilistic_forecast = poelm.predict_proba(ec_fc_reduced, rechunk=False, x_feature_dim='member').mean('ND')
        
        # Plot the forecast
        deterministic_fc_smooth = xc.gaussian_smooth(deterministic_forecast, x_sample_dim='time', x_feature_dim='member',  kernel=3)
        deterministic_fc_smooth = deterministic_fc_smooth[0,0]
        deterministic_anomaly = deterministic_fc_smooth.values - obs_wk.mean('time').mean('M').values

        
        if var in ['tmin', 'tmax']:
            cmap = 'RdBu_r'
            cmap_below = 'Blues'
            cmap_above = 'YlOrRd'
            levels = np.linspace(-10,10,21)
        elif var == 'tp':
            cmap = 'BrBG'
            cmap_below = 'YlOrRd'
            cmap_above = 'Greens'
            levels = np.linspace(-50,50,21)
            

        plt.figure(figsize=(11,8.5))
        
        # Set the axes using the specified map projection
        ax=plt.axes(projection=ccrs.PlateCarree())
         
        # Make a filled contour plot
        cs=ax.contourf(obs['Lon'], obs['Lat'], deterministic_anomaly,
                       transform = ccrs.PlateCarree(),levels = levels, cmap=cmap, extend='both')
        plt.colorbar(cs)
        
        # Add coastlines
        ax.coastlines()
        ax.add_feature(cf.BORDERS)
        
        # Define the xticks for longitude
        ax.set_xticks(np.arange(87,93,2), crs=ccrs.PlateCarree())
        lon_formatter = cticker.LongitudeFormatter()
        ax.xaxis.set_major_formatter(lon_formatter)
        
        # Define the yticks for latitude
        ax.set_yticks(np.arange(20,27,2), crs=ccrs.PlateCarree())
        lat_formatter = cticker.LatitudeFormatter()
        ax.yaxis.set_major_formatter(lat_formatter)
    
        plt.title(f'Deterministic forecast for {var} {period}')
        plt.savefig(fig_dir_fc + f'det_fc_{var}_{period}_{modeldatestr}.png')
        
        probabilistic_fc_smooth = xc.gaussian_smooth(probabilistic_forecast, x_sample_dim='time', x_feature_dim='member',  kernel=3)
        bn_fc = 100*probabilistic_fc_smooth[0,0]
        nn_fc = 100*probabilistic_fc_smooth[1,0]
        an_fc = 100*probabilistic_fc_smooth[2,0]
        
        levels_outer = np.linspace(40,80,9)
        levels_inner = np.linspace(40,50,3)
        
        cmap_nn = plt.get_cmap('Greys').copy()
        cmap_nn.set_over('lightgray')
        norm_nn = mpl.colors.Normalize(vmin=30,vmax=100)
        
        plt.figure(figsize=(15,7))
        
        # Set the axes using the specified map projection
        ax=plt.axes(projection=ccrs.PlateCarree())
         
        # Make a filled contour plot
        cs_nn=ax.contourf(obs['Lon'], obs['Lat'], nn_fc, transform = ccrs.PlateCarree(),
                          levels = levels_inner, cmap=cmap_nn, norm=norm_nn, extend='max')
        cs_bn=ax.contourf(obs['Lon'], obs['Lat'], bn_fc, transform = ccrs.PlateCarree(),
                          levels = levels_outer, cmap=cmap_below, extend='max')
        cs_an=ax.contourf(obs['Lon'], obs['Lat'], an_fc, transform = ccrs.PlateCarree(),
                          levels = levels_outer, cmap=cmap_above, extend='max')
        
        cax1 = inset_axes(ax, width='35%', height='5%', loc='lower left', 
                          bbox_to_anchor=(-0, -0.15, 1, 1), bbox_transform=ax.transAxes, borderpad=0.1)
        cax2 = inset_axes(ax, width='20%', height='5%', loc='lower center', 
                          bbox_to_anchor=(-0, -0.15, 1, 1), bbox_transform=ax.transAxes, borderpad=0.1)
        cax3 = inset_axes(ax, width='35%', height='5%', loc='lower right', 
                          bbox_to_anchor=(-0, -0.15, 1, 1), bbox_transform=ax.transAxes, borderpad=0.1)
        
        cbar1 = plt.colorbar(cs_bn, ax=ax, cax=cax1, orientation='horizontal')
        cbar2 = plt.colorbar(cs_nn, ax=ax, cax=cax2, orientation='horizontal')
        cbar3 = plt.colorbar(cs_an, ax=ax, cax=cax3, orientation='horizontal')
        
        cbar1.set_label('BN (%)')
        cbar2.set_label('NN (%)')
        cbar3.set_label('AN (%)')

        # Add coastlines
        ax.coastlines()
        ax.add_feature(cf.BORDERS)
        
        # Define the xticks for longitude
        ax.set_xticks(np.arange(87,93,2), crs=ccrs.PlateCarree())
        lon_formatter = cticker.LongitudeFormatter()
        ax.xaxis.set_major_formatter(lon_formatter)
        
        # Define the yticks for latitude
        ax.set_yticks(np.arange(20,27,2), crs=ccrs.PlateCarree())
        lat_formatter = cticker.LatitudeFormatter()
        ax.yaxis.set_major_formatter(lat_formatter)
    
        ax.set_title(f'Probabilistic forecast for {var} {period}')
        plt.savefig(fig_dir_fc + f'prob_fc_{var}_{period}_{modeldatestr}.png')
