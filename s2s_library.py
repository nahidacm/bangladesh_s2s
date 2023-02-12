#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb  6 09:01:33 2023

@author: bob
"""
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib as mpl
import cmocean
import cartopy.crs as ccrs
import cartopy.feature as cf
import cartopy.mpl.ticker as cticker
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from osgeo import osr,gdal
from pyproj import Proj, transform
import collections
import os
from shapely.geometry import shape, Polygon

def combine_models(ecmwf_available, eccc_available, var,
                   obs_ecm, obs_ecc,
                   ecm_fc_daily, ecc_fc_daily,
                   ecm_hc_daily, ecc_hc_daily):
    '''
    Function that combines the multiple models to a single dataset.
    Currently only ECMWF and ECCC are implemented. Later on, other datasets
    from other centres will be included.
    
    Input:
        ecmwf_available: boolean: states if ecmwf data is available
        eccc_available: boolean: states if eccc data is available
        var: str: set the variable name
        obs_ecm: xr.DataArray: the observations matching the ECMWF time steps
        obs_ecc: xr.DataArray: the observations matching the ECCC time steps
        ecm_fc_daily: xr.DataArray: the ECMWF forecast
        ecc_fc_daily: xr.DataArray: the ECCC forecast
        ecm_hc_daily: xr.DataArray: the ECMWF hindcasts
        ecc_hc_daily: xr.DataArray: the ECCC hindcasts
    
    Output:
        obs: xr.DataArray: the observations matching the hindcast data timesteps
        fc_daily: xr.DataArray: the multi-model forecast
        hc_daily: xr.DataArray: the multi-model hindcasts
        models: str: string with the models that are used
    '''
    
    if ecmwf_available == True and eccc_available == True:
        # Combine the ECMWF and ECCC data
    
        # Check for overlapping dates in the observations and hindcasts
        days_to_take_ecm = [obs_ecm.time[ii].values in obs_ecc.time.values for ii in range(len(obs_ecm.time))]
        days_to_take_ecc = [obs_ecc.time[ii].values in obs_ecm.time.values for ii in range(len(obs_ecc.time))]
        
        # And take the overlapping forecast time steps
        days_to_take_ecm_fc = [ecm_fc_daily.time[ii].values in ecc_fc_daily.time.values for ii in range(len(ecm_fc_daily.time))]
        days_to_take_ecc_fc = [ecc_fc_daily.time[ii].values in ecm_fc_daily.time.values for ii in range(len(ecc_fc_daily.time))]
        
        # Take out the overlapping dates
        obs_ecm = obs_ecm[days_to_take_ecm]
        obs_ecc = obs_ecc[days_to_take_ecc]
        
        ecm_hc_daily = ecm_hc_daily[days_to_take_ecm]
        ecc_hc_daily = ecc_hc_daily[days_to_take_ecc]
        
        ecm_fc_daily = ecm_fc_daily[days_to_take_ecm_fc]
        ecc_fc_daily = ecc_fc_daily[days_to_take_ecc_fc]
        
        # Set eccc members after ecmwf members to avoid conflicts with merging
        ecc_fc_daily = ecc_fc_daily.assign_coords(member=ecc_fc_daily['member']+ecm_fc_daily['member'][-1]+1)
        ecc_hc_daily = ecc_hc_daily.assign_coords(member=ecc_hc_daily['member']+ecm_hc_daily['member'][-1]+1)
        
        # Set names to merge datasets
        ecm_fc_daily.name = var
        ecc_fc_daily.name = var
        ecm_hc_daily.name = var
        ecc_hc_daily.name = var
        
        # Merge datasets
        obs = obs_ecm.copy()
        fc_daily = xr.merge((ecm_fc_daily, ecc_fc_daily))
        hc_daily = xr.merge([ecm_hc_daily, ecc_hc_daily])
        
        # Set back to DataArray
        fc_daily = fc_daily.to_array()[0]
        hc_daily = hc_daily.to_array()[0]
    
        # Remove old variables
        del obs_ecm, obs_ecc, ecm_fc_daily, ecc_fc_daily, ecm_hc_daily, ecc_hc_daily
        
        models = 'multi_model'
    
    elif ecmwf_available == True and eccc_available == False:

        # Only take ECMWF datasets
        obs = obs_ecm.copy()
        fc_daily = ecm_fc_daily.copy()
        hc_daily = ecm_hc_daily.copy()
        
        models = 'ecmwf'
        
    elif ecmwf_available == False and eccc_available == True:

        # Only take ECCC datasets
        obs = obs_ecc.copy()
        fc_daily = ecc_fc_daily.copy()
        hc_daily = ecc_hc_daily.copy()
    
        models = 'eccc'
    
    else:
        
        # Return Nones
        obs = None
        fc_daily = None
        hc_daily = None
        
        models = 'No data'
    
    return obs, fc_daily, hc_daily, models

def resample_data(obs, fc_daily, hc_daily, var, start_end_times, period, resample):
    '''
    Function to resample the data (observations, forecast and hindcast ) to 
    weekly values, based on the resample method for the variable and the start
    and end times.
    
    Input:
        obs: xr.DataArray: the daily observations
        fc_daily: xr.DataArray: the daily forecast data
        hc_daily: xr.DataArray: the daily hindcast data
        var: str: the variable name
        start_end_times: dict: dictionary with start and end times (in days) per period
        period: str: period to resample to
        resample: str: resample method. Can be max, min, mean or sum
    
    Output:
        obs_wk: xr.DataArray: weekly observational values
        fc_wk: xr.DataArray: weekly forecast values
        hc_wk: xr.DataArray: weekly hindcast values
    '''
    
    # Select and agregate to period
    start = start_end_times[period]['start']
    end = start_end_times[period]['end']
    
    len_yr = len(fc_daily)
    n_years = int(len(obs)/len(fc_daily))
    
    if resample == 'max':
        fc_wk = fc_daily[start:end].max('time')
        for yy in range(n_years):
            obs_wk_yr = obs[len_yr*yy+start:len_yr*yy+end].max('time')
            hc_wk_yr = hc_daily[len_yr*yy+start:len_yr*yy+end].max('time')
            
            # Add variable name and expand dimensions
            obs_wk_yr.name = var
            hc_wk_yr.name = var
            obs_wk_yr = obs_wk_yr.expand_dims({'time':obs.time.values[len_yr*yy+1:len_yr*yy+2]}).to_dataset()
            hc_wk_yr = hc_wk_yr.expand_dims({'time':obs.time.values[len_yr*yy+1:len_yr*yy+2]}).to_dataset()
            
            if yy == 0:
                obs_wk = obs_wk_yr
                hc_wk = hc_wk_yr
            else:
                obs_wk = xr.merge((obs_wk, obs_wk_yr))
                hc_wk = xr.merge((hc_wk, hc_wk_yr))
    
    elif resample == 'min':
        fc_wk = fc_daily[start:end].min('time')
        for yy in range(n_years):
            obs_wk_yr = obs[len_yr*yy+start:len_yr*yy+end].min('time')
            hc_wk_yr = hc_daily[len_yr*yy+start:len_yr*yy+end].min('time')
            
            # Add variable name and expand dimensions
            obs_wk_yr.name = var
            hc_wk_yr.name = var
            obs_wk_yr = obs_wk_yr.expand_dims({'time':obs.time.values[len_yr*yy+1:len_yr*yy+2]}).to_dataset()
            hc_wk_yr = hc_wk_yr.expand_dims({'time':obs.time.values[len_yr*yy+1:len_yr*yy+2]}).to_dataset()
            
            if yy == 0:
                obs_wk = obs_wk_yr
                hc_wk = hc_wk_yr
            else:
                obs_wk = xr.merge((obs_wk, obs_wk_yr))
                hc_wk = xr.merge((hc_wk, hc_wk_yr))

    elif resample == 'mean':
        fc_wk = fc_daily[start:end].mean('time')
        for yy in range(n_years):
            obs_wk_yr = obs[len_yr*yy+start:len_yr*yy+end].mean('time')
            hc_wk_yr = hc_daily[len_yr*yy+start:len_yr*yy+end].mean('time')
            
            # Add variable name and expand dimensions
            obs_wk_yr.name = var
            hc_wk_yr.name = var
            obs_wk_yr = obs_wk_yr.expand_dims({'time':obs.time.values[len_yr*yy+1:len_yr*yy+2]}).to_dataset()
            hc_wk_yr = hc_wk_yr.expand_dims({'time':obs.time.values[len_yr*yy+1:len_yr*yy+2]}).to_dataset()
            
            if yy == 0:
                obs_wk = obs_wk_yr
                hc_wk = hc_wk_yr
            else:
                obs_wk = xr.merge((obs_wk, obs_wk_yr))
                hc_wk = xr.merge((hc_wk, hc_wk_yr))
                
    elif resample == 'sum':
        fc_wk = fc_daily[start:end].sum('time')
        for yy in range(n_years):
            obs_wk_yr = obs[len_yr*yy+start:len_yr*yy+end].sum('time')
            hc_wk_yr = hc_daily[len_yr*yy+start:len_yr*yy+end].sum('time')
            
            # Add variable name and expand dimensions
            obs_wk_yr.name = var
            hc_wk_yr.name = var
            obs_wk_yr = obs_wk_yr.expand_dims({'time':obs.time.values[len_yr*yy+1:len_yr*yy+2]}).to_dataset()
            hc_wk_yr = hc_wk_yr.expand_dims({'time':obs.time.values[len_yr*yy+1:len_yr*yy+2]}).to_dataset()
            
            if yy == 0:
                obs_wk = obs_wk_yr
                hc_wk = hc_wk_yr
            else:
                obs_wk = xr.merge((obs_wk, obs_wk_yr))
                hc_wk = xr.merge((hc_wk, hc_wk_yr))
    else:
        raise(Exception(f'Unkown resample type {resample}'))

    obs_wk = obs_wk.to_array()[0]
    hc_wk = hc_wk.to_array()[0]
    
    return obs_wk, fc_wk, hc_wk
        
# Define a function to plot skill scores
def plot_skill_score(value, obs, levels, cmap, extend, title, fig_dir, filename):
    '''
    Plot the skill scores of hindcast skill analysis.
    
    Input:
        value: np.array: array with the values to plot
        obs: xr.DataArray: the observational data, will be used for the coordinates
        levels: list: the levels to show in the figrue
        cmap: str or mpl.colormap: the colormap to use
        extend: str: extend option of matplotlib
        title: str: title of the figure
        fig_dir: str: directory to store the figure
        filename: str: filename of the figure
    
    Output:
        The figure is stored under filename in fig_dir
    '''

    plt.figure(figsize=(10,8.5))

    # Set the axes using the specified map projection
    ax=plt.axes(projection=ccrs.PlateCarree())
    ax.set_extent([87,93,20,27])
     
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
    plt.tight_layout()
    plt.savefig(fig_dir + filename)

def plot_forecast(var, period, deterministic_fc_smooth, deterministic_anomaly, 
                  probabilistic_fc_smooth, fig_dir_fc, modeldatestr):
    '''
    Make a figure with the deterministic, anomaly and probabilistic forecast 
    as png and as eps format.
    
    Input:
        var: str: variable code
        deterministic_fc_smooth: xr.DataArray: the smoothed deterministic forecast
        deterministic_anomaly: xr.DataArray: the deterministic anomaly forecast
        probabilictic_fc_smooth: xr.DataArray: the smoothed probabilistic forecast
        fig_dir_fc: str: the directory to store the figures
        modeldatestr: str: string with the modeldate
    
    Ouput:
        2 figures with the forecast are stored in fig_dir_fc:
            a png figure
            a eps figure (vector format)
    '''
    
    # Set the metadata for the figures
    if var in ['tmin', 'tmax']:
        cmap_det = cmocean.cm.thermal
        cmap_anom = 'RdBu_r'
        cmap_below = 'Blues'
        cmap_above = 'YlOrRd'
        levels_det = np.linspace(5,45,41)
        ticks_det = np.linspace(5,45,5)
        levels_anom = np.linspace(-6,6,25)
        ticks_anom = np.linspace(-6,6,5)
        label_det = u'Temperature (\N{DEGREE SIGN}C)'
        label_anom = u'Temperature anomaly (\N{DEGREE SIGN}C)'
    elif var == 'tp':
        cmap_det = cmocean.cm.haline_r
        cmap_anom = 'BrBG'
        cmap_below = 'YlOrRd'
        cmap_above = 'Greens'
        levels_det = np.linspace(0,400,17)
        ticks_det = np.linspace(0,400,5)
        levels_anom = np.linspace(-50,50,21)
        ticks_anom = np.linspace(-50,50,5)
        label_det = 'Precipitation (mm)'
        label_anom = 'Precipitation amomaly (mm)'

    # Preprocess the probabilistic data
    bn_fc = 100*probabilistic_fc_smooth[0,0]
    nn_fc = 100*probabilistic_fc_smooth[1,0]
    an_fc = 100*probabilistic_fc_smooth[2,0]
    
    levels_outer = np.linspace(40,80,9)
    levels_inner = np.linspace(40,50,3)
    
    cmap_nn = plt.get_cmap('Greys').copy()
    cmap_nn.set_over('lightgray')
    norm_nn = mpl.colors.Normalize(vmin=30,vmax=100)
    
    # Make the figure
    fig, axes = plt.subplots(1,3, figsize=(12,5), subplot_kw={'projection': ccrs.PlateCarree()})
    
    # Set the axes using the specified map projection
    for ax in axes:
        ax.set_extent([88.0,92.7,20.6,26.7])
        ax.set_axis_off()
        ax.coastlines(zorder=5)
        ax.add_feature(cf.BORDERS, zorder=5)
     
    # Make a filled contour plot
    cs0=axes[0].contourf(deterministic_fc_smooth['longitude'], deterministic_fc_smooth['latitude'], deterministic_fc_smooth,
                         transform = ccrs.PlateCarree(), levels = levels_det, 
                         cmap=cmap_det, extend='both')

    cax0 = inset_axes(axes[0], width='100%', height='5%', loc='lower left', 
                      bbox_to_anchor=(0., -0.05, 1, 1), bbox_transform=axes[0].transAxes, borderpad=0.1)
    cbar0 = plt.colorbar(cs0, ax=axes[0], cax=cax0, orientation='horizontal', ticks=ticks_det)
    cbar0.set_label(label_det, fontsize=14)
    cbar0.ax.tick_params(labelsize=12)

    # Make a filled contour plot
    cs1=axes[1].contourf(deterministic_anomaly['longitude'], deterministic_anomaly['latitude'], deterministic_anomaly,
                         transform = ccrs.PlateCarree(), levels = levels_anom, 
                         cmap=cmap_anom, extend='both')
    cax1 = inset_axes(axes[1], width='100%', height='5%', loc='lower left', 
                      bbox_to_anchor=(0., -0.05, 1, 1), bbox_transform=axes[1].transAxes, borderpad=0.1)
    cbar1 = plt.colorbar(cs1, ax=axes[1], cax=cax1, orientation='horizontal', ticks=ticks_anom)
    cbar1.set_label(label_anom, fontsize=14)
    cbar1.ax.tick_params(labelsize=12)
    
    # Make a filled contour plot
    cs_nn=axes[2].contourf(nn_fc['longitude'], nn_fc['latitude'], nn_fc, transform = ccrs.PlateCarree(),
                           levels = levels_inner, cmap=cmap_nn, norm=norm_nn, extend='max')
    cs_bn=axes[2].contourf(bn_fc['longitude'], bn_fc['latitude'], bn_fc, transform = ccrs.PlateCarree(),
                           levels = levels_outer, cmap=cmap_below, extend='max')
    cs_an=axes[2].contourf(an_fc['longitude'], an_fc['latitude'], an_fc, transform = ccrs.PlateCarree(),
                           levels = levels_outer, cmap=cmap_above, extend='max')
    
    cax2a = inset_axes(axes[2], width='35%', height='5%', loc='lower left', 
                       bbox_to_anchor=(0., -0.05, 1, 1), bbox_transform=axes[2].transAxes, borderpad=0.1)
    cax2b = inset_axes(axes[2], width='20%', height='5%', loc='lower center', 
                       bbox_to_anchor=(0., -0.05, 1, 1), bbox_transform=axes[2].transAxes, borderpad=0.07)
    cax2c = inset_axes(axes[2], width='35%', height='5%', loc='lower right', 
                       bbox_to_anchor=(0., -0.05, 1, 1), bbox_transform=axes[2].transAxes, borderpad=0.1)
    
    cbar2a = plt.colorbar(cs_bn, ax=axes[2], cax=cax2a, orientation='horizontal', ticks=[40,60,80])
    cbar2b = plt.colorbar(cs_nn, ax=axes[2], cax=cax2b, orientation='horizontal', ticks=[40,50])
    cbar2c = plt.colorbar(cs_an, ax=axes[2], cax=cax2c, orientation='horizontal', ticks=[40,60,80])

    cbar2a.ax.tick_params(labelsize=12)
    cbar2b.ax.tick_params(labelsize=12)
    cbar2c.ax.tick_params(labelsize=12)
    
    cbar2a.set_label('BN (%)', fontsize=14)
    cbar2b.set_label('NN (%)', fontsize=14)
    cbar2c.set_label('AN (%)', fontsize=14)

    axes[0].set_title('Deterministic forecast', fontsize=18)
    axes[1].set_title('Forecast anomaly', fontsize=18)
    axes[2].set_title('Probabilistic forecast', fontsize=18)

    plt.subplots_adjust(wspace=0.025, hspace=0.025, bottom=0.15)
    plt.tight_layout()
    plt.savefig(fig_dir_fc + f'fc_{var}_{period}_{modeldatestr}.eps', format='eps', bbox_inches='tight')    
    plt.savefig(fig_dir_fc + f'fc_{var}_{period}_{modeldatestr}.png', format='png', bbox_inches='tight')
    
def save_forecast(varname, varunit, data, lat, lon, datapath_output,
                  filename, projection='epsg:3857', fill_value=-9999):
    '''
    This function writes output data to a tif file
    
    Input
    -----
    varname: string
        contains the name of the variable
    varunit: string
        contains the variable unit
    data: array
        contains the data to be saved. The format is (time, lat, lon), where 
        the latitudes decrease
    lat: array
        contains the latitudes, from 90 N to 90 S.
    lon: array
        contains the longitudes
    timevec: array
        contains the time steps to be saved
    datapath_output: string
        contains the output directory where the tiff needs to be saved
    filename: string
        contains the filename of the tiff file (without .tiff)
    fill_value: float
        if no data is available at a point, the fill_value is used
    
    Returns
    -------
    The function writes a tif file with name 'varname.tiff' in the folder
    datapath_output
    '''
    
    if not filename[-4:] == 'tiff':
        filename = datapath_output + filename + '.tiff'
    else:
        filename = datapath_output + filename
        
    # set geotransform
    # projection 4326 means lat-lon coordinates
    # out projection can be chosen to preference: EPSG 3857 is WGS84  
    outProj = Proj(projection)
    inProj = Proj('epsg:4326')

    # Find max and min coordinates in new projection
    xmin,ymin = transform(inProj,outProj,lon.min(),lat.min())
    xmax,ymax = transform(inProj,outProj,lon.max(),lat.max())
    
    # Numbers of gridpoints in x and y
    nx = data.shape[1]
    ny = data.shape[0]
    
    # Find the resolution of the new grid and make the new grid
    # Divide by number minus one because one cell too little is taken into account
    # by taking xmax and xmin
    xres = (xmax - xmin) / float(nx-1)
    yres = (ymax - ymin) / float(ny-1)
    geotransform = (xmin, xres, 0, ymax, 0, -yres)
    
    # create the raster file
    dst_ds = gdal.GetDriverByName('GTiff').Create(filename, nx, ny, 1, 
                                                  gdal.GDT_Float32)
    dst_ds.SetDescription(varname) # Set varname as description
    dst_ds.SetGeoTransform(geotransform)    # specify coords
    srs = osr.SpatialReference()            # establish encoding
    srs.ImportFromEPSG(int(projection[-4:])) # Use correct epsg from outproj
    dst_ds.SetProjection(srs.ExportToWkt()) # export coords to file
    
    # Write the data to the raster band
    dst_ds.GetRasterBand(1).WriteArray(data[:,:])
    dst_ds.GetRasterBand(1).SetNoDataValue(fill_value)
    dst_ds.GetRasterBand(1).SetUnitType(varunit)
    dst_ds.FlushCache()                     # write to disk
    dst_ds = None                           # save, close




def integrate_grid_to_polygon(input_grid, mask_2d, time_axis = 0, method = 'average', quantile = None):
    """
    Function translates gridded data into values per polygon
    
    Input:
        input_grid - the data on a grid that needs to be interpoplated to polygon values
        mask_2d - mask that tells for each grid-cell how much it overlaps with a certain polygon.
                  created by the function mask_from_polygon. maks2d is unique for each polygon that
                  you want to calculate
        time_axis - axis that indicates if input_grid data is 2d or 3d
        method - integration technique, possible options;
                 average:  take the average values of the gridboxes in the polygon
                 max:      take the maximum value of the gridboxes in the polygon. 
                           In this method, only the gridboxes that are significantly 
                           overlapping with the polygon are considered
                 quentile: take a certain quantile value of the gridboxes in the polygon
    Returns: an integrated value per polygon (where the polygon is represented by mask_2d)
    
    """    

    if time_axis != 0: 
        input_grid = np.moveaxis(input_grid, time_axis, 0)
    
    mask_3d = np.repeat(          
            mask_2d[np.newaxis, :, :], 
            input_grid.shape[time_axis], 
            axis=0)
        
    if method == 'average':
        value_integrated = np.sum(input_grid * mask_3d, axis=(1, 2)) / np.sum(mask_2d)
        
    elif method == 'max':
        max_mask = np.max(mask_3d)
        mask_3d[mask_3d >= 0.5*max_mask] = 1.
        mask_3d[mask_3d < 0.5*max_mask] = 0.
        value_integrated = np.max(input_grid * mask_3d, axis=(1, 2))  
          
    elif method == 'quantile': 
        if quantile is None: 
            raise Exception('quantile not selected')
        if type(input_grid) is xr.core.dataarray.DataArray: 
            input_grid = input_grid.values
        weight = mask_2d.flatten()
        value_integrated = [weighted_quantile(input_grid[i_fc].flatten(), quantile, weight) for i_fc in range(input_grid.shape[time_axis])]
        value_integrated = np.array(value_integrated)
    else: 
        raise Exception('Invalid method.') 
        
        
    return(value_integrated)
      

def weighted_quantile(values, quantiles, sample_weight=None, values_sorted=False, old_style=False):      
    """ 
    This function calculates a weighted quantile. It is similar to numpy.percentile, but supports weights.
    This is used for obtaining a weighted quantile of the gridded risk values in each township
    The weights are based on the percentual overlap of each grid cell with the township shape.
    
    NOTE: quantiles should be in [0, 1]!
    
    Input
        values: numpy.array with data
        quantiles: array with the desired quantiles [0,1]
        sample_weight: array (same lenght as values) with the weights
        values_sorted: bool, if True, then will avoid sorting of initial array
        old_style: if True, will correct output to be consistent with numpy.percentile.
        
     Returns   
         numpy.array with computed quantiles.
    
    source: https://stackoverflow.com/questions/21844024/weighted-percentile-using-numpy
    theoretical background: https://stackoverflow.com/questions/21844024/weighted-percentile-using-numpy
    """
    
    values = np.array(values)
    quantiles = np.array(quantiles)
    if sample_weight is None:
        sample_weight = np.ones(len(values))
    sample_weight = np.array(sample_weight)
    assert np.all(quantiles >= 0) and np.all(quantiles <= 1), 'quantiles should be in [0, 1]'

    if not values_sorted:
        sorter = np.argsort(values)
        values = values[sorter]
        sample_weight = sample_weight[sorter]

    weighted_quantiles = np.cumsum(sample_weight) - 0.5 * sample_weight
    if old_style:
        # To be convenient with numpy.percentile
        weighted_quantiles -= weighted_quantiles[0]
        weighted_quantiles /= weighted_quantiles[-1]
    else:
        weighted_quantiles /= np.sum(sample_weight)
        
    return np.interp(quantiles, weighted_quantiles, values)

def load_polygons(shpfile, name_column, name_column_higher_level, 
                  lats=None, lons=None, shape_mask_dir=None, make_shapemask=True):
    '''
    Function to load the names of the polygons and to create shape masks from 
    a shapefile, if non-existent.
    
    For each polygon in the provided shape file, the name will be given. If the
    corresponding shape mask is not existent, the shape masks will be created.
    The shape mask gives the amount of overlap for each polygon in the shape 
    file on the provided input grid. The shape mask has values ranging from 0
    (no overlap between polygon and grid cell) to 1 (100% overlap).
    
    Input:
        shpfile: shapefile.Reader() object: the shapefile contents
        name_column: int: the column number which contains the name of the polygon
        name_column_higher_level: int: the column number of the name of one 
                administrative level higher. This is added if double names occur
        lats: array: the latitudes of the input grid
        lons: array: the longitudes of the input grid
        shape_mask_dir: str: the directory where the shape masks are saved
        
    Output:
        polygon_names: list: list with names of the polygons
        polygon_shapes: list: list with the shapes
    
    '''
    
    # Load the names of the polygons
    polygon_names = []
    polygon_shapes = []
    for i_t in range(shpfile.numRecords):
        polygon_name = shpfile.record(i_t)[name_column].title()
        polygon_names.append(polygon_name.replace('/','-'))
        polygon_shape = shpfile.shapeRecords()[i_t].shape
        polygon_shapes.append(polygon_shape)

    if not len(polygon_names) == len(set(polygon_names)):
        # If there are duplicates in the township names, add the name of the 
        # administrative level above
        duplicate_names = [item for item, count in collections.Counter(polygon_names).items() if count > 1]
        
        # Loop again over all township names:
        for i_t in range(len(polygon_names)):
            if polygon_names[i_t] in duplicate_names:
                # Add the higher level name
                polygon_names[i_t]  = polygon_names[i_t]  + '_' + shpfile.record(i_t)[name_column_higher_level]
                
    # Check again if all township names are unique
    assert len(polygon_names) == len(set(polygon_names))
     
    if make_shapemask == True:
        for i_p in range(shpfile.numRecords):
            if not os.path.isfile(shape_mask_dir + 'shape_mask_' + polygon_names[i_p].replace('/','-') + '.npy'):
                town_shape = shpfile.shapeRecords()[i_p]
                # We want a mask with just zeroes and ones, as landuse resolution is high enough (so fraction overlap not really important)
                # If more than 50% of a grid cell overlaps with township, this is rounded to 1 (integer). 
                # Otherwise, the overlap is rounded to 0 (integer)
                # This creates, for every township, a 2d mask (size of landuse file) with zeroes and ones.
                township_mask = mask_from_polygon(lons, lats, town_shape)
                
                
                if np.max(township_mask)<0.5:
                    # If the township is so small that it only falls into 1 gridpoint: ceil
                    township_mask = np.ceil(township_mask)
                township_mask = np.round(township_mask).astype(int)
                # township_mask = data_sanity_check(township_mask,0,1, 'township mask calculation')
                print('finished mask for shape nr ' + str(i_p+1))
                
                # make sure every township has overlap with at least one grid cell
                assert np.any(township_mask > 0.0)
                
                if not os.path.exists(shape_mask_dir):
                    os.makedirs(shape_mask_dir)
                
                np.save(shape_mask_dir + '/shape_mask_' + polygon_names[i_p].replace('/','-'),
                        township_mask)  
        
    # Extra check on '/'
    for ts in range(len(polygon_names)):
        if '/' in polygon_names[ts]:
            polygon_names[ts] = polygon_names[ts].replace('/','-')    
    
    return polygon_names, polygon_shapes


    
def mask_from_polygon(lon_in, lat_in, input_shape):   
    """
    calculate mask for every township containing per grid cell the percentage of overlap
    Ranges from 0 to 1
    0 = forecast grid cell and township do not overlap
    1 = forecast grid cell and township fully overlap
    0.3 = township covers 30% of the grid cell
    
    input:  latitude and longitude arrays
            shape from a shapefile of the form:
                <class 'shapefile.ShapeRecord'> or 
                <class 'shapely.geometry.polygon.Polygon'
    output: mask_2D; numpy array of the shape (lon, lat) containing percentage of overlap
    
    """
    
    llons, llats = np.meshgrid(lon_in, lat_in)
    mask_2D = np.zeros_like(llons)
    res = np.around(np.diff(llons)[0, 0], decimals=2)
    
    if str(type(input_shape)) == "<class 'shapefile.ShapeRecord'>": 
        multi = shape(input_shape.shape.__geo_interface__)
    elif str(type(input_shape)) == "<class 'shapely.geometry.polygon.Polygon'>": 
        multi = input_shape
    else:
       raise ValueError('this shape is not of the right class. Function only works \
                        for <class shapefile.ShapeRecord> or <class shapely.geometry.polygon.Polygon')

    for xx in range(llons.shape[0]):
        for yy in range(llons.shape[1]):
            x1 = llons[xx, yy] - res / 2
            x2 = llons[xx, yy] + res / 2
            y1 = llats[xx, yy] - res / 2
            y2 = llats[xx, yy] + res / 2
            poly_grid = Polygon([(x1, y1), (x1, y2), (x2, y2), (x2, y1)])
            mask_2D[xx, yy] = multi.intersection(
                    poly_grid).area / poly_grid.area
    
    return(mask_2D)
