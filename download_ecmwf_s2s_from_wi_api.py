#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 30 14:26:26 2022

Download the ECMWF S2S files from the Weather Impact API.
The data is available on Monday and Thursday evening around 22 UTC.

@author: bob
"""
import requests
import datetime
import os
from configparser import ConfigParser

# Read the basic paths from the config file
config = ConfigParser()
config.read('config_bd_s2s.ini')

# Set the directories from the config file
output_dir = config['paths']['s2s_dir'] + 'input_ecmwf/'

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

variables = ['tmax', 'tmin', 'tp']
resolutions = ['02','04']
fc_types = ['fc','hc']

base_url = "https://service.weatherimpact.com/api/data/bangladesh_s2s/"
header = {"authkey": "b19e4b0bf632513ab7e4637a696ba247"}

# Set modeldate at yesterday
modeldate = datetime.datetime.today() - datetime.timedelta(1)
modeldate = datetime.datetime(modeldate.year,
                              modeldate.month,
                              modeldate.day,
                              0,0)
modeldatestr_api = modeldate.strftime("%Y-%m-%d")
modeldatestr_out = modeldate.strftime("%Y%m%d")

# Download the files for all variables, resolutions and forecast types
for var in variables:
    for res in resolutions:
        for fc_type in fc_types:
            
            # Create the download url
            url_add = f"ecmwf_{fc_type}_{var}_{res}?datetime={modeldatestr_api}&format=netcdf"
            url = base_url + url_add
            
            # Do the api request
            r = requests.get(url, headers=header)
            
            # Write the response in a file
            output_fn = f'{output_dir}ecmwf_{fc_type}_{var}_{modeldatestr_out}_{res}.nc'
            
            file = open(output_fn, "wb")
            file.write(r.content)
            file.close()
