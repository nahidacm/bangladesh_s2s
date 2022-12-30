# S2S Bangladesh
This repository contains scripts to create an S2S forecast for Bangladesh.

## Installation
The scripts run in Python 3.8 and require packages 
 - datetime
 - numpy
 - xarray
 - matplotlib
 - mpl_toolkits
 - cartopy
 - requests
 - configparser
 - xcast
 
Next to the scripts, the repository contains a configuration file (config_bd_s2s.ini). Please edit this file before using the scripts. The config file contains the directories where the data is stored. Most important directory is the s2s_directory

## Scripts
The current script uses ECMWF data and creates a forecast. Without the ECMWF grib files, the procedure of producing a forecast is as follows:
 1. Run the script download_ecmwf_s2s_from_wi_api.py to collect the ECMWF netcdf input files
 2. Run the script s2s_operational_forecast.py for the forecast

Note that the ECMWF data is available twice a week: on Monday and Thursday. The two scripts are build to run one day later (Tuesday and Friday) early in the day and use the forecast of the day before.

