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
 - pandas 
 - geopandas
 - configparser
 - xcast (0.5.8)
 - cmocean
 - docxptl
 
Next to the scripts, the repository contains a configuration file (config_bd_s2s.ini). Please edit this file before using the scripts. The config file contains the directories where the data is stored. Most important directory is the s2s_directory

## Scripts
The current script uses ECMWF data and creates a forecast. Without the ECMWF grib files, the procedure of producing a forecast is as follows:
 1. Run the script download_ecmwf_s2s_from_wi_api.py to collect the ECMWF netcdf input files
 2. Run the script download_s2s_iri_datalibrary_operational.py to collect the ECCC netcdf input files
 3. Run prepare_ecmwf_data.py to prepare the ECMWF data input
 4. Run prepare_eccc_data.py to prepare the ECCC data input
 5. Run the script s2s_operational_forecast.py for the forecast
 6. Run the script generate_bulletin.py to generate a bulletin with the latest forecast

Note that the ECMWF data is available twice a week: on Monday and Thursday. The ECCC data is available every Thursday. Other models will be added in a later phase.

The operational forecast script checks which data is available and creates a multi-model output if both models are available, or a single model output if only one is available.

The scripts are build to run one day later (Tuesday and Friday) early in the day and use the forecast of the day before.

## Output
The scripts have several output files:
 - Figures with the skill analysis of the hindcast
 - Figures with the current forecast
 - JSON and CSV file with the forecast values on division level
 - Bulletin with forecast

