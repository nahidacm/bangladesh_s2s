#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb  7 15:34:52 2023

Gerenate the operational S2S forecast bulletin for Bangladesh.

The code uses the output (figures [png] and data [json]) from 
s2s_operational_forecast.py and the bulletin template. The figures and data
is filled in at the keys and the bulletin will be saved as .docx file.

@author: bob
"""
import datetime
import json
import os

# Import properties for the generation of the document
from docxtpl import DocxTemplate, InlineImage
from docx.shared import Cm

import warnings
warnings.filterwarnings('ignore')

from configparser import ConfigParser

# Read the basic paths from the config file
config = ConfigParser()
config.read('config_bd_s2s.ini')

# Set the directories from the config file
direc = config['paths']['s2s_dir'] 
template_dir = 'bulletin_template/'
input_fig_dir = direc + 'output_figures_forecast/'
output_dir = direc + 'output_bulletin/'

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Set the modeldate
# Take the modelrun of yesterday
modeldate = datetime.datetime.today() - datetime.timedelta(1)
modeldate = datetime.datetime(modeldate.year,
                              modeldate.month,
                              modeldate.day,
                              0,0)
modeldatestr = modeldate.strftime("%Y%m%d")

# Define start and end of the periods
issue_date = modeldate + datetime.timedelta(1)
wk1_start = modeldate + datetime.timedelta(1)
wk1_end = modeldate + datetime.timedelta(7)
wk2_start = modeldate + datetime.timedelta(8)
wk2_end = modeldate + datetime.timedelta(14)
wk34_start = modeldate + datetime.timedelta(15)
wk34_end = modeldate + datetime.timedelta(28)

print('Generate the bulletin')
template_name = "S2S_forecast_bulletin_template.docx"

# Generate the document
template = DocxTemplate(template_dir+template_name)

# Create image object from images
fig_tmax_wk1 = InlineImage(template, input_fig_dir+f"fc_tmax_week1_{modeldatestr}.png", Cm(15))
fig_tmin_wk1 = InlineImage(template, input_fig_dir+f"fc_tmin_week1_{modeldatestr}.png", Cm(15))
fig_tp_wk1 = InlineImage(template, input_fig_dir+f"fc_tp_week1_{modeldatestr}.png", Cm(15))
fig_tmax_wk2 = InlineImage(template, input_fig_dir+f"fc_tmax_week2_{modeldatestr}.png", Cm(15))
fig_tmin_wk2 = InlineImage(template, input_fig_dir+f"fc_tmin_week2_{modeldatestr}.png", Cm(15))
fig_tp_wk2 = InlineImage(template, input_fig_dir+f"fc_tp_week2_{modeldatestr}.png", Cm(15))
fig_tmax_wk34 = InlineImage(template, input_fig_dir+f"fc_tmax_week3+4_{modeldatestr}.png", Cm(15))
fig_tmin_wk34 = InlineImage(template, input_fig_dir+f"fc_tmin_week3+4_{modeldatestr}.png", Cm(15))
fig_tp_wk34 = InlineImage(template, input_fig_dir+f"fc_tp_week3+4_{modeldatestr}.png", Cm(15))

# Load the divisional forecast data (to add in the tables)
with open(direc+f'output_forecast/divisional_forecast_{modeldatestr}.json') as jsfile:
    forecast_json = json.load(jsfile)

# Declare template variables
context = {
    # The dates
    'issue_date': issue_date.strftime('%d %B %Y'),
    'wk1_start': wk1_start.strftime('%d.%m.%Y'),
    'wk1_end': wk1_end.strftime('%d.%m.%Y'),
    'wk2_start': wk2_start.strftime('%d.%m.%Y'),
    'wk2_end': wk2_end.strftime('%d.%m.%Y'),
    'wk34_start': wk34_start.strftime('%d.%m.%Y'),
    'wk34_end': wk34_end.strftime('%d.%m.%Y'),
    
    # The figures
    'fig_tmax_wk1': fig_tmax_wk1,
    'fig_tmin_wk1': fig_tmin_wk1,
    'fig_tp_wk1': fig_tp_wk1,
    'fig_tmax_wk2': fig_tmax_wk2,
    'fig_tmin_wk2': fig_tmin_wk2,
    'fig_tp_wk2': fig_tp_wk2,
    'fig_tmax_wk34': fig_tmax_wk34,
    'fig_tmin_wk34': fig_tmin_wk34,
    'fig_tp_wk34': fig_tp_wk34
    
    }

# Add the divisional summary to the context
context.update(forecast_json)

# Render automated report
template.render(context)

# Save the report
filename = f'{output_dir}S2S_forecast_bulletin_{modeldatestr}.docx'

template.save(filename)

print(f"Bulletin for {issue_date.strftime('%d %B %Y')} is generated and saved.")
