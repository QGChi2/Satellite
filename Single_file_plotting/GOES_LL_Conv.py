########################################
# GOES projection conversion to Lat/Lon
# Function to be read in by other code
#######################################

import pandas as pd
import numpy as np
import geopandas as gpd
import matplotlib.pyplot as plt
import os
import netCDF4 as nc
from matplotlib.backends.backend_pdf import PdfPages
import cartopy.crs as ccrs
import cartopy
import cartopy.feature as cfeature
import matplotlib as mpl


def lat_lon_reproj(root,file):
	ds = nc.Dataset(os.path.join(root, file))
	########################################
	# Designate dataset
	#######################################
	var_names = [ii for ii in ds.variables]
	var_name = var_names[0]
	try:
		band_id = ds.variables['band_id'][:]
		band_id = ' (Band: {},'.format(band_id[0])
		band_wavelength = ds.variables['band_wavelength']
		band_wavelength_units = band_wavelength.units
		band_wavelength_units = '{})'.format(band_wavelength_units)
		band_wavelength = ' {0:.2f} '.format(band_wavelength[:][0])
		print('Band ID: {}'.format(band_id))
		print('Band Wavelength: {} {}'.format(band_wavelength,band_wavelength_units))
	except:
		band_id = ''
		band_wavelength = ''
		band_wavelength_units = ''
	############################################################
	# GOES-R projection info and retrieving relevant constants
	############################################################
	proj_info = ds.variables['goes_imager_projection']
	lon_origin = proj_info.longitude_of_projection_origin
	H = proj_info.perspective_point_height+proj_info.semi_major_axis
	r_eq = proj_info.semi_major_axis
	r_pol = proj_info.semi_minor_axis
	########################################
	# grid info
	########################################
	lat_rad_1d = ds.variables['x'][:]
	lon_rad_1d = ds.variables['y'][:]
	########################################
	# data info
	########################################
	data = ds.variables[var_name][:]
	data_units = ds.variables[var_name].units
	data_time_grab = ((ds.time_coverage_end).replace('T',' ')).replace('Z','')
	data_long_name = ds.variables[var_name].long_name
	########################################
	# create meshgrid filled with radian angles
	########################################
	lat_rad,lon_rad = np.meshgrid(lat_rad_1d,lon_rad_1d)
	########################################
	# lat/lon calc routine from satellite radian angle vectors
	########################################
	lambda_0 = (lon_origin*np.pi)/180.0
	#
	a_var = np.power(np.sin(lat_rad),2.0) + (np.power(np.cos(lat_rad),2.0)*(np.power(np.cos(lon_rad),2.0)+(((r_eq*r_eq)/(r_pol*r_pol))*np.power(np.sin(lon_rad),2.0))))
	b_var = -2.0*H*np.cos(lat_rad)*np.cos(lon_rad)
	c_var = (H**2.0)-(r_eq**2.0)
	#
	r_s = (-1.0*b_var - np.sqrt((b_var**2)-(4.0*a_var*c_var)))/(2.0*a_var)
	#
	s_x = r_s*np.cos(lat_rad)*np.cos(lon_rad)
	s_y = - r_s*np.sin(lat_rad)
	s_z = r_s*np.cos(lat_rad)*np.sin(lon_rad)
	################################
	# Get Lat/Lon
	################################
	lat = (180.0/np.pi)*(np.arctan(((r_eq*r_eq)/(r_pol*r_pol))*((s_z/np.sqrt(((H-s_x)*(H-s_x))+(s_y*s_y))))))
	lon = (lambda_0 - np.arctan(s_y/(H-s_x)))*(180.0/np.pi)
	################################
	# Return vars
	################################
	return lon,lat,data,data_units,data_time_grab,data_long_name,band_id,band_wavelength,band_wavelength_units,var_name, lat_rad, lon_rad, lat_rad_1d, lon_rad_1d