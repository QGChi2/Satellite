#################################################################
# GOES-16 Daytime Microphysics RGB plotting code example
# Uses channels 7, 13, 3 to build 
#
# BEFORE RUNNING THIS CODE: Make sure data types match up...
# If there are 36 CH3 files from 16 UTC to 19 UTC, there damn well # better be 36 for CH 2 and CH 3 at the same times!!!!
# Future iterations will figure out how for you to input a time
# or an array of times and it will spit out the plot(s)
#################################################################

#Basic 
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
import xarray
import metpy
from pyorbital import astronomy
from datetime import datetime
from pyspectral.near_infrared_reflectance import Calculator

# Custom...WARNING: make sure you have already cd'ed into the 
# directory this function resides!!!!  
from GOES_LL_Conv import lat_lon_reproj


###################################################################
# USER Modification section
###################################################################

#Set date
date='20210517'


################################################################
# Modify only if you know what you are doing

cwd = os.getcwd()

# Define root directory 
root=cwd+'\\REPO\\'

# Define output directory
output_root=cwd+'\Plots\\'

directory=root+date+'\\'


# Create reshape function
def rebin(a, shape):
    sh = shape[0],a.shape[0]//shape[0],shape[1],a.shape[1]//shape[1]
    return a.reshape(sh).mean(-1).mean(1)
    

# Get geos projection
FILE = ('https://ramadda.scigw.unidata.ucar.edu/repository/opendap'
        '/4ef52e10-a7da-4405-bff4-e48f68bb6ba2/entry.das#fillmismatch')
C = xarray.open_dataset(FILE)
dat = C.metpy.parse_cf('CMI_C02')
geos = dat.metpy.cartopy_crs
x = dat.x # corresponds to lat_rad_1d_CH2
y = dat.y # corresponds to lon_rad_1d_CH2

pc = ccrs.PlateCarree()


#################################################################
# Need 3 channels to build Daytime Microphysics RGB:
# File 1 = L2 ABI CMIP M6 C07 G16
# File 2 = L2 ABI CMIP M6 C13 G16
# File 3 = L2 ABI CMIP M6 C03 G16
#################################################################

f1=np.array([])
f2=np.array([])
f3=np.array([])

for file in os.listdir(directory):
    filename = os.fsdecode(file)
    if filename.startswith("OR_ABI-L2-CMIPC-M6C07_G16_"):
        f1=np.append(f1, filename)
    if filename.startswith("OR_ABI-L2-CMIPC-M6C13_G16_"):
        f2=np.append(f2,filename)
    if filename.startswith("OR_ABI-L2-CMIPC-M6C03_G16_"):
        f3=np.append(f3, filename)


for i in range(0, len(f1)):
    file1=f1[i]
    file2=f2[i]
    file3=f3[i]
    print("Starting File: ", i+1)
    ##############################################
    # file 1: CH. 2
    lon_CH7,lat_CH7,data_CH7,data_units_CH7,data_time_grab_CH7,data_long_name_CH7,band_id_CH7,band_wavelength_CH7,band_units_CH7,var_name_CH7, lat_rad_CH7, lon_rad_CH7, lat_rad_1d_CH7, lon_rad_1d_CH7 = lat_lon_reproj(directory, file1)
    ##############################################
    # file 2: CH. 3
    lon_CH13,lat_CH13,data_CH13,data_units_CH13,data_time_grab_CH13,data_long_name_CH13,band_id_CH13,band_wavelength_CH13,band_units_CH13,var_name_CH13, lat_rad_CH13, lon_rad_CH13, lat_rad_1d_CH13, lon_rad_1d_CH13  = lat_lon_reproj(directory, file2)
    ##############################################
    # file 3: CH. 1
    lon_CH3,lat_CH3,data_CH3,data_units_CH3,data_time_grab_CH3,data_long_name_CH3,band_id_CH3,band_wavelength_CH3,band_units_CH3,var_name_CH3, lat_rad_CH3, lon_rad_CH3, lat_rad_1d_CH3, lon_rad_1d_CH3  = lat_lon_reproj(directory, file3)
    ###################################################
    # Resize CH7 
    data_CH3_new=rebin(data_CH3,(1500,2500))
    ###########################################################
    # Solar angle correction
    utc_time = datetime(np.int(data_time_grab_CH7[0:4]), np.int(data_time_grab_CH7[5:7]), np.int(data_time_grab_CH7[8:10]), np.int(data_time_grab_CH7[11:13]), np.int(data_time_grab_CH7[14:16]))
    extent = [-138, 5.0, -112, 24.0]
    lat = np.linspace(np.max(lat_CH7.data), np.min(lat_CH7.data), data_CH7.shape[0])
    lon = np.linspace(np.min(lon_CH7.data), np.max(lon_CH7.data), data_CH7.shape[1])
    zenith = np.zeros((data_CH7.shape[0], data_CH7.shape[1]))
     #
    for x in range(len(lat)):
        for y in range(len(lon)):
           zenith[x,y] = astronomy.sun_zenith_angle(utc_time, lon[y], lat[x])
    #
    refl39 = Calculator('GOES-16', 'abi', 'ch7')
    data1b = refl39.reflectance_from_tbs(zenith, data_CH7, data_CH13)
    ###################################################
    # Create RGB components
    R = data_CH3_new
    G = data1b
    B = data_CH13
    # Minimuns and Maximuns
    Rmin = 0
    Rmax = 1
    Gmin = 0
    Gmax = 0.6
    Bmin = 203
    Bmax = 323
    R[R > Rmax] = Rmax
    G[G > Gmax] = Gmax
    B[B > Bmax] = Bmax
    ###################################################
    # Normalize the RGBs
    # Choose the gamma
    gamma_R = 1
    gamma_G = 2.5
    gamma_B = 1
    # Normalize the data
    R = ((R - Rmin) / (Rmax - Rmin)) ** (1/gamma_R)
    G = ((G - Gmin) / (Gmax - Gmin)) ** (1/gamma_G)
    B = ((B - Bmin) / (Bmax - Bmin)) ** (1/gamma_B) 
    ###################################################
    # Stack the RGBs
    RGB = np.stack([R, G, B], axis=2)
    ###################################################
    # Plot CONUS PlateCarree projection
    date_string = data_time_grab_CH3[5:7] + '_' + data_time_grab_CH3[8:10] + '_' + data_time_grab_CH3[0:4] + '_'+ data_time_grab_CH3[11:13]+data_time_grab_CH3[14:16]+data_time_grab_CH3[17:19]
    file_string='G16_RGB_DayMicro_'+date_string
    zoom_string='Zoom_G16_RGB_DayMicro_'+date_string
    output1=output_root+date+'\\DayMicro\\Full\\'
    output2=output_root+date+'\\DayMicro\\Zoom\\'
    if not os.path.exists(output1):
        os.makedirs(output1)
    if not os.path.exists(output2):
        os.makedirs(output2)
    #
    #
    with PdfPages(os.path.join(output1, file_string + '.pdf')) as export_pdf:
        fig = plt.figure(figsize=(15, 12))
        ax = fig.add_subplot(1, 1, 1, projection=pc)
        ax.set_extent([-135, -65, 15, 55], crs=pc)
        ax.get_xaxis().set_visible(True)
        ax.get_yaxis().set_visible(True)
        ax.imshow(RGB, origin='upper', extent=(x.min(), x.max(), y.min(), y.max()), transform=geos, interpolation='none')
        ax.coastlines(resolution='50m', color='black', linewidth=1)
        ax.add_feature(ccrs.cartopy.feature.STATES)
        ax.gridlines(color='black', alpha=0.5, linestyle='--', linewidth=0.5)
        ax.set_title('GOES-16 Daytime Microphysics', fontweight='bold', loc='left', fontsize=30)
        ax.set_title(data_time_grab_CH3[0:19] + ' UTC',loc='right', fontsize=24)
        plt.xticks(fontsize=24)
        plt.yticks(fontsize=24)
        plt.xlabel('Longitude [deg]', fontsize=24)
        plt.ylabel('Latitude  [deg]', fontsize=24)
        export_pdf.savefig(fig)
        plt.close()
    #
    #
    if date=='20210517':       
        Zoom_extent=np.array([-110,-90,25,38])
    elif date=='20210518': 
        Zoom_extent=np.array([-105,-85,25,35])
    elif date=='20210519': 
        Zoom_extent=np.array([-110,-90,25,40])
    elif date=='20210520': 
        Zoom_extent=np.array([-110,-100,38,50])
    elif date=='20210524': 
        Zoom_extent=np.array([-105,-95,35,45])
    elif date=='20210525': 
        Zoom_extent=np.array([-98,-85,42,50])
    elif date=='20210526': 
        Zoom_extent=np.array([-110,-98,32,43])
    elif date=='20210527': 
        Zoom_extent=np.array([-100,-88,32,42])
    else:
        Zoom_extent=np.array([-135, -65, 15, 55])
    #
    #
    with PdfPages(os.path.join(output2, zoom_string + '.pdf')) as export_pdf:
        fig = plt.figure(figsize=(15, 12))
        ax = fig.add_subplot(1, 1, 1, projection=pc)
        ax.set_extent(Zoom_extent, crs=pc)
        ax.get_xaxis().set_visible(True)
        ax.get_yaxis().set_visible(True)
        ax.imshow(RGB, origin='upper', extent=(x.min(), x.max(), y.min(), y.max()), transform=geos, interpolation='none')
        ax.coastlines(resolution='50m', color='black', linewidth=1)
        ax.add_feature(ccrs.cartopy.feature.STATES)
        ax.gridlines(color='black', alpha=0.5, linestyle='--', linewidth=0.5)
        ax.set_title('GOES-16 Daytime Microphysics', fontweight='bold', loc='left', fontsize=30)
        ax.set_title(data_time_grab_CH3[0:19] + ' UTC',loc='right', fontsize=24)
        plt.xticks(fontsize=24)
        plt.yticks(fontsize=24)
        plt.xlabel('Longitude [deg]', fontsize=24)
        plt.ylabel('Latitude  [deg]', fontsize=24)
        export_pdf.savefig(fig)
        plt.close()
        
       

