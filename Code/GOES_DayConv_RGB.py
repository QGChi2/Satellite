#################################################################
# GOES-16 Daytime Convection RGB plotting code example
# Uses channels 8, 10, 7, 13, 5, 2 to build 
#
# BEFORE RUNNING THIS CODE: Make sure data types match up...
# If there are 36 CH7 files from 16 UTC to 19 UTC, there damn well # better be 36 for CH 2 and CH 3 at the same times!!!!
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
# Need 6 channels to build Airmass RGB:
# File 1 = L2 ABI CMIP M6 C08 G16
# File 2 = L2 ABI CMIP M6 C10 G16
# File 3 = L2 ABI CMIP M6 C07 G16
# File 4 = L2 ABI CMIP M6 C13 G16
# File 5 = L2 ABI CMIP M6 C05 G16
# File 6 = L2 ABI CMIP M6 C02 G16
#################################################################

f1=np.array([])
f2=np.array([])
f3=np.array([])
f4=np.array([])
f5=np.array([])
f6=np.array([])

for file in os.listdir(directory):
    filename = os.fsdecode(file)
    if filename.startswith("OR_ABI-L2-CMIPC-M6C08_G16_"):
        f1=np.append(f1, filename)
    if filename.startswith("OR_ABI-L2-CMIPC-M6C10_G16_"):
        f2=np.append(f2,filename)
    if filename.startswith("OR_ABI-L2-CMIPC-M6C07_G16_"):
        f3=np.append(f3, filename)
    if filename.startswith("OR_ABI-L2-CMIPC-M6C13_G16_"):
        f4=np.append(f4, filename)
    if filename.startswith("OR_ABI-L2-CMIPC-M6C05_G16_"):
        f5=np.append(f5, filename)
    if filename.startswith("OR_ABI-L2-CMIPC-M6C02_G16_"):
        f6=np.append(f6, filename)
       


for i in range(0, len(f1)):
    file1=f1[i]
    file2=f2[i]
    file3=f3[i]
    file4=f4[i]
    file5=f5[i]
    file6=f6[i]
    print("Starting File: ", i+1)
    ##############################################
    # file 1: CH. 8
    lon_CH8,lat_CH8,data_CH8,data_units_CH8,data_time_grab_CH8,data_long_name_CH8,band_id_CH8,band_wavelength_CH8,band_units_CH8,var_name_CH8, lat_rad_CH8, lon_rad_CH8, lat_rad_1d_CH8, lon_rad_1d_CH8 = lat_lon_reproj(directory, file1)
    ##############################################
    # file 2: CH. 10
    lon_CH10,lat_CH10,data_CH10,data_units_CH10,data_time_grab_CH10,data_long_name_CH10,band_id_CH10,band_wavelength_CH10,band_units_CH10,var_name_CH10, lat_rad_CH10, lon_rad_CH10, lat_rad_1d_CH10, lon_rad_1d_CH10  = lat_lon_reproj(directory, file2)
    ##############################################
    # file 3: CH. 7
    lon_CH7,lat_CH7,data_CH7,data_units_CH7,data_time_grab_CH7,data_long_name_CH7,band_id_CH7,band_wavelength_CH7,band_units_CH7,var_name_CH7, lat_rad_CH7, lon_rad_CH7, lat_rad_1d_CH7, lon_rad_1d_CH7  = lat_lon_reproj(directory, file3)
    ##############################################
    # file 4: CH. 13
    lon_CH13,lat_CH13,data_CH13,data_units_CH13,data_time_grab_CH13,data_long_name_CH13,band_id_CH13,band_wavelength_CH13,band_units_CH13,var_name_CH13, lat_rad_CH13, lon_rad_CH13, lat_rad_1d_CH13, lon_rad_1d_CH13 = lat_lon_reproj(directory, file4)
    ##############################################
    # file 5: CH. 5
    lon_CH5,lat_CH5,data_CH5,data_units_CH5,data_time_grab_CH5,data_long_name_CH5,band_id_CH5,band_wavelength_CH5,band_units_CH5,var_name_CH5, lat_rad_CH5, lon_rad_CH5, lat_rad_1d_CH5, lon_rad_1d_CH5  = lat_lon_reproj(directory, file5)
    ##############################################
    # file 6: CH. 2
    lon_CH2,lat_CH2,data_CH2,data_units_CH2,data_time_grab_CH2,data_long_name_CH2,band_id_CH2,band_wavelength_CH2,band_units_CH2,var_name_CH2, lat_rad_CH2, lon_rad_CH2, lat_rad_1d_CH2, lon_rad_1d_CH2  = lat_lon_reproj(directory, file6)
    ###################################################
    # Resize CH8 
    data_CH2_new=rebin(data_CH2,(1500,2500))
    data_CH5_new=rebin(data_CH5,(1500,2500))
    ###################################################
    # Convert to Celsius
    data_CH8 = data_CH8 - 273.15
    data_CH10 = data_CH10 - 273.15
    data_CH7 = data_CH7 - 273.15
    data_CH13 = data_CH13 - 273.15
    ###################################################
    # Create RGB components
    R = data_CH8 - data_CH10
    G = data_CH7 - data_CH13
    B = data_CH5_new - data_CH2_new
    # Minimuns and Maximuns
    Rmin = -35.0
    Rmax = 5 
    Gmin = -5.0
    Gmax = 60
    Bmin = -0.75
    Bmax = 0.25
    R[R > Rmax] = Rmax
    G[G > Gmax] = Gmax
    B[B > Bmax] = Bmax
    ###################################################
    # Normalize the RGBs
    # Choose the gamma
    gamma = 1
    # Normalize the data
    R = ((R - Rmin) / (Rmax - Rmin)) ** (1/gamma)
    G = ((G - Gmin) / (Gmax - Gmin)) ** (1/gamma)
    B = ((B - Bmin) / (Bmax - Bmin)) ** (1/gamma) 
    ###################################################
    # Stack the RGBs
    RGB = np.stack([R, G, B], axis=2)
    ###################################################
    # Plot CONUS PlateCarree projection
    date_string = data_time_grab_CH7[5:7] + '_' + data_time_grab_CH7[8:10] + '_' + data_time_grab_CH7[0:4] + '_'+ data_time_grab_CH7[11:13]+data_time_grab_CH7[14:16]+data_time_grab_CH7[17:19]
    file_string='G16_RGB_DayConv_'+date_string
    zoom_string='Zoom_G16_RGB_DayConv_'+date_string
    output1=output_root+date+'\\DayConv\\Full\\'
    output2=output_root+date+'\\DayConv\\Zoom\\'
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
        ax.set_title('GOES-16 Daytime Convection', fontweight='bold', loc='left', fontsize=30)
        ax.set_title(data_time_grab_CH7[0:19] + ' UTC',loc='right', fontsize=24)
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
        ax.set_title('GOES-16 Daytime Convection', fontweight='bold', loc='left', fontsize=30)
        ax.set_title(data_time_grab_CH7[0:19] + ' UTC',loc='right', fontsize=24)
        plt.xticks(fontsize=24)
        plt.yticks(fontsize=24)
        plt.xlabel('Longitude [deg]', fontsize=24)
        plt.ylabel('Latitude  [deg]', fontsize=24)
        export_pdf.savefig(fig)
        plt.close()
        
        



