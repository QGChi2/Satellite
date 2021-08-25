#################################################################
# GOES-16 True Color RGB plotting code example
# Uses channels 2, 3, 1 to build 
#
# BEFORE RUNNING THIS CODE: Make sure data types match up...
# If there are 36 CH1 files from 16 UTC to 19 UTC, there damn well # better be 36 for CH 2 and CH 3 at the same times!!!!
# Future iterations will figure out how for you to input a time
# or an array of times and it will spit out the plot(s)
#################################################################

#Basic 
import pandas as pd
import numpy as np
import geopandas as gpd
import matplotlib.pyplot as plt
import netCDF4 as nc
from matplotlib.backends.backend_pdf import PdfPages
import cartopy.crs as ccrs
import cartopy
import cartopy.feature as cfeature
import matplotlib as mpl
import xarray
import metpy
from osgeo import gdal
from osgeo import osr
import os, sys

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
# Need 3 channels to build Airmass RGB:
# File 1 = L2 ABI CMIP M6 C02 G16
# File 2 = L2 ABI CMIP M6 C03 G16
# File 3 = L2 ABI CMIP M6 C01 G16
#################################################################

f1=np.array([])
f2=np.array([])
f3=np.array([])

for file in os.listdir(directory):
    filename = os.fsdecode(file)
    if filename.startswith("OR_ABI-L2-CMIPC-M6C02_G16_"):
        f1=np.append(f1, filename)
    if filename.startswith("OR_ABI-L2-CMIPC-M6C03_G16_"):
        f2=np.append(f2,filename)
    if filename.startswith("OR_ABI-L2-CMIPC-M6C01_G16_"):
        f3=np.append(f3, filename)


for i in range(0, len(f1)):
    file1=f1[i]
    file2=f2[i]
    file3=f3[i]
    print("Starting File: ", i+1)
    ##############################################
    # file 1: CH. 2
    lon_CH2,lat_CH2,data_CH2,data_units_CH2,data_time_grab_CH2,data_long_name_CH2,band_id_CH2,band_wavelength_CH2,band_units_CH2,var_name_CH2, lat_rad_CH2, lon_rad_CH2, lat_rad_1d_CH2, lon_rad_1d_CH2 = lat_lon_reproj(directory, file1)
    ##############################################
    # file 2: CH. 3
    lon_CH3,lat_CH3,data_CH3,data_units_CH3,data_time_grab_CH3,data_long_name_CH3,band_id_CH3,band_wavelength_CH3,band_units_CH3,var_name_CH3, lat_rad_CH3, lon_rad_CH3, lat_rad_1d_CH3, lon_rad_1d_CH3  = lat_lon_reproj(directory, file2)
    ##############################################
    # file 3: CH. 1
    lon_CH1,lat_CH1,data_CH1,data_units_CH1,data_time_grab_CH1,data_long_name_CH1,band_id_CH1,band_wavelength_CH1,band_units_CH1,var_name_CH1, lat_rad_CH1, lon_rad_CH1, lat_rad_1d_CH1, lon_rad_1d_CH1  = lat_lon_reproj(directory, file3)
    ###################################################
    # Resize CH2 
    data_CH2_new=rebin(data_CH2,(3000,5000))
    ###################################################
    # Create RGB components
    R = np.clip(data_CH2_new, 0, 1)
    G = np.clip(data_CH3, 0, 1)
    B = np.clip(data_CH1, 0, 1)
    ###################################################
    # Normalize the RGBs
    # Choose the gamma
    gamma = 2.2
    # Normalize the data
    R = np.power(R, 1/gamma)
    G = np.power(G, 1/gamma)
    B = np.power(B, 1/gamma)
    # Get new green
    G_true = 0.45*R + 0.1*G + 0.45*B
    G_true=np.clip(G_true, 0, 1)
    ###################################################
    # Stack the RGBs
    RGB = np.stack([R, G_true, B], axis=2)
    ###################################################
    # Plot CONUS PlateCarree projection
    date_string = data_time_grab_CH1[5:7] + '_' + data_time_grab_CH1[8:10] + '_' + data_time_grab_CH1[0:4] + '_'+ data_time_grab_CH1[11:13]+data_time_grab_CH1[14:16]+data_time_grab_CH1[17:19]
    file_string='G16_RGB_GeoColor_'+date_string
    zoom_string='Zoom_G16_RGB_GeoColor_'+date_string
    output1=output_root+date+'\\GeoColor\\Full\\'
    output2=output_root+date+'\\GeoColor\\Zoom\\'
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
        ax.set_title('GOES-16 GeoColor', fontweight='bold', loc='left', fontsize=30)
        ax.set_title(data_time_grab_CH1[0:19] + ' UTC',loc='right', fontsize=24)
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
        ax.set_title('GOES-16 GeoColor', fontweight='bold', loc='left', fontsize=30)
        ax.set_title(data_time_grab_CH1[0:19] + ' UTC',loc='right', fontsize=24)
        plt.xticks(fontsize=24)
        plt.yticks(fontsize=24)
        plt.xlabel('Longitude [deg]', fontsize=24)
        plt.ylabel('Latitude  [deg]', fontsize=24)
        export_pdf.savefig(fig)
        plt.close()
    #
    #
    #
    #################################################################################################
    # Create GeoTiff
    file_string2='G16_RGB_GeoColor_'+date_string
    output3=output_root+date+'\\GeoColor\\GTIFF\\'
    if not os.path.exists(output3):
        os.makedirs(output3)
    #  Initialize the Image Size
    image_size = R.shape
    # set geotransform
    nx = image_size[0]
    ny = image_size[1]
    xmin, ymin, xmax, ymax = [np.min(lon_CH1), np.min(lat_CH1), np.max(lon_CH1), np.max(lat_CH1)]
    xres = (xmax - xmin) / float(nx)
    yres = (ymax - ymin) / float(ny)
    geotransform = (xmin, xres, 0, ymax, 0, -yres)
    # create the 3-band raster file
    dst_ds = gdal.GetDriverByName('GTiff').Create(os.path.join(output3, file_string2 + '.tif'), ny, nx, 3, gdal.GDT_Float32)
    dst_ds.SetGeoTransform(geotransform)    # specify coords
    srs = osr.SpatialReference()            # establish encoding
    srs.ImportFromEPSG(3857)                # WGS84 lat/long
    dst_ds.SetProjection(srs.ExportToWkt()) # export coords to file
    dst_ds.GetRasterBand(1).WriteArray(R)   # write r-band to the raster
    dst_ds.GetRasterBand(2).WriteArray(G_true)   # write g-band to the raster
    dst_ds.GetRasterBand(3).WriteArray(B)   # write b-band to the raster
    dst_ds.FlushCache()                     # write to disk
    dst_ds = None

