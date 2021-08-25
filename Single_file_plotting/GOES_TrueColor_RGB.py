#################################################################
# GOES-16 True Color RGB plotting code example
# Uses channels 2, 3, 1 to build 
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

# Custom
from GOES_LL_Conv import lat_lon_reproj


cwd = os.getcwd()

# Define root directory 
root=cwd+'\\'

# Define output directory
output1=root

#################################################################
# Need 3 channels to build Airmass RGB:
# File 1 = L2 ABI CMIP M6 C02 G16
# File 2 = L2 ABI CMIP M6 C03 G16
# File 3 = L2 ABI CMIP M6 C01 G16
#################################################################

file1='OR_ABI-L2-CMIPC-M6C02_G16_s20211371656164_e20211371658537_c20211371659018.nc'

file2='OR_ABI-L2-CMIPC-M6C03_G16_s20211371656164_e20211371658537_c20211371659006.nc'

file3='OR_ABI-L2-CMIPC-M6C01_G16_s20211371656164_e20211371658537_c20211371659006.nc'


##############################################
# file 1: CH. 2

lon_CH2,lat_CH2,data_CH2,data_units_CH2,data_time_grab_CH2,data_long_name_CH2,band_id_CH2,band_wavelength_CH2,band_units_CH2,var_name_CH2, lat_rad_CH2, lon_rad_CH2, lat_rad_1d_CH2, lon_rad_1d_CH2 = lat_lon_reproj(root, file1)


##############################################
# file 2: CH. 3

lon_CH3,lat_CH3,data_CH3,data_units_CH3,data_time_grab_CH3,data_long_name_CH3,band_id_CH3,band_wavelength_CH3,band_units_CH3,var_name_CH3, lat_rad_CH3, lon_rad_CH3, lat_rad_1d_CH3, lon_rad_1d_CH3  = lat_lon_reproj(root, file2)


##############################################
# file 3: CH. 1

lon_CH1,lat_CH1,data_CH1,data_units_CH1,data_time_grab_CH1,data_long_name_CH1,band_id_CH1,band_wavelength_CH1,band_units_CH1,var_name_CH1, lat_rad_CH1, lon_rad_CH1, lat_rad_1d_CH1, lon_rad_1d_CH1  = lat_lon_reproj(root, file3)


###################################################
# Resize CH2 
###################################################

def rebin(a, shape):
    sh = shape[0],a.shape[0]//shape[0],shape[1],a.shape[1]//shape[1]
    return a.reshape(sh).mean(-1).mean(1)



data_CH2_new=rebin(data_CH2,(3000,5000))


###################################################
# Create RGB components
###################################################

R = np.clip(data_CH2_new, 0, 1)
G = np.clip(data_CH3, 0, 1)
B = np.clip(data_CH1, 0, 1)

###################################################
# Normalize the RGBs
###################################################

# Choose the gamma
gamma = 2.2
 
# Normalize the data
R = np.power(R, 1/gamma)
G = np.power(G, 1/gamma)
B = np.power(B, 1/gamma)

G_true = 0.45*R + 0.1*G + 0.45*B
G_true=np.clip(G_true, 0, 1)
 
###################################################
# Stack the RGBs
###################################################
RGB = np.stack([R, G_true, B], axis=2)

# Get geos projection
FILE = ('https://ramadda.scigw.unidata.ucar.edu/repository/opendap'
        '/4ef52e10-a7da-4405-bff4-e48f68bb6ba2/entry.das#fillmismatch')
C = xarray.open_dataset(FILE)
dat = C.metpy.parse_cf('CMI_C02')
geos = dat.metpy.cartopy_crs
x = dat.x # corresponds to lat_rad_1d_CH2
y = dat.y # corresponds to lon_rad_1d_CH2



###################################################
# Plot CONUS PlateCarree projection
###################################################


pc = ccrs.PlateCarree()


date_string = data_time_grab_CH1[5:7] + '_' + data_time_grab_CH1[8:10] + '_' + data_time_grab_CH1[0:4] + '_'+ data_time_grab_CH1[11:13]+data_time_grab_CH1[14:16]+data_time_grab_CH1[17:19]
file_string='G16_RGB_GeoColor_'+date_string

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



Zoom_string='Zoom_G16_RGB_GeoColor_'+date_string
Zoom_extent=np.array([-110,-90,25,38])

with PdfPages(os.path.join(output1, Zoom_string + '.pdf')) as export_pdf:
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



#################################################################################################
# Create GeoTiff

file_string2='G16_RGB_GeoColor_'+date_string


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
dst_ds = gdal.GetDriverByName('GTiff').Create(os.path.join(output1, file_string2 + '.tif'), ny, nx, 3, gdal.GDT_Float32)

dst_ds.SetGeoTransform(geotransform)    # specify coords
srs = osr.SpatialReference()            # establish encoding
srs.ImportFromEPSG(3857)                # WGS84 lat/long
dst_ds.SetProjection(srs.ExportToWkt()) # export coords to file
dst_ds.GetRasterBand(1).WriteArray(R)   # write r-band to the raster
dst_ds.GetRasterBand(2).WriteArray(G_true)   # write g-band to the raster
dst_ds.GetRasterBand(3).WriteArray(B)   # write b-band to the raster
dst_ds.FlushCache()                     # write to disk
dst_ds = None

