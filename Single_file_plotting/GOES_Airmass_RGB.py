#################################################################
# GOES-16 Airmass RGB plotting code example
# Uses channels 8, 10, 12, 13 to build #################################################################

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
# Need 4 channels to build Airmass RGB:
# File 1 = L2 ABI CMIP M6 C08 G16
# File 2 = L2 ABI CMIP M6 C10 G16
# File 3 = L2 ABI CMIP M6 C12 G16
# File 4 = L2 ABI CMIP M6 C13 G16
#################################################################

file1='OR_ABI-L2-CMIPC-M6C08_G16_s20210501201021_e20210501203394_c20210501203509.nc'

file2='OR_ABI-L2-CMIPC-M6C10_G16_s20210501201021_e20210501203406_c20210501203488.nc'

file3='OR_ABI-L2-CMIPC-M6C12_G16_s20210501201021_e20210501203400_c20210501203495.nc'

file4='OR_ABI-L2-CMIPC-M6C13_G16_s20210501201021_e20210501203406_c20210501203544.nc'


##############################################
# file 1: CH. 8

lon_CH8,lat_CH8,data_CH8,data_units_CH8,data_time_grab_CH8,data_long_name_CH8,band_id_CH8,band_wavelength_CH8,band_units_CH8,var_name_CH8, lat_rad_CH8, lon_rad_CH8, lat_rad_1d_CH8, lon_rad_1d_CH8 = lat_lon_reproj(root, file1)


##############################################
# file 2: CH. 10

lon_CH10,lat_CH10,data_CH10,data_units_CH10,data_time_grab_CH10,data_long_name_CH10,band_id_CH10,band_wavelength_CH10,band_units_CH10,var_name_CH10, lat_rad_CH10, lon_rad_CH10, lat_rad_1d_CH10, lon_rad_1d_CH10  = lat_lon_reproj(root, file2)

##############################################
# file 3: CH. 12

lon_CH12,lat_CH12,data_CH12,data_units_CH12,data_time_grab_CH12,data_long_name_CH12,band_id_CH12,band_wavelength_CH12,band_units_CH12,var_name_CH12, lat_rad_CH12, lon_rad_CH12, lat_rad_1d_CH12, lon_rad_1d_CH12  = lat_lon_reproj(root, file3)


##############################################
# file 4: CH. 13

lon_CH13,lat_CH13,data_CH13,data_units_CH13,data_time_grab_CH13,data_long_name_CH13,band_id_CH13,band_wavelength_CH13,band_units_CH13,var_name_CH13, lat_rad_CH13, lon_rad_CH13, lat_rad_1d_CH13, lon_rad_1d_CH13  = lat_lon_reproj(root, file4)


###################################################
# Convert to Celsius
###################################################

data_CH8 = data_CH8 - 273.15
data_CH10 = data_CH10 - 273.15
data_CH12 = data_CH12 - 273.15
data_CH13 = data_CH13 - 273.15


###################################################
# Create RGB components
###################################################

R = data_CH8 - data_CH10
G = data_CH12 - data_CH13
B = data_CH8
 
# Minimuns and Maximuns
Rmin = -26.2
Rmax = 0.6
 
Gmin = -43.2
Gmax = 6.7
 
Bmin = -29.25
Bmax = -64.65
 
R[R > Rmax] = Rmax
 
G[G > Gmax] = Gmax
 
B[B < Bmin] = Bmin

###################################################
# Normalize the RGBs
###################################################

# Choose the gamma
gamma = 1
 
# Normalize the data
R = ((R - Rmin) / (Rmax - Rmin)) ** (1/gamma)
G = ((G - Gmin) / (Gmax - Gmin)) ** (1/gamma)
B = ((B - Bmin) / (Bmax - Bmin)) ** (1/gamma) 
 
###################################################
# Stack the RGBs
###################################################
RGB = np.stack([R, G, B], axis=2)

# Get geos projection
FILE = ('https://ramadda.scigw.unidata.ucar.edu/repository/opendap'
        '/4ef52e10-a7da-4405-bff4-e48f68bb6ba2/entry.das#fillmismatch')
C = xarray.open_dataset(FILE)
dat = C.metpy.parse_cf('CMI_C02')
geos = dat.metpy.cartopy_crs
x = dat.x # corresponds to lat_rad_1d_CH8
y = dat.y # corresponds to lon_rad_1d_CH8

###################################################
# Plot CONUS PlateCarree projection
###################################################

pc = ccrs.PlateCarree()

date_string = data_time_grab_CH13[5:7] + '_' + data_time_grab_CH13[8:10] + '_' + data_time_grab_CH13[0:4] + '_'+ data_time_grab_CH13[11:13]+data_time_grab_CH13[14:16]+data_time_grab_CH13[17:19]
file_string='G16_RGB_Airmass_'+date_string

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
    ax.set_title('GOES-16 RGB Airmass', fontweight='bold', loc='left', fontsize=30)
    ax.set_title(data_time_grab_CH13[0:19] + ' UTC',loc='right', fontsize=24)
    plt.xticks(fontsize=24)
    plt.yticks(fontsize=24)
    plt.xlabel('Longitude [deg]', fontsize=24)
    plt.ylabel('Latitude  [deg]', fontsize=24)
    export_pdf.savefig(fig)
    plt.close()

#################################################################################################
# Create GeoTiff

file_string2='G16_RGB_Airmass_'+date_string


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
dst_ds.GetRasterBand(2).WriteArray(G)   # write g-band to the raster
dst_ds.GetRasterBand(3).WriteArray(B)   # write b-band to the raster
dst_ds.FlushCache()                     # write to disk
dst_ds = None