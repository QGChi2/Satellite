#################################################################
# GOES-16 Daytime Microphysics RGB plotting code example
# Uses channels 7, 13, 3 to build #################################################################

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
from pyorbital import astronomy
from datetime import datetime
from pyspectral.near_infrared_reflectance import Calculator
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
# File 1 = L2 ABI CMIP M6 C07 G16
# File 2 = L2 ABI CMIP M6 C13 G16
# File 3 = L2 ABI CMIP M6 C03 G16
#################################################################

file1='OR_ABI-L2-CMIPC-M6C07_G16_s20211371656164_e20211371658549_c20211371659023.nc'

file2='OR_ABI-L2-CMIPC-M6C13_G16_s20211371656164_e20211371658549_c20211371659038.nc'


file3='OR_ABI-L2-CMIPC-M6C03_G16_s20211371656164_e20211371658537_c20211371659006.nc'

##############################################
# file 1: CH. 7

lon_CH7,lat_CH7,data_CH7,data_units_CH7,data_time_grab_CH7,data_long_name_CH7,band_id_CH7,band_wavelength_CH7,band_units_CH7,var_name_CH7, lat_rad_CH7, lon_rad_CH7, lat_rad_1d_CH7, lon_rad_1d_CH7  = lat_lon_reproj(root, file1)

##############################################
# file 2: CH. 13

lon_CH13,lat_CH13,data_CH13,data_units_CH13,data_time_grab_CH13,data_long_name_CH13,band_id_CH13,band_wavelength_CH13,band_units_CH13,var_name_CH13, lat_rad_CH13, lon_rad_CH13, lat_rad_1d_CH13, lon_rad_1d_CH13  = lat_lon_reproj(root, file2)

##############################################
# file 2: CH. 3

lon_CH3,lat_CH3,data_CH3,data_units_CH3,data_time_grab_CH3,data_long_name_CH3,band_id_CH3,band_wavelength_CH3,band_units_CH3,var_name_CH3, lat_rad_CH3, lon_rad_CH3, lat_rad_1d_CH3, lon_rad_1d_CH3  = lat_lon_reproj(root, file3)


###################################################
# Resize CH3
###################################################

def rebin(a, shape):
    sh = shape[0],a.shape[0]//shape[0],shape[1],a.shape[1]//shape[1]
    return a.reshape(sh).mean(-1).mean(1)


data_CH3_new=rebin(data_CH3,(1500,2500))



###########################################################
# Solar angle correction
###########################################################
utc_time = datetime(np.int(data_time_grab_CH3[0:4]), np.int(data_time_grab_CH3[5:7]), np.int(data_time_grab_CH3[8:10]), np.int(data_time_grab_CH3[11:13]), np.int(data_time_grab_CH3[14:16]))
#utc_time = datetime(2021, 5, 17, 16, 59)
extent = [-138, 5.0, -112, 24.0]
lat = np.linspace(np.max(lat_CH7.data), np.min(lat_CH7.data), data_CH7.shape[0])
lon = np.linspace(np.min(lon_CH7.data), np.max(lon_CH7.data), data_CH7.shape[1])
zenith = np.zeros((data_CH7.shape[0], data_CH7.shape[1]))
 
for x in range(len(lat)):
    for y in range(len(lon)):
       zenith[x,y] = astronomy.sun_zenith_angle(utc_time, lon[y], lat[x])
#zenith[zenith > 90] = np.nan
 


print("Solar Zenith Angle calculus finished")

##############################################################
# Calculate the solar component (band 3.7 um)
##############################################################

from pyspectral.near_infrared_reflectance import Calculator
refl39 = Calculator('GOES-16', 'abi', 'ch7')
data1b = refl39.reflectance_from_tbs(zenith, data_CH7, data_CH13)
print("Solar Component calculus finished")




###################################################
# Create RGB components
###################################################

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
###################################################

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
###################################################
RGB = np.stack([R, G, B], axis=2)

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


date_string = data_time_grab_CH13[5:7] + '_' + data_time_grab_CH13[8:10] + '_' + data_time_grab_CH13[0:4] + '_'+ data_time_grab_CH13[11:13]+data_time_grab_CH13[14:16]+data_time_grab_CH13[17:19]
file_string='G16_RGB_DayMicro_'+date_string

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
    ax.set_title(data_time_grab_CH13[0:19] + ' UTC',loc='right', fontsize=24)
    plt.xticks(fontsize=24)
    plt.yticks(fontsize=24)
    plt.xlabel('Longitude [deg]', fontsize=24)
    plt.ylabel('Latitude  [deg]', fontsize=24)
    export_pdf.savefig(fig)
    plt.close()

#################################################################################################
# Create GeoTiff

file_string2='G16_RGB_DayMicro_'+date_string


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