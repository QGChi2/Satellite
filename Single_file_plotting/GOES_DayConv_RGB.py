#################################################################
# GOES-16 Day Convection RGB plotting code example
# Uses channels 8, 10, 7, 13, 5, 2 to build #################################################################

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
# File 1 = L2 ABI CMIP M6 C08 G16
# File 2 = L2 ABI CMIP M6 C10 G16
# File 3 = L2 ABI CMIP M6 C07 G16
# File 4 = L2 ABI CMIP M6 C13 G16
# File 5 = L2 ABI CMIP M6 C05 G16
# File 6 = L2 ABI CMIP M6 C02 G16
#################################################################


file1='OR_ABI-L2-CMIPC-M6C08_G16_s20211371656164_e20211371658537_c20211371659023.nc'

file2='OR_ABI-L2-CMIPC-M6C10_G16_s20211371656164_e20211371658549_c20211371659006.nc'

file3='OR_ABI-L2-CMIPC-M6C07_G16_s20211371656164_e20211371658549_c20211371659023.nc'

file4='OR_ABI-L2-CMIPC-M6C13_G16_s20211371656164_e20211371658549_c20211371659038.nc'

file5='OR_ABI-L2-CMIPC-M6C05_G16_s20211371656164_e20211371658537_c20211371659007.nc'

file6='OR_ABI-L2-CMIPC-M6C02_G16_s20211371656164_e20211371658537_c20211371659018.nc'

##############################################
# file 1: CH. 13

lon_CH8,lat_CH8,data_CH8,data_units_CH8,data_time_grab_CH8,data_long_name_CH8,band_id_CH8,band_wavelength_CH8,band_units_CH8,var_name_CH8, lat_rad_CH8, lon_rad_CH8, lat_rad_1d_CH8, lon_rad_1d_CH8  = lat_lon_reproj(root, file1)


##############################################
# file 2: CH. 10

lon_CH10,lat_CH10,data_CH10,data_units_CH10,data_time_grab_CH10,data_long_name_CH10,band_id_CH10,band_wavelength_CH10,band_units_CH10,var_name_CH10, lat_rad_CH10, lon_rad_CH10, lat_rad_1d_CH10, lon_rad_1d_CH10  = lat_lon_reproj(root, file2)

##############################################
# file 3: CH. 7

lon_CH7,lat_CH7,data_CH7,data_units_CH7,data_time_grab_CH7,data_long_name_CH7,band_id_CH7,band_wavelength_CH7,band_units_CH7,var_name_CH7, lat_rad_CH7, lon_rad_CH7, lat_rad_1d_CH7, lon_rad_1d_CH7  = lat_lon_reproj(root, file3)


##############################################
# file 4: CH. 13

lon_CH13,lat_CH13,data_CH13,data_units_CH13,data_time_grab_CH13,data_long_name_CH13,band_id_CH13,band_wavelength_CH13,band_units_CH13,var_name_CH13, lat_rad_CH13, lon_rad_CH13, lat_rad_1d_CH13, lon_rad_1d_CH13  = lat_lon_reproj(root, file4)

##############################################
# file 5: CH. 5

lon_CH5,lat_CH5,data_CH5,data_units_CH5,data_time_grab_CH5,data_long_name_CH5,band_id_CH5,band_wavelength_CH5,band_units_CH5,var_name_CH5, lat_rad_CH5, lon_rad_CH5, lat_rad_1d_CH5, lon_rad_1d_CH5  = lat_lon_reproj(root, file5)


##############################################
# file 6: CH. 2

lon_CH2,lat_CH2,data_CH2,data_units_CH2,data_time_grab_CH2,data_long_name_CH2,band_id_CH2,band_wavelength_CH2,band_units_CH2,var_name_CH2, lat_rad_CH2, lon_rad_CH2, lat_rad_1d_CH2, lon_rad_1d_CH2 = lat_lon_reproj(root, file6)




###################################################
# Convert to Celsius
###################################################


data_CH8 = data_CH8 - 273.15

data_CH10 = data_CH10 - 273.15

data_CH7 = data_CH7 - 273.15

data_CH13 = data_CH13 - 273.15


###################################################
# Resize CH2 and CH 5###################################################

def rebin(a, shape):
    sh = shape[0],a.shape[0]//shape[0],shape[1],a.shape[1]//shape[1]
    return a.reshape(sh).mean(-1).mean(1)


data_CH2_new=rebin(data_CH2,(1500,2500))
data_CH5_new=rebin(data_CH5,(1500,2500))


###################################################
# Create RGB components
###################################################

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
x = dat.x # corresponds to lat_rad_1d_CH2
y = dat.y # corresponds to lon_rad_1d_CH2



###################################################
# Plot CONUS PlateCarree projection
###################################################


pc = ccrs.PlateCarree()


date_string = data_time_grab_CH13[5:7] + '_' + data_time_grab_CH13[8:10] + '_' + data_time_grab_CH13[0:4] + '_'+ data_time_grab_CH13[11:13]+data_time_grab_CH13[14:16]+data_time_grab_CH13[17:19]
file_string='G16_RGB_DayConv_'+date_string

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
    ax.set_title(data_time_grab_CH13[0:19] + ' UTC',loc='right', fontsize=24)
    plt.xticks(fontsize=24)
    plt.yticks(fontsize=24)
    plt.xlabel('Longitude [deg]', fontsize=24)
    plt.ylabel('Latitude  [deg]', fontsize=24)
    export_pdf.savefig(fig)
    plt.close()

#################################################################################################
# Create GeoTiff

file_string2='G16_RGB_DayConv_'+date_string


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