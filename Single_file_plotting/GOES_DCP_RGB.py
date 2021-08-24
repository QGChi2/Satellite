#################################################################
# GOES-16 Day Cloud Phase RGB plotting code example
# Uses channels 13, 2, 5 to build #################################################################

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

# Custom
from GOES_LL_Conv import lat_lon_reproj


cwd = os.getcwd()

# Define root directory 
root=cwd+'\\'

# Define output directory
output1=root

#################################################################
# Need 3 channels to build Airmass RGB:
# File 1 = L2 ABI CMIP M6 C13 G16
# File 2 = L2 ABI CMIP M6 C02 G16
# File 3 = L2 ABI CMIP M6 C05 G16
#################################################################

file1='OR_ABI-L2-CMIPC-M6C13_G16_s20211371656164_e20211371658549_c20211371659038.nc'

file2='OR_ABI-L2-CMIPC-M6C02_G16_s20211371656164_e20211371658537_c20211371659018.nc'

file3='OR_ABI-L2-CMIPC-M6C05_G16_s20211371656164_e20211371658537_c20211371659007.nc'


##############################################
# file 1: CH. 13

lon_CH13,lat_CH13,data_CH13,data_units_CH13,data_time_grab_CH13,data_long_name_CH13,band_id_CH13,band_wavelength_CH13,band_units_CH13,var_name_CH13, lat_rad_CH13, lon_rad_CH13, lat_rad_1d_CH13, lon_rad_1d_CH13  = lat_lon_reproj(root, file1)


##############################################
# file 2: CH. 2

lon_CH2,lat_CH2,data_CH2,data_units_CH2,data_time_grab_CH2,data_long_name_CH2,band_id_CH2,band_wavelength_CH2,band_units_CH2,var_name_CH2, lat_rad_CH2, lon_rad_CH2, lat_rad_1d_CH2, lon_rad_1d_CH2 = lat_lon_reproj(root, file2)


##############################################
# file 3: CH. 5

lon_CH5,lat_CH5,data_CH5,data_units_CH5,data_time_grab_CH5,data_long_name_CH5,band_id_CH5,band_wavelength_CH5,band_units_CH5,var_name_CH5, lat_rad_CH5, lon_rad_CH5, lat_rad_1d_CH5, lon_rad_1d_CH5  = lat_lon_reproj(root, file3)


###################################################
# Convert to Celsius
###################################################

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

R = data_CH13
G = data_CH2_new
B = data_CH5_new
 
# Minimuns and Maximuns
Rmin = -53.5
Rmax = 7.5
 
Gmin = 0.0
Gmax = 0.78
 
Bmin = 0.01
Bmax = 0.59
 
R[R > Rmax] = Rmax
 
G[G > Gmax] = Gmax
 
B[B > Bmax] = Bmax

###################################################
# Normalize the RGBs
###################################################

# Choose the gamma
gamma = 1
 
# Normalize the data
R = ((R - Rmax) / (Rmin - Rmax)) ** (1/gamma)
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
file_string='G16_RGB_DCP_'+date_string

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
    ax.set_title('GOES-16 Day Cloud Phase', fontweight='bold', loc='left', fontsize=30)
    ax.set_title(data_time_grab_CH13[0:19] + ' UTC',loc='right', fontsize=24)
    plt.xticks(fontsize=24)
    plt.yticks(fontsize=24)
    plt.xlabel('Longitude [deg]', fontsize=24)
    plt.ylabel('Latitude  [deg]', fontsize=24)
    export_pdf.savefig(fig)
    plt.close()





