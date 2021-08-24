#################################################################
# GOES-16 True Color RGB plotting code example
# Uses channels 2, 3, 1 to build 
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












##################################################################
# Below here is Connor's lame attempt to get the data output to 
# AGOL
##################################################################


##################################################################
# Trying to make a kml


from simplekml import (Kml, OverlayXY, ScreenXY, Units, RotationXY,
                       AltitudeMode, Camera)


def make_kml(llcrnrlon, llcrnrlat, urcrnrlon, urcrnrlat,
             figs, colorbar=None, **kw):
    """TODO: LatLon bbox, list of figs, optional colorbar figure,
    and several simplekml kw..."""

    kml = Kml()
    altitude = kw.pop('altitude', 2e7)
    roll = kw.pop('roll', 0)
    tilt = kw.pop('tilt', 0)
    altitudemode = kw.pop('altitudemode', AltitudeMode.relativetoground)
    camera = Camera(latitude=np.mean([urcrnrlat, llcrnrlat]),
                    longitude=np.mean([urcrnrlon, llcrnrlon]),
                    altitude=altitude, roll=roll, tilt=tilt,
                    altitudemode=altitudemode)

    kml.document.camera = camera
    draworder = 0
    for fig in figs:  # NOTE: Overlays are limited to the same bbox.
        draworder += 1
        ground = kml.newgroundoverlay(name='GroundOverlay')
        ground.draworder = draworder
        ground.visibility = kw.pop('visibility', 1)
        ground.name = kw.pop('name', 'overlay')
        ground.color = kw.pop('color', '9effffff')
        ground.atomauthor = kw.pop('author', 'ocefpaf')
        ground.latlonbox.rotation = kw.pop('rotation', 0)
        ground.description = kw.pop('description', 'Matplotlib figure')
        ground.gxaltitudemode = kw.pop('gxaltitudemode',
                                       'clampToSeaFloor')
        ground.icon.href = fig
        ground.latlonbox.east = llcrnrlon
        ground.latlonbox.south = llcrnrlat
        ground.latlonbox.north = urcrnrlat
        ground.latlonbox.west = urcrnrlon

    if colorbar:  # Options for colorbar are hard-coded (to avoid a big mess).
        screen = kml.newscreenoverlay(name='ScreenOverlay')
        screen.icon.href = colorbar
        screen.overlayxy = OverlayXY(x=0, y=0,
                                     xunits=Units.fraction,
                                     yunits=Units.fraction)
        screen.screenxy = ScreenXY(x=0.015, y=0.075,
                                   xunits=Units.fraction,
                                   yunits=Units.fraction)
        screen.rotationXY = RotationXY(x=0.5, y=0.5,
                                       xunits=Units.fraction,
                                       yunits=Units.fraction)
        screen.size.x = 0
        screen.size.y = 0
        screen.size.xunits = Units.fraction
        screen.size.yunits = Units.fraction
        screen.visibility = 1

    kmzfile = kw.pop('kmzfile', 'overlay.kmz')
    kml.savekmz(kmzfile)



make_kml(llcrnrlon=lon.min(), llcrnrlat=lat.min(),
         urcrnrlon=lon.max(), urcrnrlat=lat.max(),
         figs=['overlay1.png', 'overlay2.png'], colorbar=None)










##################################################################
# Trying to make a GeoJSON

col = ['lat', 'long','R', 'G', 'B']
lat_out=lat_CH1.flatten().data
lon_out=lon_CH1.flatten().data
R_out=R.flatten().data
G_out=G_true.flatten().data
B_out=B.flatten().data


df = pd.DataFrame(list(zip(lat_out, lon_out, R_out, G_out, B_out)), columns=col)





import geojson



def data2geojson(df):
    features = []
    insert_features = lambda X: features.append(
            geojson.Feature(geometry=geojson.Point((X["long"],
                                                    X["lat"],
                                                    X["R"],
                                                    X["G"],
                                                    X["B"])),
                            properties=()))
    df.apply(insert_features, axis=1)
    with open('map1.geojson', 'w', encoding='utf8') as fp:
        geojson.dump(geojson.FeatureCollection(features), fp, sort_keys=True, ensure_ascii=False)



data2geojson(df)

