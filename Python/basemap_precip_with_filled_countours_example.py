from mpl_toolkits.basemap import Basemap, cm
# requires netcdf4-python (netcdf4-python.googlecode.com)
from netCDF4 import Dataset as NetCDFFile
import numpy as np
import matplotlib.pyplot as plt

# plot rainfall from NWS using special precipitation
# colormap used by the NWS, and included in basemap.

'''nc = NetCDFFile('../../../examples/nws_precip_conus_20061222.nc')'''
#nc = NetCDFFile('/home/dp/Documents/FWP/nws_precip_netcdf4/nws_precip_1day_20190311_conus.nc')
#--- Enter a date of 20190211, 20190212, or 20190213
#nc = NetCDFFile('/home/dp/Documents/FWP/nws_precip_netcdf4/nws_precip_1day_20190213_conus.nc')
#nc = NetCDFFile('/home/dp/Documents/FWP/nws_precip_netcdf4/out.nc')
nc = NetCDFFile('/home/dp/Documents/FWP/gridMET/erc_1979.nc')

# data from http://water.weather.gov/precip/
print('netCDF4 file variables:\n', nc.variables)

# Synoptic data from NARR:
# H500 = nc.variables['HGT_221_ISBL'] # observation = precipitation (inches)
# print('H500:\n', H500)
# data = H500[12] # Convert inches to millimeters precip
# print('data:\n', data)

# ERC data from gridMET:
ERC = nc.variables['energy_release_component-g'] # observation = precipitation (inches)
print('ERC:\n', ERC)
data = ERC
print('data:\n', data)

print('np.shape(data):', np.shape(data))

data_min = int(np.nanmin(data))
data_max = int(np.nanmax(data))
print('data nanmin:',data_min)
print('data nanmax:',data_max)

# H500 lon lat:
# lat = nc.variables['gridlat_221'][:]
# lon = nc.variables['gridlon_221'][:]

# ERC lon lat:
lat = nc.variables['lat'][:]
lon = nc.variables['lon'][:]
print('lon:\n', lon)
print('lat:\n', lat)

lon_0 = -126#latcorners.mean()#-nc.variables['crs'].getValue('+lon_0')
lat_0 = 40#loncorners.mean()#nc.variables['crs'].getValue('+lat_0')
lat_ts = 60

# create figure and axes instances
fig = plt.figure(figsize=(8,8))
ax = fig.add_axes([0.1,0.1,0.8,0.8])
# create polar stereographic Basemap instance.
''' m = Basemap(projection='stere',lon_0=lon_0,lat_0=90.,lat_ts=lat_0,\
            llcrnrlat=latcorners[0],urcrnrlat=latcorners[2],\
            llcrnrlon=loncorners[0],urcrnrlon=loncorners[2],\
            rsphere=6371200.,resolution='l',area_thresh=10000) '''
''' m = Basemap(projection='stere',lon_0=lon_0,lat_0=lat_0,lat_ts=lat_ts,\
            llcrnrlat=llcrnrlat,urcrnrlat=urcrnrlat,\
            llcrnrlon=llcrnrlon,urcrnrlon=urcrnrlon,\
            rsphere=6371200.0,resolution='l',area_thresh=10000) '''
m = Basemap(width=12000000,height=9000000,
            rsphere=(6378137.00,6356752.3142),\
            resolution='l',area_thresh=1000.,projection='lcc',\
            lat_1=45.,lat_2=55,lat_0=50,lon_0=-107.)

# draw coastlines, state and country boundaries, edge of map.
m.drawcoastlines()
m.drawstates()
m.drawcountries()
# draw parallels.
parallels = np.arange(0.,90,10.)
m.drawparallels(parallels,labels=[1,0,0,0],fontsize=10)
# draw meridians
meridians = np.arange(180.,360.,10.)
m.drawmeridians(meridians,labels=[0,0,0,1],fontsize=10)
ny = data.shape[0]; nx = data.shape[1]
print('nx:', nx)
print('ny:', ny)
lons, lats = m.makegrid(nx, ny) # get lat/lons of ny by nx evenly space grid.
print('lons:\n', lons)
print('lats:\n', lats)
x, y = m(lons, lats) # compute map proj coordinates.
# draw filled contours.
#clevs = [0,1,2.5,5,7.5,10,15,20,30,40,50,70,100,150,200,250,300,400,500,600,750]
clevs = list(range(data_min, data_max, 10))
cs = m.contourf(x,y,data,clevs,cmap=cm.s3pcpn)
# add colorbar.
cbar = m.colorbar(cs,location='bottom',pad="5%")
cbar.set_label('mm')
# add title
'''plt.title(prcpvar.long_name+' for period ending '+prcpvar.dateofdata)'''
plt.show()

cs = m.contourf(data(x,y),30,cmap=plt.cm.jet)
plt.show()

# setup north polar stereographic basemap.
# The longitude lon_0 is at 6-o'clock, and the
# latitude circle boundinglat is tangent to the edge
# of the map at lon_0. Default value of lat_ts
# (latitude of true scale) is pole.
m = Basemap(projection='npstere',boundinglat=10,lon_0=-105,resolution='l')
m.drawcoastlines()
m.fillcontinents(color='coral',lake_color='aqua')
# draw parallels and meridians.
parallels = np.arange(0.,90,10.)
m.drawparallels(parallels,labels=[1,0,0,0],fontsize=10)
# draw meridians
meridians = np.arange(180.,360.,10.)
m.drawmeridians(meridians,labels=[0,0,0,1],fontsize=10)
m.drawmapboundary(fill_color='aqua')
# draw tissot's indicatrix to show distortion.
ax = plt.gca()
for y in np.linspace(m.ymax/20,19*m.ymax/20,10):
    for x in np.linspace(m.xmax/20,19*m.xmax/20,10):
        lon, lat = m(x,y,inverse=True)
        poly = m.tissot(lon,lat,2.5,100,\
                        facecolor='green',zorder=10,alpha=0.5)
plt.title("North Polar Stereographic Projection")
plt.show()