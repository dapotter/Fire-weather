import numpy as np
from sompy.sompy import SOMFactory
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import math
import glob
import os
import pygrib
import pickle

print(np.__version__)

infile = open('data.pkl','rb')
data = pickle.load(infile)
infile.close()
print('data[1][1]:', data[1][1])
print('data[1][1][0]:', data[1][1][0])
''' Weather variables and levels in data list: '''
'''
Temp, IsothermZero level 0:    level_var = -4, weather_var = 1
Geopotential Height, 500 hPa:  level_var = -25, weather_var = 2
Pres, 80 m height:             level_var = -41, weather_var = 3
'''
level_var = -41
weather_var = 3
Z = data[level_var][weather_var]

''' Getting Z directly from pickle: '''
##infile = open('Z.pkl','rb')
##Z = pickle.load(infile)
##infile.close()
print('Z[0]:', Z[0])

infile = open('X.pkl','rb')
X = pickle.load(infile)
infile.close()
print('X[0]:', X[0])

infile = open('Y.pkl','rb')
Y = pickle.load(infile)
infile.close()
print('Y[0]:', Y[0])

Z_list = []
for l in Z:
    Z_list.extend(l)

''' Longitude (x-axis) '''
X_list = []
for l in X:
    l -= 360
    X_list.extend(l)

''' Latitude (y-axis) '''
Y_list = []
for l in Y:
    l -= 90
    l *= -1 # y-axis (latitude) needs to be flipped
    Y_list.extend(l)

# Convert Kelvin to def. F:
if weather_var == 1:
    Z = ((Z-273.15)*(9/5)) + 32


XYZ_list = [[x,y,z] for x,y,z in zip(X_list,Y_list,Z_list)]

df_XYZ = pd.DataFrame(XYZ_list, columns=['Lon','Lat','GpH'])
print('df_XYZ:\n', df_XYZ)

''' Plotting '''
Z_flat = [x for sublist in Z for x in sublist]
Z_flat_max = max(Z_flat)
Z_flat_min = min(Z_flat)
print('Z_flat_max:', Z_flat_max)
print('Z_flat_min:', Z_flat_min)
cbar_res = (Z_flat_max - Z_flat_min)/280 # A cbar_res of 0.5 results in a good Temp plot
print('cbar_res:', cbar_res)
plt.figure()
clev = np.arange(Z_flat_min, Z_flat_max, cbar_res)
plt.contourf(X,Y,Z,clev,cmap=plt.cm.jet)
#plt.imshow(Z, vmin = -43, vmax = 90, cmap=plt.cm.jet)#,extent=[X.min(), X.max(), Y.min(), Y.max()])
plt.colorbar()
plt.xlabel('Longitude, deg'); plt.ylabel('Latitude, deg')
plt.show()






##''' Trying to open a grib file with pygrib:
##grbs = pygrib.open('sampledata/flux.grb')
##print('grbs:\n', grbs)
###C:\Users\Dan\Desktop\pygrib-master\pygrib-master\sampledata\flux.grb
##'''

''' Script for importing weather files from rad.ucar.edu.
    Add file names to "Listoffiles" '''

##import sys
##import os
##import urllib2
##import cookielib
##
##if (len(sys.argv) != 2):
##  print "usage: "+sys.argv[0]+" [-q] password_on_RDA_webserver"
##  print "-q suppresses the progress message for each file that is downloaded"
##  sys.exit(1)
##
##passwd_idx=1
##verbose=True
##if (len(sys.argv) == 3 and sys.argv[1] == "-q"):
##  passwd_idx=2
##  verbose=False
##
##cj=cookielib.MozillaCookieJar()
##opener=urllib2.build_opener(urllib2.HTTPCookieProcessor(cj))
##
### check for existing cookies file and authenticate if necessary
##do_authentication=False
##if (os.path.isfile("auth.rda.ucar.edu")):
##  cj.load("auth.rda.ucar.edu",False,True)
##  for cookie in cj:
##    if (cookie.name == "sess" and cookie.is_expired()):
##      do_authentication=True
##else:
##  do_authentication=True
##if (do_authentication):
##  login=opener.open("https://rda.ucar.edu/cgi-bin/login","email=daniel.b.potter@gmail.com&password="+sys.argv[1]+"&action=login")
##
### save the authentication cookies for future downloads
### NOTE! - cookies are saved for future sessions because overly-frequent authentication to our server can cause your data access to be blocked
##  cj.clear_session_cookies()
##  cj.save("auth.rda.ucar.edu",True,True)
##
### download the data file(s)
##listoffiles=["3HRLY/1979/NARR3D_197901_0103.tar"]
##for file in listoffiles:
##  idx=file.rfind("/")
##  if (idx > 0):
##    ofile=file[idx+1:]
##  else:
##    ofile=file
##  if (verbose):
##    sys.stdout.write("downloading "+ofile+"...")
##    sys.stdout.flush()
##  infile=opener.open("http://rda.ucar.edu/data/ds608.0/"+file)
##  outfile=open(ofile,"wb")
##  outfile.write(infile.read())
##  outfile.close()
##  if (verbose):
##    sys.stdout.write("done.\n")

''' ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ '''
####Run to plot heatmap of CAPE:
####df = pd.read_csv('model data CAPE 180-0 mb.csv')
####print(df.head())
####df2 = df[['lon','lat','value']]
####df2 = df2.loc[(df2['lon'] >= -125) & (df2['lon'] <= -117) & (df2['lat'] >= 39) & (df2['lat'] <= 49), ['lat','lon','value']]
####df2.set_index(['lat','lon'], inplace=True, drop=True)
####print('df2:\n', df2)
####df_piv = pd.pivot_table(df2, values=['value'], index=['lat'], columns=['lon'], fill_value=0)
####print('df_piv:\n', df_piv)
####xticks = df_piv.columns[[1]]
####yticks = np.arange(min(df_piv.index), max(df_piv.index))
####''' Plotting: '''
####ax = sns.heatmap(df_piv)
####ax.invert_yaxis()
#####ax = sns.heatmap(df_piv, )
####ax.set_xlabel('Longitude')
####ax.set_ylabel('Latitude')
#####print('ax.get_xlim()[1]:\n', ax.get_xlim()[1])
#####print('ax.get_ylim()[1]:\n', ax.get_ylim()[1])
#####ax.set_xticks(xticks*ax.get_xlim()[1])
#####ax.set_yticks(yticks*ax.get_ylim()[1])
####plt.setp(ax.xaxis.get_majorticklabels(), rotation=90)
####plt.tight_layout()
####plt.title('CAPE - (CA,OR,WA), 180-0 mb, - 1/16/19')
####plt.show()
''' ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ '''

### read in all csvs from folder
##path = '..\\..\\data\\'
##path = 'C:\\Users\Dan\Downloads\SOMPY_robust_clustering-master\SOMPY_robust_clustering-master\data\\'
##all_files = glob.glob(os.path.join(path, "*.csv"))
##print('all_files:\n' ,all_files[0:10])
##
### concat into one df
##df_from_each_file = (pd.read_csv(f, skiprows = 31) for f in all_files)
##concatenated_df   = pd.concat(df_from_each_file, ignore_index=True)
##
##print('concatenated df head:\n', concatenated_df.head)
##print('concatenated df columns:\n', concatenated_df.columns)
##
### get columns Lat, Long, Mean Temp, Max Temp, Min temp, Precipitation
##data = concatenated_df[['Lat', 'Long', 'Tm', 'Tx', 'Tn', 'P']]
##data = data.apply(pd.to_numeric,  errors='coerce') # Converting to floats
##data = data.dropna(how='any')
##names = ['Latitude', "longitude", 'Monthly Median temperature (C)','Monthly Max temperature (C)', 'Monthly Min temperature (C)', 'Monthly total precipitation (mm)']
##
##print(data.head())
##
### create the SOM network and train it. You can experiment with different normalizations and initializations
##sm = SOMFactory().build(data.values, normalization = 'var', initialization='pca', component_names=names)
##sm.train(n_job=1, verbose=False, train_rough_len=2, train_finetune_len=5)
##
### The quantization error: average distance between each data vector and its BMU.
### The topographic error: the proportion of all data vectors for which first and second BMUs are not adjacent units.
##topographic_error = sm.calculate_topographic_error()
##quantization_error = np.mean(sm._bmu[1])
##print ("Topographic error = %s; Quantization error = %s" % (topographic_error, quantization_error))
##
### component planes view
##from sompy.visualization.mapview import View2D
##view2D  = View2D(10,10,"rand data",text_size=12)
##view2D.show(sm, col_sz=4, which_dim="all", denormalize=True)
##
### U-matrix plot
##from sompy.visualization.umatrix import UMatrixView
##
##umat  = UMatrixView(width=10,height=10,title='U-matrix')
##umat.show(sm)
##
##
##
### do the K-means clustering on the SOM grid, sweep across k = 2 to 20
##from sompy.visualization.hitmap import HitMapView
##K = 20 # stop at this k for SSE sweep
##K_opt = 18 # optimal K already found
##[labels, km, norm_data] = sm.cluster(K,K_opt)
##hits  = HitMapView(20,20,"Clustering",text_size=12)
##a=hits.show(sm)
##
##import gmplot
##
##gmap = gmplot.GoogleMapPlotter(54.2, -124.875224, 6)
##j = 0
##for i in km.cluster_centers_:
##    gmap.marker(i[0],i[1],'red', title="Centroid " + str(j))
##    j += 1
##
##gmap.draw("centroids_map.html")
##
##
##from bs4 import BeautifulSoup
##
##def insertapikey(fname, apikey):
##    """put the google api key in a html file"""
##    def putkey(htmltxt, apikey, apistring=None):
##        """put the apikey in the htmltxt and return soup"""
##        if not apistring:
##            apistring = "https://maps.googleapis.com/maps/api/js?key=%s&callback=initMap"
##        soup = BeautifulSoup(htmltxt, 'html.parser')
##        body = soup.body
##        src = apistring % (apikey, )
##        tscript = soup.new_tag("script", src=src, async="defer")
##        body.insert(-1, tscript)
##        return soup
##    htmltxt = open(fname, 'r').read()
##    soup = putkey(htmltxt, apikey)
##    newtxt = soup.prettify()
##    open(fname, 'w').write(newtxt)
##API_KEY= 'YOUR API KEY HERE'
##insertapikey("centroids_map.html", API_KEY)
##
##
##gmap = gmplot.GoogleMapPlotter(54.2, -124.875224, 6)
##j = 0
##for i in km.cluster_centers_:
##    gmap.marker(i[0],i[1],'red', title="Centroid " + str(j))
##    j += 1
##
##gmap.draw("centroids_map.html")
