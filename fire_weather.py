from __future__ import division
import numpy as np
import pandas as pd
import seaborn as sns
from datetime import datetime

import importlib
print(importlib.import_module('mpl_toolkits').__path__)

#import matplotlib
#matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.tri as tri
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import matplotlib.animation as Animation
from matplotlib import cm
import mpl_toolkits.mplot3d.axes3d as p3
from mpl_toolkits.basemap import Basemap

from scipy.interpolate import griddata

import math
import glob
import os
import pickle

import numpy as np

print(np.__version__)


''' ~~~~~~~~~~~~~~~~~~~~~ import_SYNVAR_csv ~~~~~~~~~~~~~~~~~~~~~ '''
def import_SYNVAR_csv(lon_min, lon_max, lat_min, lat_max):
    #read in all csvs from folder
    #path = '..\\..\\data\\'
    #path = 'C:\\Users\Dan\Downloads\SOMPY_robust_clustering-master\SOMPY_robust_clustering-master\data\\'
    # path = ['/home/dp/Documents/FWP/NARR/csv_CAPE/',\
    #         '/home/dp/Documents/FWP/NARR/csv_PMSL/']
    # path_csv_H500 = '/home/dp/Documents/FWP/NARR/csv_H500/'
    path_all_csv = '/home/dp/Documents/FWP/NARR/csv/'

    ''' csv file names must end with either H500.csv, CAPE.csv, or PMSL.csv '''
    H500_files = glob.glob(os.path.join(path_all_csv, '*H500.csv')) # H500 file paths in a list
    print('H500_files:\n', H500_files)
    CAPE_files = glob.glob(os.path.join(path_all_csv, '*CAPE.csv'))
    PMSL_files = glob.glob(os.path.join(path_all_csv, '*PMSL.csv'))
    all_files = [H500_files, CAPE_files, PMSL_files]    # All file paths in a list of lists
    print('all_files:\n', all_files)
    SYNABBR_shortlist = ['H500', 'CAPE', 'PMSL']
    ''' Creating  '''
    SYNABBR_list = []
    for i,var_list in enumerate(all_files):
        print('var_list:\n', var_list)
        SYNABBR_list.append([SYNABBR_shortlist[i]]*len(var_list))
    print('SYNABBR_list:\n', SYNABBR_list)

    # Example files:
    # all_files     = [['2_H500.csv', '1_H500.csv'], ['1_CAPE.csv', '2_CAPE.csv'], ['1_PMSL.csv', '2_PMSL.csv']]
    # SYNABBR_list  = [['H500', 'H500'], ['CAPE', 'CAPE'], ['PMSL', 'PMSL']]

    # Looping through all_files = [[synvar_files],[synvar_files],[synvar_files]]. i goes from 0 to 2
    i_list = list(range(0,len(all_files)))
    df_all_csv = pd.DataFrame([])
    for i, SYNVAR_files, SYNABBR in zip(i_list, all_files, SYNABBR_shortlist):
        # When i = 0, all_files = ['/path/to/1_H500.csv', '/path/to/2_H500.csv'], SYNABBR_shortlist='H500'
        # When i = 1, all_files = ['/path/to/1_CAPE.csv', '/path/to/2_CAPE.csv'], SYNABBR_shortlist='CAPE'

        # Creating a dataframe generator for one type of synoptic variable on each loop through all_files
        # e.g. When i = 0, df_from_each_file contains all H500 data that concatenates into df
        df_from_each_file = (pd.read_csv(file, header='infer', index_col=['lon', 'lat', 'time']) for file in SYNVAR_files)
        print('df from each file:\n', df_from_each_file)
        df = pd.concat(df_from_each_file, axis=0)  # 
        print('concatenated df head:\n', df.head)
        print('concatenated df columns:\n', df.columns)
        # Resetting index, may not be necessary
        df.reset_index(inplace=True)
        print('df after reset_index:\n', df)
        print('Length of df after reset_index:\n', len(df))#[SYNABBR_shortlist[i]]))
        df['time'] = pd.to_datetime(df['time'], format='%m/%d/%Y (%H:%M)')
        df.set_index(['lon', 'lat', 'time'], inplace=True)
        df.sort_index(level=2, inplace=True)
        print('Length of df after reset_index:\n', len(df))
        if i == 0: # First time through loop, append df to columns
            # When i = 0, all H500 files in df are processed:
            df_all_csv = df_all_csv.append(df)
            print('11111111111111111111111111111 First df_all_csv concatenation:\n', df_all_csv)
        else: # Concat df to rows of df_all_csv
            df_all_csv = pd.concat((df_all_csv, df), axis=1, join='inner')
            print('22222222222222222222222222222 Second df_all_csv concatenation:\n', df_all_csv)
            print('22222222222222222222222222222 Columns of df_all_csv concatenation:\n', df_all_csv.columns)

        arr = df.values
        print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ arr:\n', arr)
        print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ np.shape(arr):', np.shape(arr))

    print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~Final df_all_csv:\n', df_all_csv)
    print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~Final df_all_csv w/ index:\n', df_all_csv)
    print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~Length of final df_all_csv w/index:', len(df_all_csv['CAPE']))

    print('df_all_csv.loc[index values]:\n', df_all_csv.loc[-131.602, 49.7179, '1979-01-01 00:00:00'])
    print('df_all_csv.loc[[index values]]:\n', df_all_csv.loc[[-131.602, 49.7179, '1979-01-01 00:00:00']])

    # get columns Lat, Lon, Mean Temp, Max Temp, Min temp, Precipitation
    data = df[[]]
    data = data.apply(pd.to_numeric,  errors='coerce') # Converting to floats
    data = data.dropna(how='any')
    #names = ['Latitude', 'Longitude', 'Monthly Median temperature (C)', 'Monthly Max temperature (C)', 'Monthly Min temperature (C)', 'Monthly total precipitation (mm)']
    print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ data.head():\n', data.head())
    

    '''
    SYNVAR = syn_var_str
    print('SYNVAR: ', SYNVAR)
    #df = df[['time','lon','lat','H500']]
    df = df[['time','lon','lat',SYNVAR]]
    cols = df.columns
    '''


    # Shouldn't need this conversion: dt_list = pd.to_datetime(t)
    #print('t datetime formatted:\n', t)
    
    ''' Referencing specific index values in df: '''
    print('type(df.index.get_level_values(0)):\n', type(df.index.get_level_values(0)))  # Referencing time type
    print('df.index.get_level_values(0)[0]:\n', df.index.get_level_values(0)[0])        # Referencing time index values
    print('type(df.index.get_level_values(1)):\n', type(df.index.get_level_values(1)))  # Referencing lon type
    print('df.index.get_level_values(1)[0]:\n', df.index.get_level_values(1)[0])        # Referencing lon index values
    print('type(df.index.get_level_values(2)):\n', type(df.index.get_level_values(2)))  # Referencing lon type
    print('df.index.get_level_values(2)[0]:\n', df.index.get_level_values(2)[0])        # Referencing lon index values


    SYNVAR = 'PMSL'
    t = df.index.get_level_values(0).tolist()
    x = df.index.get_level_values(1).tolist()
    y = df.index.get_level_values(2).tolist()
    z = df[SYNVAR].values.tolist()
    d = [i for i in zip(t,x,y,z)]
    df = pd.DataFrame(data=d, columns=['time','lon','lat',SYNVAR])#cols)

    df.set_index('time', inplace=True)
    #print('df.index:\n', df.index)
    #print('df.index[10]:\n', df.index[10])
    
    ''' ~~~~~~~~~~~~~~~  SYNVAR Tripcolor & Tricontourf: Unevenly spaced grid  ~~~~~~~~~~~~~~ '''

    # Looking at content of df index values:
    df2 = df[(df.index == datetime(1979,1,1,0,0,0))]#.values.tolist()
    df3 = df[(df.index == datetime(1979,1,1,3,0,0))]#.values.tolist()
    print('df2:\n', df2)
    print('df3:\n', df3)

    x_t0 = df2['lon'].values.tolist()
    y_t0 = df2['lat'].values.tolist()
    z_t0 = df2[SYNVAR].values.tolist()
    print('Shape x_t0:\n', np.shape(x_t0))
    print('Shape y_t0:\n', np.shape(y_t0))
    print('Shape z_t0:\n', np.shape(z_t0))

    #df_list = list(df.groupby('time'))
    #print('df_list:\n', df_list)
    SYNVAR_3d = np.array(list(df.groupby('time').apply(pd.DataFrame.to_numpy))) # For some reason to_numpy can be called without () and it works. Why is this allowed?
    print('SYNVAR_3d[0][10][2]:\n', SYNVAR_3d[0][10][2])
    #print('SYNVAR_3d[0,:,:]:\n', SYNVAR_3d[0,:,:])
    #print('SYNVAR_3d:\n', SYNVAR_3d)
    print('SYNVAR_3d shape:\n', np.shape(SYNVAR_3d))
    
    f, ax = plt.subplots(1,2, sharex=True, sharey=True)
    ax[0].tripcolor(x,y,z)
    ax[1].tricontourf(x_t0,y_t0,z_t0,20)#, cmap=cm.jet) # 20 contour levels is good quality

    ax[0].plot(x,y, 'ko ', markersize=1)
    ax[0].set_xlabel('Longitude'); ax[0].set_ylabel('Latitude')
    ax[1].plot(x_t0,y_t0, 'ko ', markersize=1)
    ax[1].set_xlabel('Longitude'); ax[1].set_ylabel('Latitude')

##   plt.savefig('SYNVAR_OR_WA_pyplot.png')
    plt.show()
    
    ''' ~~~~~~~~~~~~~~~  SYNVAR Surface Plot: Evenly spaced grid - DOESN'T WORK  ~~~~~~~~~~~~~~ '''
    npts = 200
    N = 200

    # -----------------------
    # Interpolation on a grid
    # -----------------------
    # A contour plot of irregularly spaced data coordinates
    # via interpolation on a grid.

    # Create grid values first.
    xi = np.linspace(lon_min, lon_max, N) #-126, -115
    yi = np.linspace(lat_min, lat_max, N) # 40, 51

    # Perform linear interpolation of the data (x,y)
    # on a grid defined by (xi,yi)
##    triang = tri.Triangulation(x, y)
##    interpolator = tri.LinearTriInterpolator(triang, z)
    Xi, Yi = np.meshgrid(xi, yi)
##    zi = interpolator(Xi, Yi)

    ''' Convert to a regular grid '''
    Zi = griddata((x_t0, y_t0), z_t0, (Xi,Yi), method='linear') #(xi[None,:], yi[:,None]), method='linear', fill_value=0)
    Z_grad_x = np.gradient(Zi, axis=1)
    Z_grad_y = np.gradient(Zi, axis=0)

    # print('xi:\n', xi)
    # print('yi:\n', yi)
    # print('Xi:\n', Xi)
    # print('Yi:\n', Yi)
    # print('zi:\n', Zi)
    
##    fig, (ax1, ax2) = plt.subplots(nrows=2)
##    ax1.contour(Xi, Yi, zi, levels=14, linewidths=0.5, colors='k')
##    cntr1 = ax1.contourf(Xi, Yi, zi, levels=14, cmap="RdBu_r")
##
##    fig.colorbar(cntr1, ax=ax1)
##    ax1.plot(x, y, 'ko', ms=3)
##    ax1.axis((-2, 2, -2, 2))
##    ax1.set_title('grid and contour (%d points, %d grid points)' %
##                  (npts, ngridx * ngridy))
##
##    plt.subplots_adjust(hspace=0.5)
##    plt.show()

    ''' ~~~~~~~~~~~~~~~~~~~ SYNVAR Surface plot ~~~~~~~~~~~~~~~~~~~ '''
    X = Xi
    Y = Yi
    Z = Zi

    Z_flat = np.ravel(Z)
    Z_min = np.nanmin(Z_flat)
    Z_max = np.nanmax(Z_flat)

    Z_grad_x_flat = np.ravel(Z_grad_x)
    Z_grad_x_min = np.nanmin(Z_grad_x_flat)
    Z_grad_x_max = np.nanmax(Z_grad_x_flat)

    Z_grad_y_flat = np.ravel(Z_grad_y)
    Z_grad_y_min = np.nanmin(Z_grad_y_flat)
    Z_grad_y_max = np.nanmax(Z_grad_y_flat)

    fig = plt.figure()
    ax = fig.add_subplot(131, projection='3d')
    ax.set_zlim(Z_min, Z_max)
    surf = ax.plot_surface(X,Y,Z, cmap=cm.jet, vmin=Z_min, vmax=Z_max, linewidth=0, antialiased=False)
    ax.set_zlim(np.nanmin(Z_flat), np.nanmax(Z_flat))
    ax.zaxis.set_major_locator(LinearLocator(6))
    ax.zaxis.set_major_formatter(FormatStrFormatter('%d'))

    ax = fig.add_subplot(132, projection='3d')
    ax.set_zlim(Z_grad_x_min, Z_grad_x_max)
    surf = ax.plot_surface(X,Y,Z_grad_x, cmap=cm.jet, vmin=Z_grad_x_min, vmax=Z_grad_x_max, linewidth=0, antialiased=False)
    ax.set_zlim(np.nanmin(Z_grad_x_flat), np.nanmax(Z_grad_x_flat))
    ax.zaxis.set_major_locator(LinearLocator(6))
    ax.zaxis.set_major_formatter(FormatStrFormatter('%d'))

    ax = fig.add_subplot(133, projection='3d')
    ax.set_zlim(Z_grad_y_min, Z_grad_y_max)
    surf = ax.plot_surface(X,Y,Z_grad_y, cmap=cm.jet, vmin=Z_grad_y_min, vmax=Z_grad_y_max, linewidth=0, antialiased=False)
    ax.set_zlim(np.nanmin(Z_grad_y_flat), np.nanmax(Z_grad_y_flat))
    ax.zaxis.set_major_locator(LinearLocator(6))
    ax.zaxis.set_major_formatter(FormatStrFormatter('%d'))

    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')
    ax.set_zlabel(SYNVAR)
    # Color bar which maps values to colors:
    fig.colorbar(surf, shrink=0.5, aspect=5)
    plt.show()
    ''' ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ '''
    
    
    ''' ~~~~~~~~~~~~~~~~~~~~~~~~~ Example 3D Animation ~~~~~~~~~~~~~~~~~~~~~~~ '''
    def update_plot(frame_number, zarray, plot):
        plot[0].remove()
        plot[0] = ax.plot_surface(x, y, zarray[:,:,frame_number], cmap="magma")

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    N = 100
    nmax = 20
    x = np.linspace(-4,4,N+1)           # [-4, -3.428, -2.857, ..., 2.857, 3.428, 4]
    x, y = np.meshgrid(x, x)            # Make x list into a mesh
    print('x:\n', x)
    print('y:\n', y)
    print('(x,y):\n', (x,y))
    print('x shape:\n', np.shape(x))
    print('y shape:\n', np.shape(y))
    print('(x,y) shape:\n', np.shape((x,y)))
    zarray = np.zeros((N+1, N+1, nmax)) # Z values set to zero, filled below
    print('zarray shape:\n', np.shape(zarray))

    f = lambda x,y,sig : 1/np.sqrt(sig)*np.exp(-(x**2+y**2)/sig**2)

    for i in range(nmax):
        zarray[:,:,i] = f(x,y,1.5+np.sin(i*2*np.pi/nmax))

    plot = [ax.plot_surface(x, y, zarray[:,:,0], color='0.75', rstride=1, cstride=1)]
    ax.set_zlim(0,1.5)
    animate = Animation.FuncAnimation(fig, update_plot, nmax, interval=40, fargs=(zarray, plot))
    plt.show()
    ''' ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ ''' 
    
    
    ''' ~~~~~~~~~~~~ Animation: SYNVAR and SYNVAR Gradient ~~~~~~~~~~~~ '''
    def update_plot(frame_number, Zi_grad_x, plot):
        plot[0].remove()
        plot[0] = ax.plot_surface(Xi, Yi, Zi_grad_x[:,:,frame_number], cmap="magma")

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    N = 200       # x and y grid number of points. 100 x 100 = 10,000 grid points
    nmax = 17     # number of z matrices (e.g. frames or unique datetime objects)
    
    xi = np.linspace(lon_min, lon_max, N)       # -126, -115
    yi = np.linspace(lat_min, lat_max, N)       # 40, 51
    Xi, Yi = np.meshgrid(xi, yi)                # Xi, Yi meshgrid
    Xi_Yi = np.c_[Xi,Yi]
    print('(Xi,Yi):\n', (Xi,Yi))
    print('Xi_Yi:\n', Xi_Yi)
    Zi = np.zeros((N, N, nmax))                 # zi values set to zero, filled below
    Zi_grad_unknown = np.zeros((N, N, nmax))          # zi values set to zero, filled below
    Zi_grad_x = np.zeros((N, N, nmax))          # zi values set to zero, filled below
    Zi_grad_y = np.zeros((N, N, nmax))          # zi values set to zero, filled below

    # iterate through every x,y grid, differs for every datetime stamp
    
    z_min_list = []; z_max_list = []
    layers, row, cols = np.shape(SYNVAR_3d)
    for j in range(0,layers):
        xt = SYNVAR_3d[j,:,0]
        yt = SYNVAR_3d[j,:,1]
        zt = SYNVAR_3d[j,:,2]
        #print('zt for layer {}:\n'.format(j, zt))
        #xt = df_t['lon'].values.tolist()
        #yt = df_t['lat'].values.tolist()
        #zt = df_t[SYNVAR].values.tolist()
        # (xt, yt) is 1122 x 2
        # zt is 1122
        # (Xi, Yi) is N x N = 100 x 100
        xt_yt = np.c_[xt,yt]
        Zi[:,:,j] = griddata((xt,yt), zt, (Xi,Yi), method='linear') # Converts z data to a regular grid
        z_min_list.append(np.nanmin(Zi[:,:,j]))
        z_max_list.append(np.nanmax(Zi[:,:,j]))
        # print('np.gradient(Zi[:,:,j])[0]:\n', np.gradient(Zi[:,:,j])[0])
        #Zi_grad_unknown[:,:,j] = np.gradient(Zi[:,:,j])[0]
        Zi_grad_x[:,:,j] = np.gradient(Zi[:,:,j], axis=1)
        Zi_grad_y[:,:,j] = np.gradient(Zi[:,:,j], axis=0)

        print('~~~~~~~~~~~~~~~~~ nan min Zi[:,:,j]):\n', np.nanmin(Zi[:,:,j]))
        #print('Xi[0,:,j]:\n', Xi[0,:])
        #print('Yi[0,:,j]:\n', Yi[0,:])
        print('Zi[:,:,j]:\n', Zi[:,:,j])

    print('Zi[:,:,0]:\n', Zi[:,:,0])
    print('Zi_grad_unknown[:,:,0]:\n', Zi_grad_unknown[:,:,0])
    print('Zi_grad_x[:,:,0]:\n', Zi_grad_x[:,:,0])
    print('Zi_grad_y[:,:,0]:\n', Zi_grad_y[:,:,0])
    print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n')

    print('np.nanmin(z_min_list):', np.nanmin(z_min_list))
    print('np.nanmax(z_max_list):', np.nanmax(z_max_list))

    print('np.nanmin(Zi):', np.nanmin(Zi))
    print('np.nanmax(Zi):', np.nanmax(Zi))

    print('Zi all-layers min:', np.nanmin(Zi[:,:,j]))
    print('Zi all-layers max:', np.nanmax(Zi[:,:,j]))

    print('(xt,yt) shape:', np.shape((xt,yt)))
    print('xt_yt shape:', np.shape(xt_yt))
    print('zt shape:', np.shape(zt))
    print('Xi shape:', np.shape(Xi))
    print('Yi shape:', np.shape(Yi))
    print('(Xi,Yi) shape:', np.shape((Xi,Yi)))
    print('Zi shape:', np.shape(Zi))
    print('Zi[:,:,10] shape:\n', np.shape(Zi[:,:,10]))

    #Z_flat = np.ravel(zi)
    Z_min = np.nanmin(Zi_grad_x[:,:,j]) # Used to set 3D animation vertical range
    Z_max = np.nanmax(Zi_grad_x[:,:,j]) # Used to set 3D animation vertical range
    
    #plot = [ax.plot_surface(x, y, zarray[:,:,0], color='0.75', rstride=1, cstride=1)]
    plot = [ax.plot_surface(Xi, Yi, Zi_grad_x[:,:,0], vmin=Z_min, vmax=Z_max, linewidth=0, antialiased=False, color='0.75', rstride=1, cstride=1)]

    ax.set_zlim(Z_min, Z_max)
    ax.zaxis.set_major_locator(LinearLocator(6))
    ax.zaxis.set_major_formatter(FormatStrFormatter('%d'))
    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')
    ax.set_zlabel(SYNVAR+' x gradient')
    # Color bar which maps values to colors:
    #fig.colorbar(surf, shrink=0.5, aspect=5)
    # NEED THIS BUT NOT WORKING: fig.colorbar(plot, shrink=0.5, aspect=5)
    
    # Set animation figure parameters
    fig.set_size_inches(2, 2, True) # width (inches), height (inches), forward = True or False
    dpi = 300
    # Animation.FuncAnimation(figure reference, function that updates plot, time interval between frames in milliseconds, (data, plot reference))
    animate = Animation.FuncAnimation(fig, update_plot, nmax, interval=200, fargs=(Zi_grad_x, plot))
    
    '''--- Save Animation as mp4:'''
    plt.rcParams['animation.ffmpeg_path'] = '/usr/bin/ffmpeg'
    writer = Animation.FFMpegWriter(fps=3, codec='libx264', bitrate=1800)
    filename = SYNVAR + '_Jan_1_to_3_1979.mp4'
    animate.save(filename=filename, writer=writer, dpi=dpi)
    ''' ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ '''

    plt.show() # Animation is saved by the time it shows up.

    if syn_var_str == 'H500':
        SYNVAR_3d

    return SYNVAR_3d, df, t, x, y, z

''' ~~~~~~~~~~~~~~~~~~~~~ Run import_SYNVAR_csv ~~~~~~~~~~~~~~~~~~~~~ '''
SYNVAR_3d, df, t, x, y, z = import_SYNVAR_csv(-125,-116,41,50)
#                                        (csv filename, synoptic weather variable name, longitude min, longitude max, latitude min, latitude max)




''' Interpolation '''
def interpgrid():
    z = np.array([-5.,-15.,-25.,-36.,-49.,-65.,-84.,-105.5,-130.5,-159.5,-192.5,
                  -230.,-273.,-322.5,-379.,-443.,-515.,-596.,-688.,-792.,-909.5,
                  -1042.5,-1192.5,-1362.,-1553.5,-1770.,-2015.,-2285.,-2565.,-2845.])
    y = np.arange(0,100)
    print('z:\n', z)
    print('y:\n', y)
    ''' np.random.ran(nrows, ncols) results in an nrows x ncols matrix:
        np.cumsum(array, axis=0) sums array row-to-row for each column.
        e.g. within column 0, the rows are summed downward so that the
        bottom row of column 0 is a summation of all of the values above. '''
    yz_matrix = np.cumsum(np.random.rand(len(z), len(y)), axis=0)
    ''' With yz_matrix full of unique values, it's possible to  '''
    print('yz_matrix:\n', yz_matrix)

    fig, (ax, ax2) = plt.subplots(ncols=2)

    # plot raw data as pcolormesh
    Y,Z = np.meshgrid(y,z[::-1])
    print('Z:\n', Z)
    print('Y:\n', Y)
    print('YI Shape:', np.shape(YI)) # 2840 x 100
    ax.pcolormesh(Y,Z, yz_matrix, cmap='inferno')
    ax.set_title("pcolormesh data")

    # now interpolate data to new grid 
    zi = np.arange(-2845,-5) # A new, higher resolution z, increment is 1 from -2845 to -5
    print('zi:\n', zi)
    YI,ZI = np.meshgrid(y,zi)
    print('ZI:\n', ZI)
    print('ZI Shape:', np.shape(ZI))
    print('YI:\n', YI)
    print('YI Shape:', np.shape(YI)) # 2840 x 100
    points = np.c_[Y.flatten(),Z.flatten()] # Flattens the original 2-D Y array 30x100 to 1-D 1x3000

    ''' scipy.interpolate.griddate():
        points = original y,z points
        yz_matrix = data that fits into y,z points grid.
        (YI,ZI) = points at which to interpolate data
        method = linear interpolation '''
    interp = griddata(points, yz_matrix.flatten(), (YI,ZI), method='linear')
    print('interp:\n', interp)
    
    ax2.pcolormesh(YI,ZI, interp, cmap='inferno')
    ax2.set_title("pcolormesh interpolated")

    plt.show()
    return Y, Z
''' ~~~~~~~~~~~~~~~~~ '''

''' Contour '''
def simple_contour():
    x = [1,2,4,5,8,9,10]
    y = [1,3,5,7,8,14,18]
    z = np.random.randn(len(y), len(x))

    xi = np.arange(min(x), max(x), 0.5)
    yi = np.arange(min(y), max(y), 0.5)
    print('yi:\n', yi)
    
    x, y = np.meshgrid(x, y)
    xx, yy = np.meshgrid(xi, yi)
    print('xx:\n', xx)
    print('yy:\n', yy)

    # Interpolation
    points = np.c_[x.flatten(),y.flatten()]
    zi = griddata(points, z.flatten(), (xx, yy), method='linear')

    fig, (ax, ax2) = plt.subplots(ncols=2)

    ax.pcolormesh(x,y, z, cmap='inferno')
    ax.set_title("pcolormesh data")

    ax2.pcolormesh(xi,yi, zi, cmap='inferno')
    ax2.set_title("pcolormesh interpolated")

    plt.show()
    return



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
