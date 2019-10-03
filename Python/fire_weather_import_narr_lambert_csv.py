from __future__ import division
import numpy as np
import pandas as pd
import seaborn as sns
from datetime import datetime
from datetime import date
from datetime import timedelta
import importlib
print(importlib.import_module('mpl_toolkits').__path__)

#import matplotlib
#matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.tri as tri
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import matplotlib.animation as Animatiston
from matplotlib import cm
import mpl_toolkits.mplot3d.axes3d as p3
from mpl_toolkits.basemap import Basemap

from scipy.interpolate import griddata
from scipy import spatial
from sklearn import neighbors

import math
import glob
import os
import pickle

import numpy as np

print(np.__version__)

import psycopg2



def import_NARR_lambert_csv(lon_min, lon_max, lat_min, lat_max, import_interval, export_interval, NARR_csv_in_dir, NARR_pkl_out_dir):
    # ----------------------------------------------------------------------------------------------------
    # WARNING:  This function incorrectly calculates gradients for synoptic variables by using lambert
    #           irregularly gridded lambert conformal data. The data has been regridded to a rectilinear
    #           grid in NCL, gradients calculated in NCL, and is imported by import_NARR_lambert_csv(), which
    #           supersedes this function.
    #           Also, this function (import_NARR_lambert_csv) creates an animated surface plot with an
    #           interpolated 200 x 200 regular grid. This function should be copied to its own unique function that
    #           uses the rectilinear data output by import_NARR_lambert_csv() below.
    #
    # Parameters:
    #
    # lon_min: subsetting the data to OR and WA
    # lon_max: ""
    # lat_min: ""
    # lat_max: ""
    #
    # import_interval:      Number of csv files to import from the NARR_csv_in_dir import path per loop.
    #                       If import_interval is 2, the script will import all csv's starting with
    #                       0_ and 1_ (e.g. 0_H500.csv, 0_PMSL.csv, etc) then the next group
    #                       of csv files (e.g. 2_H500.csv, 2_PMSL.csv, ..., 3_H500.csv, 3_PMSL.csv), etc.
    #                       what happens if the number of csv files doesn't evenly divide by the 
    #                       import_interval (e.g. 11 files with an interval of 2)?
    #
    # export_interval:      Number of times through the csv import loop before exporting the data as a
    #                       pickle file. If import_interval is 2 and export_interval is 10, then 20 csv
    #                       files will be imported before the data is exported as a pickle file, then
    #                       another 20, an so on.
    #
    # NARR_csv_in_dir:      Path where csv files of NARR data are located and imported from
    #
    # NARR_pkl_out_dir:     Path where the final dataframe of formatted synoptic variables are exported to
    # 
    #
    # Script Function:
    # 
    # 1)    Read in import_interval number of NARR csv file in which data is in lambert conformal format,
    #       a type of irregular grid for meterological data
    #
    # 2)    Drop all H500 and PMSL data, keep their gradients  
    #  
    # 3)    Export all NARR data (df_narr) to csv and pickle files
    #
    # Note: The reason that it's necessary to modulate the number of csv files imported at once
    #       using 'import_interal' is because all csv data is held in RAM, which quickly becomes
    #       overwhelmed when more than 2 csv files are imported if each one has 100 grib files
    #       (roughly 300 hrs = 12.5 days) worth of data.
    #
    #
    # Old import paths:
    # NARR_csv_in_dir = '/home/dp/Documents/FWP/NARR/csv/'
    # NARR_csv_in_dir = '/mnt/seagate/NARR/csv/'
    #
    # Old export paths:
    # NARR_pkl_out_dir = '/home/dp/Documents/FWP/NARR/pickle/'
    # ----------------------------------------------------------------------------------------------------

    # NARR csv file names must end with either H500.csv, CAPE.csv, or PMSL.csv:
    H500_files = glob.glob(os.path.join(NARR_csv_in_dir, '*H500.csv'))
    PMSL_files = glob.glob(os.path.join(NARR_csv_in_dir, '*PMSL.csv'))
    CAPE_files = glob.glob(os.path.join(NARR_csv_in_dir, '*CAPE.csv'))
    print('H500_files:\n', H500_files)
    print('PMSL_files:\n', PMSL_files)
    print('CAPE_files:\n', CAPE_files)

    all_NARR_files = [H500_files, PMSL_files, CAPE_files] # All file hpaths in a list of lists
    print('H500 file paths:\n', H500_files)
    print('All NARR file paths :\n', all_NARR_files)
    all_NARR_files = [sorted(file_list) for file_list in all_NARR_files]
    print('All NARR file paths - after sorting:\n', all_NARR_files)

    num_csv_files = len(H500_files)
    csv_file_index_list = list(range(0, num_csv_files-import_interval, import_interval))
    print('csv_file_index_list:\n', csv_file_index_list)
    start_end_list = [(s,s+import_interval) for s in csv_file_index_list]
    print('start_end_list:\n', start_end_list)

    for pkl_counter, (s,e) in enumerate(start_end_list):
        # This loop imports import_interval number of csv files at a time,
        # creating a dataframe, and exporting it as a pickle file on each iteration.
        # pkl_counter is used to create the naming prefix for exporting pickle files.
        select_NARR_files = [file_list[s:e] for file_list in all_NARR_files]
        print('Select NARR file paths - start index to end index:\n', select_NARR_files)

        SYNABBR_shortlist = ['H500', 'PMSL', 'CAPE']
        ''' Creating '''
        SYNABBR_list = []
        for i,var_list in enumerate(select_NARR_files):
            print('var_list:\n', var_list)
            SYNABBR_list.append([SYNABBR_shortlist[i]]*len(var_list))
        print('SYNABBR_list:\n', SYNABBR_list)

        # Example files:
        # select_NARR_files = [['1_H500.csv', '2_H500.csv'], ['1_CAPE.csv', '2_CAPE.csv'], ['1_PMSL.csv', '2_PMSL.csv']]

        # SYNABBR_list = [['H500', 'H500'], ['CAPE', 'CAPE'], ['PMSL', 'PMSL']]

        # Looping through select_NARR_files = [[synvar_files],[synvar_files],[synvar_files]]. i goes from 0 to 2
        i_list = list(range(0,len(select_NARR_files)))
        df_all_synvar = pd.DataFrame([])

        for i, SYNVAR_files, SYNABBR in zip(i_list, select_NARR_files, SYNABBR_shortlist):
            # Loops through list of file paths, combines all csv file data into
            # one dataframe df_all_synvar

            # When i = 0, SYNVAR_files = ['/path/to/1_H500.csv', '/path/to/2_H500.csv', ...], SYNABBR_shortlist='H500'
            # When i = 1, SYNVAR_files = ['/path/to/1_CAPE.csv', '/path/to/2_CAPE.csv', ...], SYNABBR_shortlist='CAPE'

            # Creating a dataframe generator for one type of synoptic variable on each loop through select_NARR_files
            # e.g. When i = 0, df_from_each_file contains all H500 data that concatenates into df
            df_from_each_file = (pd.read_csv(file, header='infer', index_col=['lon', 'lat', 'time']) for file in SYNVAR_files)
            print('df from each file:\n', df_from_each_file)
            df = pd.concat(df_from_each_file, axis=0)
            print('concatenated df head:\n', df.head)
            print('concatenated df columns:\n', df.columns)
            # # Resetting index, may not be necessary
            # df.reset_index(inplace=True)
            # df.set_index('lon', inplace=True)
            # print('df after reset_index:\n', df)
            # print('Length of df after reset_index:\n', len(df))

            # Concatenating all H500 csv's together, then all PMSL csv's to it, and so on.
            # df is either all H500 csv's concatenated, or all PMSL csv's, and so on. See
            # the dataframe generator above for how df is created.
            if i == 0: # First time through loop, append df to columns
                # When i = 0, all H500 files in df are processed:
                df_all_synvar = df_all_synvar.append(df)
                print('First df_all_synvar concatenation:\n', df_all_synvar)
            else: # Concat df to rows of df_all_synvar
                df_all_synvar = pd.concat((df_all_synvar, df), axis=1, join='outer')
                print('Second df_all_synvar concatenation:\n', df_all_synvar)
                print('Columns of df_all_synvar concatenation:\n', df_all_synvar.columns)

            arr = df.values
            print(' arr:\n', arr)
            print('np.shape(arr):', np.shape(arr))

        print('Final df_all_synvar:\n', df_all_synvar)
        print('Length of final df_all_synvar w/index:', len(df_all_synvar['CAPE']))

        # Setting multi-index's time index to datetime. To do this, the index must be recreated.
        # https://stackoverflow.com/questions/45243291/parse-pandas-multiindex-to-datetime
        # Note that the format provided is the format in the csv file. pd.to_datetime converts
        # it to the format it thinks is appropriate.
        df_all_synvar.index = df_all_synvar.index.set_levels([df.index.levels[0], df.index.levels[1], pd.to_datetime(df.index.levels[2], format='%m/%d/%Y (%H:%M)')])
        # This doesn't work if the date is out of range for the csv file that is being read in
        # print('**** df_all_synvar.loc[index values]:\n', df_all_synvar.loc[-131.602, 49.7179, '1979-01-01 00:00:00'])
        # print('**** df_all_synvar.loc[[index values]]:\n', df_all_synvar.loc[[-131.602, 49.7179, '1979-01-01 00:00:00']])
        print('df_all_synvar: Contains all synoptic variables from csv files:\n', df_all_synvar)

        # Pickle out:
        df_all_synvar_pkl = df_all_synvar.to_pickle('/home/dp/Documents/FWP/NARR/pickle/df_all_synvar.pkl')

        # get columns Lat, Lon, Mean Temp, Max Temp, Min temp, Precipitation
        data = df[[]]
        data = data.apply(pd.to_numeric,  errors='coerce') # Converting data to floats
        data = data.dropna(how='any')
        #names = ['Latitude', 'Longitude', 'Monthly Median temperature (C)', 'Monthly Max temperature (C)', 'Monthly Min temperature (C)', 'Monthly total precipitation (mm)']
        print('data.head():\n', data.head())


        '''--- Plots a select synoptic variable from df_all_synvar ---'''

        # Checking index referencing:
        print('type(df.index.get_level_values(0)):\n', type(df.index.get_level_values(0)))  # Referencing lon type
        print('df.index.get_level_values(0)[0]:\n', df.index.get_level_values(0)[0])        # Referencing lon index values
        print('type(df.index.get_level_values(1)):\n', type(df.index.get_level_values(1)))  # Referencing lat type
        print('df.index.get_level_values(1)[0]:\n', df.index.get_level_values(1)[0])        # Referencing lat index values
        print('type(df.index.get_level_values(2)):\n', type(df.index.get_level_values(2)))  # Referencing time type
        print('df.index.get_level_values(2)[0]:\n', df.index.get_level_values(2)[0])        # Referencing time index values


        x = df_all_synvar.index.get_level_values(0).tolist()
        y = df_all_synvar.index.get_level_values(1).tolist()
        t = df_all_synvar.index.get_level_values(2).tolist()
        # print('x values from df_all_synvar index level 0:\n', x)
        # print('y values from df_all_synvar index level 1:\n', y)
        # print('t values from df_all_synvar index level 2:\n', t)

        df_narr = pd.DataFrame([])
        df_all_synvar_grid_interp = pd.DataFrame([])
        for i, SYNVAR in enumerate(SYNABBR_shortlist):
            # This loops through all of the synoptic variables (H500, etc)
            # and puts them individually into dataframes 
            print('Now processing: ', SYNVAR)

            # Getting z values. x, y, t values gathered just before loop.
            z = df_all_synvar[SYNVAR].values.tolist()
            d = [i for i in zip(t,x,y,z)]

            df = pd.DataFrame(data=d, columns=['time','lon','lat',SYNVAR])

            df.set_index('time', inplace=True)
            print('df.index:\n', df.index)
            print('df.index[10]:\n', df.index[10])


            ''' --------------- SYNVAR Tripcolor & Tricontourf: Unevenly spaced grid --------------- '''
            # # COMMENTING OUT BECAUSE THIS HAS BEEN MOSTLY MOVED INTO THE LOOP CALCULATING GRADIENTS
            # # BELOW USING SYNVAR_3d DATA.

            # Looking at content of df index values:
            datetime_to_plot = df.index[0] #datetime(1979,1,1,0,0,0)
            datetime_to_plot_str = datetime_to_plot.strftime('%b %d, %Y')
            # Getting all lon-lat pairs whose dates match
            df_datetime = df[(df.index == datetime_to_plot)]
            print('df_datetime:\n', df_datetime)

            # Used to calculate X gradient. Data is already
            # ordered such that each data point's neighbor is
            # east or west of it.
            x_t0 = df_datetime['lon'].values.tolist()
            y_t0 = df_datetime['lat'].values.tolist()
            z_t0 = df_datetime[SYNVAR].values.tolist()
            print('Shape x_t0:\n', np.shape(x_t0))
            print('Shape y_t0:\n', np.shape(y_t0))
            print('Shape z_t0:\n', np.shape(z_t0))

            # Used to calculate Y gradient. Reshaping the data so that
            # every data point's neighbor is higher or lower in latitude
            x_t0_2 = np.array(x_t0).reshape((40,40)).T.ravel()
            y_t0_2 = np.array(y_t0).reshape((40,40)).T.ravel()
            z_t0_2 = np.array(z_t0).reshape((40,40)).T.ravel()
            print('Shape x_t0_2:\n', np.shape(x_t0_2))
            print('Shape y_t0_2:\n', np.shape(y_t0_2))
            print('Shape z_t0_2:\n', np.shape(z_t0_2))

            # # Checking to make sure the data is reordered:
            # xyz = [[x,y,z] for x,y,z in zip(x_t0_2, y_t0_2, z_t0_2)]
            # df_t0_2 = pd.DataFrame(xyz, columns=['lon','lat','H500'])
            # print('df_t0_2:\n', df_t0_2)

            # # Calculating gradients. df_all_synvar currently contains H500, CAPE, PMSL.
            # # print('x_t0:\n', x_t0)
            # # print('y_t0:\n', y_t0)
            # # print('z_t0:\n', z_t0)

            # # Calculating X gradient. Y Gradient (z_grad_y) is incorrect
            # z_grad_x = np.gradient(z_t0, x_t0)
            # z_grad_y = np.gradient(z_t0, y_t0)
            # print('z_grad_x:\n', z_grad_x[0:30])
            # print('z_grad_y:\n', z_grad_y[0:30])

            # # Calculating Y gradient. X Gradient (z_grad_x_2) is incorrect
            # z_grad_x_2 = np.gradient(z_t0_2, x_t0_2)
            # z_grad_y_2 = np.gradient(z_t0_2, y_t0_2)
            # print('z_grad_x_2:\n', z_grad_x_2[0:30])
            # print('z_grad_y_2:\n', z_grad_y_2[0:30])

            # f, ax = plt.subplots(3,3, figsize=(12,12))
            # ax[0,0].plot(x_t0, y_t0, marker='o', markersize=0.5, linewidth=0.1)
            # ax[0,1].plot(x_t0, z_t0, marker='o', markersize=0.5, linewidth=0.1)
            # ax[0,2].plot(y_t0, z_t0, marker='o', markersize=0.5, linewidth=0.1)

            # ax[1,0].plot(x_t0, z_grad_x, marker='o', markersize=0.5, linewidth=0.1)
            # ax[1,1].plot(y_t0, z_grad_x, marker='o', markersize=0.5, linewidth=0.1)
            # ax[1,2].plot(z_t0, z_grad_x, marker='o', markersize=0.5, linewidth=0.1)

            # ax[2,0].plot(x_t0, z_grad_y, marker='o', markersize=0.5, linewidth=0.1)
            # ax[2,1].plot(y_t0, z_grad_y, marker='o', markersize=0.5, linewidth=0.1)
            # ax[2,2].plot(z_t0, z_grad_y, marker='o', markersize=0.5, linewidth=0.1)

            # ax[0,0].set_title('y_t0 vs x_t0')
            # ax[0,1].set_title('z_t0 vs x_t0')
            # ax[0,2].set_title('z_t0 vs y_t0')
            # ax[1,0].set_title('z_grad_x vs x_t0')
            # ax[1,1].set_title('z_grad_x vs y_t0')
            # ax[1,2].set_title('z_grad_x vs z_t0')
            # ax[2,0].set_title('z_grad_y vs x_t0')
            # ax[2,1].set_title('z_grad_y vs y_t0')
            # ax[2,2].set_title('z_grad_y vs z_t0')

            # plt.show()

            # f, ax = plt.subplots(3,3, figsize=(12,12))
            # ax[0,0].plot(x_t0_2, y_t0_2, marker='o', markersize=0.5, linewidth=0.1)
            # ax[0,1].plot(x_t0_2, z_t0_2, marker='o', markersize=0.5, linewidth=0.1)
            # ax[0,2].plot(y_t0_2, z_t0_2, marker='o', markersize=0.5, linewidth=0.1)

            # ax[1,0].plot(x_t0_2, z_grad_x_2, marker='o', markersize=0.5, linewidth=0.1)
            # ax[1,1].plot(y_t0_2, z_grad_x_2, marker='o', markersize=0.5, linewidth=0.1)
            # ax[1,2].plot(z_t0_2, z_grad_x_2, marker='o', markersize=0.5, linewidth=0.1)

            # ax[2,0].plot(x_t0_2, z_grad_y_2, marker='o', markersize=0.5, linewidth=0.1)
            # ax[2,1].plot(y_t0_2, z_grad_y_2, marker='o', markersize=0.5, linewidth=0.1)
            # ax[2,2].plot(z_t0_2, z_grad_y_2, marker='o', markersize=0.5, linewidth=0.1)

            # ax[0,0].set_title('y_t0_2 vs x_t0_2')
            # ax[0,1].set_title('z_t0_2 vs x_t0_2')
            # ax[0,2].set_title('z_t0_2 vs y_t0_2')
            # ax[1,0].set_title('z_grad_x_2 vs x_t0_2')
            # ax[1,1].set_title('z_grad_x_2 vs y_t0_2')
            # ax[1,2].set_title('z_grad_x_2 vs z_t0_2')
            # ax[2,0].set_title('z_grad_y_2 vs x_t0_2')
            # ax[2,1].set_title('z_grad_y_2 vs y_t0_2')
            # ax[2,2].set_title('z_grad_y_2 vs z_t0_2')

            # plt.show()


            # '''--------------- SYNVAR Contour Plots ---------------'''
            # print('\n')
            # print('******************************************************')
            # print('Plotting SYNVAR Values:')
            # f, ax = plt.subplots(1,2, figsize=(9,4), sharex=True, sharey=True)
            # ax[0].tripcolor(x_t0,y_t0,z_t0,20,cmap=cm.jet) # Use to use this: (x,y,z,20)
            # tcf = ax[1].tricontourf(x_t0,y_t0,z_t0,20, cmap=cm.jet) # 20 contour levels is good quality
            # f.colorbar(tcf)

            # ax[0].plot(x_t0,y_t0, 'ko ', markersize=1) # Used to use this: (x,y, 'ko ', markersize=1)
            # ax[0].set_xlabel('Longitude'); ax[0].set_ylabel('Latitude')
            # ax[1].plot(x_t0,y_t0, 'ko ', markersize=1)
            # ax[1].set_xlabel('Longitude'); ax[1].set_ylabel('Latitude')

            # plt.suptitle('Contour Plots: '+SYNVAR+' ('+datetime_to_plot_str+')')

            # plt.savefig(SYNVAR+'_contour_plots.png')
            # plt.show()


            # '''--------------- SYNVAR Gradient Contour Plots ---------------'''
            # print('\n')
            # print('******************************************************')
            # print('Plotting SYNVAR X and Y Gradients:')
            # f, ax = plt.subplots(1,2, figsize=(9,4), sharex=True, sharey=True)
            # ax[0].tricontourf(x_t0,y_t0,z_grad_x,20,cmap=cm.jet) # Use to use this: (x,y,z,20)
            # tcf = ax[1].tricontourf(x_t0_2,y_t0_2,z_grad_y_2,20, cmap=cm.jet) # 20 contour levels is good quality
            # f.colorbar(tcf)

            # ax[0].plot(x_t0,y_t0, 'ko ', markersize=1)
            # ax[0].set_xlabel('Longitude'); ax[0].set_ylabel('Latitude')
            # # title_str = SYNVAR+' X Gradient'
            # # print('title_str:', title_str)
            # ax[0].set_title(SYNVAR+' X Gradient')

            # ax[1].plot(x_t0_2,y_t0_2, 'ko ', markersize=1)
            # ax[1].set_xlabel('Longitude'); ax[1].set_ylabel('Latitude')
            # # title_str = SYNVAR+' Y Gradient'
            # ax[1].set_title(SYNVAR+' Y Gradient')

            # plt.suptitle('Contour Plots: '+SYNVAR+' X and Y Gradients'+' ('+datetime_to_plot_str+')')
            # plt.savefig(SYNVAR+'_contour_plots_gradient.png')
            # plt.show()


            # NOTE: The code below creates a numpy array with data grouped by time.
            # xt = SYNVAR_3d[j,:,0]
            # yt = SYNVAR_3d[j,:,1]
            # zt = SYNVAR_3d[j,:,2]
            # SYNVAR_3d[time, row, column] where column 0 = lon, 1 = lat, 2 = synvar (H500, PMSL, etc)
            # SYNVAR_3d[time][row][column]
            # SYNVAR_datetimes is used in the loop below to iterate through the layers of SYNVAR_3d
            SYNVAR_all_datetimes = df.index.tolist()
            # print('SYNVAR_all_datetimes:\n', SYNVAR_all_datetimes)
            SYNVAR_datetimes = df.index.unique()
            print('SYNVAR_datetimes:\n', SYNVAR_datetimes)
            SYNVAR_3d = np.array(list(df.groupby('time').apply(pd.DataFrame.to_numpy))) # For some reason to_numpy can be called without () and it works. Why is this allowed?
            print('-------------------------------------------')
            print('SYNVAR_3d[0][10][2]:\n', SYNVAR_3d[0][10][2])
            print('SYNVAR_3d[0,:,:]:\n', SYNVAR_3d[0,:,:])
            print('SYNVAR_3d:\n', SYNVAR_3d)
            SYNVAR_3d_shape = np.shape(SYNVAR_3d)
            print('SYNVAR_3d shape:\n', SYNVAR_3d_shape)
            m,n,r = SYNVAR_3d.shape

            num_datetimes = SYNVAR_3d_shape[0]  # Used to specify number of layers in Z and Zi


            '''--------------- SYNVAR surf plot w/ evenly spaced grid - not working ---------------'''
            # Interpolating an irregular grid onto a regular grid for
            # making contour plots
            npts = 200
            N = 200

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

            ''' --- SYNVAR Surface plot --- '''
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

            # The below plotting functions result in one plotting window
            # with three 3D surface plots:
            print('***************************** Plotting SYNVAR, X and Y Gradients...')
            fig = plt.figure(figsize=(12,3))
            ax = fig.add_subplot(131, projection='3d')
            ax.set_zlim(Z_min, Z_max)
            surf = ax.plot_surface(X,Y,Z, cmap=cm.jet, vmin=Z_min, vmax=Z_max, linewidth=0, antialiased=False)
            ax.set_zlim(np.nanmin(Z_flat), np.nanmax(Z_flat))
            ax.zaxis.set_major_locator(LinearLocator(6))
            ax.zaxis.set_major_formatter(FormatStrFormatter('%d'))
            plt.title(SYNVAR)

            ax = fig.add_subplot(132, projection='3d')
            ax.set_zlim(Z_grad_x_min, Z_grad_x_max)
            surf = ax.plot_surface(X,Y,Z_grad_x, cmap=cm.jet, vmin=Z_grad_x_min, vmax=Z_grad_x_max, linewidth=0, antialiased=False)
            ax.set_zlim(np.nanmin(Z_grad_x_flat), np.nanmax(Z_grad_x_flat))
            ax.zaxis.set_major_locator(LinearLocator(6))
            ax.zaxis.set_major_formatter(FormatStrFormatter('%d'))
            plt.title(SYNVAR+' X Gradient')

            ax = fig.add_subplot(133, projection='3d')
            ax.set_zlim(Z_grad_y_min, Z_grad_y_max)
            surf = ax.plot_surface(X,Y,Z_grad_y, cmap=cm.jet, vmin=Z_grad_y_min, vmax=Z_grad_y_max, linewidth=0, antialiased=False)
            ax.set_zlim(np.nanmin(Z_grad_y_flat), np.nanmax(Z_grad_y_flat))
            ax.zaxis.set_major_locator(LinearLocator(6))
            ax.zaxis.set_major_formatter(FormatStrFormatter('%d'))
            plt.title(SYNVAR+' Y Gradient')

            ax.set_xlabel('Longitude')
            ax.set_ylabel('Latitude')
            ax.set_zlabel(SYNVAR)
            # Color bar which maps values to colors:
            fig.colorbar(surf, shrink=0.5, aspect=5)
            plt.suptitle(SYNVAR+' Interpolated X and Y Gradients'+' ('+datetime_to_plot_str+')')
            plt.savefig(SYNVAR+'_surface_plots_x_and_y_gradients.png')
            plt.show()


            '''--------------- Example 3D Animation (UNCOMMENT TO RUN) ---------------'''
            # def update_plot(frame_number, zarray, plot):
            #     plot[0].remove()
            #     plot[0] = ax.plot_surface(x, y, zarray[:,:,frame_number], cmap="magma")

            # fig = plt.figure()
            # ax = fig.add_subplot(111, projection='3d')

            # N = 100
            # nmax = 20
            # x = np.linspace(-4,4,N+1)           # [-4, -3.428, -2.857, ..., 2.857, 3.428, 4]
            # x, y = np.meshgrid(x, x)            # Make x list into a mesh
            # print('x:\n', x)
            # print('y:\n', y)
            # print('(x,y):\n', (x,y))
            # print('x shape:\n', np.shape(x))
            # print('y shape:\n', np.shape(y))
            # print('(x,y) shape:\n', np.shape((x,y)))
            # zarray = np.zeros((N+1, N+1, nmax)) # Z values set to zero, filled below
            # print('zarray shape:\n', np.shape(zarray))

            # f = lambda x,y,sig : 1/np.sqrt(sig)*np.exp(-(x**2+y**2)/sig**2)

            # for i in range(nmax):
            #     zarray[:,:,i] = f(x,y,1.5+np.sin(i*2*np.pi/nmax))

            # plot = [ax.plot_surface(x, y, zarray[:,:,0], color='0.75', rstride=1, cstride=1)]
            # ax.set_zlim(0,1.5)
            # animate = Animation.FuncAnimation(fig, update_plot, nmax, interval=40, fargs=(zarray, plot))
            # plt.show()
            '''--------------- ^ EXAMPLE ANIMATION: DO NOT DELETE. UNCOMMENT TO RUN ^ ---------------'''


            '''--------------- Animation: SYNVAR and SYNVAR gradient ---------------'''
            print('******************************** Animating each synoptic variable for each 3-hour period...')
            def update_plot(frame_number, Zi_grad_x, plot):
                plot[0].remove()
                plot[0] = ax.plot_surface(Xi, Yi, Zi_grad_x[:,:,frame_number], cmap="magma")

            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')

            N = 200       # x and y grid number of points. 100 x 100 = 10,000 grid points
            nmax = num_datetimes     # number of z matrices (e.g. frames or unique datetime objects)

            xi = np.linspace(lon_min, lon_max, N)       # -126, -115
            yi = np.linspace(lat_min, lat_max, N)       # 40, 51
            Xi, Yi = np.meshgrid(xi, yi)                # Xi, Yi meshgrid
            Xi_Yi = np.c_[Xi,Yi]
            print('(Xi,Yi):\n', (Xi,Yi))
            print('Xi_Yi:\n', Xi_Yi)


            ''' Step through list of variables in df and display each.
                Also calculating gradients for H500 and PMSL and creating
                dataframe to house these variables '''

            Zi = np.zeros((N, N, nmax))                 # zi values set to zero, filled below
            Zi_grad_unknown = np.zeros((N, N, nmax))          # zi values set to zero, filled below
            Zi_grad_x = np.zeros((N, N, nmax))          # zi values set to zero, filled below
            Zi_grad_y = np.zeros((N, N, nmax))          # zi values set to zero, filled below

            # iterate through every x,y grid, differs for every datetime stamp

            # Setting up arrays to fill with gradient data calculated inside loop
            # dates, rows, cols  = np.shape(SYNVAR_3d)
            # zt_grad_x_arr = np.zeros((dates*rows,1))
            # zt_grad_y_2_arr = np.zeros((dates*rows,1))

            # Empty lists for X gradient:
            xt_all = []
            yt_all = []
            zt_all = []
            zt_grad_x_all = []
            # Empty lists for Y gradient:
            xt2_all = []
            yt2_all = []
            zt2_all = []
            zt_grad_y_all = []


            z_min_list = []; z_max_list = []
            z_grad_x_min_list = []; z_grad_x_max_list = []; z_grad_y_min_list = []; z_grad_y_max_list = []
            layers, row, cols = np.shape(SYNVAR_3d)
            layer_num_list = list(range(0,layers))
            for j, dt in zip(layer_num_list, SYNVAR_datetimes): # Step through Zi-layers
                # This loop is inside of the SYNVAR loop above, meaning
                # this loop iterates through all layers of one specific
                # synoptic variable, such as H500, at a time.
                print('*********************** Analyzing datetime :\n', dt)
                datetime_to_plot_str = dt.strftime('%b %d, %Y')
                print('SYNVAR_3d datetime_to_plot_str:', datetime_to_plot_str)
                # SYNVAR_3d[datetime:row:column]
                # jth time point
                # all rows within jth timepoint
                # 0th, 1st, or 2nd
                xt = SYNVAR_3d[j,:,0]
                yt = SYNVAR_3d[j,:,1]
                zt = SYNVAR_3d[j,:,2]
                print('xt:\n', xt)
                print('yt:\n', yt)
                print('zt:\n', zt)

                ''' ******************************** Calculating Gradients On Original NARR Data ******************************** '''

                # Used to calculate Y gradient. Reshaping the data so that
                # every data point's neighbor is higher or lower in latitude
                xt2 = np.array(xt).reshape((40,40)).T.ravel()
                yt2 = np.array(yt).reshape((40,40)).T.ravel()
                zt2 = np.array(zt).reshape((40,40)).T.ravel()
                print('Shape xt2:\n', np.shape(xt2))
                print('Shape yt2:\n', np.shape(yt2))
                print('Shape zt2:\n', np.shape(zt2))

                # Checking to make sure the data is reordered:
                xyz = [[x,y,z] for x,y,z in zip(xt2, yt2, zt2)]
                dft2 = pd.DataFrame(xyz, columns=['lon','lat', SYNVAR])
                print('dft2:\n', dft2)

                # Calculating gradients. df_all_synvar currently contains H500, CAPE, PMSL.
                # print('x_t0:\n', x_t0)
                # print('y_t0:\n', y_t0)
                # print('z_t0:\n', z_t0)

                # Calculating X gradient. Y Gradient (zt_grad_y) is incorrect
                zt_grad_x = np.gradient(zt, xt)
                zt_grad_y = np.gradient(zt, yt)
                print('zt_grad_x:\n', zt_grad_x[0:30])
                print('zt_grad_y:\n', zt_grad_y[0:30])

                # Calculating Y gradient. X Gradient (zt_grad_x_2) is incorrect
                zt_grad_x_2 = np.gradient(zt2, xt2)
                zt_grad_y_2 = np.gradient(zt2, yt2)
                print('zt_grad_x_2:\n', zt_grad_x_2[0:30])
                print('zt_grad_y_2:\n', zt_grad_y_2[0:30])

                if j == layer_num_list[-1]:
                    # If the loop is on its last layer (last date) then
                    # create and save some contour plots.

                    #******************************** Synvar Contour Plots:
                    print('\n')
                    print('******************************************************')
                    print('Plotting SYNVAR Values:')
                    f, ax = plt.subplots(1,2, figsize=(9,4), sharex=True, sharey=True)
                    ax[0].tripcolor(xt,yt,zt,20,cmap=cm.jet) # Use to use this: (x,y,z,20)
                    tcf = ax[1].tricontourf(xt,yt,zt,20, cmap=cm.jet) # 20 contour levels is good quality
                    f.colorbar(tcf)

                    ax[0].plot(xt,yt, 'ko ', markersize=1) # Used to use this: (x,y, 'ko ', markersize=1)
                    ax[0].set_xlabel('Longitude'); ax[0].set_ylabel('Latitude')
                    ax[1].plot(xt,yt, 'ko ', markersize=1)
                    ax[1].set_xlabel('Longitude'); ax[1].set_ylabel('Latitude')

                    plt.suptitle('Contour Plots: '+SYNVAR+' ('+datetime_to_plot_str+')')

                    # plt.savefig(SYNVAR+'_contour_plots.png')
                    plt.show()


                    #****************************** Synvar Gradient Contour Plots:
                    print('\n')
                    print('******************************************************')
                    print('Plotting SYNVAR X and Y Gradients:')
                    f, ax = plt.subplots(1,2, figsize=(9,4), sharex=True, sharey=True)
                    ax[0].tricontourf(xt,yt,zt_grad_x,20,cmap=cm.jet) # Use to use this: (x,y,z,20)
                    tcf = ax[1].tricontourf(xt2,yt2,zt_grad_y_2,20, cmap=cm.jet) # 20 contour levels is good quality
                    f.colorbar(tcf)

                    ax[0].plot(xt,yt, 'ko ', markersize=1)
                    ax[0].set_xlabel('Longitude'); ax[0].set_ylabel('Latitude')
                    # title_str = SYNVAR+' X Gradient'
                    # print('title_str:', title_str)
                    ax[0].set_title(SYNVAR+' X Gradient')

                    ax[1].plot(xt2,yt2, 'ko ', markersize=1)
                    ax[1].set_xlabel('Longitude'); ax[1].set_ylabel('Latitude')
                    # title_str = SYNVAR+' Y Gradient'
                    ax[1].set_title(SYNVAR+' Y Gradient')

                    plt.suptitle('Contour Plots: '+SYNVAR+' X and Y Gradients'+' ('+datetime_to_plot_str+')')
                    # plt.savefig(SYNVAR+'_contour_plots_gradient.png')
                    plt.show()

                zt_grad_x_shape = np.shape(zt_grad_x)
                zt_grad_y_2_shape = np.shape(zt_grad_y_2)
                print('zt_grad_x_shape:\n', zt_grad_x_shape)
                print('zt_grad_y_2_shape:\n', zt_grad_y_2_shape)

                # Writing the gradients and their coordinates for original NARR grid to a list
                xt_all.extend(xt)
                yt_all.extend(yt)
                zt_all.extend(zt)
                zt_grad_x_all.extend(zt_grad_x)

                xt2_all.extend(xt2)
                yt2_all.extend(yt2)
                zt2_all.extend(zt2)
                zt_grad_y_all.extend(zt_grad_y_2)


                xt_yt = np.c_[xt,yt]

                # --- Array shapes:
                # (xt, yt)  = 1122 x 2
                # zt        = 1122
                # (Xi, Yi)  = 100 x 100 (e.g. N x N)

                ''' Calculating gridded interpolated Zi and its gradients.
                    CAPE gradients will also be calculated but won't be
                    included in the final df. '''
                Zi[:,:,j] = griddata((xt,yt), zt, (Xi,Yi), method='linear')
                Zi_grad_x[:,:,j] = np.gradient(Zi[:,:,j], axis=1)
                Zi_grad_y[:,:,j] = np.gradient(Zi[:,:,j], axis=0)

                # Min and max lists in the event that taking
                # np.nanmin(Zi) doesn't work, use np.nanmin(z_min_list)
                z_min_list.append(np.nanmin(Zi[:,:,j]))
                z_max_list.append(np.nanmax(Zi[:,:,j]))
                z_grad_x_min_list.append(np.nanmin(Zi_grad_x[:,:,j]))
                z_grad_x_max_list.append(np.nanmax(Zi_grad_x[:,:,j]))
                z_grad_y_min_list.append(np.nanmin(Zi_grad_y[:,:,j]))
                z_grad_y_max_list.append(np.nanmax(Zi_grad_y[:,:,j]))

            print('zt_grad_x_all[0:100]:\n', zt_grad_x_all[0:100])
            print('zt_grad_y_all[0:100]:\n', zt_grad_y_all[0:100])

            # Convert X gradient related lists to numpy arrays:
            xt_all_arr = np.array(xt_all)
            yt_all_arr = np.array(yt_all)
            zt_all_arr = np.array(zt_all)
            zt_grad_x_all_arr = np.array(zt_grad_x_all)
            
            # Convert Y gradient related lists to numpy arrays:
            xt2_all_arr = np.array(xt2_all)
            yt2_all_arr = np.array(yt2_all)
            zt2_all_arr = np.array(zt2_all)
            zt_grad_y_all_arr = np.array(zt_grad_y_all)

            t_all_arr = np.array(SYNVAR_all_datetimes)
            print('shape of t_all_arr:\n', np.shape(t_all_arr)) # Should be correct size
            print('shape of xt_all:\n', np.shape(xt_all))
            print('shape of yt_all:\n', np.shape(yt_all))
            print('shape of zt_all:\n', np.shape(zt_all))
            print('shape of zt_grad_x_all:\n', np.shape(zt_grad_x_all_arr)) # Should be correct size
            print('shape of zt_grad_y_all:\n', np.shape(zt_grad_y_all_arr)) # Should be correct size

            # NEED TO APPEND ALL xt, yt, zt data to their own lists
            arr_grad_x = np.column_stack((xt_all, yt_all, t_all_arr, zt_all,  zt_grad_x_all_arr)) # Contains H500 and H500 Grad X
            arr_grad_y = np.column_stack((xt2_all, yt2_all, t_all_arr,  zt_grad_y_all_arr)) # Contains H500 Grad Y with different x,y pair ordering
            cols_x = ['lon','lat','time',SYNVAR,SYNVAR+' Grad X']
            cols_y = ['lon','lat','time',SYNVAR+' Grad Y']
            # Make dataframes from data
            df_x = pd.DataFrame(data=arr_grad_x, columns=cols_x)
            df_y = pd.DataFrame(data=arr_grad_y, columns=cols_y)
            print('df_x:\n', df_x.head())
            print('df_y:\n', df_y.head())
            # Set indices before using indices to merge
            df_x.set_index(['lon','lat','time'], inplace=True)
            df_y.set_index(['lon','lat','time'], inplace=True)
            print('df_x after setting index to lon lat time:\n', df_x.head())
            print('df_y after setting index to lon lat time:\n', df_y.head())
            print('verifying df_x["time"] index is in datetime format:\n', df_x.index.levels[2])

           
            # Merge df_x and df_y:
            # Left and right indices are set to True to merge on both.
            # how = 'Left' to retain the order of the left indices (df_x indices) while
            # the right indices (df_y indices) are sorted according to left indices.
            # This is done because df_y's order may be different after reordering xt, yt
            # and zt to create xt2, yt2 and zt2 to calculate the Y gradient (zt_grad_y_2)
            # in the loop above.
            df_synvar_and_gradients = df_x.merge(df_y, how='left', left_index=True, right_index=True)
            print('df_synvar_and_gradients for '+SYNVAR+':\n', df_synvar_and_gradients)

            if i == 0:
                df_narr = df_narr.append(df_synvar_and_gradients)
                print('************* df_synvar_and_gradients for '+SYNVAR+' append:\n', df_synvar_and_gradients)
            else:
                df_narr = pd.concat((df_narr, df_synvar_and_gradients), axis=1, join='inner')
                print('************* df_synvar_and_gradients for '+SYNVAR+' concatenation:\n', df_synvar_and_gradients)

            print('df_narr:\n', df_narr)



            ''' The code below is for the animation using NARR data interpolated to a regular grid.
                The data isn't for ML training purposes. '''

            # Array shapes:
            print('(xt,yt) shape:', np.shape((xt,yt)))
            print('xt_yt shape:', np.shape(xt_yt))
            print('zt shape:', np.shape(zt))

            # Formatting interpolated grid (Xi,Yi,Zi,Zi_grad_x,Zi_grad_y) and time into df:
            m,n,r = np.shape(Zi)
            Xi_shape = np.shape(Xi)
            Yi_shape = np.shape(Yi)
            t_unique = np.unique(t)
            num_unique_times = len(t_unique)
            t_unique_exp = np.repeat(t_unique, m*n)
            print('Xi shape:', np.shape(Xi))
            print('Yi shape:', np.shape(Yi))
            print('Zi shape:', np.shape(Zi))
            print('Zi[:,:,10] shape:\n', np.shape(Zi[:,:,10]))
            print('t_unique_exp shape:', np.shape(t_unique_exp))

            Xi_tiled = np.tile(Xi, num_unique_times)
            Xi_tiled_flat = np.ravel(Xi_tiled)
            Yi_tiled = np.tile(Yi, num_unique_times)
            Yi_tiled_flat = np.ravel(Yi_tiled)
            Zi_flat = np.ravel(Zi)
            Zi_grad_x_flat = np.ravel(Zi_grad_x)
            Zi_grad_y_flat = np.ravel(Zi_grad_y)

            print('Xi_tiled_flat:', np.shape(Xi_tiled_flat))
            print('Yi_tiled_flat:', np.shape(Yi_tiled_flat))
            print('Zi_flat:', np.shape(Zi_flat))
            print('Zi_grad_x_flat:', np.shape(Zi_grad_x_flat))
            print('Zi_grad_y_flat:', np.shape(Zi_grad_y_flat))

            '''
            Building each synoptic variable array into a dataframe and concatenating the dataframes
            on 'inner' indices. Catches any disagreements with index values that array column stack won't.
            '''

            #SYNABBR_finallist = ['H500 Grad X', 'H500 Grad Y', 'PMSL Grad X', 'PMSL Grad Y']
            # Gridded interpolated data is put into array, then df_all_grid_interp
            arr = np.column_stack((Xi_tiled_flat, Yi_tiled_flat, t_unique_exp, Zi_flat, Zi_grad_x_flat, Zi_grad_y_flat))
            print('arr shape:', np.shape(arr))
            # df_synvar_grid_interp contains Zi and gradients for a specific synvar (e.g. H500):
            cols = ['lon','lat','time',SYNVAR,SYNVAR+' Grad X',SYNVAR+' Grad Y']
            df_synvar_grid_interp = pd.DataFrame(data=arr, columns=cols)
            # Set index to allow inner joining when concatenating latest df_synvar_grid_interp to df_all_synvar_grid_interp:
            df_synvar_grid_interp.set_index(['lon','lat','time'], inplace=True)

            # Sort index by time
            #df_synvar_grid_interp.sort_index(level=2, inplace=True)
            print('df_synvar_grid_interp sorted by time:\n', df_synvar_grid_interp)

            '''--------------- IF THERE'S A WAY TO REWRITE THIS SECTION WITHOUT LOOP, IT SHOULD BE DONE: ---------------'''
            # This loop may not be necessary. As an alternative, appending all
            # df_synvar_grid_interp dataframes to a list called 'df_synvar_grid_interp_list',
            # then when i == len(SYNABBR_shortlist)-1,
            # do the following:
            # df_all_synvar_grid_interp = pd.concat([df_all_synvar_grid_interp, df_synvar_grid_interp), axis=1, join='inner')
            if i == 0:
                # df_all_synvar_grid_interp contains all grid interpolated synoptic variables
                df_all_synvar_grid_interp = df_all_synvar_grid_interp.append(df_synvar_grid_interp)
            else:
                df_all_synvar_grid_interp = pd.concat((df_all_synvar_grid_interp, df_synvar_grid_interp), axis=1, join='inner')
            ''' ########################################### '''

            print('df_all_synvar_grid_interp:\n', df_all_synvar_grid_interp)
            '''---------------------------------------------------------------------------------------------------------'''

            # # Used to set 3D animation vertical range:
            # Z_min = int(np.nanmin(Zi[:,:,:]))
            # Z_max = int(np.nanmax(Zi[:,:,:]))
            # Z_grad_x_min = int(np.nanmin(Zi_grad_x[:,:,:]))
            # Z_grad_x_max = int(np.nanmax(Zi_grad_x[:,:,:]))
            # Z_grad_y_min = int(np.nanmin(Zi_grad_y[:,:,:]))
            # Z_grad_y_max = int(np.nanmax(Zi_grad_y[:,:,:]))

            # #plot = [ax.plot_surface(x, y, zarray[:,:,0], color='0.75', rstride=1, cstride=1)]
            # print('*********************** Plotting surface of Xi, Yi, Zi_grad_x')
            # plot = [ax.plot_surface(Xi, Yi, Zi_grad_x[:,:,0], vmin=Z_min, vmax=Z_max, linewidth=0, antialiased=True, color='0.75', rstride=1, cstride=1)]

            # ax.set_zlim(Z_min, Z_max)
            # ax.zaxis.set_major_locator(LinearLocator(6))
            # ax.zaxis.set_major_formatter(FormatStrFormatter('%d'))
            # ax.set_xlabel('Longitude')
            # ax.set_ylabel('Latitude')
            # ax.set_zlabel(SYNVAR+': X gradient')
            # # Color bar which maps values to colors:
            # #fig.colorbar(surf, shrink=0.5, aspect=5)
            # # NEED THIS BUT NOT WORKING: fig.colorbar(plot, shrink=0.5, aspect=5)

            ''' --- Animation parameters --- '''
            # #---Set animation figure parameters:
            # fig.set_size_inches(2, 2, True) # width (inches), height (inches), forward = True or False
            # dpi = 300
            # # Animation.FuncAnimation(figure reference, function that updates plot, time interval between frames in milliseconds, (data, plot reference))
            # animate = Animation.FuncAnimation(fig, update_plot, nmax, interval=200, fargs=(Zi_grad_x, plot))

            # #--- Save Animation as mp4:
            # plt.rcParams['animation.ffmpeg_path'] = '/usr/bin/ffmpeg'
            # writer = Animation.FFMpegWriter(fps=3, codec='libx264', bitrate=1800)
            # initial_date = df_all_synvar_grid_interp.index.get_level_values(2)[0].strftime('%b_%d_%Y')
            # print('initial date:', initial_date)
            # final_date   = df_all_synvar_grid_interp.index.get_level_values(2)[-1].strftime('%b_%d_%Y')
            # print('final date:', final_date)

            # filename = SYNVAR + '_' + initial_date + '_to_' + final_date + '.mp4' #'_Jan_1_to_14_1979.mp4'
            # print('filename:', filename)
            # animate.save(filename=filename, writer=writer, dpi=dpi)

            # plt.show() # Animation is saved by the time it shows up.
            ''' --------------------------- '''

        # Drop non-gradient data for H500 and PMSL, and CAPE gradients:
        # Pressure gradients, not the pressures themselves, drive surface level winds and contribute to fire weather.
        # CAPE levels, not gradients, drive thunderstorm activity associated with fire ignition.
        df_narr.drop(columns=['H500', 'PMSL', 'CAPE Grad X', 'CAPE Grad Y'], inplace=True)
        df_all_synvar_grid_interp.drop(columns=['H500', 'PMSL', 'CAPE Grad X', 'CAPE Grad Y'], inplace=True)
        # Verify columns have been removed:
        print('df_narr:\n', df_narr.head())
        print('df_all_synvar_grid_interp:\n', df_all_synvar_grid_interp.head())


        # Pickle out:
        print('Exporting to Pickle...')
        print('df_narr.columns:\n', df_narr.columns)
        print('df_all_synvar_grid_interp.columns:\n', df_all_synvar_grid_interp.columns)

        # If the pickle number is single digit, add a prefix 0, otherwise use i for prefix:
        if pkl_counter < export_interval:
            pickle_name_narr = '0' + str(pkl_counter) + '_df_narr.pkl'
            pickle_name_all_synvar_grid_interp = '0' + str(pkl_counter) + '_df_all_synvar_grid_interp.pkl'
        else:
            pickle_name_narr = str(pkl_counter) + '_df_narr.pkl'
            pickle_name_all_synvar_grid_interp = str(pkl_counter) + '_df_all_synvar_grid_interp.pkl'

        df_narr_pkl = df_narr.to_pickle(NARR_pkl_out_dir + pickle_name_narr)
        df_all_synvar_grid_interp_pkl = df_all_synvar_grid_interp.to_pickle(NARR_pkl_out_dir + pickle_name_all_synvar_grid_interp)

        ######## DON'T EXPORT THE FINAL DF TO CSV BECAUSE ONLY THE PICKLE VERSIONS ARE USED IN merge_NARR_gridMET().
        # # Export to csv:
        # print('Exporting to CSV... (This could take a minute) ******************************')
        # df_narr.to_csv(export_csv_dir + 'df_narr.csv', index=True, header=True)
        # df_all_synvar_grid_interp.to_csv(export_csv_dir + 'df_all_synvar_grid_interp.csv', index=True, header=True)

        # NOTE: Current sample size for Jan 1-14 from SOMPY's point of view is 98 unique maps

    return

# ----------------------------------------
# import_NARR_lambert_csv(-125,-116,41,50, 2, '/home/dp/Documents/FWP/NARR/csv/','/home/dp/Documents/FWP/NARR/pickle/')
# ----------------------------------------