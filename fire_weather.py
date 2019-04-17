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


def import_gridMET_csv():
    # Imports all csv files ending in ERC and concatenates them together into
    # df_erc, sets index to lon, lat, time, and converts days since Jan 1, 1900 
    # to date format, 

    # Path to gridMET outcome variables (ERC, BI, etc) for supervision:
    path_all_gridMET_csv = '/home/dp/Documents/FWP/gridMET/csv/'

    # gridMET csv file names must end with ERC.csv:
    all_gridMET_files = glob.glob(os.path.join(path_all_gridMET_csv, '*ERC.csv'))
    # print('all_gridMET_files:\n', all_gridMET_files)

    nfiles = len(all_gridMET_files)
    print('Building ERC dataframe, could take a minute. Relax.')
    df_gen = (pd.read_csv(file, header='infer', index_col=['lon','lat','time']) for file in all_gridMET_files)
    df_erc = pd.concat(df_gen, axis=0)
    # print('df: gridMET ERC:\n', df_erc)

    df_erc.reset_index(inplace=True)

    # You can add a datetime object to an entire dataframe column of timedelta objects,
    # as done below. This is done because df_erc time column is in days since Jan 1, 1900,
    # and needs to be converted to a relevant datetime object.
    df_erc['time'] = pd.to_timedelta(df_erc['time'], unit='D') + datetime(1900,1,1,0,0)
    df_erc.set_index(['lon','lat','time'], inplace=True)
    df_erc.sort_index(level=2, inplace=True)
    print('df_erc with timedelta:\n', df_erc)

    df_erc_pkl = df_erc.to_pickle('/home/dp/Documents/FWP/gridMET/pickle/df_erc.pkl')

    return

''' ------ Import all gridMET CSVs ------ '''
# import_gridMET_csv()
''' ------------------------------------- '''


def merge_NARR_gridMET_csv(start_date, end_date):#lon_min, lon_max, lat_min, lat_max):
    # Imports df_erc and df_all_synvar pickle files, converts to dataframes,
    # sets index of each to time column, aggregates NARR data from 3hrs
    # to 24hrs to match 24 hour ERC data.
    # For day D of ERC data, the temporal aggregation is the 24 hour period from
    # 2100 UTC of day D-1 to 1800 UTC of day D.
    # The merge is then done by interpolating the gridMET grid (4km) to
    # the NARR grid (32km) using 5 gridMET neighbors for each NARR grid
    # point. The ERC values at those 5 points are interpolated linearly
    # to the location of the NARR grid point, resulting in array 'zi',
    # which is then written to the NARR dataframe as a new column 'ERC'.
    # It's plotted and then pickled.

    df_gridMET = pd.read_pickle('/home/dp/Documents/FWP/gridMET/pickle/df_erc.pkl')
    print('df_gridMET:\n', df_gridMET)
    df_NARR = pd.read_pickle('/home/dp/Documents/FWP/NARR/pickle/df_all_synvar.pkl')
    print('df_NARR:\n', df_NARR)
    time_window_hrs = 24 # Average NARR data over 24 hr period to match gridMET

    # gridMET converted to dates. Would add 2100 UTC to gridMET but not necessary
    # because the analysis will be done on daily, not sub-daily, data.
    df_gridMET.reset_index(inplace=True)
    df_gridMET['time'] = df_gridMET['time'].dt.date # Converting time to date
    print('df_gridMET reduced to dates **************************************:\n', df_gridMET)

    # NARR times are in UTC
    df_NARR.reset_index(inplace=True)
    print('df_NARR reset index:\n', df_NARR)
    print('df_NARR index is lon:\n', df_NARR)

    print('df_NARR lon lat grouping mean:\n', df_NARR.groupby(['lon','lat']).mean())

    df_NARR.set_index('lon', inplace=True)
    print('df_NARR.loc[-131.602]:\n', df_NARR.loc[-131.602])

    return

    df_NARR['time'] = df_NARR['time'].dt.date # Converting time to date
    print('df_NARR reduced to dates **************************************:\n', df_NARR)

    # Date range strings parsed to datetime objects
    start_dt_object = datetime.strptime(start_date, '%Y,%m,%d')
    end_dt_object = datetime.strptime(end_date, '%Y,%m,%d')
    print('start_dt_object:\n', start_dt_object)
    print('end_dt_object:\n', end_dt_object)

    # Convert start and end datetimes to dates
    start_date_object = datetime.date(start_dt_object)
    end_date_object = datetime.date(end_dt_object)
    print('start_date_object:\n', start_date_object)
    print('end_date_object:\n', end_date_object)

    # Timedelta object for number of days from start_date to end_date.
    # Used to make timedelta list for making date list
    num_days = (end_date_object - start_date_object)
    print('num_days in days:\n', num_days)

    # Creating timedelta_list, then using it to make date_list
    timedelta_list = [timedelta(days=days) for days in list(range(0,1000)) if timedelta(days=days) <= num_days]
    print('timedelta_list:\n', timedelta_list)
    date_list = [td+start_date_object for td in timedelta_list]
    print('date_list:\n', date_list)
    # Removes first day. NARR average needs averages across day 1 and 2, thus gridMET data must start on day 2
    date_list = date_list[1:]


    df_NARR_ERC = pd.DataFrame([])
    for i, d in enumerate(date_list):
        # This loop goes through every date and interpolates gridMET ERC
        # values to the NARR grid using nearest neighbor interpolation.
        # df_gridMET and df_NARR matching the date are copied to
        # df_gridMET_date and df_NARR_date. Interpolation is performed
        # on these dataframes.
        # The interpolated ERC values are copied to df_NARR_date['ERC']
        # and then concatenated to df_NARR_ERC.
        df_gridMET_date = df_gridMET[df_gridMET['time'] == d]
        df_NARR_date = df_NARR[df_NARR['time'] == d]
        print('df_gridMET_date:\n', df_gridMET_date)
        print('df_NARR_date:\n', df_NARR_date)
        print('df_gridMET_date shape:\n', np.shape(df_gridMET_date.values))
        print('df_NARR_date shape:\n', np.shape(df_NARR_date.values))

        # # PLOT 1: Plotting NARR grid over gridMET grid:
        # plt.figure()
        # plt.scatter(x=df_gridMET_date.lon, y=df_gridMET_date.lat, c='white', s=8, marker='o', edgecolors='g', label='gridMET grid')
        # plt.scatter(x=df_NARR_date.lon, y=df_NARR_date.lat, c='k', s=8, marker='+', edgecolors='g', label='NARR grid')
        # plt.xlabel('Longitude, deg'); plt.ylabel('Latitude, deg')
        # plt.title('NARR grid after removing out-of-bounds points')
        # plt.legend()
        # plt.savefig('NARR_gridMET_complete_grids.png', bbox_inches='tight')
        # plt.show()

        # Changing invalid data points to values the interpolation
        # algorithm can interpolate to:
        ########################## df_gridMET_date.replace(-32767, 1.0123456789, inplace=True)

        # Clip NARR grid to min and max lon lat values in the gridMET grid
        x_min = min(df_gridMET_date['lon'].values); x_max = max(df_gridMET_date['lon'].values)
        y_min = min(df_gridMET_date['lat'].values); y_max = max(df_gridMET_date['lat'].values)
        print('x_min:\n', x_min)
        print('x_max:\n', x_max)
        print('y_min:\n', y_min)
        print('y_max:\n', y_max)

        # Select all rows that are inside the lon-lat window of the ERC dataset:
        criteria = (x_max >= df_NARR_date['lon']) & (df_NARR_date['lon'] >= x_min) & (y_max >= df_NARR_date['lat']) & (df_NARR_date['lat'] >= y_min)
        print('NARR rows before cutting out-of-bounds lon-lat points:', df_NARR_date.count())
        df_NARR_date = df_NARR_date.loc[criteria]
        print('NARR rows after cutting out-of-bounds lon-lat points:', df_NARR_date.count())
        print('df_NARR_date after removing out-of-bounds Lon-Lat points:\n', df_NARR_date)

        # # PLOT 2: Plotting NARR grid where it overlaps with gridMET:
        # plt.figure()
        # plt.scatter(x=df_gridMET_date.lon, y=df_gridMET_date.lat, c='white', s=8, marker='o', edgecolors='g', label='gridMET grid')
        # plt.scatter(x=df_NARR_date.lon, y=df_NARR_date.lat, c='k', s=8, marker='+', edgecolors='g', label='NARR grid')
        # plt.xlabel('Longitude, deg'); plt.ylabel('Latitude, deg')
        # plt.title('NARR grid after removing out-of-bounds points')
        # plt.legend()
        # plt.savefig('NARR_gridMET_complete_grids.png', bbox_inches='tight')
        # plt.show()

        # Define x y and z for interpolation:
        x = df_gridMET_date.lon.values
        y = df_gridMET_date.lat.values
        z = df_gridMET_date.erc.values
        xi = df_NARR_date.lon.values
        yi = df_NARR_date.lat.values
        print('x:\n{}\n y:\n{}\n z:\n{}\n'.format(x, y, z))
        print('xi:\n{}\n yi:\n{}\n'.format(xi,yi))
        print('x shape:\n{}\n y shape:\n{}\n z shape:\n{}\n'.format(np.shape(x), np.shape(y), np.shape(z)))
        print('xi shape:\n{}\n yi shape:\n{}\n'.format(np.shape(xi),np.shape(yi)))
        
        # Interpolation:
        would_you_be_my_neighbor = 5
        gridMET_shape = np.shape(df_gridMET_date.values[:,0:2])
        NARR_shape = np.shape(df_NARR_date.values[:,0:2])
        print('gridMET shape:', gridMET_shape)
        print('NARR shape:', NARR_shape)
        tree = neighbors.KDTree(df_gridMET_date.values[:,0:2], leaf_size=2)
        dist, ind = tree.query(df_NARR_date.values[:,0:2], k=would_you_be_my_neighbor)
        print('indices:', ind)  # indices of 3 closest neighbors
        print('distances:', dist)  # distances to 3 closest neighbors
        print('df_NARR_date with ERC rounded to nearest int and invalid rows removed:\n', df_NARR_date)

        # Create ERC data (zi) from interpolated grid
        zi = griddata((x,y),z,(xi,yi),method='nearest')
        print('zi:\n', zi)
        print('zi shape:\n{}\n'.format(np.shape(zi)))

        # Plotting before and after interpolation of gridMET ERC to NARR grid:
        # Plots gridMET grid before and after interpolation. It uses gridMET grid,
        # so the plotting is done here rather than inside of plot_NARR_ERC() which
        # already has the gridMET ERC data within df_NARR_date_ERC.

        if d == date_list[-1]: # Only plot if on last date in list
            plt.close()
            plt.figure()
            plt.scatter(x=df_gridMET_date.lon, y=df_gridMET_date.lat, color='white', marker='o', edgecolors='g', s=df_gridMET_date.erc, label='gridMET')
            plt.scatter(x=df_gridMET_date.lon.iloc[np.ravel(ind)], y=df_gridMET_date.lat.iloc[np.ravel(ind)], color='r', marker='x', s=7, label='nearest gridMET')
            plt.scatter(x=df_NARR_date.lon, y=df_NARR_date.lat, color='k', marker='+', s=7, label='NARR')
            plt.xlabel('Longitude, deg'); plt.ylabel('Latitude, deg')
            plt.title('Nearest gridMET points using interpolated indices')
            plt.legend()
            plt.savefig('NARR_gridMET_before_interp.png', bbox_inches='tight')
            plt.show()

            plt.scatter(x=df_gridMET_date.lon, y=df_gridMET_date.lat, color='white', marker='o', edgecolors='g', s=df_gridMET_date.erc, label='gridMET')
            plt.scatter(x=df_gridMET_date.lon.iloc[np.ravel(ind)], y=df_gridMET_date.lat.iloc[np.ravel(ind)], color='r', marker='x', s=7, label='nearest gridMET')
            plt.scatter(x=xi, y=yi, color='y', edgecolors='y', alpha=0.6, marker='o', s=zi, label='interp NARR')
            plt.scatter(x=df_NARR_date.lon, y=df_NARR_date.lat, color='k', marker='+', s=7, label='NARR')
            plt.xlabel('Longitude, deg'); plt.ylabel('Latitude, deg')
            plt.title('Interpolated ERC values')
            plt.legend()
            plt.savefig('NARR_gridMET_after_interp.png', bbox_inches='tight')
            plt.show()

        # Add interpolated ERC values to df_NARR_date:
        df_NARR_date['ERC'] = zi
        print('df_NARR_date with ERC:\n', df_NARR_date)

        if i == 0: # First time through loop, append df_NARR_date to columns
            # When i = 0, all H500 files in df are processed:
            df_NARR_ERC = df_NARR_ERC.append(df_NARR_date)
            print('First df_NARR_ERC concatenation:\n', df_NARR_ERC)
        else: # Concat df_NARR_date to rows of df_NARR_ERC
            df_NARR_ERC = pd.concat((df_NARR_ERC, df_NARR_date), axis=0)
            # print('Second df_NARR_ERC concatenation:\n', df_NARR_date)
            # print('df_NARR_date.columns:\n', df_NARR_date.columns)
            print('***************************** Analyzing {} *****************************'.format(d))


    # Remove invalid values:
    # print('df_NARR rows before rounding ERC:\n', df_NARR_ERC.count())
    df_NARR_ERC = df_NARR_ERC[df_NARR_ERC['ERC'] > 0]
    # print('df_NARR_ERC rows after rounding ERC:\n', df_NARR_ERC.count())
    df_NARR_ERC = df_NARR_ERC.round({'ERC':0})
    print('df_NARR_ERC rows after rounding ERC and removing invalid values:\n', df_NARR_ERC)

    df_NARR_ERC_pkl = df_NARR_ERC.to_pickle('/home/dp/Documents/FWP/NARR/pickle/df_NARR_ERC.pkl')
    df_NARR_ERC.to_csv('/home/dp/Documents/FWP/NARR/df_NARR_ERC.csv', header=True, index=False) # Includes index columns, names all columns at top of file

    return

''' ------ Import all gridMET CSVs ------ '''
merge_NARR_gridMET_csv('1979,1,1','1979,1,13')
''' ------------------------------------- '''


def plot_NARR_ERC(ERC_date):
    # This function makes a contour plot containing tripcolour and tricontourf subplots of ERC data.

    df_NARR_ERC = pd.read_pickle('/home/dp/Documents/FWP/NARR/pickle/df_NARR_ERC.pkl')
    print('df_NARR with ERC:\n', df_NARR_ERC)

    # WARNING: These x,y,t values are for all days in df_NARR_ERC.
    # Currently, df_NARR_ERC only covers Jan 1, 1979, and this
    # means the plot below that uses these values works. It won't
    # work when there are multiple dates.
    x = df_NARR_ERC.lon.values.tolist()     # Longitude
    y = df_NARR_ERC.lat.values.tolist()     # Latitude
    t = df_NARR_ERC.time.values.tolist()    # Time
    # print('x values from df_NARR_ERC.lon:\n', x)
    # print('y values from df_NARR_ERC.lat:\n', y)
    # print('t values from df_NARR_ERC.time:\n', t)

    # Getting z values and building new dataframe with time index and ERC data.
    z = df_NARR_ERC['ERC'].values.tolist()
    d = [i for i in zip(t,x,y,z)]
    df = pd.DataFrame(data=d, columns=['time','lon','lat','ERC'])
    df.set_index('time', inplace=True)
    print('df.index:\n', df.index)
    print('df.index[10]:\n', df.index[10])

    # Convert timepoint to build contour plot from ERC data
    ERC_date = datetime.strptime(ERC_date, '%Y,%m,%d')
    ERC_date = datetime.date(ERC_date)
    print('ERC_date:', ERC_date)
    # Get the ERC data for the day specified
    df_t0 = df[(df.index == ERC_date)]
    # Split into constituents
    x_t0 = df_t0['lon'].values.tolist()
    y_t0 = df_t0['lat'].values.tolist()
    z_t0 = df_t0['ERC'].values.tolist()
    print('df_t0:\n', df_t0)
    # print('Shape x_t0:\n', np.shape(x_t0))
    # print('Shape y_t0:\n', np.shape(y_t0))
    # print('Shape z_t0:\n', np.shape(z_t0))

    plt.close()
    f, ax = plt.subplots(1,2, figsize=(9,4), sharex=True, sharey=True)

    ax[0].tripcolor(x,y,z) # Plots across all timepoints?
    ax[0].plot(x,y, 'ko ', markersize=1)
    ax[0].set_xlabel('Longitude'); ax[0].set_ylabel('Latitude')

    ax[1].tricontourf(x_t0,y_t0,z_t0, 30, cmap=cm.jet) # 20 contour levels is good quality
    ax[1].plot(x_t0,y_t0, 'ko ', markersize=1)
    ax[1].set_xlabel('Longitude'); ax[1].set_ylabel('Latitude')

    date_str = ERC_date.strftime('%b %d, %Y')
    plt.suptitle('ERC Contour Plots: '+date_str)
    
    plt.savefig('ERC_contour.png', bbox_inches='tight')
    plt.show()

    return


''' ------ Import all gridMET CSVs ------ '''
#plot_NARR_ERC('1979,1,13')
''' ------------------------------------- '''


''' --- Import NARR data from csv files and animate it --- '''
def import_NARR_csv(lon_min, lon_max, lat_min, lat_max):
    # Read in all csvs from folder:
    # Not sure what the double slashes are for: path = '..\\..\\data\\'
    # SOMPY path: path = 'C:\\Users\Dan\Downloads\SOMPY_robust_clustering-master\SOMPY_robust_clustering-master\data\\'

    # Path to NARR feature variables (H500, PMSL, CAPE, etc):
    path_all_NARR_csv = '/home/dp/Documents/FWP/NARR/csv/'

    # NARR csv file names must end with either H500.csv, CAPE.csv, or PMSL.csv:
    H500_files = glob.glob(os.path.join(path_all_NARR_csv, '*H500.csv')) # H500 file paths in a list
    CAPE_files = glob.glob(os.path.join(path_all_NARR_csv, '*CAPE.csv'))
    PMSL_files = glob.glob(os.path.join(path_all_NARR_csv, '*PMSL.csv'))
    all_NARR_files = [H500_files, CAPE_files, PMSL_files] # All file paths in a list of lists
    print('H500_files:\n', H500_files)
    print('all_NARR_files:\n', all_NARR_files)

    SYNABBR_shortlist = ['H500', 'CAPE', 'PMSL']
    ''' Creating '''
    SYNABBR_list = []
    for i,var_list in enumerate(all_NARR_files):
        print('var_list:\n', var_list)
        SYNABBR_list.append([SYNABBR_shortlist[i]]*len(var_list))
    print('SYNABBR_list:\n', SYNABBR_list)

    # Example files:
    # all_NARR_files     = [['2_H500.csv', '1_H500.csv'], ['1_CAPE.csv', '2_CAPE.csv'], ['1_PMSL.csv', '2_PMSL.csv']]
    # SYNABBR_list  = [['H500', 'H500'], ['CAPE', 'CAPE'], ['PMSL', 'PMSL']]

    # Looping through all_NARR_files = [[synvar_files],[synvar_files],[synvar_files]]. i goes from 0 to 2
    i_list = list(range(0,len(all_NARR_files)))
    df_all_synvar = pd.DataFrame([])

    # Loops through list of file paths
    for i, SYNVAR_files, SYNABBR in zip(i_list, all_NARR_files, SYNABBR_shortlist):
        # When i = 0, SYNVAR_files = ['/path/to/1_H500.csv', '/path/to/2_H500.csv', ...], SYNABBR_shortlist='H500'
        # When i = 1, SYNVAR_files = ['/path/to/1_CAPE.csv', '/path/to/2_CAPE.csv', ...], SYNABBR_shortlist='CAPE'

        # Creating a dataframe generator for one type of synoptic variable on each loop through all_NARR_files
        # e.g. When i = 0, df_from_each_file contains all H500 data that concatenates into df
        df_from_each_file = (pd.read_csv(file, header='infer', index_col=['lon', 'lat', 'time']) for file in SYNVAR_files)
        print('df from each file:\n', df_from_each_file)
        df = pd.concat(df_from_each_file, axis=0)
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
            df_all_synvar = df_all_synvar.append(df)
            print('First df_all_synvar concatenation:\n', df_all_synvar)
        else: # Concat df to rows of df_all_synvar
            df_all_synvar = pd.concat((df_all_synvar, df), axis=1, join='inner')
            print('Second df_all_synvar concatenation:\n', df_all_synvar)
            print('Columns of df_all_synvar concatenation:\n', df_all_synvar.columns)

        arr = df.values
        print(' arr:\n', arr)
        print('np.shape(arr):', np.shape(arr))

    print('Final df_all_synvar:\n', df_all_synvar)
    print('Final df_all_synvar w/ index:\n', df_all_synvar)
    print('Length of final df_all_synvar w/index:', len(df_all_synvar['CAPE']))

    print('df_all_synvar.loc[index values]:\n', df_all_synvar.loc[-131.602, 49.7179, '1979-01-01 00:00:00'])
    print('df_all_synvar.loc[[index values]]:\n', df_all_synvar.loc[[-131.602, 49.7179, '1979-01-01 00:00:00']])

    # get columns Lat, Lon, Mean Temp, Max Temp, Min temp, Precipitation
    data = df[[]]
    data = data.apply(pd.to_numeric,  errors='coerce') # Converting to floats
    data = data.dropna(how='any')
    #names = ['Latitude', 'Longitude', 'Monthly Median temperature (C)', 'Monthly Max temperature (C)', 'Monthly Min temperature (C)', 'Monthly total precipitation (mm)']
    print('data.head():\n', data.head())

    # Pickle out:
    df_all_synvar_pkl = df_all_synvar.to_pickle('/home/dp/Documents/FWP/NARR/pickle/df_all_synvar.pkl')

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

    df_all_synvar_grid_interp = pd.DataFrame([])
    for i, SYNVAR in enumerate(SYNABBR_shortlist):
        print('Now processing: ', SYNVAR)

        # Getting z values. x, y, t values gathered just before loop.
        z = df_all_synvar[SYNVAR].values.tolist()
        d = [i for i in zip(t,x,y,z)]

        df = pd.DataFrame(data=d, columns=['time','lon','lat',SYNVAR])

        df.set_index('time', inplace=True)
        print('df.index:\n', df.index)
        print('df.index[10]:\n', df.index[10])


        ''' --------------- SYNVAR Tripcolor & Tricontourf: Unevenly spaced grid --------------- '''

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
        print('-------------------------------------------')
        print('SYNVAR_3d[0][10][2]:\n', SYNVAR_3d[0][10][2])
        print('SYNVAR_3d[0,:,:]:\n', SYNVAR_3d[0,:,:])
        print('SYNVAR_3d:\n', SYNVAR_3d)
        SYNVAR_3d_shape = np.shape(SYNVAR_3d)
        print('SYNVAR_3d shape:\n', SYNVAR_3d_shape)
        m,n,r = SYNVAR_3d.shape
        out_arr = np.column_stack((np.repeat(np.arange(m),n),SYNVAR_3d.reshape(m*n,-1)))
        out_df = pd.DataFrame(out_arr)
        #out_df.columns = SYNABBR_shortlist
        print('out_arr:\n', out_arr)
        print('out_df:\n', out_df)
        our_arr_with_time = np.column_stack((out_arr,t))
        print('our_arr_with_time:\n', our_arr_with_time)

        num_datetimes = SYNVAR_3d_shape[0]  # Used to specify number of layers in Z and Zi

        f, ax = plt.subplots(1,2, sharex=True, sharey=True)
        ax[0].tripcolor(x,y,z)
        ax[1].tricontourf(x_t0,y_t0,z_t0,20)#, cmap=cm.jet) # 20 contour levels is good quality

        ax[0].plot(x,y, 'ko ', markersize=1)
        ax[0].set_xlabel('Longitude'); ax[0].set_ylabel('Latitude')
        ax[1].plot(x_t0,y_t0, 'ko ', markersize=1)
        ax[1].set_xlabel('Longitude'); ax[1].set_ylabel('Latitude')

    ##   plt.savefig('SYNVAR_OR_WA_pyplot.png')
        plt.show()

        ''' --------------- SYNVAR surf plot w/ evenly spaced grid - not working --------------- '''
        npts = 200
        N = 200

        # --- Interpolation on a grid:
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


        ''' --- Example 3D Animation --- '''
        ''' Uncomment to run '''
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


        ''' --- Animation: SYNVAR and SYNVAR gradient --- '''
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

        z_min_list = []; z_max_list = []
        z_grad_x_min_list = []; z_grad_x_max_list = []; z_grad_y_min_list = []; z_grad_y_max_list = []
        layers, row, cols = np.shape(SYNVAR_3d)

        for j in range(0,layers): # Step through Zi-layers
            # SYNVAR_3d[datetime:row:column]
            # jth time point
            # all rows within jth timepoint
            # 0th, 1st, or 2nd
            xt = SYNVAR_3d[j,:,0]
            yt = SYNVAR_3d[j,:,1]
            zt = SYNVAR_3d[j,:,2]
            xt_yt = np.c_[xt,yt]

            # --- Array shapes:
            # (xt, yt)  = 1122 x 2
            # zt        = 1122
            # (Xi, Yi)  = 100 x 100 (e.g. N x N)

            ''' Calculating gridded interpolated Zi and its gradients.
                CAPE gradients will also be calculated but won't be
                included in the final df. '''
            Zi[:,:,j] = griddata((xt,yt), zt, (Xi,Yi), method='linear') # Converts z data to a regular grid
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



        ''' --- Building each synoptic variable array into a dataframe and concatenating the dataframes
            on 'inner' indices. Catches any disagreements with index values that array column stack won't. --- '''

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

        ''' ADDRESS THIS SECTION. REWRITE WITHOUT LOOP: '''
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
        ''' -------------------------------------------------------------------------------------------------- '''

        # Used to set 3D animation vertical range:
        Z_min = int(np.nanmin(Zi[:,:,:]))
        Z_max = int(np.nanmax(Zi[:,:,:]))
        Z_grad_x_min = int(np.nanmin(Zi_grad_x[:,:,:]))
        Z_grad_x_max = int(np.nanmax(Zi_grad_x[:,:,:]))
        Z_grad_y_min = int(np.nanmin(Zi_grad_y[:,:,:]))
        Z_grad_y_max = int(np.nanmax(Zi_grad_y[:,:,:]))

        #plot = [ax.plot_surface(x, y, zarray[:,:,0], color='0.75', rstride=1, cstride=1)]
        plot = [ax.plot_surface(Xi, Yi, Zi_grad_x[:,:,0], vmin=Z_min, vmax=Z_max, linewidth=0, antialiased=True, color='0.75', rstride=1, cstride=1)]

        ax.set_zlim(Z_min, Z_max)
        ax.zaxis.set_major_locator(LinearLocator(6))
        ax.zaxis.set_major_formatter(FormatStrFormatter('%d'))
        ax.set_xlabel('Longitude')
        ax.set_ylabel('Latitude')
        ax.set_zlabel(SYNVAR+': X gradient')
        # Color bar which maps values to colors:
        #fig.colorbar(surf, shrink=0.5, aspect=5)
        # NEED THIS BUT NOT WORKING: fig.colorbar(plot, shrink=0.5, aspect=5)

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

    # Pickle out:
    df_all_synvar_grid_interp_pkl = df_all_synvar_grid_interp.to_pickle('/home/dp/Documents/FWP/NARR/pickle/df_all_synvar_grid_interp.pkl')

    # df with only the columns desired. Gradients for CAPE are being dropped.
    # Current sample size for Jan 1-14 from SOMPY's point of view is 98 unique maps
    df_chosen = df_all_synvar_grid_interp.drop(columns=['CAPE Grad X', 'CAPE Grad Y'])

    return df_chosen, df_all_synvar_grid_interp, df_all_synvar, SYNVAR_3d, df, t, x, y, z


''' Import pickle file, export as csv for SOM training in Julia '''
def synvarPickleToCSV(pickle_in_filename, csv_out_filename, cols_list):
    # What is file size of pickle vs csv?
    pickle_dir  = '~/Documents/FWP/NARR/pickle/' + pickle_in_filename
    csv_dir     = '~/Documents/FWP/NARR/' + csv_out_filename # Writing to FWP/NARR

    # Read in pickle:
    df = pd.read_pickle(pickle_dir)
    df = df[cols_list[0]] # Currently exporting first element of cols_list
    print('Imported pickle as df:\n', df.head())
    # Export df as csv:
    df.to_csv(csv_dir, index=True, header=True) # Includes index columns, names all columns at top of file
    return

''' --- Run import_csv, synvar_plot --- '''
#df_chosen, df_all_synvar_grid_interp, df_all_synvar, SYNVAR_3d, df, t, x, y, z = import_NARR_gridMET_csv(-125,-116,41,50)
''' ----------------------------------- '''

''' --- Run synvarPickleToCSV --- '''
#df = synvarPickleToCSV('df_all_synvar_grid_interp.pkl','df_all_synvar_grid_interp.csv',['H500 Grad X'])
''' ----------------------------- '''

# Read in all csvs from folder
# path = '..\\..\\data\\'
# path = 'C:\\Users\Dan\Downloads\SOMPY_robust_clustering-master\SOMPY_robust_clustering-master\data\\'
# all_NARR_files = glob.glob(os.path.join(path, "*.csv"))
# print('all_NARR_files:\n' ,all_NARR_files[0:10])

# # concat into one df
# df_from_each_file = (pd.read_csv(f, skiprows = 31) for f in all_NARR_files)
# concatenated_df   = pd.concat(df_from_each_file, ignore_index=True)

# print('concatenated df head:\n', concatenated_df.head)
# print('concatenated df columns:\n', concatenated_df.columns)

# # get columns Lat, Long, Mean Temp, Max Temp, Min temp, Precipitation
# data = concatenated_df[['Lat', 'Long', 'Tm', 'Tx', 'Tn', 'P']]
# data = data.apply(pd.to_numeric,  errors='coerce') # Converting to floats
# data = data.dropna(how='any')
# names = ['Latitude', "longitude", 'Monthly Median temperature (C)','Monthly Max temperature (C)', 'Monthly Min temperature (C)', 'Monthly total precipitation (mm)']

# print(data.head())

# # create the SOM network and train it. You can experiment with different normalizations and initializations
# sm = SOMFactory().build(data.values, normalization = 'var', initialization='pca', component_names=names)
# sm.train(n_job=1, verbose=False, train_rough_len=2, train_finetune_len=5)

# # The quantization error: average distance between each data vector and its BMU.
# # The topographic error: the proportion of all data vectors for which first and second BMUs are not adjacent units.
# topographic_error = sm.calculate_topographic_error()
# quantization_error = np.mean(sm._bmu[1])
# print ("Topographic error = %s; Quantization error = %s" % (topographic_error, quantization_error))

# # component planes view
# from sompy.visualization.mapview import View2D
# view2D  = View2D(10,10,"rand data",text_size=12)
# view2D.show(sm, col_sz=4, which_dim="all", denormalize=True)

# # U-matrix plot
# from sompy.visualization.umatrix import UMatrixView

# umat  = UMatrixView(width=10,height=10,title='U-matrix')
# umat.show(sm)



# # do the K-means clustering on the SOM grid, sweep across k = 2 to 20
# from sompy.visualization.hitmap import HitMapView
# K = 20 # stop at this k for SSE sweep
# K_opt = 18 # optimal K already found
# [labels, km, norm_data] = sm.cluster(K,K_opt)
# hits  = HitMapView(20,20,"Clustering",text_size=12)
# a=hits.show(sm)

# import gmplot

# gmap = gmplot.GoogleMapPlotter(54.2, -124.875224, 6)
# j = 0
# for i in km.cluster_centers_:
#    gmap.marker(i[0],i[1],'red', title="Centroid " + str(j))
#    j += 1

# gmap.draw("centroids_map.html")


# from bs4 import BeautifulSoup

# def insertapikey(fname, apikey):
#    """put the google api key in a html file"""
#    def putkey(htmltxt, apikey, apistring=None):
#        """put the apikey in the htmltxt and return soup"""
#        if not apistring:
#            apistring = "https://maps.googleapis.com/maps/api/js?key=%s&callback=initMap"
#        soup = BeautifulSoup(htmltxt, 'html.parser')
#        body = soup.body
#        src = apistring % (apikey, )
#        tscript = soup.new_tag("script", src=src, async="defer")
#        body.insert(-1, tscript)
#        return soup
#    htmltxt = open(fname, 'r').read()
#    soup = putkey(htmltxt, apikey)
#    newtxt = soup.prettify()
#    open(fname, 'w').write(newtxt)
# API_KEY= 'YOUR API KEY HERE'
# insertapikey("centroids_map.html", API_KEY)


# gmap = gmplot.GoogleMapPlotter(54.2, -124.875224, 6)
# j = 0
# for i in km.cluster_centers_:
#    gmap.marker(i[0],i[1],'red', title="Centroid " + str(j))
#    j += 1

# gmap.draw("centroids_map.html")




''' ---------------------- Plotting Code ---------------------- '''
# ''' Interpolation '''
# def interpgrid():
#     z = np.array([-5.,-15.,-25.,-36.,-49.,-65.,-84.,-105.5,-130.5,-159.5,-192.5,
#                   -230.,-273.,-322.5,-379.,-443.,-515.,-596.,-688.,-792.,-909.5,
#                   -1042.5,-1192.5,-1362.,-1553.5,-1770.,-2015.,-2285.,-2565.,-2845.])
#     y = np.arange(0,100)
#     print('z:\n', z)
#     print('y:\n', y)
#     ''' np.random.ran(nrows, ncols) results in an nrows x ncols matrix:
#         np.cumsum(array, axis=0) sums array row-to-row for each column.
#         e.g. within column 0, the rows are summed downward so that the
#         bottom row of column 0 is a summation of all of the values above. '''
#     yz_matrix = np.cumsum(np.random.rand(len(z), len(y)), axis=0)
#     ''' With yz_matrix full of unique values, it's possible to  '''
#     print('yz_matrix:\n', yz_matrix)

#     fig, (ax, ax2) = plt.subplots(ncols=2)

#     # plot raw data as pcolormesh
#     Y,Z = np.meshgrid(y,z[::-1])
#     print('Z:\n', Z)
#     print('Y:\n', Y)
#     print('YI Shape:', np.shape(YI)) # 2840 x 100
#     ax.pcolormesh(Y,Z, yz_matrix, cmap='inferno')
#     ax.set_title("pcolormesh data")

#     # now interpolate data to new grid
#     zi = np.arange(-2845,-5) # A new, higher resolution z, increment is 1 from -2845 to -5
#     print('zi:\n', zi)
#     YI,ZI = np.meshgrid(y,zi)
#     print('ZI:\n', ZI)
#     print('ZI Shape:', np.shape(ZI))
#     print('YI:\n', YI)
#     print('YI Shape:', np.shape(YI)) # 2840 x 100
#     points = np.c_[Y.flatten(),Z.flatten()] # Flattens the original 2-D Y array 30x100 to 1-D 1x3000

#     ''' scipy.interpolate.griddate():
#         points = original y,z points
#         yz_matrix = data that fits into y,z points grid.
#         (YI,ZI) = points at which to interpolate data
#         method = linear interpolation '''
#     interp = griddata(points, yz_matrix.flatten(), (YI,ZI), method='linear')
#     print('interp:\n', interp)

#     ax2.pcolormesh(YI,ZI, interp, cmap='inferno')
#     ax2.set_title("pcolormesh interpolated")

#     plt.show()
#     return Y, Z
# ''' ~~~~~~~~~~~~~~~~~ '''

# ''' Contour '''
# def simple_contour():
#     x = [1,2,4,5,8,9,10]
#     y = [1,3,5,7,8,14,18]
#     z = np.random.randn(len(y), len(x))

#     xi = np.arange(min(x), max(x), 0.5)
#     yi = np.arange(min(y), max(y), 0.5)
#     print('yi:\n', yi)

#     x, y = np.meshgrid(x, y)
#     xx, yy = np.meshgrid(xi, yi)
#     print('xx:\n', xx)
#     print('yy:\n', yy)

#     # Interpolation
#     points = np.c_[x.flatten(),y.flatten()]
#     zi = griddata(points, z.flatten(), (xx, yy), method='linear')

#     fig, (ax, ax2) = plt.subplots(ncols=2)

#     ax.pcolormesh(x,y, z, cmap='inferno')
#     ax.set_title("pcolormesh data")

#     ax2.pcolormesh(xi,yi, zi, cmap='inferno')
#     ax2.set_title("pcolormesh interpolated")

#     plt.show()
#     return
''' --------------------------------------------------- '''
