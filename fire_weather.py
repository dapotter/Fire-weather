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
from matplotlib import rcParams
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
from sklearn import preprocessing
from sklearn.model_selection import train_test_split

import math
import glob
import os
import pickle

import numpy as np

print(np.__version__)

import psycopg2



def fire_weather_db(table):
    # ----------------------------------------------------------------------------
    # PARAMETERS:
    # 
    # table:                    Name of table to create in postgres in pgAdmin4:
    #                           'narr', 'narr_erc', or 'narr_erc_categorical'
    #
    # SCRIPT FUNCTION:          
    #
    # 1)    Copies CSV files (df_NARR.csv, df_NARR_ERC.csv, or 
    #       df_NARR_ERC_categorical) from their directories to
    #       fire_weather_db database in Postgres. Automatically
    #       drops a table if it exists.
    # ----------------------------------------------------------------------------

    conn = psycopg2.connect(
        database='fire_weather_db',
        # database='sample_db',
        user='postgres',
        password='test123',
        host='127.0.0.1',
        port='5432')

    cur = conn.cursor()
    # cur.execute('select * from narr_erc limit 20') # Executes the cursor but won't return rows
    # cur.execute('insert into people (id, first_name, last_name, age, occupation) values(%s, %s, %s, %s, %s)', (21, "Dan", "Potter", 34, "Data Scientist")) # Executes the cursor but won't return rows

    # cur.execute('select * from people')
    # rows = cur.fetchall()
    # for r in rows: # Returns multiple tuples (lat, lon, time)
    #                  #                         (lat, lon, time)
    #     # print(f'lat {r[0]} lon {r[1]} date {r[2]} h500_grad_x {r[3]} h500_grad_y {r[4]} pmsl_grad_x {r[5]} pmsl_grad_y {r[5]} cape {r[6]} erc {r[7]}')
    #     print(f'id {r[0]} first_name {r[1]} last_name {r[2]} age {r[3]} occupation {r[4]}')
    
    # ----------------------------------------------------------------------------
    if table == 'narr':
        print('Creating {}...'.format(str(table)))
        # Drop the table. Need to check if it exists first.
        cur.execute("drop table narr")
        # Create narr table:
        cur.execute("create table narr \
                    ( \
                    lat double precision, \
                    lon double precision, \
                    time timestamp, \
                    h500 double precision, \
                    h500_grad_x double precision, \
                    h500_grad_y double precision, \
                    pmsl double precision, \
                    pmsl_grad_x double precision, \
                    pmsl_grad_y double precision, \
                    cape double precision \
                    ) \
                    ")
        # Copy df_NARR.csv into narr table
        cur.execute("copy narr from '/home/dp/Documents/FWP/NARR/csv/df_NARR.csv' delimiter ',' csv header;")
        # Add primary key
        cur.execute("alter table narr add column id serial primary key")
        # Read some of the data. Need to find an alternative to fetchall for printing to screen.
        #cur.execute('select * from narr_erc limit 20') # Executes the cursor but won't return rows
    # ----------------------------------------------------------------------------
    elif table == 'narr_erc':
        print('Creating {}...'.format(str(table)))
        # Drop the table. Need to check if it exists first.
        cur.execute("drop table narr_erc")
        # Create narr_erc table:
        cur.execute("create table narr_erc \
                    ( \
                    lat double precision, \
                    lon double precision, \
                    date date, \
                    h500 double precision, \
                    h500_grad_x double precision, \
                    h500_grad_y double precision, \
                    pmsl double precision, \
                    pmsl_grad_x double precision, \
                    pmsl_grad_y double precision, \
                    cape double precision, \
                    erc real \
                    ) \
                    ")
        # Copy df_NARR_ERC.csv into narr_erc table
        cur.execute("copy narr_erc from '/home/dp/Documents/FWP/NARR_gridMET/csv/df_NARR_ERC.csv' delimiter ',' csv header;")
        # Add primary key
        cur.execute("alter table narr_erc add column id serial primary key")
    # ----------------------------------------------------------------------------
    elif table == 'narr_erc_categorical':
        print('Creating {}...'.format(str(table)))
        # Drop the table. Need to check if it exists first.
        cur.execute("drop table narr_erc_categorical")
        # Create narr_erc_categorical table:
        cur.execute("create table narr_erc_categorical \
                    ( \
                    lat double precision, \
                    lon double precision, \
                    date date, \
                    h500 double precision, \
                    h500_grad_x double precision, \
                    h500_grad_y double precision, \
                    pmsl double precision, \
                    pmsl_grad_x double precision, \
                    pmsl_grad_y double precision, \
                    cape double precision, \
                    erc varchar (255) \
                    ) \
                    ")
        # Copy df_NARR_ERC_categorical.csv into narr_erc_categorical table
        cur.execute("copy narr_erc_categorical from '/home/dp/Documents/FWP/NARR_gridMET/csv/df_NARR_ERC_categorical.csv' delimiter ',' csv header;")
        # Add primary key
        cur.execute("alter table narr_erc_categorical add column id serial primary key")
    conn.commit()

    cur.close()
    conn.close()

    print('Done')

    return


def import_NARR_csv(lon_min, lon_max, lat_min, lat_max, import_interval, export_interval, multi, NARR_csv_in_dir, NARR_csv_out_dir, NARR_pkl_out_dir):
    # ----------------------------------------------------------------------------------------------------
    # PARAMETERS:
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
    # multi:                If XX_MULTI.csv files are present, they contain not only H500, H500 Grad X,
    #                       and H500 Grad Y, but also the following extra variables:
    #                           TEMP (500 hPa temperature, K)
    #                           PVEL (vertical pressure velocity, Pa/s)
    #                           SPFH (Specific humidity, kg/kg)
    #                           CWTR (Cloud water, kg/kg)
    #                       These variables can be imported by setting multi=True. None of the extra
    #                       variables have gradients.
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
    # SCRIPT FUNCTION:
    # 
    # 1)    Read in import_interval number of NARR csv files in rectilinear grid format. All XX_H500.csv
    #       files are stacked, insensitive to the number of columns of data present. Then all XX_PMSL.csv
    #       files are stacked and concatenated to the H500 data. Finally all XX_CAPE.csv data is
    #       concatenated.
    #
    # 2)    Drop all CAPE gradients, keep everything else
    #  
    # 3)    Export all NARR data (df_narr) to csv and pickle files
    #
    # NOTE: The reason that it's necessary to modulate the number of csv files imported at once
    #       using 'import_interal' is because all csv data is held in RAM, which quickly becomes
    #       overwhelmed when more than 2 csv files are imported if each one has 100 grib files
    #       (roughly 300 hrs = 12.5 days) worth of meteorological data.
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
    PMSL_files = glob.glob(os.path.join(NARR_csv_in_dir, '*PMSL.csv'))
    CAPE_files = glob.glob(os.path.join(NARR_csv_in_dir, '*CAPE.csv'))
    print('PMSL_files:\n', PMSL_files)
    print('CAPE_files:\n', CAPE_files)

    if multi == True:
        MULTI_files = glob.glob(os.path.join(NARR_csv_in_dir, '*MULTI.csv'))
        print('MULTI_files:\n', MULTI_files)
        # List of all files:
        all_NARR_files = [MULTI_files, PMSL_files, CAPE_files] # All file paths in a list of lists
        SYNABBR_shortlist = ['MULTI', 'PMSL', 'CAPE']

    else:
        H500_files = glob.glob(os.path.join(NARR_csv_in_dir, '*H500.csv'))
        print('H500_files:\n', H500_files)
        # List of all files:
        all_NARR_files = [H500_files, PMSL_files, CAPE_files] # All file hpaths in a list of lists
        SYNABBR_shortlist = ['H500', 'PMSL', 'CAPE']

        
    # print('H500 file paths:\n', H500_files)
    print('All NARR file paths - before sorting:\n', all_NARR_files)
    all_NARR_files = [sorted(file_list) for file_list in all_NARR_files]
    print('All NARR file paths - after sorting:\n', all_NARR_files)

    num_csv_files = len(PMSL_files)
    csv_file_index_list = list(range(0, num_csv_files, import_interval)) # Import num_csv_files number of files
    start_end_list = [(s,s+import_interval) for s in csv_file_index_list]
    # print('csv_file_index_list:\n', csv_file_index_list)
    # print('start_end_list:\n', start_end_list)
    
    first_datetime_list = []
    last_datetime_list = []
    df_narr = pd.DataFrame([])

    for pkl_counter, (s,e) in enumerate(start_end_list):
        # This loop imports import_interval number of csv files at a time,
        # creating a dataframe, and exporting it as a pickle file on each iteration.
        # pkl_counter is used to create the naming prefix for exporting pickle files.
        select_NARR_files = [file_list[s:e] for file_list in all_NARR_files]
        print('Select NARR file paths - start index to end index:\n', select_NARR_files)

        ''' Creating '''
        SYNABBR_list = []
        for i,var_list in enumerate(select_NARR_files):
            print('var_list:\n', var_list)
            SYNABBR_list.append([SYNABBR_shortlist[i]]*len(var_list))
        print('SYNABBR_list:\n', SYNABBR_list)

        # Example files:
        # If import_factor = 2, 2 files per sublist:
        # select_NARR_files = [['1_H500.csv', '2_H500.csv'], ['1_CAPE.csv', '2_CAPE.csv'], ['1_PMSL.csv', '2_PMSL.csv']]

        # SYNABBR_list = [['H500', 'H500'], ['CAPE', 'CAPE'], ['PMSL', 'PMSL']]

        # Looping through select_NARR_files = [[synvar_files],[synvar_files],[synvar_files]]. i goes from 0 to 2
        i_list = list(range(0,len(select_NARR_files)))
        
        df_narr_partial = pd.DataFrame([])

        for i, SYNVAR_files, SYNABBR in zip(i_list, select_NARR_files, SYNABBR_shortlist):
            # Loops through list of file paths, combines all csv file data into
            # one dataframe df_narr_partial. Outside of this loop, it combines
            # all df_narr_partial data into one csv file df_narr.

            # When i = 0, SYNVAR_files = ['/path/to/01_H500.csv', '/path/to/02_H500.csv', ...], SYNABBR_shortlist='H500'
            # When i = 2, SYNVAR_files = ['/path/to/01_CAPE.csv', '/path/to/02_CAPE.csv', ...], SYNABBR_shortlist='CAPE'
            # if multi == True:
            # When i = 0, SYNVAR_files = ['/path/to/01_MULTI.csv', '/path/to/02_MULTI.csv', ...], SYNABBR_shortlist='MULTI'

            print('i:\n', i)
            print('SYNVAR_files:\n', SYNVAR_files)
            print('SYNABBR:\n', SYNABBR)
            # Creating a dataframe generator for one type of synoptic variable on each loop through select_NARR_files
            # e.g. When i = 0, df_from_each_file contains all H500 data that concatenates into df
            df_from_each_file = (pd.read_csv(file, header='infer', index_col=['lon', 'lat', 'time']) for file in SYNVAR_files)
            print('df from each file... (This could take a minute. Reading every csv file in /NARR/csv/ into one dataframe.)\n', df_from_each_file)
            df = pd.concat(df_from_each_file, axis=0)
            print('concatenated df head:\n', df.head)
            print('concatenated df columns:\n', df.columns)

            # Reordering indices from csv format (time, lon, lat) to time, lat, lon:
            df.reset_index(inplace=True)
            df.set_index(['time','lat','lon'], inplace=True)
            print('df after reset index:\n', df)

            # Converting time to pandas datetime:
            df.index = df.index.set_levels([pd.to_datetime(df.index.levels[0], format='%m/%d/%Y (%H:%M)'), df.index.levels[1], df.index.levels[2]])
            # print('df after converting time index to datetime:\n', df)

            # CHECK FOR TEMPORAL INCONTINUITIES:
            # Checking for gaps in time indicative of NARR file download error:
            # df_index_diff = df.index.get_level_values(2).diff()
            # if any element of df_index_diff > 3 hours, throw error and abort

            # Storing when H500 data starts and ends. Trying to manage
            # time range disagreements between H500, PMSL and CAPE
            if s == start_end_list[0][0]: # Gets the first time value in the first H500 csv (e.g. 1_H500.csv)
                first_datetime = df.index.get_level_values(2)[0]
                first_datetime_list.append(first_datetime)
            elif e == start_end_list[-1][-1]: # Gets the last time value in the last H500 csv (e.g. 10_H500.csv)
                last_datetime = df.index.get_level_values(2)[-1]
                last_datetime_list.append(last_datetime)

            # Resetting index, may not be necessary
            # df.reset_index(inplace=True)
            # df.set_index('lon', inplace=True)
            # print('df after reset_index:\n', df)
            # print('Length of df after reset_index:\n', len(df))

            # Concatenating all H500 csv's together, then all PMSL csv's to it, and so on.
            # df is either all H500 csv's concatenated, or all PMSL csv's, and so on. See
            # the dataframe generator above for how df is created.
            print('df_narr_partial PRIOR TO df CONCATENATION:\n', df_narr_partial)
            if i == 0: # First time through loop, append df to columns
                # When i = 0, all H500 files in df are processed:
                df_narr_partial = df_narr_partial.append(df)
                print('First df_narr_partial concatenation:\n', df_narr_partial)
            else: # Concat df to rows of df_narr_partial
                # df_narr_partial = pd.concat((df_narr_partial, df), axis=1, join='inner', sort=False)
                # print('df_narr_partial concatenation:\n', df_narr_partial)
                # print('Columns of df_narr_partial concatenation:\n', df_narr_partial.columns)
                df_narr_partial = df_narr_partial.join(df) # Joins index-on-index, in this case on=['time','lat','lon'], how='left'
                print('df_narr_partial join:\n', df_narr_partial)
                print('Columns of df_narr_partial join:\n', df_narr_partial.columns)

            arr = df.values
            print(' arr:\n', arr)
            print('np.shape(arr):', np.shape(arr))

        print('Final df_narr_partial:\n', df_narr_partial)
        print('Length of final df_narr_partial w/index:', len(df_narr_partial['CAPE']))


        # Setting multi-index's time index to datetime. To do this, the index must be recreated.
        # https://stackoverflow.com/questions/45243291/parse-pandas-multiindex-to-datetime
        # NOTE that the format provided is the format in the csv file. pd.to_datetime converts
        # it to the format it thinks is appropriate.
        # Use if index is lon, lat, time:
        # df_narr_partial.index = df_narr_partial.index.set_levels([df.index.levels[0], df.index.levels[1], pd.to_datetime(df.index.levels[2], format='%m/%d/%Y (%H:%M)')])
        # Use if index is time, lon, lat:
        # df_narr_partial.index = df_narr_partial.index.set_levels([pd.to_datetime(df.index.levels[0], format='%m/%d/%Y (%H:%M)'), df.index.levels[1], df.index.levels[2]])
        # Use if index is time, lat, lon:
        # df_narr_partial.index = df_narr_partial.index.set_levels([pd.to_datetime(df.index.levels[0], format='%m/%d/%Y (%H:%M)'), df.index.levels[1], df.index.levels[2]])

        # df_narr_partial.reset_index(inplace=True)
        # df_narr_partial.set_index(['lon','lat','time'], inplace=True)
        # print('df_narr_partial :\n', df_narr_partial)

        # This doesn't work if the date is out of range for the csv file that is being read in:
        # print('**** df_narr_partial.loc[index values]:\n', df_narr_partial.loc[-131.602, 49.7179, '1979-01-01 00:00:00'])
        # print('**** df_narr_partial.loc[[index values]]:\n', df_narr_partial.loc[[-131.602, 49.7179, '1979-01-01 00:00:00']])
        print('df_narr_partial: Contains all synoptic variables from csv files:\n', df_narr_partial)


        # get columns Lat, Lon, Mean Temp, Max Temp, Min temp, Precipitation
        data = df[[]]
        data = data.apply(pd.to_numeric,  errors='coerce') # Converting data to floats
        data = data.dropna(how='any')
        #names = ['Latitude', 'Longitude', 'Monthly Median temperature (C)', 'Monthly Max temperature (C)', 'Monthly Min temperature (C)', 'Monthly total precipitation (mm)']
        print('data.head():\n', data.head())


        '''--- Plots a select synoptic variable from df_narr_partial ---'''

        # # Checking index referencing:
        # print('type(df.index.get_level_values(0)):\n', type(df.index.get_level_values(0)))  # Referencing lon type
        # print('df.index.get_level_values(0)[0]:\n', df.index.get_level_values(0)[0])        # Referencing lon index values
        # print('type(df.index.get_level_values(1)):\n', type(df.index.get_level_values(1)))  # Referencing lat type
        # print('df.index.get_level_values(1)[0]:\n', df.index.get_level_values(1)[0])        # Referencing lat index values
        # print('type(df.index.get_level_values(2)):\n', type(df.index.get_level_values(2)))  # Referencing time type
        # print('df.index.get_level_values(2)[0]:\n', df.index.get_level_values(2)[0])        # Referencing time index values


        df_time_point = df_narr_partial.reset_index()
        df_time_point.set_index('time', inplace=True)
        datetime_to_plot = df_time_point.index[-1]
        datetime_to_plot_str = datetime_to_plot.strftime('%b %d, %Y')
        df_time_point = df_time_point.loc[datetime_to_plot]
        # print('df_time_point:\n', df_time_point)

        x = df_time_point.lon.tolist()
        y = df_time_point.lat.tolist()
        H500 = df_time_point['H500'].tolist()
        H500_Grad_X = df_time_point['H500 Grad X'].tolist()
        H500_Grad_Y = df_time_point['H500 Grad Y'].tolist()


        # CONTOUR PLOTTING. WORKS. UNCOMMENT TO UNLEASH ITS BEAUTIFUL PLOTS.
        # ----------------------------------------------------------------------------------------------------
        # # Contour Plots: H500, H500 Grad X, H500 Grad Y for last 3-hr timepoint in df_narr_partial
        # print('Plotting SYNVAR X and Y Gradients:')
        # f, ax = plt.subplots(1,3, figsize=(15,4), sharex=True, sharey=True)
        # im1 = ax[0].tricontourf(x,y,H500, 20, cmap=cm.jet) # Use to use this: (x,y,z,20)
        # im2 = ax[1].tricontourf(x,y,H500_Grad_X, 20, cmap=cm.jet) # Use to use this: (x,y,z,20)
        # im3 = ax[2].tricontourf(x,y,H500_Grad_Y, 20, cmap=cm.jet) # 20 contour levels is good quality
        # f.colorbar(im1, ax=ax[0])
        # f.colorbar(im2, ax=ax[1])
        # f.colorbar(im3, ax=ax[2])

        # ax[0].plot(x, y, 'ko ', markersize=1)
        # ax[0].set_xlabel('Longitude'); ax[0].set_ylabel('Latitude')
        # # title_str = SYNVAR+' X Gradient'
        # # print('title_str:', title_str)
        # ax[0].set_title('H500')

        # ax[1].plot(x, y, 'ko ', markersize=1)
        # ax[1].set_xlabel('Longitude'); ax[1].set_ylabel('Latitude')
        # # title_str = SYNVAR+' X Gradient'
        # # print('title_str:', title_str)
        # ax[1].set_title('H500 X Gradient')

        # ax[2].plot(x, y, 'ko ', markersize=1)
        # ax[2].set_xlabel('Longitude'); ax[2].set_ylabel('Latitude')
        # # title_str = SYNVAR+' Y Gradient'
        # ax[2].set_title('H500 Y Gradient')

        # plt.suptitle('Contour Plots: H500, X and Y Gradients'+' ('+datetime_to_plot_str+')')
        # plt.savefig('H500_contour_with_gradient_rectilinear.png')
        # plt.show()
        # ----------------------------------------------------------------------------------------------------


        # Drop CAPE gradients:
        # Pressure gradients drive surface level winds and contribute to fire weather,
        # however I am keeping the pressure levels because they can be correlated with stormy
        # weather. H500 is correlated 0.77 with ERC. PMSL is much less correlated with ERC, why
        # is this?
        df_narr_partial.drop(columns=['CAPE Grad X', 'CAPE Grad Y'], inplace=True)
        print('df_narr_partial:\n', df_narr_partial.head())

        # Index values (lon and lat) carry out to too many decimals (e.g. 42.058800000000000002)
        # Round to four decimals
        decimals = 4
        df_narr_partial.reset_index(inplace=True)
        df_narr_partial['lat'] = df_narr_partial['lat'].apply(lambda x: round(x, decimals))
        df_narr_partial['lon'] = df_narr_partial['lon'].apply(lambda x: round(x, decimals))
        df_narr_partial.set_index(['lat','lon','time'], inplace=True)

        # Pickle naming (If the pickle number is single digit, add a prefix 0, otherwise use i for prefix):
        if pkl_counter < export_interval:
            pickle_name_narr = '0' + str(pkl_counter) + '_df_narr_partial.pkl'
        else:
            pickle_name_narr = str(pkl_counter) + '_df_narr_partial.pkl'
        # Pickle out:
        df_narr_partial.to_pickle(NARR_pkl_out_dir + pickle_name_narr)

        # CSV write (Unlike the pickle out above, here the data is written to one enormous csv file):
        df_narr = pd.concat((df_narr, df_narr_partial), axis=0)
        print('df_narr:\n', df_narr)

    # # IF SHORT ON TIME, YOU MAY NOT WANT TO EXPORT THE ENTIRE 6 GB CSV FILE:
    # print('Exporting to CSV... (This could take a minute) ******************************')
    # df_narr.to_csv(NARR_csv_out_dir + 'df_NARR.csv', index=True, header=True)

    # NOTE: Current sample size for Jan 1-14 from SOMPY's point of view is 98 unique maps
    # Need to change the columns these access:
    print('first_datetime_list:\n', first_datetime_list)
    print('last_datetime_list:\n', last_datetime_list)

    return


def import_gridMET_csv(gridMET_csv_in_dir, gridMET_pkl_out_dir):
    # ----------------------------------------------------------------------------
    # PARAMETERS:
    #
    # gridMET_csv_in_dir:       Directory of raw csv gridMET data from gridMET website.
    #
    # gridMET_pkl_out_dir:      Directory to pickle out the gridMET data.
    # 
    # SCRIPT FUNCTION:
    # 
    # 1)    gridMET data exists in csv files for each day of the downloaded year.
    #       This function imports all of those csv files ending in ERC.csv and
    #       concatenates them into df_erc, sets index to lon, lat, time. 
    # 2)    Converts days since Jan 1, 1900 to date format
    # 3)    Exports to df_erc.pkl. No csv export as it's not necessary.
    #
    # NOTE: The exported pickle df_erc.pkl will later be imported by merge_NARR_ERC().
    # ----------------------------------------------------------------------------

    # gridMET csv file names must end with ERC.csv:
    all_gridMET_files = glob.glob(os.path.join(gridMET_csv_in_dir, '*ERC.csv'))
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

    df_erc.to_pickle(gridMET_pkl_out_dir + 'df_erc.pkl')

    return


def import_multi_NARR_csv(lon_min, lon_max, lat_min, lat_max, import_interval, export_interval, multi, NARR_csv_in_dir, NARR_csv_out_dir, NARR_pkl_out_dir):
    # ----------------------------------------------------------------------------------------------------
    # PARAMETERS:
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
    # multi:                If XX_MULTI.csv files are present, they contain not only H500, H500 Grad X,
    #                       and H500 Grad Y, but also the following extra variables:
    #                           TEMP (500 hPa temperature, K)
    #                           PVEL (vertical pressure velocity, Pa/s)
    #                           SPFH (Specific humidity, kg/kg)
    #                           CWTR (Cloud water, kg/kg)
    #                       These variables can be imported by setting multi=True. None of the extra
    #                       variables have gradients.
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
    # SCRIPT FUNCTION:
    # 
    # 1)    Read in import_interval number of NARR csv files in rectilinear grid format. All XX_H500.csv
    #       files are stacked, insensitive to the number of columns of data present. Then all XX_PMSL.csv
    #       files are stacked and concatenated to the H500 data. Finally all XX_CAPE.csv data is
    #       concatenated.
    #
    # 2)    Drop all CAPE gradients, keep everything else
    #  
    # 3)    Export all NARR data (df_narr) to csv and pickle files
    #
    # NOTE: The reason that it's necessary to modulate the number of csv files imported at once
    #       using 'import_interal' is because all csv data is held in RAM, which quickly becomes
    #       overwhelmed when more than 2 csv files are imported if each one has 100 grib files
    #       (roughly 300 hrs = 12.5 days) worth of meteorological data.
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
    # MULTI_files = glob.glob(os.path.join(NARR_csv_in_dir, '*MULTI.csv'))
    MULTI_files = glob.glob(os.path.join(NARR_csv_in_dir, '*H500.csv'))
    # PMSL_files = glob.glob(os.path.join(NARR_csv_in_dir, '*PMSL.csv'))
    # CAPE_files = glob.glob(os.path.join(NARR_csv_in_dir, '*CAPE.csv'))
    print('MULTI_files:\n', MULTI_files)
    # print('PMSL_files:\n', PMSL_files)
    # print('CAPE_files:\n', CAPE_files)
    # List of all files:
    
    # all_NARR_files = [MULTI_files, PMSL_files, CAPE_files] # All file paths in a list of lists
    all_NARR_files = [MULTI_files] # All file paths in a list of lists

    # SYNABBR_shortlist = ['MULTI', 'PMSL', 'CAPE']
    SYNABBR_shortlist = ['MULTI']

    # print('H500 file paths:\n', H500_files)
    print('All NARR file paths - before sorting:\n', all_NARR_files)
    all_NARR_files = [sorted(file_list) for file_list in all_NARR_files]
    print('All NARR file paths - after sorting:\n', all_NARR_files)

    num_csv_files = len(MULTI_files)
    csv_file_index_list = list(range(0, num_csv_files, import_interval)) # Import num_csv_files number of files
    start_end_list = [(s,s+import_interval) for s in csv_file_index_list]
    # print('csv_file_index_list:\n', csv_file_index_list)
    # print('start_end_list:\n', start_end_list)
    
    first_datetime_list = []
    last_datetime_list = []

    # df_narr = pd.DataFrame([])
    df_MULTI = pd.DataFrame([])

    # df_narr_partial = pd.DataFrame([])
    df_MULTI_partial = pd.DataFrame([])
    df_PMSL_partial = pd.DataFrame([])
    df_CAPE_partial = pd.DataFrame([])

    for pkl_counter, (s,e) in enumerate(start_end_list):
        # This loop imports import_interval number of csv files at a time,
        # creating a dataframe, and exporting it as a pickle file on each iteration.
        # pkl_counter is used to create the naming prefix for exporting pickle files.
        select_NARR_files = [file_list[s:e] for file_list in all_NARR_files]
        print('Select NARR file paths - start index to end index:\n', select_NARR_files)

        ''' Creating '''
        SYNABBR_list = []
        for i,var_list in enumerate(select_NARR_files):
            print('var_list:\n', var_list)
            SYNABBR_list.append([SYNABBR_shortlist[i]]*len(var_list))
        print('SYNABBR_list:\n', SYNABBR_list)

        # Example files:
        # If import_factor = 2, 2 files per sublist:
        # select_NARR_files = [['1_H500.csv', '2_H500.csv'], ['1_CAPE.csv', '2_CAPE.csv'], ['1_PMSL.csv', '2_PMSL.csv']]

        # SYNABBR_list = [['H500', 'H500'], ['CAPE', 'CAPE'], ['PMSL', 'PMSL']]

        # Looping through select_NARR_files = [[synvar_files],[synvar_files],[synvar_files]]. i goes from 0 to 2
        i_list = list(range(0,len(select_NARR_files)))

        # for i, SYNVAR_files, SYNABBR in zip(i_list, select_NARR_files, SYNABBR_shortlist):
        # Loops through list of file paths, combines all csv file data into
        # one dataframe df_narr_partial. Outside of this loop, it combines
        # all df_narr_partial data into one csv file df_narr.

        # When i = 0, SYNVAR_files = ['/path/to/01_H500.csv', '/path/to/02_H500.csv', ...], SYNABBR_shortlist='H500'
        # When i = 2, SYNVAR_files = ['/path/to/01_CAPE.csv', '/path/to/02_CAPE.csv', ...], SYNABBR_shortlist='CAPE'
        # if multi == True:
        # When i = 0, SYNVAR_files = ['/path/to/01_MULTI.csv', '/path/to/02_MULTI.csv', ...], SYNABBR_shortlist='MULTI'
        i = 0
        SYNVAR_files = select_NARR_files[0]
        SYNABBR = SYNABBR_shortlist[0]
        print('i:\n', i)
        print('SYNVAR_files:\n', SYNVAR_files)
        print('SYNABBR:\n', SYNABBR)
        # Creating a dataframe generator for one type of synoptic variable on each loop through select_NARR_files
        # e.g. When i = 0, df_from_each_file contains all H500 data that concatenates into df
        df_from_each_file = (pd.read_csv(file, header='infer', index_col=['lon', 'lat', 'time']) for file in SYNVAR_files)
        print('df from each file... (This could take a minute. Reading every csv file in /NARR/csv/ into one dataframe.)\n', df_from_each_file)
        df = pd.concat(df_from_each_file, axis=0)
        print('concatenated df head:\n', df.head)
        print('concatenated df columns:\n', df.columns)

        print('df:\n', df)

        df.reset_index(inplace=True)
        df.set_index(['lat','lon','time'], inplace=True)
        print('df after reset index:\n', df)
        df.index = df.index.set_levels([df.index.levels[0], df.index.levels[1], pd.to_datetime(df.index.levels[2], format='%m/%d/%Y (%H:%M)')])
        print('df after index formatting:\n', df)

        # Storing when H500 data starts and ends. Trying to manage
        # time range disagreements between H500, PMSL and CAPE
        if s == start_end_list[0][0]: # Gets the first time value in the first H500 csv (e.g. 1_H500.csv)
            first_datetime = df.index.get_level_values(2)[0]
            first_datetime_list.append(first_datetime)
            print('first_datetime_list:\n', first_datetime_list)
        elif e == start_end_list[-1][-1]: # Gets the last time value in the last H500 csv (e.g. 10_H500.csv)
            last_datetime = df.index.get_level_values(2)[-1]
            last_datetime_list.append(last_datetime)
            print('last_datetime_list:\n', last_datetime_list)

        # # Resetting index, may not be necessary
        # df.reset_index(inplace=True)
        # df.set_index('lon', inplace=True)
        # print('df after reset_index:\n', df)
        # print('Length of df after reset_index:\n', len(df))

        # Concatenating all H500 csv's together, then all PMSL csv's to it, and so on.
        # df is either all H500 csv's concatenated, or all PMSL csv's, and so on. See
        # the dataframe generator above for how df is created.
        print('df_MULTI_partial JUST BEFORE CONCATENATION WITH df:\n', df_MULTI_partial)
        ##if i == 0: # First time through loop, append df to columns
            # When i = 0, all H500 files in df are processed:
            # df_narr_partial = df_narr_partial.append(df)
        df_MULTI_partial = df_MULTI_partial.append(df)
        # print('df_narr_partial concatenation along rows:\n', df_narr_partial)
        print('df_MULTI_partial concatenation along rows:\n', df_MULTI_partial)
        # else: # Concat df to rows of df_narr_partial
        #     df_narr_partial = pd.concat((df_narr_partial, df), axis=1, join='inner', sort=False)
        #     print('Second df_narr_partial concatenation:\n', df_narr_partial)
        #     print('Columns of df_narr_partial concatenation:\n', df_narr_partial.columns)
        
        # arr = df.values
        # print(' arr:\n', arr)
        # print('np.shape(arr):', np.shape(arr))
        arr = df_MULTI_partial.values
        print(' arr:\n', arr)
        print('np.shape(arr):', np.shape(arr))

        # print('Final df_narr_partial:\n', df_narr_partial)
        # print('Length of final df_narr_partial w/index:', len(df_narr_partial['CAPE']))
        print('Final df_MULTI_partial:\n', df_MULTI_partial)
        print('Length of final df_MULTI_partial w/index:', len(df_MULTI_partial['H500']))

        # Setting multi-index's time index to datetime. To do this, the index must be recreated.
        # https://stackoverflow.com/questions/45243291/parse-pandas-multiindex-to-datetime
        # NOTE that the format provided is the format in the csv file. pd.to_datetime converts
        # it to the format it thinks is appropriate.
        # Use if index is lon, lat, time:
        # df_narr_partial.index = df_narr_partial.index.set_levels([df.index.levels[0], df.index.levels[1], pd.to_datetime(df.index.levels[2], format='%m/%d/%Y (%H:%M)')])
        # Use if index is time, lon, lat:
        # df_narr_partial.index = df_narr_partial.index.set_levels([pd.to_datetime(df.index.levels[0], format='%m/%d/%Y (%H:%M)'), df.index.levels[1], df.index.levels[2]])
        # Use if index is time, lat, lon:
        # df_narr_partial.index = df_narr_partial.index.set_levels([pd.to_datetime(df.index.levels[0], format='%m/%d/%Y (%H:%M)'), df.index.levels[1], df.index.levels[2]])
        
        # print('df_MULTI_partial:\n', df_MULTI_partial)
        # print('df:\n', df)

        # df_MULTI_partial.index = df_MULTI_partial.index.set_levels([df.index.levels[0], df.index.levels[1], pd.to_datetime(df.index.levels[2], format='%m/%d/%Y (%H:%M)')])
        # print('df_MULTI_partial:\n', df_MULTI_partial)

        # df_narr_partial.reset_index(inplace=True)
        # df_narr_partial.set_index(['lon','lat','time'], inplace=True)
        # print('df_narr_partial :\n', df_narr_partial)

        # This doesn't work if the date is out of range for the csv file that is being read in:
        # print('**** df_narr_partial.loc[index values]:\n', df_narr_partial.loc[-131.602, 49.7179, '1979-01-01 00:00:00'])
        # print('**** df_narr_partial.loc[[index values]]:\n', df_narr_partial.loc[[-131.602, 49.7179, '1979-01-01 00:00:00']])
        # print('df_narr_partial: Contains all synoptic variables from csv files:\n', df_narr_partial)
        print('df_MULTI_partial: Contains all synoptic variables from csv files:\n', df_MULTI_partial)

        '''--- Plots a select synoptic variable from df_narr_partial ---'''
        # # Checking index referencing:
        # print('type(df.index.get_level_values(0)):\n', type(df.index.get_level_values(0)))  # Referencing lon type
        # print('df.index.get_level_values(0)[0]:\n', df.index.get_level_values(0)[0])        # Referencing lon index values
        # print('type(df.index.get_level_values(1)):\n', type(df.index.get_level_values(1)))  # Referencing lat type
        # print('df.index.get_level_values(1)[0]:\n', df.index.get_level_values(1)[0])        # Referencing lat index values
        # print('type(df.index.get_level_values(2)):\n', type(df.index.get_level_values(2)))  # Referencing time type
        # print('df.index.get_level_values(2)[0]:\n', df.index.get_level_values(2)[0])        # Referencing time index values

        df_time_point = df_MULTI_partial.reset_index()
        print('df_time_point:\n', df_time_point)
        df_time_point.set_index('time', inplace=True)
        print('df_time_point:\n', df_time_point)
        datetime_to_plot = df_time_point.index[-1]
        print('datetime_to_plot:\n', datetime_to_plot)
        datetime_to_plot_str = datetime_to_plot.strftime('%b %d, %Y')
        df_time_point = df_time_point.loc[datetime_to_plot]
        # print('df_time_point:\n', df_time_point)

        x = df_time_point.lon.tolist()
        y = df_time_point.lat.tolist()
        H500 = df_time_point['H500'].tolist()
        H500_Grad_X = df_time_point['H500 Grad X'].tolist()
        H500_Grad_Y = df_time_point['H500 Grad Y'].tolist()

        # CONTOUR PLOTTING. WORKS. UNCOMMENT TO UNLEASH ITS BEAUTIFUL PLOTS.
        # ----------------------------------------------------------------------------------------------------
        # # Contour Plots: H500, H500 Grad X, H500 Grad Y for last 3-hr timepoint in df_narr_partial
        # print('Plotting SYNVAR X and Y Gradients:')
        # f, ax = plt.subplots(1,3, figsize=(15,4), sharex=True, sharey=True)
        # im1 = ax[0].tricontourf(x,y,H500, 20, cmap=cm.jet) # Use to use this: (x,y,z,20)
        # im2 = ax[1].tricontourf(x,y,H500_Grad_X, 20, cmap=cm.jet) # Use to use this: (x,y,z,20)
        # im3 = ax[2].tricontourf(x,y,H500_Grad_Y, 20, cmap=cm.jet) # 20 contour levels is good quality
        # f.colorbar(im1, ax=ax[0])
        # f.colorbar(im2, ax=ax[1])
        # f.colorbar(im3, ax=ax[2])

        # ax[0].plot(x, y, 'ko ', markersize=1)
        # ax[0].set_xlabel('Longitude'); ax[0].set_ylabel('Latitude')
        # # title_str = SYNVAR+' X Gradient'
        # # print('title_str:', title_str)
        # ax[0].set_title('H500')

        # ax[1].plot(x, y, 'ko ', markersize=1)
        # ax[1].set_xlabel('Longitude'); ax[1].set_ylabel('Latitude')
        # # title_str = SYNVAR+' X Gradient'
        # # print('title_str:', title_str)
        # ax[1].set_title('H500 X Gradient')

        # ax[2].plot(x, y, 'ko ', markersize=1)
        # ax[2].set_xlabel('Longitude'); ax[2].set_ylabel('Latitude')
        # # title_str = SYNVAR+' Y Gradient'
        # ax[2].set_title('H500 Y Gradient')

        # plt.suptitle('Contour Plots: H500, X and Y Gradients'+' ('+datetime_to_plot_str+')')
        # plt.savefig('H500_contour_with_gradient_rectilinear.png')
        # plt.show()
        # ----------------------------------------------------------------------------------------------------

        # Drop CAPE gradients:
        # Pressure gradients drive surface level winds and contribute to fire weather,
        # however I am keeping the pressure levels because they can be correlated with stormy
        # weather. H500 is correlated 0.77 with ERC. PMSL is much less correlated with ERC, why
        # is this?
        print('df_MULTI_partial:\n', df_MULTI_partial.head())

        # Index values (lon and lat) carry out to too many decimals (e.g. 42.058800000000000002)
        # Round to four decimals
        decimals = 4
        df_MULTI_partial.reset_index(inplace=True)
        df_MULTI_partial['lat'] = df_MULTI_partial['lat'].apply(lambda x: round(x, decimals))
        df_MULTI_partial['lon'] = df_MULTI_partial['lon'].apply(lambda x: round(x, decimals))
        df_MULTI_partial.set_index(['lat','lon','time'], inplace=True)
        print('df_MULTI_partial:\n', df_MULTI_partial)

        # BLOCKING THIS BECAUSE df_MULTI_partial GETS BIGGER WITH EACH ITERATION, THUS
        # NO NEED TO PICKLE OUT ON EACH ITERATION, PICKLING OUT AT THE END ONCE THE LOOP
        # IS COMPLETE:
        # Pickle naming (If the pickle number is single digit, add a prefix 0, otherwise use i for prefix):
        # if pkl_counter < export_interval:
        #     pickle_name_narr = '0' + str(pkl_counter) + '_df_MULTI_partial.pkl'
        # else:
        #     pickle_name_narr = str(pkl_counter) + '_df_MULTI_partial.pkl'
        # # Pickle out:
        # df_MULTI_partial.to_pickle(NARR_pkl_out_dir + pickle_name_narr)

        # CSV write (Unlike the pickle out above, here the data is written to one enormous csv file):
        df_MULTI = pd.concat((df_MULTI, df_MULTI_partial), axis=0)
        print('df_MULTI:\n', df_MULTI)

    print('Exporting to pickle... *****************************')
    df_MULTI.to_pickle(NARR_pkl_out_dir + 'df_MULTI.pkl')
    # EXPORTING TO CSV TAKES SEVERAL MINUTES. IF TIME IS AN ISSUE, DON'T RUN THIS:
    # print('Exporting to CSV... (This could take a minute) ******************************')
    # df_MULTI.to_csv(NARR_csv_out_dir + 'df_MULTI.csv', index=True, header=True)

    # NOTE: Current sample size for Jan 1-14 from SOMPY's point of view is 98 unique maps
    # Need to change the columns these access:
    print('first_datetime_list:\n', first_datetime_list)
    print('last_datetime_list:\n', last_datetime_list)

    return


# I'm not sure why this is here so I've commented it out. I think it
# got accidentally duplicated:
# def import_gridMET_csv(gridMET_csv_in_dir, gridMET_pkl_out_dir):
    # ----------------------------------------------------------------------------
    # PARAMETERS:
    #
    # gridMET_csv_in_dir:       Directory of raw csv gridMET data from gridMET website.
    #
    # gridMET_pkl_out_dir:      Directory to pickle out the gridMET data.
    # 
    # SCRIPT FUNCTION:
    # 
    # 1)    gridMET data exists in csv files for each day of the downloaded year.
    #       This function imports all of those csv files ending in ERC.csv and
    #       concatenates them into df_erc, sets index to lon, lat, time. 
    # 2)    Converts days since Jan 1, 1900 to date format
    # 3)    Exports to df_erc.pkl. No csv export as it's not necessary.
    #
    # NOTE: The exported pickle df_erc.pkl will later be imported by merge_NARR_ERC().
    # ----------------------------------------------------------------------------

    # gridMET csv file names must end with ERC.csv:
    all_gridMET_files = glob.glob(os.path.join(gridMET_csv_in_dir, '*ERC.csv'))
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

    df_erc.to_pickle(gridMET_pkl_out_dir + 'df_erc.pkl')

    return


def merge_NARR_gridMET(start_date, end_date, multi, gridMET_pkl_in_dir, NARR_pkl_in_dir, NARR_gridMET_pkl_out_dir, NARR_gridMET_csv_out_dir):
    # ----------------------------------------------------------------------------
    # PARAMETERS:
    #
    # PURPOSE:                  You have NARR and ERC data for a particular year and would like to merge
    #                           the data into a single file. The two dataset grids don't align, so ERC's
    #                           data is interpolated to the NARR grid so that both datasets have their
    #                           values at the same locatoins. The resulting file for a single year pushes
    #                           an 8 GB RAM limit, so it's recommended that only a single year is merged.
    #
    # start_date:               Beginning of date range to merge. Can be any date as long as it exists
    #                           in both gridMET (ERC) and NARR pickle files. Typically would be 01/02/YEAR
    #                           for the start of the year to merge.
    #
    # end_date:                 End of date range to merge. Can be any date as long as it exists
    #                           in both gridMET (ERC) and NARR pickle files. Typically would be 12/31/YEAR
    #                           for the end of the year to merge.
    #
    # gridMET_pkl_in_dir:       Directory and file name of pickle file containing gridMET data
    #                           data such as daily ERC data.
    #
    # NARR_pkl_in_dir:          Directory that contains all of the NARR pickle files. Example
    #                           NARR pickle file: '00_df_narr.pkl'.
    #
    # NARR_gridMET_pkl_out_dir: Directory to pickle out the combined NARR-gridMET 24hr resolution
    #                           dataframe.
    # 
    # SCRIPT FUNCTION:
    # 
    # 1)    Imports df_erc and df_narr pickle files, converts to dataframes,
    #       sets index of each to time column, aggregates NARR data from 3hrs
    #       to 24hrs to match 24 hour ERC data.
    # 2)    For day D of ERC data, the temporal aggregation is the 24 hour period from
    #       2100 UTC of day D-1 to 1800 UTC of day D.
    # 3)    The merge is then done by interpolating the gridMET grid (4km) to
    #       the NARR grid (32km) using 5 gridMET neighbors for each NARR grid
    #       point. The ERC values at those 5 points are interpolated linearly
    #       to the location of the NARR grid point, resulting in array 'zi',
    #       which is then written to the NARR dataframe as a new column 'ERC'.
    # 4)    The new daily dataframe is plotted and then pickled out.
    # ----------------------------------------------------------------------------


    # Import gridMET ERC data for each day of 1979
    df_gridMET = pd.read_pickle(gridMET_pkl_in_dir + 'df_erc.pkl')
    print('df_gridMET:\n', df_gridMET)

    # Import NARR synoptic variable data for all of 1979
    # NOTE: NARR_pkl_files are all pickle files named 'XX_df_narr_partial.pkl'
    NARR_pkl_files = glob.glob(os.path.join(NARR_pkl_in_dir, '*_df_narr_partial.pkl'))
    # Sorting the files so they go into the dataframe in the correct order
    NARR_pkl_files = sorted(NARR_pkl_files)
    print('NARR_pkl_files:\n', NARR_pkl_files)
    # NARR_pkl_files = NARR_pkl_files[0:2] # Only select the first two
    print('NARR_pkl_files:\n', NARR_pkl_files)
    df_from_each_pickle_file = (pd.read_pickle(file) for file in NARR_pkl_files)
    df_NARR = pd.concat(df_from_each_pickle_file, axis=0)
    print('df_NARR:\n', df_NARR)
    print('df_NARR columns:\n', df_NARR.columns)
    
    # df_NARR currently has time, lat, lon (not lon, lat, time as previously) as indices:
    df_NARR.reset_index(inplace=True)

    # Trim off any NARR data that goes beyond the last time value in gridMET:
    trim = df_gridMET.index.get_level_values(2)[-1]
    # All NARR data rows whose timestamps are before or equal to ERC's last date are kept:
    df_NARR = df_NARR.loc[(df_NARR['time'] <= trim)]
    print('df_NARR after trimming data:\n', df_NARR)

    time_window_hrs = 24 # Average NARR data over 24 hr period to match gridMET

    # gridMET converted to dates. Would add 2100 UTC to gridMET but not necessary
    # because the analysis will be done on daily, not sub-daily, data.
    df_gridMET.reset_index(inplace=True)
    df_gridMET['time'] = df_gridMET['time'].dt.date # Converting time to date
    # Adjusting lon values to match NARR rectilinear grid: e.g. -124.767 + 360 = 235.233
    df_gridMET.lon += 360
    print('df_gridMET: datetimes converted to dates **************************************:\n', df_gridMET)

    # ----------------------------------------------------------------------------
    # Creating a time range list over which averaging is done:

    # Date range strings parsed to datetime objects
    start_dt_object = datetime.strptime(start_date, '%Y,%m,%d')
    end_dt_object = datetime.strptime(end_date, '%Y,%m,%d')
    # print('start_dt_object:\n', start_dt_object)
    # print('end_dt_object:\n', end_dt_object)

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
    # print('timedelta_list:\n', timedelta_list)
    date_list = [td+start_date_object for td in timedelta_list]
    print('date_list:\n', date_list)
    
    # Removes first day. NARR average needs averages across day 1 and 2,
    # thus gridMET data must start on day 2
    date_list = date_list[1:]

    # Creating time range list of tuples used to select a time range for each
    # NARR grid point and then average the data associated with those grid points.
    # NOTE that date_list now starts at day 2.
    time_range_list = [(pd.Timestamp(d)-timedelta(hours=3),pd.Timestamp(d)+timedelta(hours=18)) for d in date_list]
    print('time_range_list:\n', time_range_list)
    # ----------------------------------------------------------------------------

    # Setting the index to time to select all data in a particular time range before
    # grouping on longitude:
    # Selecting a time range is easier with one index. Using a multiindex requires
    # knowing the "coordinates" of the index, e.g. .loc[(first lon, first lat, first time):(last lon, lat lat, last time)]
    # I don't know first and last lon and lat but could get them. Alternatively, just use time index then reset to
    # lon lat time index later.
    df_NARR.set_index('time', inplace=True)

    # time_range_list is a list of tuples of times:
    # time_range_list = [(Timestamp(1979,1,1,21,0,0), Timestamp(1979,1,2,18,0,0)),...]
    df_NARR_all_time_avg = pd.DataFrame([])
    for tr in time_range_list:
        # This loop goes through all of the NARR data and gets each grid point's
        # data for datetime ranges from day-1 2100 UTC to day 1800 UTC and
        # averages that data, then moves onto to the same time range for the
        # next, concatenating the dataframes together.

        # # Running this section prints each of the NARR grid point's
        # # data for the specified time range. There are a lot of grid points,
        # # takes some time to print.
        # # Grouping by longitude after a select time range is selected:
        # for name, group in df_NARR_time_range.groupby('lon'):
        #     print(name)
        #     print(group)
        # df_NARR_time_avg = df_NARR_time_range.groupby('lon').mean()
        
        # time range start and end
        tr_start = tr[0]
        tr_end = tr[1]
        print('tr_start is {}, tr_end is {}\n'.format(tr_start, tr_end))
        # lon lat range

        D = datetime.date(tr_end) # NARR date associated with gridMET date

        # Average time across range, assign a date column to specify date
        df_NARR_time_range = df_NARR.loc[tr_start:tr_end]
        # print('df_NARR_time_range:\n', df_NARR_time_range)

        df_NARR_time_range.reset_index(inplace=True)
        cols = df_NARR_time_range.columns
        if multi == True:
            df_NARR_time_range[['lat','lon','H500','H500 Grad X','H500 Grad Y','PVEL','SPFH','CWTR','PMSL','PMSL Grad X','PMSL Grad Y','CAPE']] = df_NARR_time_range[['lat','lon','H500','H500 Grad X','H500 Grad Y','PVEL','SPFH','CWTR','PMSL','PMSL Grad X','PMSL Grad Y','CAPE']].apply(pd.to_numeric)
        else:
            df_NARR_time_range[['lat','lon','H500','H500 Grad X','H500 Grad Y','PMSL','PMSL Grad X','PMSL Grad Y','CAPE']] = df_NARR_time_range[['lat','lon','H500','H500 Grad X','H500 Grad Y','PMSL','PMSL Grad X','PMSL Grad Y','CAPE']].apply(pd.to_numeric)
        
        # print('df_NARR_time_range after resetting index and making columns numeric:\n', df_NARR_time_range)
        df_NARR_time_range['time'] = pd.to_datetime(df_NARR_time_range['time'])
        df_NARR_time_range.set_index(['lat','lon','time'], inplace=True)
        # print('df_NARR_time_range after setting index to lat, lon, time (not lon, lat, time as previously):\n', df_NARR_time_range)
        # ---------------------------------------------------------------------------------
        # WARNING: It might complain that 233.000 doesn't exist:
        print('df_NARR_time_range.loc[39.0, 233.000]:\n', df_NARR_time_range.loc[39.0,233.000]) # [242.941,42.3137] is very close to Medford, OR
        # ---------------------------------------------------------------------------------
        print('* * * * * THE SCRIPT MAY FAIL HERE with the error "no numeric types to aggregate."\nCheck that the end date exists in the last csv file made in import_NARR_csv()')
        df_NARR_time_avg = df_NARR_time_range.groupby(['lat','lon']).mean()
        # print('df_NARR_time_avg before assigning day D to time column:\n', df_NARR_time_avg)
        df_NARR_time_avg['time'] = D
        # print('df_NARR_time_avg after assigning day D to time column:\n', df_NARR_time_avg)
        
        df_NARR_all_time_avg = pd.concat((df_NARR_all_time_avg, df_NARR_time_avg), axis=0)

        # This is being replaced by the concatenation above. Remember to remove enumeration at start of loop
        # if i == 0: # First time through loop, append df_NARR_date to columns
        #     # When i = 0, all H500 files in df are processed:
        #     df_NARR_all_time_avg = df_NARR_all_time_avg.append(df_NARR_time_avg)
        #     # print('First df_NARR_all_time_avg concatenation:\n', df_NARR_all_time_avg)
        # else: # Concat df_NARR_date to rows of df_NARR_ERC
        #     df_NARR_all_time_avg = pd.concat((df_NARR_all_time_avg, df_NARR_time_avg), axis=0)

        #     # print('Second df_NARR_all_time_avg concatenation:\n', df_NARR_time_avg)
        #     # print('df_NARR_time_avg.columns:\n', df_NARR_time_avg.columns)
        print('***************************** Analyzing {} - {} *****************************'.format(tr_start,tr_end))

    df_NARR = df_NARR_all_time_avg
    
    # Right now time (not lon, as previously) is the index value, reset it
    df_NARR.reset_index(inplace=True)
    # print('df_NARR after time range averaging:\n', df_NARR)

    df_NARR_ERC = pd.DataFrame([])

    for d in date_list:
        # -------------------------------------------------------------------
        # This loop goes through every date in both dataframes and interpolates gridMET ERC
        # values to the NARR grid using nearest neighbor interpolation.
        # df_gridMET and df_NARR matching the date are copied to
        # df_plot_date and df_NARR_date. Interpolation is performed
        # on these dataframes.
        # The interpolated ERC values are copied to df_NARR_date['ERC']
        # and then concatenated to df_NARR_ERC.
        #
        # df_NARR and df_gridMET have no indices in this loop.
        # -------------------------------------------------------------------

        df_plot_date = df_gridMET[df_gridMET['time'] == d]
        df_NARR_date = df_NARR[df_NARR['time'] == d]
        # print('df_plot_date:\n', df_plot_date)
        # print('df_NARR_date:\n', df_NARR_date)
        # print('df_plot_date shape:\n', np.shape(df_plot_date.values))
        # print('df_NARR_date shape:\n', np.shape(df_NARR_date.values))

        # -------------------------------------------------------------------
        # # PLOT 1: Plotting NARR grid over gridMET grid:
        # plt.figure()
        # plt.scatter(x=df_plot_date.lon, y=df_plot_date.lat, c='white', s=8, marker='o', edgecolors='g', label='gridMET grid')
        # plt.scatter(x=df_NARR_date.lon, y=df_NARR_date.lat, c='k', s=8, marker='+', edgecolors='g', label='NARR grid')
        # plt.xlabel('Longitude, deg'); plt.ylabel('Latitude, deg')
        # plt.title('NARR grid after removing out-of-bounds points')
        # plt.legend()
        # plt.savefig('NARR_gridMET_complete_grids.png', bbox_inches='tight')
        # plt.show()
        # -------------------------------------------------------------------

        # Changing invalid data points to values the interpolation
        # algorithm can interpolate to:
        ########################## df_plot_date.replace(-32767, 1.0123456789, inplace=True)

        # Clip NARR grid to min and max lon lat values in the gridMET grid
        x_min = min(df_plot_date['lon'].values); x_max = max(df_plot_date['lon'].values)
        y_min = min(df_plot_date['lat'].values); y_max = max(df_plot_date['lat'].values)
        # print('x_min:\n', x_min)
        # print('x_max:\n', x_max)
        # print('y_min:\n', y_min)
        # print('y_max:\n', y_max)

        # Select all rows that are inside the lon-lat window of the ERC dataset:
        criteria = (x_max >= df_NARR_date['lon']) & (df_NARR_date['lon'] >= x_min) & (y_max >= df_NARR_date['lat']) & (df_NARR_date['lat'] >= y_min)
        # print('NARR rows before cutting out-of-bounds lon-lat points:', df_NARR_date.count())        
        df_NARR_date = df_NARR_date.loc[criteria]
        # print('NARR rows after cutting out-of-bounds lon-lat points:', df_NARR_date.count())
        # print('df_NARR_date after removing out-of-bounds Lon-Lat points:\n', df_NARR_date)

        # -------------------------------------------------------------------
        # # PLOT 2: Plotting NARR grid where it overlaps with gridMET:
        # plt.figure()
        # plt.scatter(x=df_plot_date.lon, y=df_plot_date.lat, c='white', s=8, marker='o', edgecolors='g', label='gridMET grid')
        # plt.scatter(x=df_NARR_date.lon, y=df_NARR_date.lat, c='k', s=8, marker='+', edgecolors='g', label='NARR grid')
        # plt.xlabel('Longitude, deg'); plt.ylabel('Latitude, deg')
        # plt.title('NARR grid after removing out-of-bounds points')
        # plt.legend()
        # plt.savefig('NARR_gridMET_complete_grids.png', bbox_inches='tight')
        # plt.show()
        # -------------------------------------------------------------------

        # Define x y and z for interpolation:
        x = df_plot_date.lon.values
        y = df_plot_date.lat.values
        z = df_plot_date.erc.values
        xi = df_NARR_date.lon.values
        yi = df_NARR_date.lat.values
        # print('ERC values:\n x:\n{}\n y:\n{}\n z:\n{}\n'.format(x, y, z))
        # print('NARR values:\n xi:\n{}\n yi:\n{}\n'.format(xi,yi))
        # print('x shape:\n{}\n y shape:\n{}\n z shape:\n{}\n'.format(np.shape(x), np.shape(y), np.shape(z)))
        # print('xi shape:\n{}\n yi shape:\n{}\n'.format(np.shape(xi),np.shape(yi)))
        
        # Interpolation:
        would_you_be_my_neighbor = 5
        gridMET_shape = np.shape(df_plot_date.values[:,0:2])
        NARR_shape = np.shape(df_NARR_date.values[:,0:2])
        tree = neighbors.KDTree(df_plot_date.values[:,0:2], leaf_size=2)
        dist, ind = tree.query(df_NARR_date.values[:,0:2], k=would_you_be_my_neighbor)
        # print('gridMET shape:', gridMET_shape)
        # print('NARR shape:', NARR_shape)
        # print('indices:', ind)  # indices of 3 closest neighbors
        # print('distances:', dist)  # distances to 3 closest neighbors
        # print('df_NARR_date with ERC rounded to nearest int and invalid rows removed:\n', df_NARR_date)

        # Create ERC data (zi) from interpolated grid
        zi = griddata((x,y),z,(xi,yi),method='nearest')
        # print('zi:\n', zi)
        # print('zi shape:\n{}\n'.format(np.shape(zi)))

        # Plotting before and after interpolation of gridMET ERC to NARR grid:
        # Plots gridMET grid before and after interpolation. It uses gridMET grid,
        # so the plotting is done here rather than inside of plot_NARR_ERC() which
        # already has the gridMET ERC data within df_NARR_date_ERC.

        if d == date_list[-1]: # Only plot if on last date in date_list
            plt.close()
            plt.figure()
            plt.scatter(x=df_plot_date.lon, y=df_plot_date.lat, color='white', marker='o', edgecolors='g', s=df_plot_date.erc/3, label='gridMET')
            plt.scatter(x=df_plot_date.lon.iloc[np.ravel(ind)], y=df_plot_date.lat.iloc[np.ravel(ind)], color='r', marker='x', s=7, label='nearest gridMET')
            plt.scatter(x=df_NARR_date.lon, y=df_NARR_date.lat, color='k', marker='+', s=7, label='NARR')
            plt.xlabel('Longitude, deg'); plt.ylabel('Latitude, deg')
            plt.title('Nearest gridMET points using interpolated indices')
            plt.legend()
            plt.savefig('NARR_gridMET_before_interp.png', bbox_inches='tight')
            plt.show()

            plt.scatter(x=df_plot_date.lon, y=df_plot_date.lat, color='white', marker='o', edgecolors='g', s=df_plot_date.erc/3, label='gridMET')
            plt.scatter(x=df_plot_date.lon.iloc[np.ravel(ind)], y=df_plot_date.lat.iloc[np.ravel(ind)], color='r', marker='x', s=7, label='nearest gridMET')
            plt.scatter(x=xi, y=yi, color='y', edgecolors='y', alpha=0.6, marker='o', s=zi, label='interp NARR')
            plt.scatter(x=df_NARR_date.lon, y=df_NARR_date.lat, color='k', marker='+', s=7, label='NARR')
            plt.xlabel('Longitude, deg'); plt.ylabel('Latitude, deg')
            plt.title('Interpolated ERC values')
            plt.legend()
            plt.savefig('NARR_gridMET_after_interp.png', bbox_inches='tight')
            plt.show()

        # Add interpolated ERC values (contained in list zi) to a new df_NARR_date column.
        # This is where the merge takes place, no need to align on indices using df.merge().
        # There are no indices to align merge_NARRon anyways because zi was created with the same lon-lat
        # order as the NARR data:
        df_NARR_date['ERC'] = zi
        # print('df_NARR_date with ERC:\n', df_NARR_date)

        df_NARR_ERC = pd.concat((df_NARR_ERC, df_NARR_date), axis=0)
        # Replaced by concatenation above:
        # if i == 0: # First time through loop, append df_NARR_date to columns
        #     # When i = 0, all H500 files in df are processed:
        #     df_NARR_ERC = df_NARR_ERC.append(df_NARR_date)
        #     # print('First df_NARR_ERC concatenation:\n', df_NARR_ERC)
        #     print('***************************** Analyzing {} *****************************'.format(d))
        # else: # Concat df_NARR_date to rows of df_NARR_ERC
        #     df_NARR_ERC = pd.concat((df_NARR_ERC, df_NARR_date), axis=0)
        #     # print('Second df_NARR_ERC concatenation:\n', df_NARR_date)
        #     # print('df_NARR_date.columns:\n', df_NARR_date.columns)
        #     print('***************************** Analyzing {} *****************************'.format(d))
        print('***************************** Analyzing {} *****************************'.format(d))

    # Getting lon, lat, time all on the left hand side so order is correct in the
    # csv for Julia import
    df_NARR_ERC.set_index(['lat','lon','time'], inplace=True) # Not ['lon','lat','time'] as previously
    df_NARR_ERC.reset_index(inplace=True)
    # Remove invalid values (fill values) all of which are -32767:
    # print('df_NARR rows before rounding ERC:\n', df_NARR_ERC.count())
    # REMOVING THIS PROCESS: ##############################
    # df_NARR_ERC = df_NARR_ERC[df_NARR_ERC['ERC'] > 0]
    #######################################################
    df_NARR_ERC = df_NARR_ERC.round({'ERC':0})
    print('df_NARR_ERC row count after rounding ERC:\n', df_NARR_ERC.count())
    print('df_NARR_ERC rows after rounding ERC to nearest integer and removing invalid values:\n', df_NARR_ERC)

    # Index values (lon and lat) have carry out to many decimals (e.g. 42.058800000000000002)
    # Round to four decimals
    n_decimals = 4
    # df_NARR_ERC.reset_index(inplace=True)
    df_NARR_ERC['lat'] = df_NARR_ERC['lat'].apply(lambda x: round(x, n_decimals))
    df_NARR_ERC['lon'] = df_NARR_ERC['lon'].apply(lambda x: round(x, n_decimals))
    # df_NARR_ERC.set_index(['lat','lon'], inplace=True)
    print('df_NARR_ERC after rounding lat lon values:\n', df_NARR_ERC)

    # Bin ERC values. Export below as df_NARR_ERC_categorical:
    erc_bins = [-32768,-1,19,27,35,44,500]
    erc_labels = ['invalid','low','moderate','high','very high','extreme']
    # Cutting returns a series with categorical ERC values
    s_ERC_categorical = pd.cut(df_NARR_ERC['ERC'], bins=erc_bins, labels=erc_labels)
    print('s_ERC_categorical:\n', s_ERC_categorical)
    # Concatenate df_NARR_ERC (minus its ERC data) to the categorical ERC data
    df_NARR_ERC_categorical = pd.concat((df_NARR_ERC.drop('ERC', axis=1), s_ERC_categorical), axis=1)
    print('df_NARR_ERC_categorical:\n', df_NARR_ERC_categorical)

    print('Exporting continuous and categorical NARR ERC data to pickle and csv... **************************')
    # Pickle export:
    df_NARR_ERC.to_pickle(NARR_gridMET_pkl_out_dir + 'df_NARR_ERC.pkl')
    df_NARR_ERC.to_csv(NARR_gridMET_csv_out_dir + 'df_NARR_ERC.csv', header=True, index=False) # Includes index columns, names all columns at top of file
    
    # CSV export:
    df_NARR_ERC_categorical.to_pickle(NARR_gridMET_pkl_out_dir + 'df_NARR_ERC_categorical.pkl')
    df_NARR_ERC_categorical.to_csv(NARR_gridMET_csv_out_dir + 'df_NARR_ERC_categorical.csv', header=True, index=False) # Includes index columns, names all columns at top of file
    
    return


def stack_NARR_gridMET(NARR_gridMET_pkl_in_dir, NARR_gridMET_pkl_out_dir, NARR_gridMET_csv_out_dir):
    # Might build one single large pickle and csv file
    # for all years' of data. I would prefer to keep this
    # functionality here, but it could be put into
    # merge_NARR_gridMET later.
    # Note that build_monthly_NARR_gridMET below
    # does a build of all years' of data for each
    # individual month as well as a single large csv
    # file of all years' data. The creation of one
    # file containing all data can be deleted at some
    # point as this function (stack_NARR_gridMET) will
    # do this here.

    # ----------------------------------------------------------------------------
    # NOTE: After running merge_NARR_gridMET(), go into /FWP/NARR_gridMET/, pull out
    #       all df_NARR_ERC_categorical.pkl files from their year folders,
    #       rename to df_NARR_ERC_categorical_1979.pkl, etc for every year, and put
    #       into /FWP/NARR_gridMET/. Then run this function to stack all years'
    #       data into one df and export as a pickle and csv file.
    #
    # PARAMETERS:
    #
    # NARR_gridMET_pkl_in_dir:      Directory of pickle NARR-gridMET directories
    #                               containing each year of data.
    #
    # NARR_gridMET_pkl_out_dir:     Directory of export for all years of NARR-gridMET
    #                               data in pkl format.
    #
    # NARR_gridMET_csv_out_dir:     Directory of export for all years of NARR-gridMET
    #                               data in csv format.
    # 
    # SCRIPT FUNCTION:
    # 
    # 1)    NARR-gridMET data exists in the directory /FWP/NARR_gridMET.
    #       This function imports all pickle files ending in *YEAR.pkl and
    #       joins them into df_NARR_gridMET_all_years containing every year
    #       of NARR-gridMET data (this dataset can be massive).
    #
    # 2)    If any of the superfluous columns are present: TEMP, SPFH, TEMP, or
    #       CWTR, they are dropped.
    #
    # 3)    
    #           
    # 4)    Exports to the same directory they were imported from and is placed in:
    #           /Training/Categorical
    #           /Training/Continuous
    #       depending on whether the ERC data is in categorical or continuous format.
    #       Exports in pickle and csv formats.
    #
    # NOTE: The exported pickle files are for backup purposes and to import for
    #       processing in Python if necessary.
    # ----------------------------------------------------------------------------

    # Import NARR-gridMET data for all years
    # NOTE: NARR_pkl_files are all pickle files named 'df_NARR_ERC_XXXX.pkl' (XXXX = year)
    continuous_path = NARR_gridMET_pkl_in_dir + 'Continuous/'
    categorical_path = NARR_gridMET_pkl_in_dir + 'Categorical/'

    NARR_gridMET_cont_pkl_files = glob.glob(os.path.join(continuous_path, '*.pkl'))
    NARR_gridMET_cat_pkl_files = glob.glob(os.path.join(categorical_path, '*.pkl'))

    print('NARR_gridMET_cont_pkl_files:\n', NARR_gridMET_cont_pkl_files)
    print('NARR_gridMET_cat_pkl_files:\n', NARR_gridMET_cat_pkl_files)

    # Sorting the files so they go into the dataframe in the correct order
    NARR_gridMET_cont_pkl_files = sorted(NARR_gridMET_cont_pkl_files)
    NARR_gridMET_cat_pkl_files = sorted(NARR_gridMET_cat_pkl_files)
    print('NARR_gridMET_cont_pkl_files after sort:\n', NARR_gridMET_cont_pkl_files)
    print('NARR_gridMET_cat_pkl_files after sort:\n', NARR_gridMET_cat_pkl_files)

    # Read in each year's pickle file and concatenate them:
    df_cont = (pd.read_pickle(file) for file in NARR_gridMET_cont_pkl_files)
    df_cat = (pd.read_pickle(file) for file in NARR_gridMET_cat_pkl_files)

    # Concatenate:
    df_NARR_gridMET_cont_all_years = pd.concat(df_cont, axis=0, sort=False)
    df_NARR_gridMET_cat_all_years = pd.concat(df_cat, axis=0, sort=False)
    print('df_NARR_gridMET_cont_all_years:\n', df_NARR_gridMET_cont_all_years)
    print('df_NARR_gridMET_cont_all_years columns:\n', df_NARR_gridMET_cont_all_years.columns)

    # Drop superfluous columns:
    print('Dropping multi columns...\n')
    df_NARR_gridMET_cont_all_years.drop(['CAPE','TEMP','SPFH','PVEL','CWTR'], axis=1, inplace=True)
    df_NARR_gridMET_cat_all_years.drop(['CAPE','TEMP','SPFH','PVEL','CWTR'], axis=1, inplace=True)

    # ----------------------------------------

    # NaN Check:
    # Column NaNs
    cont_column_nans = df_NARR_gridMET_cont_all_years.isnull().sum(axis=0)
    cat_column_nans = df_NARR_gridMET_cat_all_years.isnull().sum(axis=0)
    # All NaNs:
    cont_nans = df_NARR_gridMET_cont_all_years.isnull().sum().sum()
    cat_nans = df_NARR_gridMET_cat_all_years.isnull().sum().sum()
    print('cont_nans:\n', cont_nans)
    if cont_nans > 0 or cat_nans > 0:
        print('Continuous NaNs:\n', cont_column_nans)
        print('Categorical NaNs:\n', cat_column_nans)
        print('Warning: NaNs are present, terminating function.')
        return
    else:
        print('NaN check: Data is free of missing values')

    # ----------------------------------------

    # Sorting by lat and lon:
    print('Sorting values...\n')
    df_NARR_gridMET_cont_all_years.sort_values(['lat','lon'], inplace=True)
    df_NARR_gridMET_cat_all_years.sort_values(['lat','lon'], inplace=True)
    print('df_NARR_gridMET_cont_all_years lat, lon index:\n', df_NARR_gridMET_cont_all_years)
    print('df_NARR_gridMET_cat_all_years lat, lon index:\n', df_NARR_gridMET_cat_all_years)

    # Convert time column to datetime:
    df_NARR_gridMET_cont_all_years['time'] = pd.to_datetime(df_NARR_gridMET_cont_all_years['time'])

    # Set lat and lon as index:
    print('Setting index...\n')
    df_NARR_gridMET_cont_all_years.set_index(['lat','lon'], inplace=True)
    df_NARR_gridMET_cat_all_years.set_index(['lat','lon'], inplace=True)

    print('df_NARR_gridMET_cont_all_years lat, lon index:\n', df_NARR_gridMET_cont_all_years)
    print('df_NARR_gridMET_cat_all_years lat, lon index:\n', df_NARR_gridMET_cat_all_years)

    # ----------------------------------------
    
    # PLOTTING:
    # Get one location for plotting:
    df_NARR_gridMET_cont_all_years_loc = df_NARR_gridMET_cont_all_years.loc[(42.0588, 236.0590), :]
    df_NARR_gridMET_cont_all_years_loc.reset_index(inplace=True)

    fig, (ax1, ax2) = plt.subplots(1,2, figsize=(9,4))
    df_NARR_gridMET_cont_all_years_loc.plot(x='time', y='H500', ax=ax1, color='cyan')
    df_NARR_gridMET_cont_all_years_loc.plot(x='time', y='ERC', ax=ax2, color='red')
    ax1.set_title('H500')
    ax2.set_title('ERC')
    plt.suptitle('Kalmiopsis Wilderness')
    plt.show()
    
    # ----------------------------------------

    # PKL EXPORT:
    # Export all years' data to pickle format:
    cont_pkl_export_file = NARR_gridMET_pkl_out_dir + 'df_NARR_gridMET_all_years.pkl'
    cat_pkl_export_file = NARR_gridMET_pkl_out_dir + 'df_NARR_gridMET_categorical_all_years.pkl'

    print('Exporting to pickle the continuous and categorical data for all years. This could take a few seconds...')
    df_NARR_gridMET_cont_all_years.to_pickle(cont_pkl_export_file)
    df_NARR_gridMET_cat_all_years.to_pickle(cat_pkl_export_file)
    print('Finished pickle export')
    
    # ----------------------------------------

    # CSV EXPORT:
    # Export all years' data to csv format:
    cont_csv_export_file = NARR_gridMET_csv_out_dir + 'df_NARR_gridMET_all_years.csv'
    cat_csv_export_file = NARR_gridMET_csv_out_dir + 'df_NARR_gridMET_categorical_all_years.csv'

    print('Exporting to csv the continuous and categorical data for all years. This could take a few minutes...')
    df_NARR_gridMET_cont_all_years.to_csv(cont_csv_export_file)
    df_NARR_gridMET_cat_all_years.to_csv(cat_csv_export_file)
    print('Finished csv export')

    return


def plot_NARR_gridMET(plot_date, multi, plot_lon, plot_lat, NARR_gridMET_pkl_in_dir):
    # -------------------------------------------------------------------
    # This function makes the following:
    # 1)    Contour plot containing tripcolour and tricontourf subplots
    #       of ERC data on a single date
    # 2)    Time series of synoptic variable data for a single lon-lat point
    #       over the dataframe's entire time range
    # -------------------------------------------------------------------

    df_NARR_ERC = pd.read_pickle(NARR_gridMET_pkl_in_dir + 'df_NARR_ERC.pkl')
    df_NARR_ERC_categorical = pd.read_pickle(NARR_gridMET_pkl_in_dir + 'df_NARR_ERC_categorical.pkl')
    print('df_NARR_ERC:\n', df_NARR_ERC)
    print('df_NARR_ERC_categorical:\n', df_NARR_ERC_categorical)

    # WARNING: These x,y,t values are for all days in df_NARR_ERC.
    # Currently, df_NARR_ERC only covers Jan 1, 1979, and this
    # means the plot below that uses these values works. It won't
    # work when there are multiple dates.
    x = df_NARR_ERC.lon.values.tolist()     # Longitude
    y = df_NARR_ERC.lat.values.tolist()     # Latitude
    t = df_NARR_ERC.time.tolist()    # Time
    # print('x values from df_NARR_ERC.lon:\n', x)
    # print('y values from df_NARR_ERC.lat:\n', y)
    # print('t values from df_NARR_ERC.time:\n', t)

    # Getting z values and building new dataframe with time index and ERC data.
    z = df_NARR_ERC['ERC'].values.tolist()
    d = [i for i in zip(t,x,y,z)]
    # print('d:\n', d)
    df = pd.DataFrame(data=d, columns=['time','lon','lat','ERC'])
    df.set_index('time', inplace=True)
    print('df.index:\n', df.index[0:100])

    # Convert timepoint to year,month,date format for building contour plot from ERC data
    plot_date = datetime.strptime(plot_date, '%Y,%m,%d')
    print('plot_date:', plot_date)
    # Get the ERC data for the day specified
    df_t = df[(df.index == plot_date)]
    # Replace ERC = -32767 values with -1:
    df_t['ERC'].replace(-32767, -1, inplace=True)
    print('df_t, -32767 replaced with -1:\n', df_t)
    # Split into constituents
    x_t = df_t['lon'].values.tolist()
    y_t = df_t['lat'].values.tolist()
    z_t = df_t['ERC'].values.tolist()
    # print('Shape x_t:\n', np.shape(x_t))
    # print('Shape y_t:\n', np.shape(y_t))
    # print('Shape z_t:\n', np.shape(z_t))

    # -------------------------------------------------------------------
    # Contour Plots: ERC
    plt.close()
    f, ax = plt.subplots(1,2, figsize=(8,3), sharex=True, sharey=True)

    ax[0].tripcolor(x_t, y_t, z_t, 30, cmap=cm.jet) # Plots across all timepoints?
    ax[0].plot(x_t, y_t, 'ko ', markersize=1)
    ax[0].set_xlabel('Longitude'); ax[0].set_ylabel('Latitude')

    tcf = ax[1].tricontourf(x_t, y_t, z_t, 30, cmap=cm.jet) # 20 contour levels is good quality
    ax[1].plot(x_t, y_t, 'ko ', markersize=1)
    ax[1].set_xlabel('Longitude'); ax[1].set_ylabel('Latitude')
    f.colorbar(tcf)

    date_str = plot_date.strftime('%b %d, %Y')
    plt.suptitle('ERC Contour Plots: '+date_str)
    plt.savefig('ERC_contour.png', bbox_inches='tight')
    plt.show()
    # -------------------------------------------------------------------

    # Creating a dataframe at one latitude-longitude point across all time points:
    df_NARR_ERC.set_index(['lat','lon'], inplace=True)
    print('df_NARR_ERC with lat lon index:\n', df_NARR_ERC)
    # Specifying a longitude point, make this an
    # actual longitude value in the future:
    print('df_NARR_ERC before sorting:\n', df_NARR_ERC)
    # df_NARR_ERC.sort_index(level=(0,1), inplace=True)
    # print('df_NARR_ERC after sorting:\n', df_NARR_ERC)

    # Making one dataframe for all locations:
    all_locations = [(x,y) for x,y in zip(plot_lat,plot_lon)]
    df_NARR_ERC_lon_lat_time_series = df_NARR_ERC.loc[all_locations]
    print('df_NARR_ERC_lon_lat_time_series:\n', df_NARR_ERC_lon_lat_time_series)
    df_NARR_ERC_lon_lat_time_series.reset_index(drop=True, inplace=True)
    df_NARR_ERC_lon_lat_time_series['time'] = pd.to_datetime(df_NARR_ERC_lon_lat_time_series['time'])
    df_NARR_ERC_lon_lat_time_series.set_index('time', inplace=True)
    df_NARR_ERC_lon_lat_time_series.loc[:,'H500 SD']        = df_NARR_ERC_lon_lat_time_series.loc[:,'H500'].rolling(30, closed='neither').std()
    df_NARR_ERC_lon_lat_time_series.loc[:,'PMSL SD']        = df_NARR_ERC_lon_lat_time_series.loc[:,'PMSL'].rolling(30, closed='neither').std()
    df_NARR_ERC_lon_lat_time_series.loc[:,'H500 Grad X MA'] = df_NARR_ERC_lon_lat_time_series.loc[:,'H500 Grad X'].rolling(30, closed='neither').mean()
    df_NARR_ERC_lon_lat_time_series.loc[:,'H500 Grad Y MA'] = df_NARR_ERC_lon_lat_time_series.loc[:,'H500 Grad Y'].rolling(30, closed='neither').mean()
    df_NARR_ERC_lon_lat_time_series.loc[:,'PMSL Grad X MA'] = df_NARR_ERC_lon_lat_time_series.loc[:,'PMSL Grad X'].rolling(30, closed='neither').mean()
    df_NARR_ERC_lon_lat_time_series.loc[:,'PMSL Grad Y MA'] = df_NARR_ERC_lon_lat_time_series.loc[:,'PMSL Grad Y'].rolling(30, closed='neither').mean()
    df_NARR_ERC_lon_lat_time_series.loc[:,'CAPE MA']        = df_NARR_ERC_lon_lat_time_series.loc[:,'CAPE'].rolling(30, closed='neither').mean()


    # OR time series (cumulative sums):
    df_NARR_ERC_lon_lat_time_series_0 = df_NARR_ERC.loc[(plot_lat[0], plot_lon[0])]
    df_NARR_ERC_lon_lat_time_series_0.reset_index(drop=True, inplace=True)
    df_NARR_ERC_lon_lat_time_series_0['time'] = pd.to_datetime(df_NARR_ERC_lon_lat_time_series_0['time'])
    df_NARR_ERC_lon_lat_time_series_0.set_index('time', inplace=True)
    # df_NARR_ERC_lon_lat_time_series_0.loc[:,'H500 Adj CS']    = pd.Series(df_NARR_ERC_lon_lat_time_series_0.loc[:,'H500']-5500).cumsum(axis=0)
    # df_NARR_ERC_lon_lat_time_series_0.loc[:,'PMSL Adj CS']    = pd.Series(df_NARR_ERC_lon_lat_time_series_0.loc[:,'PMSL']-101325).cumsum(axis=0)
    # df_NARR_ERC_lon_lat_time_series_0.loc[:,'H500 Grad X CS'] = df_NARR_ERC_lon_lat_time_series_0.loc[:,'H500 Grad X'].cumsum(axis=0)
    # df_NARR_ERC_lon_lat_time_series_0.loc[:,'H500 Grad Y CS'] = df_NARR_ERC_lon_lat_time_series_0.loc[:,'H500 Grad Y'].cumsum(axis=0)
    # df_NARR_ERC_lon_lat_time_series_0.loc[:,'PMSL Grad X CS'] = df_NARR_ERC_lon_lat_time_series_0.loc[:,'PMSL Grad X'].cumsum(axis=0)
    # df_NARR_ERC_lon_lat_time_series_0.loc[:,'PMSL Grad Y CS'] = df_NARR_ERC_lon_lat_time_series_0.loc[:,'PMSL Grad Y'].cumsum(axis=0)
    # df_NARR_ERC_lon_lat_time_series_0.loc[:,'CAPE CS']        = df_NARR_ERC_lon_lat_time_series_0.loc[:,'CAPE'].cumsum(axis=0)
    df_NARR_ERC_lon_lat_time_series_0.loc[:,'H500 SD']        = df_NARR_ERC_lon_lat_time_series_0.loc[:,'H500'].rolling(30, closed='neither').std()
    df_NARR_ERC_lon_lat_time_series_0.loc[:,'PMSL SD']        = df_NARR_ERC_lon_lat_time_series_0.loc[:,'PMSL'].rolling(30, closed='neither').std()
    df_NARR_ERC_lon_lat_time_series_0.loc[:,'H500 Grad X MA'] = df_NARR_ERC_lon_lat_time_series_0.loc[:,'H500 Grad X'].rolling(30, closed='neither').mean()
    df_NARR_ERC_lon_lat_time_series_0.loc[:,'H500 Grad Y MA'] = df_NARR_ERC_lon_lat_time_series_0.loc[:,'H500 Grad Y'].rolling(30, closed='neither').mean()
    df_NARR_ERC_lon_lat_time_series_0.loc[:,'PMSL Grad X MA'] = df_NARR_ERC_lon_lat_time_series_0.loc[:,'PMSL Grad X'].rolling(30, closed='neither').mean()
    df_NARR_ERC_lon_lat_time_series_0.loc[:,'PMSL Grad Y MA'] = df_NARR_ERC_lon_lat_time_series_0.loc[:,'PMSL Grad Y'].rolling(30, closed='neither').mean()
    df_NARR_ERC_lon_lat_time_series_0.loc[:,'CAPE MA']        = df_NARR_ERC_lon_lat_time_series_0.loc[:,'CAPE'].rolling(30, closed='neither').mean()
    print('df_NARR_ERC_lon_lat_time_series_0:\n', df_NARR_ERC_lon_lat_time_series_0.to_string())

    df_NARR_ERC_lon_lat_time_series_1 = df_NARR_ERC.loc[(plot_lat[1], plot_lon[1])]
    df_NARR_ERC_lon_lat_time_series_1.reset_index(drop=True, inplace=True)
    df_NARR_ERC_lon_lat_time_series_1['time'] = pd.to_datetime(df_NARR_ERC_lon_lat_time_series_1['time'])
    df_NARR_ERC_lon_lat_time_series_1.set_index('time', inplace=True)
    # df_NARR_ERC_lon_lat_time_series_1.loc[:,'H500 Adj CS']        = pd.Series(df_NARR_ERC_lon_lat_time_series_1.loc[:,'H500']-5500).cumsum(axis=0)
    # df_NARR_ERC_lon_lat_time_series_1.loc[:,'PMSL Adj CS']        = pd.Series(df_NARR_ERC_lon_lat_time_series_1.loc[:,'PMSL']-101325).cumsum(axis=0)
    # df_NARR_ERC_lon_lat_time_series_1.loc[:,'H500 Grad X CS'] = df_NARR_ERC_lon_lat_time_series_1.loc[:,'H500 Grad X'].cumsum(axis=0)
    # df_NARR_ERC_lon_lat_time_series_1.loc[:,'H500 Grad Y CS'] = df_NARR_ERC_lon_lat_time_series_1.loc[:,'H500 Grad Y'].cumsum(axis=0)
    # df_NARR_ERC_lon_lat_time_series_1.loc[:,'PMSL Grad X CS'] = df_NARR_ERC_lon_lat_time_series_1.loc[:,'PMSL Grad X'].cumsum(axis=0)
    # df_NARR_ERC_lon_lat_time_series_1.loc[:,'PMSL Grad Y CS'] = df_NARR_ERC_lon_lat_time_series_1.loc[:,'PMSL Grad Y'].cumsum(axis=0)
    # df_NARR_ERC_lon_lat_time_series_1.loc[:,'CAPE CS']           = df_NARR_ERC_lon_lat_time_series_1.loc[:,'CAPE'].cumsum(axis=0)
    df_NARR_ERC_lon_lat_time_series_1.loc[:,'H500 SD']        = df_NARR_ERC_lon_lat_time_series_1.loc[:,'H500'].rolling(30, closed='neither').std()
    df_NARR_ERC_lon_lat_time_series_1.loc[:,'PMSL SD']        = df_NARR_ERC_lon_lat_time_series_1.loc[:,'PMSL'].rolling(30, closed='neither').std()
    df_NARR_ERC_lon_lat_time_series_1.loc[:,'H500 Grad X MA'] = df_NARR_ERC_lon_lat_time_series_1.loc[:,'H500 Grad X'].rolling(30, closed='neither').mean()
    df_NARR_ERC_lon_lat_time_series_1.loc[:,'H500 Grad Y MA'] = df_NARR_ERC_lon_lat_time_series_1.loc[:,'H500 Grad Y'].rolling(30, closed='neither').mean()
    df_NARR_ERC_lon_lat_time_series_1.loc[:,'PMSL Grad X MA'] = df_NARR_ERC_lon_lat_time_series_1.loc[:,'PMSL Grad X'].rolling(30, closed='neither').mean()
    df_NARR_ERC_lon_lat_time_series_1.loc[:,'PMSL Grad Y MA'] = df_NARR_ERC_lon_lat_time_series_1.loc[:,'PMSL Grad Y'].rolling(30, closed='neither').mean()
    df_NARR_ERC_lon_lat_time_series_1.loc[:,'CAPE MA']        = df_NARR_ERC_lon_lat_time_series_1.loc[:,'CAPE'].rolling(30, closed='neither').mean()

    df_NARR_ERC_lon_lat_time_series_2 = df_NARR_ERC.loc[(plot_lat[2], plot_lon[2])]
    df_NARR_ERC_lon_lat_time_series_2.reset_index(drop=True, inplace=True)
    df_NARR_ERC_lon_lat_time_series_2['time'] = pd.to_datetime(df_NARR_ERC_lon_lat_time_series_2['time'])
    df_NARR_ERC_lon_lat_time_series_2.set_index('time', inplace=True)
    # df_NARR_ERC_lon_lat_time_series_2.loc[:,'H500 Adj CS']        = pd.Series(df_NARR_ERC_lon_lat_time_series_2.loc[:,'H500']-5500).cumsum(axis=0)
    # df_NARR_ERC_lon_lat_time_series_2.loc[:,'PMSL Adj CS']        = pd.Series(df_NARR_ERC_lon_lat_time_series_2.loc[:,'PMSL']-101325).cumsum(axis=0)
    # df_NARR_ERC_lon_lat_time_series_2.loc[:,'H500 Grad X CS'] = df_NARR_ERC_lon_lat_time_series_2.loc[:,'H500 Grad X'].cumsum(axis=0)
    # df_NARR_ERC_lon_lat_time_series_2.loc[:,'H500 Grad Y CS'] = df_NARR_ERC_lon_lat_time_series_2.loc[:,'H500 Grad Y'].cumsum(axis=0)
    # df_NARR_ERC_lon_lat_time_series_2.loc[:,'PMSL Grad X CS'] = df_NARR_ERC_lon_lat_time_series_2.loc[:,'PMSL Grad X'].cumsum(axis=0)
    # df_NARR_ERC_lon_lat_time_series_2.loc[:,'PMSL Grad Y CS'] = df_NARR_ERC_lon_lat_time_series_2.loc[:,'PMSL Grad Y'].cumsum(axis=0)
    # df_NARR_ERC_lon_lat_time_series_2.loc[:,'CAPE CS']           = df_NARR_ERC_lon_lat_time_series_2.loc[:,'CAPE'].cumsum(axis=0)
    df_NARR_ERC_lon_lat_time_series_2.loc[:,'H500 SD']        = df_NARR_ERC_lon_lat_time_series_2.loc[:,'H500'].rolling(30, closed='neither').std()
    df_NARR_ERC_lon_lat_time_series_2.loc[:,'PMSL SD']        = df_NARR_ERC_lon_lat_time_series_2.loc[:,'PMSL'].rolling(30, closed='neither').std()
    df_NARR_ERC_lon_lat_time_series_2.loc[:,'H500 Grad X MA'] = df_NARR_ERC_lon_lat_time_series_2.loc[:,'H500 Grad X'].rolling(30, closed='neither').mean()
    df_NARR_ERC_lon_lat_time_series_2.loc[:,'H500 Grad Y MA'] = df_NARR_ERC_lon_lat_time_series_2.loc[:,'H500 Grad Y'].rolling(30, closed='neither').mean()
    df_NARR_ERC_lon_lat_time_series_2.loc[:,'PMSL Grad X MA'] = df_NARR_ERC_lon_lat_time_series_2.loc[:,'PMSL Grad X'].rolling(30, closed='neither').mean()
    df_NARR_ERC_lon_lat_time_series_2.loc[:,'PMSL Grad Y MA'] = df_NARR_ERC_lon_lat_time_series_2.loc[:,'PMSL Grad Y'].rolling(30, closed='neither').mean()
    df_NARR_ERC_lon_lat_time_series_2.loc[:,'CAPE MA']        = df_NARR_ERC_lon_lat_time_series_2.loc[:,'CAPE'].rolling(30, closed='neither').mean()


    df_NARR_ERC_lon_lat_time_series_3 = df_NARR_ERC.loc[(plot_lat[3], plot_lon[3])]
    df_NARR_ERC_lon_lat_time_series_3.reset_index(drop=True, inplace=True)
    df_NARR_ERC_lon_lat_time_series_3['time'] = pd.to_datetime(df_NARR_ERC_lon_lat_time_series_3['time'])
    df_NARR_ERC_lon_lat_time_series_3.set_index('time', inplace=True)
    # df_NARR_ERC_lon_lat_time_series_3.loc[:,'H500 Adj CS']        = pd.Series(df_NARR_ERC_lon_lat_time_series_3.loc[:,'H500']-5500).cumsum(axis=0)
    # df_NARR_ERC_lon_lat_time_series_3.loc[:,'PMSL Adj CS']        = pd.Series(df_NARR_ERC_lon_lat_time_series_3.loc[:,'PMSL']-101325).cumsum(axis=0)
    # df_NARR_ERC_lon_lat_time_series_3.loc[:,'H500 Grad X CS'] = df_NARR_ERC_lon_lat_time_series_3.loc[:,'H500 Grad X'].cumsum(axis=0)
    # df_NARR_ERC_lon_lat_time_series_3.loc[:,'H500 Grad Y CS'] = df_NARR_ERC_lon_lat_time_series_3.loc[:,'H500 Grad Y'].cumsum(axis=0)
    # df_NARR_ERC_lon_lat_time_series_3.loc[:,'PMSL Grad X CS'] = df_NARR_ERC_lon_lat_time_series_3.loc[:,'PMSL Grad X'].cumsum(axis=0)
    # df_NARR_ERC_lon_lat_time_series_3.loc[:,'PMSL Grad Y CS'] = df_NARR_ERC_lon_lat_time_series_3.loc[:,'PMSL Grad Y'].cumsum(axis=0)
    # df_NARR_ERC_lon_lat_time_series_3.loc[:,'CAPE CS']           = df_NARR_ERC_lon_lat_time_series_3.loc[:,'CAPE'].cumsum(axis=0)
    df_NARR_ERC_lon_lat_time_series_3.loc[:,'H500 SD']        = df_NARR_ERC_lon_lat_time_series_3.loc[:,'H500'].rolling(30, closed='neither').std()
    df_NARR_ERC_lon_lat_time_series_3.loc[:,'PMSL SD']        = df_NARR_ERC_lon_lat_time_series_3.loc[:,'PMSL'].rolling(30, closed='neither').std()
    df_NARR_ERC_lon_lat_time_series_3.loc[:,'H500 Grad X MA'] = df_NARR_ERC_lon_lat_time_series_3.loc[:,'H500 Grad X'].rolling(30, closed='neither').mean()
    df_NARR_ERC_lon_lat_time_series_3.loc[:,'H500 Grad Y MA'] = df_NARR_ERC_lon_lat_time_series_3.loc[:,'H500 Grad Y'].rolling(30, closed='neither').mean()
    df_NARR_ERC_lon_lat_time_series_3.loc[:,'PMSL Grad X MA'] = df_NARR_ERC_lon_lat_time_series_3.loc[:,'PMSL Grad X'].rolling(30, closed='neither').mean()
    df_NARR_ERC_lon_lat_time_series_3.loc[:,'PMSL Grad Y MA'] = df_NARR_ERC_lon_lat_time_series_3.loc[:,'PMSL Grad Y'].rolling(30, closed='neither').mean()
    df_NARR_ERC_lon_lat_time_series_3.loc[:,'CAPE MA']        = df_NARR_ERC_lon_lat_time_series_3.loc[:,'CAPE'].rolling(30, closed='neither').mean()


    # WA time series (cumulative sums):
    df_NARR_ERC_lon_lat_time_series_4 = df_NARR_ERC.loc[(plot_lat[4], plot_lon[4])]
    df_NARR_ERC_lon_lat_time_series_4.reset_index(drop=True, inplace=True)
    df_NARR_ERC_lon_lat_time_series_4['time'] = pd.to_datetime(df_NARR_ERC_lon_lat_time_series_4['time'])
    df_NARR_ERC_lon_lat_time_series_4.set_index('time', inplace=True)
    # df_NARR_ERC_lon_lat_time_series_4.loc[:,'H500 Adj CS']        = pd.Series(df_NARR_ERC_lon_lat_time_series_4.loc[:,'H500']-5500).cumsum(axis=0)
    # df_NARR_ERC_lon_lat_time_series_4.loc[:,'PMSL Adj CS']        = pd.Series(df_NARR_ERC_lon_lat_time_series_4.loc[:,'PMSL']-101325).cumsum(axis=0)
    # df_NARR_ERC_lon_lat_time_series_4.loc[:,'H500 Grad X CS'] = df_NARR_ERC_lon_lat_time_series_4.loc[:,'H500 Grad X'].cumsum(axis=0)
    # df_NARR_ERC_lon_lat_time_series_4.loc[:,'H500 Grad Y CS'] = df_NARR_ERC_lon_lat_time_series_4.loc[:,'H500 Grad Y'].cumsum(axis=0)
    # df_NARR_ERC_lon_lat_time_series_4.loc[:,'PMSL Grad X CS'] = df_NARR_ERC_lon_lat_time_series_4.loc[:,'PMSL Grad X'].cumsum(axis=0)
    # df_NARR_ERC_lon_lat_time_series_4.loc[:,'PMSL Grad Y CS'] = df_NARR_ERC_lon_lat_time_series_4.loc[:,'PMSL Grad Y'].cumsum(axis=0)
    # df_NARR_ERC_lon_lat_time_series_4.loc[:,'CAPE CS']           = df_NARR_ERC_lon_lat_time_series_4.loc[:,'CAPE'].cumsum(axis=0)
    df_NARR_ERC_lon_lat_time_series_4.loc[:,'H500 SD']        = df_NARR_ERC_lon_lat_time_series_4.loc[:,'H500'].rolling(30, closed='neither').std()
    df_NARR_ERC_lon_lat_time_series_4.loc[:,'PMSL SD']        = df_NARR_ERC_lon_lat_time_series_4.loc[:,'PMSL'].rolling(30, closed='neither').std()
    df_NARR_ERC_lon_lat_time_series_4.loc[:,'H500 Grad X MA'] = df_NARR_ERC_lon_lat_time_series_4.loc[:,'H500 Grad X'].rolling(30, closed='neither').mean()
    df_NARR_ERC_lon_lat_time_series_4.loc[:,'H500 Grad Y MA'] = df_NARR_ERC_lon_lat_time_series_4.loc[:,'H500 Grad Y'].rolling(30, closed='neither').mean()
    df_NARR_ERC_lon_lat_time_series_4.loc[:,'PMSL Grad X MA'] = df_NARR_ERC_lon_lat_time_series_4.loc[:,'PMSL Grad X'].rolling(30, closed='neither').mean()
    df_NARR_ERC_lon_lat_time_series_4.loc[:,'PMSL Grad Y MA'] = df_NARR_ERC_lon_lat_time_series_4.loc[:,'PMSL Grad Y'].rolling(30, closed='neither').mean()
    df_NARR_ERC_lon_lat_time_series_4.loc[:,'CAPE MA']        = df_NARR_ERC_lon_lat_time_series_4.loc[:,'CAPE'].rolling(30, closed='neither').mean()


    df_NARR_ERC_lon_lat_time_series_5 = df_NARR_ERC.loc[(plot_lat[5], plot_lon[5])]
    df_NARR_ERC_lon_lat_time_series_5.reset_index(drop=True, inplace=True)
    df_NARR_ERC_lon_lat_time_series_5['time'] = pd.to_datetime(df_NARR_ERC_lon_lat_time_series_5['time'])
    df_NARR_ERC_lon_lat_time_series_5.set_index('time', inplace=True)
    # df_NARR_ERC_lon_lat_time_series_5.loc[:,'H500 Adj CS']        = pd.Series(df_NARR_ERC_lon_lat_time_series_5.loc[:,'H500']-5500).cumsum(axis=0)
    # df_NARR_ERC_lon_lat_time_series_5.loc[:,'PMSL Adj CS']        = pd.Series(df_NARR_ERC_lon_lat_time_series_5.loc[:,'PMSL']-101325).cumsum(axis=0)
    # df_NARR_ERC_lon_lat_time_series_5.loc[:,'H500 Grad X CS'] = df_NARR_ERC_lon_lat_time_series_5.loc[:,'H500 Grad X'].cumsum(axis=0)
    # df_NARR_ERC_lon_lat_time_series_5.loc[:,'H500 Grad Y CS'] = df_NARR_ERC_lon_lat_time_series_5.loc[:,'H500 Grad Y'].cumsum(axis=0)
    # df_NARR_ERC_lon_lat_time_series_5.loc[:,'PMSL Grad X CS'] = df_NARR_ERC_lon_lat_time_series_5.loc[:,'PMSL Grad X'].cumsum(axis=0)
    # df_NARR_ERC_lon_lat_time_series_5.loc[:,'PMSL Grad Y CS'] = df_NARR_ERC_lon_lat_time_series_5.loc[:,'PMSL Grad Y'].cumsum(axis=0)
    # df_NARR_ERC_lon_lat_time_series_5.loc[:,'CAPE CS']           = df_NARR_ERC_lon_lat_time_series_5.loc[:,'CAPE'].cumsum(axis=0)
    df_NARR_ERC_lon_lat_time_series_5.loc[:,'H500 SD']        = df_NARR_ERC_lon_lat_time_series_5.loc[:,'H500'].rolling(30, closed='neither').std()
    df_NARR_ERC_lon_lat_time_series_5.loc[:,'PMSL SD']        = df_NARR_ERC_lon_lat_time_series_5.loc[:,'PMSL'].rolling(30, closed='neither').std()
    df_NARR_ERC_lon_lat_time_series_5.loc[:,'H500 Grad X MA'] = df_NARR_ERC_lon_lat_time_series_5.loc[:,'H500 Grad X'].rolling(30, closed='neither').mean()
    df_NARR_ERC_lon_lat_time_series_5.loc[:,'H500 Grad Y MA'] = df_NARR_ERC_lon_lat_time_series_5.loc[:,'H500 Grad Y'].rolling(30, closed='neither').mean()
    df_NARR_ERC_lon_lat_time_series_5.loc[:,'PMSL Grad X MA'] = df_NARR_ERC_lon_lat_time_series_5.loc[:,'PMSL Grad X'].rolling(30, closed='neither').mean()
    df_NARR_ERC_lon_lat_time_series_5.loc[:,'PMSL Grad Y MA'] = df_NARR_ERC_lon_lat_time_series_5.loc[:,'PMSL Grad Y'].rolling(30, closed='neither').mean()
    df_NARR_ERC_lon_lat_time_series_5.loc[:,'CAPE MA']        = df_NARR_ERC_lon_lat_time_series_5.loc[:,'CAPE'].rolling(30, closed='neither').mean()


    df_NARR_ERC_lon_lat_time_series_6 = df_NARR_ERC.loc[(plot_lat[6], plot_lon[6])]
    df_NARR_ERC_lon_lat_time_series_6.reset_index(drop=True, inplace=True)
    df_NARR_ERC_lon_lat_time_series_6['time'] = pd.to_datetime(df_NARR_ERC_lon_lat_time_series_6['time'])
    df_NARR_ERC_lon_lat_time_series_6.set_index('time', inplace=True)
    # df_NARR_ERC_lon_lat_time_series_6.loc[:,'H500 Adj CS']        = pd.Series(df_NARR_ERC_lon_lat_time_series_6.loc[:,'H500']-5500).cumsum(axis=0)
    # df_NARR_ERC_lon_lat_time_series_6.loc[:,'PMSL Adj CS']        = pd.Series(df_NARR_ERC_lon_lat_time_series_6.loc[:,'PMSL']-101325).cumsum(axis=0)
    # df_NARR_ERC_lon_lat_time_series_6.loc[:,'H500 Grad X CS'] = df_NARR_ERC_lon_lat_time_series_6.loc[:,'H500 Grad X'].cumsum(axis=0)
    # df_NARR_ERC_lon_lat_time_series_6.loc[:,'H500 Grad Y CS'] = df_NARR_ERC_lon_lat_time_series_6.loc[:,'H500 Grad Y'].cumsum(axis=0)
    # df_NARR_ERC_lon_lat_time_series_6.loc[:,'PMSL Grad X CS'] = df_NARR_ERC_lon_lat_time_series_6.loc[:,'PMSL Grad X'].cumsum(axis=0)
    # df_NARR_ERC_lon_lat_time_series_6.loc[:,'PMSL Grad Y CS'] = df_NARR_ERC_lon_lat_time_series_6.loc[:,'PMSL Grad Y'].cumsum(axis=0)
    # df_NARR_ERC_lon_lat_time_series_6.loc[:,'CAPE CS']           = df_NARR_ERC_lon_lat_time_series_6.loc[:,'CAPE'].cumsum(axis=0)
    df_NARR_ERC_lon_lat_time_series_6.loc[:,'H500 SD']        = df_NARR_ERC_lon_lat_time_series_6.loc[:,'H500'].rolling(30, closed='neither').std()
    df_NARR_ERC_lon_lat_time_series_6.loc[:,'PMSL SD']        = df_NARR_ERC_lon_lat_time_series_6.loc[:,'PMSL'].rolling(30, closed='neither').std()
    df_NARR_ERC_lon_lat_time_series_6.loc[:,'H500 Grad X MA'] = df_NARR_ERC_lon_lat_time_series_6.loc[:,'H500 Grad X'].rolling(30, closed='neither').mean()
    df_NARR_ERC_lon_lat_time_series_6.loc[:,'H500 Grad Y MA'] = df_NARR_ERC_lon_lat_time_series_6.loc[:,'H500 Grad Y'].rolling(30, closed='neither').mean()
    df_NARR_ERC_lon_lat_time_series_6.loc[:,'PMSL Grad X MA'] = df_NARR_ERC_lon_lat_time_series_6.loc[:,'PMSL Grad X'].rolling(30, closed='neither').mean()
    df_NARR_ERC_lon_lat_time_series_6.loc[:,'PMSL Grad Y MA'] = df_NARR_ERC_lon_lat_time_series_6.loc[:,'PMSL Grad Y'].rolling(30, closed='neither').mean()
    df_NARR_ERC_lon_lat_time_series_6.loc[:,'CAPE MA']        = df_NARR_ERC_lon_lat_time_series_6.loc[:,'CAPE'].rolling(30, closed='neither').mean()


    df_NARR_ERC_lon_lat_time_series_7 = df_NARR_ERC.loc[(plot_lat[7], plot_lon[7])]
    df_NARR_ERC_lon_lat_time_series_7.reset_index(drop=True, inplace=True)
    df_NARR_ERC_lon_lat_time_series_7['time'] = pd.to_datetime(df_NARR_ERC_lon_lat_time_series_7['time'])
    df_NARR_ERC_lon_lat_time_series_7.set_index('time', inplace=True)
    # df_NARR_ERC_lon_lat_time_series_7.loc[:,'H500 Adj CS']        = pd.Series(df_NARR_ERC_lon_lat_time_series_7.loc[:,'H500']-5500).cumsum(axis=0)
    # df_NARR_ERC_lon_lat_time_series_7.loc[:,'PMSL Adj CS']        = pd.Series(df_NARR_ERC_lon_lat_time_series_7.loc[:,'PMSL']-101325).cumsum(axis=0)
    # df_NARR_ERC_lon_lat_time_series_7.loc[:,'H500 Grad X CS'] = df_NARR_ERC_lon_lat_time_series_7.loc[:,'H500 Grad X'].cumsum(axis=0)
    # df_NARR_ERC_lon_lat_time_series_7.loc[:,'H500 Grad Y CS'] = df_NARR_ERC_lon_lat_time_series_7.loc[:,'H500 Grad Y'].cumsum(axis=0)
    # df_NARR_ERC_lon_lat_time_series_7.loc[:,'PMSL Grad X CS'] = df_NARR_ERC_lon_lat_time_series_7.loc[:,'PMSL Grad X'].cumsum(axis=0)
    # df_NARR_ERC_lon_lat_time_series_7.loc[:,'PMSL Grad Y CS'] = df_NARR_ERC_lon_lat_time_series_7.loc[:,'PMSL Grad Y'].cumsum(axis=0)
    # df_NARR_ERC_lon_lat_time_series_7.loc[:,'CAPE CS']           = df_NARR_ERC_lon_lat_time_series_7.loc[:,'CAPE'].cumsum(axis=0)
    df_NARR_ERC_lon_lat_time_series_7.loc[:,'H500 SD']        = df_NARR_ERC_lon_lat_time_series_7.loc[:,'H500'].rolling(30, closed='neither').std()
    df_NARR_ERC_lon_lat_time_series_7.loc[:,'PMSL SD']        = df_NARR_ERC_lon_lat_time_series_7.loc[:,'PMSL'].rolling(30, closed='neither').std()
    df_NARR_ERC_lon_lat_time_series_7.loc[:,'H500 Grad X MA'] = df_NARR_ERC_lon_lat_time_series_7.loc[:,'H500 Grad X'].rolling(30, closed='neither').mean()
    df_NARR_ERC_lon_lat_time_series_7.loc[:,'H500 Grad Y MA'] = df_NARR_ERC_lon_lat_time_series_7.loc[:,'H500 Grad Y'].rolling(30, closed='neither').mean()
    df_NARR_ERC_lon_lat_time_series_7.loc[:,'PMSL Grad X MA'] = df_NARR_ERC_lon_lat_time_series_7.loc[:,'PMSL Grad X'].rolling(30, closed='neither').mean()
    df_NARR_ERC_lon_lat_time_series_7.loc[:,'PMSL Grad Y MA'] = df_NARR_ERC_lon_lat_time_series_7.loc[:,'PMSL Grad Y'].rolling(30, closed='neither').mean()
    df_NARR_ERC_lon_lat_time_series_7.loc[:,'CAPE MA']        = df_NARR_ERC_lon_lat_time_series_7.loc[:,'CAPE'].rolling(30, closed='neither').mean()

    # -------------------------------------------------------------------
    # Plot H500-ERC correlation contour plot
    # Use August 1979, all locations
    # Organize so every location has a time series: ['lat','lon','time']
    # Calculate 30-day running correlation for each location
    # Concatenate data into one dataframe
    # Create meshgridded data
    # Plot with plt.tricontourf()

    df_NARR_ERC_ts = df_NARR_ERC.reset_index()
    df_NARR_ERC_ts.set_index(['lat','lon','time'], inplace=True)
    df_NARR_ERC_ts.sort_index(level=1, inplace=True)
    print('df_NARR_ERC_ts:\n', df_NARR_ERC_ts)
    # -------------------------------------------------------------------


    # print('df_NARR_ERC_lon_lat_time_series:\n', df_NARR_ERC_lon_lat_time_series)

    # # -------------------------------------------------------------------
    # NOTE: COMMENTING OUT THE CUMULATIVE SUMS, THEIR CORRELATIONS WITH ERC ARE MINISCULE
    # OR time series
    plt.close()
    # Plotting three time series of df_NARR_ERC data at a lon-lat time:
    # 1) H500 X and Y Gradients
    # 2) PMSL X and Y Gradients 
    # 3) CAPE
    # 4) ERC
    fig, ax = plt.subplots(4,6, figsize=(16,12))

    # NOTE: Preserving cumulative sum plotting info of first row to be plotted somewhere else
    # df_NARR_ERC_lon_lat_time_series_0.plot(x='time', y='H500 Grad X CS', legend=True, ax=ax[0,1], color='k', alpha=0.7)
    # df_NARR_ERC_lon_lat_time_series_0.plot(x='time', y='H500 Grad Y CS', legend=True, ax=ax[0,1], color='g', alpha=0.7)
    # df_NARR_ERC_lon_lat_time_series_0.plot(x='time', y='PMSL Grad X CS', legend=True, ax=ax[0,3], color='gray', alpha=0.7)
    # df_NARR_ERC_lon_lat_time_series_0.plot(x='time', y='PMSL Grad Y CS', legend=True, ax=ax[0,3], color='m', alpha=0.7)
    # df_NARR_ERC_lon_lat_time_series_0.plot(x='time', y='CAPE CS', legend=True, ax=ax[0,5], color='orange', alpha=0.7)
    # df_NARR_ERC_lon_lat_time_series_0.plot(x='time', y='H500 Adj CS', legend=True, ax=ax[0,6], color='#304ed3', alpha=0.7)
    # df_NARR_ERC_lon_lat_time_series_0.plot(x='time', y='PMSL Adj CS', legend=True, ax=ax[0,7], color='#5e8460', alpha=0.7)

    df_NARR_ERC_lon_lat_time_series_0.plot(use_index=True, y='H500 Grad X',       legend=True, ax=ax[0,0], color='k', alpha=0.5)
    df_NARR_ERC_lon_lat_time_series_0.plot(use_index=True, y='H500 Grad X MA',    legend=True, ax=ax[0,0], color='k')
    df_NARR_ERC_lon_lat_time_series_0.plot(use_index=True, y='H500 Grad Y',       legend=True, ax=ax[0,1], color='g', alpha=0.5)
    df_NARR_ERC_lon_lat_time_series_0.plot(use_index=True, y='H500 Grad Y MA',    legend=True, ax=ax[0,1], color='g')
    df_NARR_ERC_lon_lat_time_series_0.plot(use_index=True, y='PMSL Grad X',       legend=True, ax=ax[0,2], color='blue', alpha=0.5)
    df_NARR_ERC_lon_lat_time_series_0.plot(use_index=True, y='PMSL Grad X MA',    legend=True, ax=ax[0,2], color='blue')
    df_NARR_ERC_lon_lat_time_series_0.plot(use_index=True, y='PMSL Grad Y',       legend=True, ax=ax[0,3], color='m', alpha=0.5)
    df_NARR_ERC_lon_lat_time_series_0.plot(use_index=True, y='PMSL Grad Y MA',    legend=True, ax=ax[0,3], color='m')
    df_NARR_ERC_lon_lat_time_series_0.plot(use_index=True, y='CAPE',              legend=True, ax=ax[0,4], color='orange', alpha=0.5)
    df_NARR_ERC_lon_lat_time_series_0.plot(use_index=True, y='CAPE MA',           legend=True, ax=ax[0,4], color='orange')
    df_NARR_ERC_lon_lat_time_series_0.plot(use_index=True, y='ERC',               legend=True, ax=ax[0,5], color='r')

    df_NARR_ERC_lon_lat_time_series_1.plot(use_index=True, y='H500 Grad X',       legend=True, ax=ax[1,0], color='k', alpha=0.5)
    df_NARR_ERC_lon_lat_time_series_1.plot(use_index=True, y='H500 Grad X MA',    legend=True, ax=ax[1,0], color='k')
    df_NARR_ERC_lon_lat_time_series_1.plot(use_index=True, y='H500 Grad Y',       legend=True, ax=ax[1,1], color='g', alpha=0.5)
    df_NARR_ERC_lon_lat_time_series_1.plot(use_index=True, y='H500 Grad Y MA',    legend=True, ax=ax[1,1], color='g')
    df_NARR_ERC_lon_lat_time_series_1.plot(use_index=True, y='PMSL Grad X',       legend=True, ax=ax[1,2], color='blue', alpha=0.5)
    df_NARR_ERC_lon_lat_time_series_1.plot(use_index=True, y='PMSL Grad X MA',    legend=True, ax=ax[1,2], color='blue')
    df_NARR_ERC_lon_lat_time_series_1.plot(use_index=True, y='PMSL Grad Y',       legend=True, ax=ax[1,3], color='m', alpha=0.5)
    df_NARR_ERC_lon_lat_time_series_1.plot(use_index=True, y='PMSL Grad Y MA',    legend=True, ax=ax[1,3], color='m')
    df_NARR_ERC_lon_lat_time_series_1.plot(use_index=True, y='CAPE',              legend=True, ax=ax[1,4], color='orange', alpha=0.5)
    df_NARR_ERC_lon_lat_time_series_1.plot(use_index=True, y='CAPE MA',           legend=True, ax=ax[1,4], color='orange')
    df_NARR_ERC_lon_lat_time_series_1.plot(use_index=True, y='ERC',               legend=True, ax=ax[1,5], color='r')

    df_NARR_ERC_lon_lat_time_series_2.plot(use_index=True, y='H500 Grad X',       legend=True, ax=ax[2,0], color='k', alpha=0.5)
    df_NARR_ERC_lon_lat_time_series_2.plot(use_index=True, y='H500 Grad X MA',    legend=True, ax=ax[2,0], color='k')
    df_NARR_ERC_lon_lat_time_series_2.plot(use_index=True, y='H500 Grad Y',       legend=True, ax=ax[2,1], color='g', alpha=0.5)
    df_NARR_ERC_lon_lat_time_series_2.plot(use_index=True, y='H500 Grad Y MA',    legend=True, ax=ax[2,1], color='g')
    df_NARR_ERC_lon_lat_time_series_2.plot(use_index=True, y='PMSL Grad X',       legend=True, ax=ax[2,2], color='blue', alpha=0.5)
    df_NARR_ERC_lon_lat_time_series_2.plot(use_index=True, y='PMSL Grad X MA',    legend=True, ax=ax[2,2], color='blue')
    df_NARR_ERC_lon_lat_time_series_2.plot(use_index=True, y='PMSL Grad Y',       legend=True, ax=ax[2,3], color='m', alpha=0.5)
    df_NARR_ERC_lon_lat_time_series_2.plot(use_index=True, y='PMSL Grad Y MA',    legend=True, ax=ax[2,3], color='m')
    df_NARR_ERC_lon_lat_time_series_2.plot(use_index=True, y='CAPE',              legend=True, ax=ax[2,4], color='orange', alpha=0.5)
    df_NARR_ERC_lon_lat_time_series_2.plot(use_index=True, y='CAPE MA',           legend=True, ax=ax[2,4], color='orange')
    df_NARR_ERC_lon_lat_time_series_2.plot(use_index=True, y='ERC',               legend=True, ax=ax[2,5], color='r')

    df_NARR_ERC_lon_lat_time_series_3.plot(use_index=True, y='H500 Grad X',       legend=True, ax=ax[3,0], color='k', alpha=0.5)
    df_NARR_ERC_lon_lat_time_series_3.plot(use_index=True, y='H500 Grad X MA',    legend=True, ax=ax[3,0], color='k')
    df_NARR_ERC_lon_lat_time_series_3.plot(use_index=True, y='H500 Grad Y',       legend=True, ax=ax[3,1], color='g', alpha=0.5)
    df_NARR_ERC_lon_lat_time_series_3.plot(use_index=True, y='H500 Grad Y MA',    legend=True, ax=ax[3,1], color='g')
    df_NARR_ERC_lon_lat_time_series_3.plot(use_index=True, y='PMSL Grad X',       legend=True, ax=ax[3,2], color='blue', alpha=0.5)
    df_NARR_ERC_lon_lat_time_series_3.plot(use_index=True, y='PMSL Grad X MA',    legend=True, ax=ax[3,2], color='blue')
    df_NARR_ERC_lon_lat_time_series_3.plot(use_index=True, y='PMSL Grad Y',       legend=True, ax=ax[3,3], color='m', alpha=0.5)
    df_NARR_ERC_lon_lat_time_series_3.plot(use_index=True, y='PMSL Grad Y MA',    legend=True, ax=ax[3,3], color='m')
    df_NARR_ERC_lon_lat_time_series_3.plot(use_index=True, y='CAPE',              legend=True, ax=ax[3,4], color='orange', alpha=0.5)
    df_NARR_ERC_lon_lat_time_series_3.plot(use_index=True, y='CAPE MA',           legend=True, ax=ax[3,4], color='orange')
    df_NARR_ERC_lon_lat_time_series_3.plot(use_index=True, y='ERC',               legend=True, ax=ax[3,5], color='r')

    # TITLES:
    ax[0,0].set_xlabel('Date')
    ax[0,0].set_title('H500 Grad X & MA') # gpm/deg
    ax[0,1].set_title('H500 Grad Y & MA')
    ax[0,2].set_title('PMSL Grad X & MA') # Pa/deg
    ax[0,3].set_title('PMSL Grad Y & MA')
    ax[0,4].set_title('CAPE & MA') # J/kg
    ax[0,5].set_title('ERC')
    
    # X LABELS:
    # ax[0,1].set_xlabel('Date')
    ax[3,0].set_xlabel('Date')
    # ax[2,3].set_xlabel('Date')

    # Y LABELS:
    ax[0,0].set_ylabel('Kalmiopsis')
    ax[1,0].set_ylabel('Bend')
    ax[2,0].set_ylabel('John Day')
    ax[3,0].set_ylabel('Medford')

    # HIDE X-AXIS LABELS:
    ax[0,0].set_xticks([]); ax[0,2].set_xticks([]); ax[0,3].set_xticks([]); ax[0,4].set_xticks([]); ax[0,5].set_xticks([])
    ax[0,1].set_xticks([]); ax[1,2].set_xticks([]); ax[1,3].set_xticks([]); ax[1,4].set_xticks([]); ax[1,5].set_xticks([])
    ax[0,2].set_xticks([]); ax[2,2].set_xticks([]); ax[2,3].set_xticks([]); ax[2,4].set_xticks([]); ax[2,5].set_xticks([])

    plt.subplots_adjust(wspace=0.3)
    plt.savefig('NARR_ERC_Time_Series__OR.png', bbox_inches='tight')
    plt.show()
    # -------------------------------------------------------------------


    # -------------------------------------------------------------------
    # H500 ERC Time Series:

    # NOTE: USE MELT HERE INSTEAD OF CONCAT:
    df_NARR_ERC_lon_lat_time_series_0['Location'] = 'Kalmiopsis'
    df_NARR_ERC_lon_lat_time_series_1['Location'] = 'Bend'
    df_NARR_ERC_lon_lat_time_series_2['Location'] = 'John Day'
    df_NARR_ERC_lon_lat_time_series_3['Location'] = 'Medford'

    df = pd.concat((df_NARR_ERC_lon_lat_time_series_0,
                    df_NARR_ERC_lon_lat_time_series_1,
                    df_NARR_ERC_lon_lat_time_series_2,
                    df_NARR_ERC_lon_lat_time_series_3), axis=0)

    df['H500 ERC Corr'] = df['H500'].rolling(30).corr(df['ERC'])

    df.reset_index(inplace=True)
    df.rename(columns={'time':'Date'}, inplace=True)
    df['M'] = df['Date'].dt.strftime('%b')
    print('df:\n', df)

    sns.lineplot(data=df, x="Date", y='H500 ERC Corr', hue="Location")#, palette=palette, #height=5, aspect=.75, facet_kws=dict(sharex=False))
    plt.savefig('Correlations_Time_Series_H500_ERC__OR__1979.png', bbox_inches='tight')
    plt.show()

    # -------------------------------------------------------------------
    # H500 average with error shading for all OR locations:
    plt.close()
    if multi == True:

        df_location_avgs = df.groupby(['Date'])['H500','H500 SD','H500 Grad X','H500 Grad Y','PMSL','PMSL SD','PMSL Grad X','PMSL Grad Y', 'PVEL', 'ERC'].mean().reset_index()
        df_location_stds = df.groupby(['Date'])['H500','H500 SD','H500 Grad X','H500 Grad Y','PMSL','PMSL SD','PMSL Grad X','PMSL Grad Y', 'PVEL', 'ERC'].std().reset_index()
    else:
        df_location_avgs = df.groupby(['Date'])['H500','H500 SD','H500 Grad X','H500 Grad Y','PMSL','PMSL SD','PMSL Grad X','PMSL Grad Y', 'ERC'].mean().reset_index()
        df_location_stds = df.groupby(['Date'])['H500','H500 SD','H500 Grad X','H500 Grad Y','PMSL','PMSL SD','PMSL Grad X','PMSL Grad Y', 'ERC'].std().reset_index()

    # Converts all columns to numeric. It ignores any columns that can't be converted.
    df_location_avgs.apply(pd.to_numeric, errors='ignore')
    df_location_stds.apply(pd.to_numeric, errors='ignore')

    print('df_location_avgs:\n', df_location_avgs.to_string())
    print('df_location_avgs.dtypes:\n', df_location_avgs.dtypes)
    print('df_location_stds:\n', df_location_stds.to_string())
    print('df_location_stds.dtypes:\n', df_location_stds.dtypes)

    # H500:
    # f, (ax1,ax3)= plt.subplots(1,2, figsize=(4,4))
    ax1 = df_location_avgs.plot(x='Date', y='H500', color='#5170a0')
    ax1.fill_between(df_location_avgs['Date'].dt.to_pydatetime(), df_location_avgs['H500']-df_location_stds['H500'], df_location_avgs['H500']+df_location_stds['H500'], color='#5170a0', alpha=0.35)
    ax2 = ax1.twinx()
    df_location_avgs.plot(x='Date', y='ERC', ax=ax2, alpha=0.4, color='r')
    ax2.fill_between(df_location_avgs['Date'].dt.to_pydatetime(), df_location_avgs['ERC']-df_location_stds['ERC'], df_location_avgs['ERC']+df_location_stds['ERC'], color='r', alpha=0.35)

    ax1.set_ylabel('H500', color='#5170a0')
    ax1.tick_params('y', colors='#5170a0')
    ax1.legend(labels=['H500'], loc='upper left')

    ax2.set_ylabel('ERC', color='r')
    ax2.tick_params('y', colors='r')
    ax2.legend(labels=['ERC'], loc='upper right')
    plt.savefig('NARR_H500_ERC_Time_Series__OR__1979.png', bbox_inches='tight')


    # H500 SD:
    ax7 = df_location_avgs.plot(x='Date', y='H500 SD', color='#26529b')
    ax7.fill_between(df_location_avgs['Date'].dt.to_pydatetime(), df_location_avgs['H500 SD']-df_location_stds['H500 SD'], df_location_avgs['H500 SD']+df_location_stds['H500 SD'], color='#26529b', alpha=0.35)
    ax8 = ax7.twinx()
    df_location_avgs.plot(x='Date', y='ERC', ax=ax8, alpha=0.4, color='r')
    ax8.fill_between(df_location_avgs['Date'].dt.to_pydatetime(), df_location_avgs['ERC']-df_location_stds['ERC'], df_location_avgs['ERC']+df_location_stds['ERC'], color='r', alpha=0.35)

    ax7.set_ylabel('H500 SD', color='#26529b')
    ax7.tick_params('y', colors='#26529b')
    ax7.legend(labels=['H500 SD'], loc='upper left')

    ax8.set_ylabel('ERC', color='r')
    ax8.tick_params('y', colors='r')
    ax8.legend(labels=['ERC'], loc='upper right')
    plt.savefig('NARR_H500_SD_ERC_Time_Series__OR__1979.png', bbox_inches='tight')


    if multi == True:
        # H500 Grad X and PVEL:
        # f, (ax1,ax3)= plt.subplots(1,2, figsize=(4,4))
        ax13 = df_location_avgs.plot(x='Date', y='H500 Grad X', color='#5170a0')
        ax13.fill_between(df_location_avgs['Date'].dt.to_pydatetime(), df_location_avgs['H500 Grad X']-df_location_stds['H500 Grad X'], df_location_avgs['H500 Grad X']+df_location_stds['H500 Grad X'], color='#5170a0', alpha=0.35)
        ax14 = ax13.twinx()
        df_location_avgs.plot(x='Date', y='PVEL', ax=ax14, alpha=0.4, color='#c9440c')
        ax14.fill_between(df_location_avgs['Date'].dt.to_pydatetime(), df_location_avgs['PVEL']-df_location_stds['PVEL'], df_location_avgs['PVEL']+df_location_stds['PVEL'], color='#c9440c', alpha=0.35)

        ax13.set_ylabel('H500 Grad X', color='#5170a0')
        ax13.tick_params('y', colors='#5170a0')
        ax13.legend(labels=['H500 Grad X'], loc='upper left')

        ax14.set_ylabel('PVEL', color='#c9440c')
        ax14.tick_params('y', colors='#c9440c')
        ax14.legend(labels=['PVEL'], loc='upper right')
        plt.savefig('NARR_H500_Grad_X_PVEL_Time_Series__OR__1979.png', bbox_inches='tight')

        # H500 Grad Y and PVEL:
        # f, (ax1,ax3)= plt.subplots(1,2, figsize=(4,4))
        ax15 = df_location_avgs.plot(x='Date', y='H500 Grad Y', color='#5170a0')
        ax15.fill_between(df_location_avgs['Date'].dt.to_pydatetime(), df_location_avgs['H500 Grad Y']-df_location_stds['H500 Grad Y'], df_location_avgs['H500 Grad Y']+df_location_stds['H500 Grad Y'], color='#5170a0', alpha=0.35)
        ax16 = ax15.twinx()
        df_location_avgs.plot(x='Date', y='PVEL', ax=ax16, alpha=0.4, color='#c9440c')
        ax16.fill_between(df_location_avgs['Date'].dt.to_pydatetime(), df_location_avgs['PVEL']-df_location_stds['PVEL'], df_location_avgs['PVEL']+df_location_stds['PVEL'], color='#c9440c', alpha=0.35)

        ax15.set_ylabel('H500 Grad Y', color='#5170a0')
        ax15.tick_params('y', colors='#5170a0')
        ax15.legend(labels=['H500 Grad Y'], loc='upper left')

        ax16.set_ylabel('PVEL', color='#c9440c')
        ax16.tick_params('y', colors='#c9440c')
        ax16.legend(labels=['PVEL'], loc='upper right')
        plt.savefig('NARR_H500_Grad_Y_PVEL_Time_Series__OR__1979.png', bbox_inches='tight')


    # PMSL:
    ax3 = df_location_avgs.plot(x='Date', y='PMSL', color='#548246')
    ax3.fill_between(df_location_avgs['Date'].dt.to_pydatetime(), df_location_avgs['PMSL']-df_location_stds['PMSL'], df_location_avgs['PMSL']+df_location_stds['PMSL'], color='#548246', alpha=0.35)
    ax4 = ax3.twinx()
    df_location_avgs.plot(x='Date', y='ERC', ax=ax4, alpha=0.4, color='r')
    ax4.fill_between(df_location_avgs['Date'].dt.to_pydatetime(), df_location_avgs['ERC']-df_location_stds['ERC'], df_location_avgs['ERC']+df_location_stds['ERC'], color='r', alpha=0.35)

    ax3.set_ylabel('PMSL', color='#548246')
    ax3.tick_params('y', colors='#548246')
    ax3.legend(labels=['PMSL'], loc='upper left')

    ax4.set_ylabel('ERC', color='r')
    ax4.tick_params('y', colors='r')
    ax4.legend(labels=['ERC'], loc='upper right')
    plt.savefig('NARR_PMSL_ERC_Time_Series__OR__1979.png', bbox_inches='tight')


    # PMSL SD:
    ax5 = df_location_avgs.plot(x='Date', y='PMSL SD', color='#7a652d')
    ax5.fill_between(df_location_avgs['Date'].dt.to_pydatetime(), df_location_avgs['PMSL SD']-df_location_stds['PMSL SD'], df_location_avgs['PMSL SD']+df_location_stds['PMSL SD'], color='#7a652d', alpha=0.35)
    ax6 = ax5.twinx()
    df_location_avgs.plot(x='Date', y='ERC', ax=ax6, alpha=0.4, color='r')
    ax6.fill_between(df_location_avgs['Date'].dt.to_pydatetime(), df_location_avgs['ERC']-df_location_stds['ERC'], df_location_avgs['ERC']+df_location_stds['ERC'], color='r', alpha=0.35)

    ax5.set_ylabel('PMSL SD', color='#7a652d')
    ax5.tick_params('y', colors='#7a652d')
    ax5.legend(labels=['PMSL SD'], loc='upper left')

    ax6.set_ylabel('ERC', color='r')
    ax6.tick_params('y', colors='r')
    ax6.legend(labels=['ERC'], loc='upper right')

    plt.savefig('NARR_PMSL_SD_ERC_Time_Series__OR__1979.png', bbox_inches='tight')
    plt.show()

    # PMSL Grad X:
    ax9 = df_location_avgs.plot(x='Date', y='PMSL Grad X', color='#26529b')
    ax9.fill_between(df_location_avgs['Date'].dt.to_pydatetime(), df_location_avgs['PMSL Grad X']-df_location_stds['PMSL Grad X'], df_location_avgs['PMSL Grad X']+df_location_stds['PMSL Grad X'], color='#26529b', alpha=0.35)
    ax10 = ax9.twinx()
    df_location_avgs.plot(x='Date', y='ERC', ax=ax10, alpha=0.4, color='r')
    ax10.fill_between(df_location_avgs['Date'].dt.to_pydatetime(), df_location_avgs['ERC']-df_location_stds['ERC'], df_location_avgs['ERC']+df_location_stds['ERC'], color='r', alpha=0.35)

    ax9.set_ylabel('PMSL Grad X', color='#26529b')
    ax9.tick_params('y', colors='#26529b')
    ax9.legend(labels=['PMSL Grad X'], loc='upper left')

    ax10.set_ylabel('ERC', color='r')
    ax10.tick_params('y', colors='r')
    ax10.legend(labels=['ERC'], loc='upper right')
    plt.savefig('NARR_PMSL_Grad_X_ERC_Time_Series__OR__1979.png', bbox_inches='tight')

    # PMSL Grad Y:
    ax11 = df_location_avgs.plot(x='Date', y='PMSL Grad Y', color='#26529b')
    ax11.fill_between(df_location_avgs['Date'].dt.to_pydatetime(), df_location_avgs['PMSL Grad Y']-df_location_stds['PMSL Grad Y'], df_location_avgs['PMSL Grad Y']+df_location_stds['PMSL Grad Y'], color='#26529b', alpha=0.35)
    ax12 = ax11.twinx()
    df_location_avgs.plot(x='Date', y='ERC', ax=ax12, alpha=0.4, color='r')
    ax12.fill_between(df_location_avgs['Date'].dt.to_pydatetime(), df_location_avgs['ERC']-df_location_stds['ERC'], df_location_avgs['ERC']+df_location_stds['ERC'], color='r', alpha=0.35)

    ax11.set_ylabel('PMSL Grad Y', color='#26529b')
    ax11.tick_params('y', colors='#26529b')
    ax11.legend(labels=['PMSL Grad Y'], loc='upper left')

    ax12.set_ylabel('ERC', color='r')
    ax12.tick_params('y', colors='r')
    ax12.legend(labels=['ERC'], loc='upper right')
    plt.savefig('NARR_PMSL_Grad_Y_ERC_Time_Series__OR__1979.png', bbox_inches='tight')

    # -------------------------------------------------------------------
    # # H500 average with error shading for all OR locations:
    # plt.close()
    # sns.lineplot(x='Date', y='H500', data=df)
    # plt.show()

    # -------------------------------------------------------------------
    # H500-ERC Time-lag Running Cross-correlation:
    plt.close()
    df['H500 Shift 1'] = df['H500'].shift(periods=1)
    df['H500 Shift 2'] = df['H500'].shift(periods=2)
    df['H500 Shift 3'] = df['H500'].shift(periods=3)
    df['H500 Shift 4'] = df['H500'].shift(periods=4)
    df['H500 Shift 5'] = df['H500'].shift(periods=5)
    df['H500 Shift 6'] = df['H500'].shift(periods=6)
    df['N1 Day Lag'] = pd.Series(df['H500'].shift(periods=-1)).rolling(30).corr(df['ERC'])
    df['0 Day Lag'] = pd.Series(df['H500']).rolling(30).corr(df['ERC']) # Control Variable: unshifted, rolling correlation of H500 data
    df['1 Day Lag'] = pd.Series(df['H500'].shift(periods=1)).rolling(30).corr(df['ERC'])
    df['2 Day Lag'] = pd.Series(df['H500'].shift(periods=2)).rolling(30).corr(df['ERC'])
    df['3 Day Lag'] = pd.Series(df['H500'].shift(periods=3)).rolling(30).corr(df['ERC'])
    df['4 Day Lag'] = pd.Series(df['H500'].shift(periods=4)).rolling(30).corr(df['ERC'])
    df['5 Day Lag'] = pd.Series(df['H500'].shift(periods=5)).rolling(30).corr(df['ERC'])
    print('df:\n', df.head())

    df_time_lag_corr = df[['Date', 'Location', 'N1 Day Lag', '0 Day Lag', '1 Day Lag', '2 Day Lag', '3 Day Lag', '4 Day Lag', '5 Day Lag']]
    df_time_lag_corr['Date'] = df_time_lag_corr['Date'].dt.to_pydatetime()
    print('df_time_lag_corr:\n', df_time_lag_corr)

    # df_time_lag_corr.reset_index(inplace=True)
    # df_time_lag_corr.rename(columns={'time':'Date'}, inplace=True)
    print('df_time_lag_corr:\n', df_time_lag_corr)
    df_tidy_time_lag_corr = pd.melt(df_time_lag_corr,
                                    id_vars=['Date', 'Location'],
                                    value_vars=[
                                            'N1 Day Lag',
                                            '0 Day Lag',
                                            '1 Day Lag',
                                            '2 Day Lag',
                                            '3 Day Lag',
                                            '4 Day Lag',
                                            '5 Day Lag'],
                                    var_name='Lag',
                                    value_name='Corr'
                                    )

    plt.close()
    g = sns.relplot(x='Date', y='Corr', hue='Lag', col='Location', kind='line', data=df_tidy_time_lag_corr)
    plt.savefig('Correlations_Time_Series_Lag__OR__1979.png', bbox_inches='tight')
    plt.show()
    # -------------------------------------------------------------------


    # # -------------------------------------------------------------------
    # # WA time series
    # plt.close()
    # # Plotting three time series of df_NARR_ERC data at a lon-lat time:
    # # 1) H500 X and Y Gradients
    # # 2) PMSL X and Y Gradients
    # # 3) CAPE
    # # 4) ERC
    # fig, ax = plt.subplots(4,8, figsize=(16,14))
    # df_NARR_ERC_lon_lat_time_series_4.plot(x='time', y='H500 Grad X', legend=True, ax=ax[0,0], color='k')
    # df_NARR_ERC_lon_lat_time_series_4.plot(x='time', y='H500 Grad Y', legend=True, ax=ax[0,0], color='g')
    # df_NARR_ERC_lon_lat_time_series_4.plot(x='time', y='H500 Grad X CS', legend=True, ax=ax[0,1], color='k', alpha=0.7)
    # df_NARR_ERC_lon_lat_time_series_4.plot(x='time', y='H500 Grad Y CS', legend=True, ax=ax[0,1], color='g', alpha=0.7)
    # df_NARR_ERC_lon_lat_time_series_4.plot(x='time', y='PMSL Grad X', legend=True, ax=ax[0,2], color='gray')
    # df_NARR_ERC_lon_lat_time_series_4.plot(x='time', y='PMSL Grad Y', legend=True, ax=ax[0,2], color='m')
    # df_NARR_ERC_lon_lat_time_series_4.plot(x='time', y='PMSL Grad X CS', legend=True, ax=ax[0,3], color='gray', alpha=0.7)
    # df_NARR_ERC_lon_lat_time_series_4.plot(x='time', y='PMSL Grad Y CS', legend=True, ax=ax[0,3], color='m', alpha=0.7)
    # df_NARR_ERC_lon_lat_time_series_4.plot(x='time', y='CAPE', legend=True, ax=ax[0,4], color='orange')
    # # df_NARR_ERC_lon_lat_time_series_4.plot(x='time', y='CAPE CS', legend=True, ax=ax[0,5], color='orange', alpha=0.7)
    # df_NARR_ERC_lon_lat_time_series_4.plot(x='time', y='ERC', legend=True, ax=ax[0,5], color='r')
    # df_NARR_ERC_lon_lat_time_series_4.plot(x='time', y='H500 Adj CS', legend=True, ax=ax[0,6], color='#304ed3', alpha=0.7)
    # df_NARR_ERC_lon_lat_time_series_4.plot(x='time', y='PMSL Adj CS', legend=True, ax=ax[0,7], color='#5e8460', alpha=0.7)

    # df_NARR_ERC_lon_lat_time_series_5.plot(x='time', y='H500 Grad X', legend=False, ax=ax[1,0], color='k')
    # df_NARR_ERC_lon_lat_time_series_5.plot(x='time', y='H500 Grad Y', legend=False, ax=ax[1,0], color='g')
    # df_NARR_ERC_lon_lat_time_series_5.plot(x='time', y='H500 Grad X CS', legend=False, ax=ax[1,1], color='k', alpha=0.7)
    # df_NARR_ERC_lon_lat_time_series_5.plot(x='time', y='H500 Grad Y CS', legend=False, ax=ax[1,1], color='g', alpha=0.7)
    # df_NARR_ERC_lon_lat_time_series_5.plot(x='time', y='PMSL Grad X', legend=False, ax=ax[1,2], color='gray')
    # df_NARR_ERC_lon_lat_time_series_5.plot(x='time', y='PMSL Grad Y', legend=False, ax=ax[1,2], color='m')
    # df_NARR_ERC_lon_lat_time_series_5.plot(x='time', y='PMSL Grad X CS', legend=False, ax=ax[1,3], color='gray', alpha=0.7)
    # df_NARR_ERC_lon_lat_time_series_5.plot(x='time', y='PMSL Grad Y CS', legend=False, ax=ax[1,3], color='m', alpha=0.7)
    # df_NARR_ERC_lon_lat_time_series_5.plot(x='time', y='CAPE', legend=False, ax=ax[1,4], color='orange')
    # df_NARR_ERC_lon_lat_time_series_5.plot(x='time', y='ERC', legend=True, ax=ax[1,5], color='r')
    # df_NARR_ERC_lon_lat_time_series_5.plot(x='time', y='H500 Adj CS', legend=True, ax=ax[1,6], color='#304ed3', alpha=0.7)
    # df_NARR_ERC_lon_lat_time_series_5.plot(x='time', y='PMSL Adj CS', legend=True, ax=ax[1,7], color='#5e8460', alpha=0.7)

    # df_NARR_ERC_lon_lat_time_series_6.plot(x='time', y='H500 Grad X', legend=False, ax=ax[2,0], color='k')
    # df_NARR_ERC_lon_lat_time_series_6.plot(x='time', y='H500 Grad Y', legend=False, ax=ax[2,0], color='g')
    # df_NARR_ERC_lon_lat_time_series_6.plot(x='time', y='H500 Grad X CS', legend=False, ax=ax[2,1], color='k', alpha=0.7)
    # df_NARR_ERC_lon_lat_time_series_6.plot(x='time', y='H500 Grad Y CS', legend=False, ax=ax[2,1], color='g', alpha=0.7)
    # df_NARR_ERC_lon_lat_time_series_6.plot(x='time', y='PMSL Grad X', legend=False, ax=ax[2,2], color='gray')
    # df_NARR_ERC_lon_lat_time_series_6.plot(x='time', y='PMSL Grad Y', legend=False, ax=ax[2,2], color='m')
    # df_NARR_ERC_lon_lat_time_series_6.plot(x='time', y='PMSL Grad X CS', legend=False, ax=ax[2,3], color='gray', alpha=0.7)
    # df_NARR_ERC_lon_lat_time_series_6.plot(x='time', y='PMSL Grad Y CS', legend=False, ax=ax[2,3], color='m', alpha=0.7)
    # df_NARR_ERC_lon_lat_time_series_6.plot(x='time', y='CAPE', legend=False, ax=ax[2,4], color='orange')
    # # df_NARR_ERC_lon_lat_time_series_6.plot(x='time', y='CAPE CS', legend=False, ax=ax[2,5], color='orange', alpha=0.7)
    # df_NARR_ERC_lon_lat_time_series_6.plot(x='time', y='ERC', legend=True, ax=ax[2,5], color='r')
    # df_NARR_ERC_lon_lat_time_series_6.plot(x='time', y='H500 Adj CS', legend=True, ax=ax[2,6], color='#304ed3', alpha=0.7)
    # df_NARR_ERC_lon_lat_time_series_6.plot(x='time', y='PMSL Adj CS', legend=True, ax=ax[2,7], color='#5e8460', alpha=0.7)

    # df_NARR_ERC_lon_lat_time_series_7.plot(x='time', y='H500 Grad X', legend=False, ax=ax[3,0], color='k')
    # df_NARR_ERC_lon_lat_time_series_7.plot(x='time', y='H500 Grad Y', legend=False, ax=ax[3,0], color='g')
    # df_NARR_ERC_lon_lat_time_series_7.plot(x='time', y='H500 Grad X CS', legend=False, ax=ax[3,1], color='k', alpha=0.7)
    # df_NARR_ERC_lon_lat_time_series_7.plot(x='time', y='H500 Grad Y CS', legend=False, ax=ax[3,1], color='g', alpha=0.7)
    # df_NARR_ERC_lon_lat_time_series_7.plot(x='time', y='PMSL Grad X', legend=False, ax=ax[3,2], color='gray')
    # df_NARR_ERC_lon_lat_time_series_7.plot(x='time', y='PMSL Grad Y', legend=False, ax=ax[3,2], color='m')
    # df_NARR_ERC_lon_lat_time_series_7.plot(x='time', y='PMSL Grad X CS', legend=False, ax=ax[3,3], color='gray', alpha=0.7)
    # df_NARR_ERC_lon_lat_time_series_7.plot(x='time', y='PMSL Grad Y CS', legend=False, ax=ax[3,3], color='m', alpha=0.7)
    # df_NARR_ERC_lon_lat_time_series_7.plot(x='time', y='CAPE', legend=False, ax=ax[3,4], color='orange')
    # # df_NARR_ERC_lon_lat_time_series_7.plot(x='time', y='CAPE CS', legend=False, ax=ax[3,5], color='orange', alpha=0.7)
    # df_NARR_ERC_lon_lat_time_series_7.plot(x='time', y='ERC', legend=True, ax=ax[3,5], color='r')
    # df_NARR_ERC_lon_lat_time_series_7.plot(x='time', y='H500 Adj CS', legend=True, ax=ax[3,6], color='#304ed3', alpha=0.7)
    # df_NARR_ERC_lon_lat_time_series_7.plot(x='time', y='PMSL Adj CS', legend=True, ax=ax[3,7], color='#5e8460', alpha=0.7)

    # ax[0,0].set_xlabel('Date')
    # ax[0,0].set_title('H500 Gradients, gpm/deg')
    # # ax[0,1].set_xlabel('Date')
    # ax[0,1].set_title('H500 Grad CS, gpm/deg')

    # ax[2,2].set_xlabel('Date')
    # ax[0,2].set_title('PMSL Gradients, Pa/deg')
    # # ax[2,3].set_xlabel('Date')
    # ax[0,3].set_title('PMSL Grad CS, Pa/deg')

    # ax[2,2].set_xlabel('Date')
    # # ax[0,2].set_title('CAPE, J/kg')
    # ax[0,2].set_title('H500 & PMSL Adj CS')
    # ax[2,3].set_xlabel('Date')
    # ax[0,3].set_title('ERC')
    
    # ax[0,0].set_ylabel('Olympics')
    # ax[1,0].set_ylabel('Snoqualmie')
    # ax[2,0].set_ylabel('Colville')
    # ax[3,0].set_ylabel('N Cascades')

    # plt.savefig('NARR_ERC_lon_lat_time_series_WA.png', bbox_inches='tight')
    # plt.show()
    # -------------------------------------------------------------------

    # -------------------------------------------------------------------
    # # Printing stationary time series of variables:
    # # df_NARR_ERC_lon_lat_time_series_0.set_index('time', inplace=True)
    # print('df_NARR_ERC_lon_lat_time_series_0:\n', df_NARR_ERC_lon_lat_time_series_0)
    # df_NARR_ERC_lon_lat_time_series_0.diff().plot(figsize=(9,6))
    # print('df_NARR_ERC_lon_lat_time_series_0.diff():\n', df_NARR_ERC_lon_lat_time_series_0.diff().head().to_string())
    # plt.xlabel('Time')
    # plt.title('Stationary Time Series - Kalmiopsis')
    # plt.show()
    # -------------------------------------------------------------------

    # -------------------------------------------------------------------
    # Running correlation of synvars and ERC:

    # -------------------------------------------------------------------


    # -------------------------------------------------------------------
    # Seaborn Correlation Heatmap:
    # Location: Kalmiopsis:
    # Time Series correlation and Stationary Time Series correlation:
    df_NARR_ERC_lon_lat_time_series_0.drop(columns='Location', inplace=True) # WARNING: NOT SURE WHY 'Location' IS IN THIS DATAFRAME
    df_NARR_ERC_lon_lat_time_series_0_corr = df_NARR_ERC_lon_lat_time_series_0.corr() # Correlation
    print('df_NARR_ERC_lon_lat_time_series_0:\n', df_NARR_ERC_lon_lat_time_series_0.to_string())
    df_NARR_ERC_lon_lat_time_series_0_corr.apply(pd.to_numeric, errors='ignore')
    df_NARR_ERC_lon_lat_stationary_time_series_0_corr = df_NARR_ERC_lon_lat_time_series_0.diff().corr()
    print('df_NARR_ERC_lon_lat_time_series_0_corr:\n', df_NARR_ERC_lon_lat_time_series_0_corr.to_string())
    print('df_NARR_ERC_lon_lat_stationary_time_series_0_corr:\n', df_NARR_ERC_lon_lat_stationary_time_series_0_corr.to_string())
    
    # Generate a mask for the upper triangle
    # Time series corr:
    mask = np.zeros_like(df_NARR_ERC_lon_lat_time_series_0_corr, dtype=np.bool)
    mask[np.triu_indices_from(mask)] = True
    # Stationary time series corr:
    mask = np.zeros_like(df_NARR_ERC_lon_lat_stationary_time_series_0_corr, dtype=np.bool)
    mask[np.triu_indices_from(mask)] = True
    
    # Build diagonal heat map
    sns.set(style="white")
    f, (ax1, ax2) = plt.subplots(1,2, figsize=(9, 4))
    cmap = sns.diverging_palette(220, 10, as_cmap=True)
    sns.heatmap(df_NARR_ERC_lon_lat_time_series_0_corr, mask=mask, cmap=cmap, vmax=1, vmin=-1, center=0,
                        square=True, linewidths=.5, cbar_kws={"shrink": .5}, ax=ax1)
    sns.heatmap(df_NARR_ERC_lon_lat_stationary_time_series_0_corr, mask=mask, cmap=cmap, vmax=1, vmin=-1, center=0,
                        square=True, linewidths=.5, cbar_kws={"shrink": .5}, ax=ax2)
    ax1.set_title('Corr - Time Series')
    ax1.tick_params(labelsize=9)
    ax2.set_title('Corr - Stationary Time Series')
    ax2.tick_params(labelsize=9)
    plt.subplots_adjust(left=0.2, bottom=0.25, right=0.95, top=0.9, wspace=0.7)
    plt.savefig('Correlations_Time_Series_Heatmap__Kalmiopsis__1979.png', bbox_inches='tight')
    plt.show()
    # -------------------------------------------------------------------

    # -------------------------------------------------------------------
    # Seaborn Correlation Heatmap:
    # Location: All OR and WA Location Data
    # Time Series correlation and Stationary Time Series correlation:

    df_NARR_ERC_lon_lat_time_series_corr = df_NARR_ERC_lon_lat_time_series.corr() # Correlation
    print('df_NARR_ERC_lon_lat_time_series:\n', df_NARR_ERC_lon_lat_time_series.to_string())
    df_NARR_ERC_lon_lat_time_series_corr.apply(pd.to_numeric, errors='ignore')
    df_NARR_ERC_lon_lat_stationary_time_series_corr = df_NARR_ERC_lon_lat_time_series.diff().corr()
    print('df_NARR_ERC_lon_lat_time_series_corr:\n', df_NARR_ERC_lon_lat_time_series_corr.to_string())
    print('df_NARR_ERC_lon_lat_stationary_time_series_corr:\n', df_NARR_ERC_lon_lat_stationary_time_series_corr.to_string())
    
    # Generate a mask for the upper triangle
    # Time series corr:
    mask = np.zeros_like(df_NARR_ERC_lon_lat_time_series_corr, dtype=np.bool)
    mask[np.triu_indices_from(mask)] = True
    # Stationary time series corr:
    mask = np.zeros_like(df_NARR_ERC_lon_lat_stationary_time_series_corr, dtype=np.bool)
    mask[np.triu_indices_from(mask)] = True
    
    # Build diagonal heatmap:
    sns.set(style="white")
    f, (ax1, ax2) = plt.subplots(1,2, figsize=(9, 4))
    cmap = sns.diverging_palette(220, 10, as_cmap=True)
    sns.heatmap(df_NARR_ERC_lon_lat_time_series_corr, mask=mask, cmap=cmap, vmax=1, vmin=-1, center=0,
                        square=True, linewidths=.5, cbar_kws={"shrink": .5}, ax=ax1)
    sns.heatmap(df_NARR_ERC_lon_lat_stationary_time_series_corr, mask=mask, cmap=cmap, vmax=1, vmin=-1, center=0,
                        square=True, linewidths=.5, cbar_kws={"shrink": .5}, ax=ax2)
    ax1.set_title('Corr - Time Series')
    ax1.tick_params(labelsize=9)
    ax2.set_title('Corr - Stationary Time Series')
    ax2.tick_params(labelsize=9)
    plt.suptitle('OR and WA (8 Locations)')
    plt.subplots_adjust(left=0.2, bottom=0.25, right=0.95, top=0.9, wspace=0.7)
    plt.savefig('Correlations_Time_Series_Heatmap__OR_&_WA__1979.png', bbox_inches='tight')
    plt.show()
    # -------------------------------------------------------------------


    # -------------------------------------------------------------------
    # Plot of correlation coefficients for 1979 for Kalmiopsis Wilderness:

    df_ERC_0_corr = df_NARR_ERC_lon_lat_time_series_0_corr[['ERC']]
    df_ERC_0_corr.columns = ['ERC TS Corr']
    df_ERC_stat_0_corr = df_NARR_ERC_lon_lat_stationary_time_series_0_corr[['ERC']]
    df_ERC_stat_0_corr.columns = ['ERC TS Stat Corr']
    print('ERC Time Series correlation:\n', df_ERC_0_corr)
    print('ERC Stationary Time Series correlation:\n', df_ERC_stat_0_corr)
    df_ERC_corrs = pd.concat((df_ERC_0_corr, df_ERC_stat_0_corr), axis=1)
    print('df_ERC_corrs:\n', df_ERC_corrs)
    # Remove ERC row containing correlations of 1.0:
    df_ERC_corrs.drop(labels='ERC', axis=0, inplace=True)

    f, ax = plt.subplots(figsize=(5,4))
    df_ERC_corrs.plot.bar(use_index=True, ax=ax)
    ax.set_ylabel('Correlation')
    plt.title('NARR-ERC Correlations: Kalmiopsis (1979)')
    plt.savefig('Correlations_Time_Series_Barplot__Kalmiopsis__1979.png', bbox_inches='tight')
    plt.show()
    
    # -------------------------------------------------------------------
    # Plot of correlation coefficients for 1979 for all OR and WA:

    df_ERC_corr = df_NARR_ERC_lon_lat_time_series_corr[['ERC']]
    df_ERC_corr.columns = ['Time Series Corr']
    df_ERC_stat_corr = df_NARR_ERC_lon_lat_stationary_time_series_corr[['ERC']]
    df_ERC_stat_corr.columns = ['Stat. Time Series Corr']
    print('ERC Time Series correlation:\n', df_ERC_corr)
    print('ERC Stationary Time Series correlation:\n', df_ERC_stat_corr)
    df_ERC_corrs = pd.concat((df_ERC_corr, df_ERC_stat_corr), axis=1)
    print('df_ERC_corrs:\n', df_ERC_corrs)
    # Remove ERC row containing correlations of 1.0:
    df_ERC_corrs.drop(labels='ERC', axis=0, inplace=True)

    f, ax = plt.subplots(figsize=(5,4))
    df_ERC_corrs.plot.bar(use_index=True, ax=ax)
    ax.set_ylabel('Correlation')
    plt.title('NARR-ERC Correlations: OR & WA (1979)')
    plt.savefig('Correlations_Time_Series_Barplot__OR_&_WA__1979.png', bbox_inches='tight')
    plt.show()
    

    # Making a seaborn version of the above plot:
    df_ERC_corrs.reset_index(inplace=True)
    print('df_ERC_corrs index:\n', df_ERC_corrs.index)
    df_ERC_corrs.rename(columns={'index': 'Var'}, inplace=True)
    print('df_ERC_corrs:\n', df_ERC_corrs)

    df_ERC_tidy_corrs = pd.melt(df_ERC_corrs, id_vars=['Var'], value_vars=['Time Series Corr','Stat. Time Series Corr'], var_name='Corr Type', value_name='Corr')
    print('df_ERC_tidy_corrs:\n', df_ERC_tidy_corrs)
    plt.close()
    sns.barplot(x='Var', y='Corr', hue='Corr Type', data=df_ERC_tidy_corrs)
    plt.title('NARR-ERC Correlations: OR & WA (1979)')
    plt.savefig('Correlations_Time_Series_Barplot_Seaborn__OR_&_WA__1979.png', bbox_inches='tight')
    plt.show()
    # -------------------------------------------------------------------


    # Seaborn Pairplot. WAY TOO MUCH DATA TO MAKE SENSE OF THIS PLOT:
    # sns.set(style="ticks")
    # sns.pairplot(df_NARR_ERC_categorical, hue="ERC")
    # plt.show()

    # df_NARR_ERC_categorical_lon_lat_time_series_corr = df_NARR_ERC_categorical_lon_lat_time_series.groupby(['ERC']).corr()
    # print('df_NARR_ERC_categorical_lon_lat_time_series_corr:\n', df_NARR_ERC_categorical_lon_lat_time_series_corr.to_string())

    return


def build_monthly_NARR_gridMET(NARR_gridMET_pkl_in_dir, NARR_gridMET_csv_out_dir):
    # ----------------------------------------------------------------------------
    # NOTE: After running merge_NARR_gridMET(), go into /FWP/NARR_gridMET/, pull out
    #       all df_NARR_ERC_categorical.pkl files from their year folders,
    #       rename to df_NARR_ERC_categorical_1979.pkl, etc for every year, and put
    #       into /FWP/NARR_gridMET/. Then run this function to merge all years'
    #       January data into one df and export as csv files. It will do this
    #       for every month of the year. It will also combine all years' data
    #       into one giant csv file.
    #
    # PARAMETERS:
    #
    # NARR_gridMET_pkl_in_dir:      Directory of pickle NARR-gridMET directories
    #                               containing each year of dat.
    # 
    # SCRIPT FUNCTION:
    # 
    # 1)    NARR-gridMET data exists in the directory /FWP/NARR_gridMET.
    #       This function imports all pickle files ending in *YEAR.pkl and
    #       joins them into df_NARR_gridMET_all_years containing every year
    #       of NARR-gridMET data (this dataset can be massive).
    #
    # 2)    If any of the superfluous columns are present: TEMP, SPFH, TEMP, or
    #       CWTR, they are dropped.
    #
    # 3)    df_NARR_gridMET_all_years is used to create dataframes for each month
    #       of the year across all years:
    #           df_NARR_gridMET_1
    #           df_NARR_gridMET_2
    #                   ...
    #           df_NARR_gridMET_11
    #           df_NARR_gridMET_12
    #       If the imported years run from 1979 to 1983, then df_NARR_gridMET_Aug
    #       will be data from every august of 1979 through 1983 (4 months of data total)
    #       This is used to train the SOM on August only.
    #           
    # 4)    Exports to the same directory they were imported from and is placed in:
    #           /Training/Categorical or,
    #           /Training/Continuous
    #       depending on whether the ERC data is in categorical or continuous format.
    #       Exports in csv format only at the moment.
    #
    # NOTE: The exported csv df_NARR_gridMET_8 (for August) will then be imported by Julia.
    # NOTE: The exported pickle files are for backup purposes and to import for
    #       processing in Python if necessary.
    # ----------------------------------------------------------------------------

    # Import NARR-gridMET data for all years
    # NOTE: NARR_pkl_files are all pickle files named 'df_NARR_ERC_XXXX.pkl' (XXXX = year)
    continuous_path = NARR_gridMET_pkl_in_dir + 'Continuous/'
    categorical_path = NARR_gridMET_pkl_in_dir + 'Categorical/'

    NARR_gridMET_cont_pkl_files = glob.glob(os.path.join(continuous_path, '*.pkl'))
    NARR_gridMET_cat_pkl_files = glob.glob(os.path.join(categorical_path, '*.pkl'))

    # Sorting the files so they go into the dataframe in the correct order
    NARR_gridMET_cont_pkl_files = sorted(NARR_gridMET_cont_pkl_files)
    NARR_gridMET_cat_pkl_files = sorted(NARR_gridMET_cat_pkl_files)
    print('NARR_gridMET_cont_pkl_files:\n', NARR_gridMET_cont_pkl_files)
    print('NARR_gridMET_cat_pkl_files:\n', NARR_gridMET_cat_pkl_files)

    # Read in each year's pickle file and put them into one dataframe:
    df_cont = (pd.read_pickle(file) for file in NARR_gridMET_cont_pkl_files)
    df_cat = (pd.read_pickle(file) for file in NARR_gridMET_cat_pkl_files)
    df_NARR_gridMET_cont_all_years = pd.concat(df_cont, axis=0)
    df_NARR_gridMET_cat_all_years = pd.concat(df_cat, axis=0)
    print('df_NARR_gridMET_cont_all_years:\n', df_NARR_gridMET_cont_all_years)
    print('df_NARR_gridMET_cont_all_years columns:\n', df_NARR_gridMET_cont_all_years.columns)
    
    # Drop superfluous columns:
    df_NARR_gridMET_cont_all_years.drop(['TEMP','SPFH','PVEL','CWTR'], axis=1, inplace=True)
    df_NARR_gridMET_cat_all_years.drop(['TEMP','SPFH','PVEL','CWTR'], axis=1, inplace=True) 
    print('df_NARR_gridMET_cont_all_years:\n', df_NARR_gridMET_cont_all_years)
    print('df_NARR_gridMET_cont_all_years columns:\n', df_NARR_gridMET_cont_all_years.columns)
    df_NARR_gridMET_cont_all_years.set_index(['lat','lon'], inplace=True)
    df_NARR_gridMET_cat_all_years.set_index(['lat','lon'], inplace=True)

    # Convert time column to datetime:
    df_NARR_gridMET_cont_all_years['time'] = pd.to_datetime(df_NARR_gridMET_cont_all_years['time'])


    # ----------------------------------------
    # PLOTTING:
    # Get one location for plotting:
    df_NARR_gridMET_cont_all_years_loc = df_NARR_gridMET_cont_all_years.loc[(42.0588, 236.0590)]
    df_NARR_gridMET_cont_all_years_loc['time'] = pd.to_datetime(df_NARR_gridMET_cont_all_years_loc['time'])

    # H500 vs time
    plt.figure()
    plt.plot(df_NARR_gridMET_all_years_loc['time'], df_NARR_gridMET_all_years_loc['H500'])
    plt.show()

    # ERC vs time
    plt.figure()
    plt.plot(df_NARR_gridMET_all_years_loc['time'], df_NARR_gridMET_all_years_loc['ERC'])
    plt.show()
    # ----------------------------------------


    # Create month column containing month number (1 for January, etc)
    df_NARR_gridMET_cont_all_years['month'] = df_NARR_gridMET_cont_all_years['time'].dt.month
    df_NARR_gridMET_cat_all_years['month'] = df_NARR_gridMET_cat_all_years['time'].dt.month
    print('df_NARR_gridMET_cont_all_years:\n', df_NARR_gridMET_cont_all_years)
    print('df_NARR_gridMET_cont_all_years columns:\n', df_NARR_gridMET_cont_all_years.columns)

    # Create export directories for continuous and categorical data
    continuous_export_dir = NARR_gridMET_csv_out_dir + 'Continuous/'
    categorical_export_dir = NARR_gridMET_csv_out_dir + 'Categorical/'
    print('continuous_export_dir:\n', continuous_export_dir)
    print('categorical_export_dir:\n', categorical_export_dir)


    # # EXPORT ALL YEARS' DATA TO ONE CSV:
    # continuous_export_file = continuous_export_dir+'df_NARR_gridMET_all_years.csv'
    # categorical_export_file = categorical_export_dir+'df_NARR_gridMET_categorical_all_years.csv'
    # print('Exporting to csv the continuous and categorical data for all years. This could take a few minutes...')
    # df_NARR_gridMET_cont_all_years.to_csv(continuous_export_file, index=True, header=True)
    # df_NARR_gridMET_cat_all_years.to_csv(categorical_export_file, index=True, header=True)


    # EXPORT EACH MONTH TO INDIVIDUAL CSVs:
    # Iterate through month numbers 1 through 12 
    for month in range(1,13):
        print('month:\n', month)
        # Get one month for all years of data:
        df_NARR_gridMET_cont_month = df_NARR_gridMET_cont_all_years.loc[df_NARR_gridMET_cont_all_years['month']==month]
        df_NARR_gridMET_cat_month = df_NARR_gridMET_cat_all_years.loc[df_NARR_gridMET_cat_all_years['month']==month]
        # print('df_NARR_gridMET_cont_month:\n', df_NARR_gridMET_cont_month)
        print('Exporting month {} to csv:\n'.format(month))
        # Export directories with file names:
        continuous_export_file = continuous_export_dir+'df_NARR_gridMET_'+str(month)+'.csv'
        categorical_export_file = categorical_export_dir+'df_NARR_gridMET_categorical_'+str(month)+'.csv'
        # Export as csv:
        df_NARR_gridMET_cont_month.to_csv(continuous_export_file, index=True, header=True)
        df_NARR_gridMET_cat_month.to_csv(categorical_export_file, index=True, header=True)
    return


def export_NARR_gridMET_loc_to_rnn(NARR_gridMET_pkl_in_dir, NARR_gridMET_loc_out_dir, location):
    # ----------------------------------------------------------------------------
    # PARAMETERS:
    #
    # NARR_gridMET_csv_in_dir:      Directory of pickle NARR-gridMET directories
    #                               containing each year of data.
    #
    # NARR_gridMET_loc_pkl_out_dir: Directory of pickle NARR-gridMET for location
    #                               data
    #
    # loc:  Longitude-latitude coordinates
    #
    # time_steps:   Number of time_steps. All of them are fine as we only have
    #               1979 to 1983 (about 365*5 time steps)
    # 
    # SCRIPT FUNCTION:
    # 
    # 1)    Imports continuous (non-categorical) NARR ERC pickle data for all years
    #
    # 2)    Stacks them on top of one another
    #
    # 3)    Drops some columns and selects rows based on location coordinates
    #
    # 4)    Exports to a pickle file
    #
    # ----------------------------------------------------------------------------


    df = pd.read_pickle(NARR_gridMET_pkl_in_dir)
    print('Imported df:\n', df)
    
    df.reset_index(inplace=True)
    location_mask = (df['lat'] == location[0]) & (df['lon'] == location[1])

    # Make a column of normally distributed random noise for training baseline:
    df['noise'] = np.random.randn(df.shape[0])

    cols_to_keep = ['time','noise','H500','H500 Grad X','H500 Grad Y','PMSL','PMSL Grad X','PMSL Grad Y','ERC']
    df_location = df.loc[ location_mask, cols_to_keep ]
    print('df location shape:\n', df_location.shape)
    print('df_location head:\n', df_location.head().to_string())
    print('df_location tail:\n', df_location.tail().to_string())

    # ----------------------------------------

    # PLOTTING:
    fig, (ax0, ax1, ax2, ax3, ax4) = plt.subplots(1,5, figsize=(14,3))
    df_location.plot(x='time', y='noise', ax=ax0, color='gray')
    df_location.plot(x='time', y='H500 Grad X', ax=ax1, color='purple', alpha=0.7)
    df_location.plot(x='time', y='H500 Grad Y', ax=ax2, color='green', alpha=0.7)
    df_location.plot(x='time', y='H500', ax=ax3, color='blue', alpha=0.7)
    df_location.plot(x='time', y='ERC', ax=ax4, color='red')
    plt.show()

    # ----------------------------------------

    print('Exporting location to pickle and csv files...')
    df_location.to_pickle(NARR_gridMET_loc_out_dir + '/pickle/Training/' + 'df_NARR_gridMET_Kalmiopsis_loc_all_years.pkl')
    df_location.to_csv(NARR_gridMET_loc_out_dir + '/csv/Training/' + 'df_NARR_gridMET_Kalmiopsis_loc_all_years.csv', index=False)
    print('Export complete')

    return


def export_NARR_gridMET_reg_to_rnn(NARR_gridMET_pkl_in_dir, NARR_gridMET_reg_pkl_out_dir, region, region_name):
    # ----------------------------------------------------------------------------
    # PARAMETERS:
    #
    # NARR_gridMET_pkl_in_dir:      Directory of pickle NARR-gridMET directories
    #                               containing each year of data.
    #
    # NARR_gridMET_reg_pkl_out_dir: Directory of pickle NARR-gridMET for regional
    #                               data
    #
    # region:  Longitude-latitude coordinates from SW corner to NE corner of region
    #
    # time_steps:   No ability to select the number of time steps, may be added
    #               at some point. Each time step is one day.
    # 
    # SCRIPT FUNCTION:
    # 
    # 1)    Imports continuous (non-categorical) NARR ERC pickle data for all years
    #
    # 2)    Stacks them on top of one another
    #
    # 3)    Drops some columns and selects rows based on region coordinates
    #
    # 4)    Exports to a pickle file
    #
    # ----------------------------------------------------------------------------

    print('Region:\n', region)
    df = pd.read_pickle(NARR_gridMET_pkl_in_dir)

    print('df:\n', df.head().to_string())
    print('df.shape:', df.shape)

    df.reset_index(inplace=True)

    # Entire dataset:
    print('df lat-lon min max in the dataset:', ((np.min(df.lat), np.min(df.lon)), (np.max(df.lat), np.max(df.lon))))
    unique_lat = np.unique(df.lat)
    unique_lon = np.unique(df.lon)
    n_lat = len(unique_lat)
    n_lon = len(unique_lon)
    print('Number of latitudes in OR and WA:', n_lat)
    print('Number of longitudes in OR and WA:', n_lon)
    print('Number of locations in OR and WA:', n_lat*n_lon)

    # Select region:
    region_mask = (df['lat'] <= region[1][0]) & (df['lat'] >= region[0][0]) & (df['lon'] <= region[1][1]) & (df['lon'] >= region[0][1])
    cols_to_keep = ['time','lat','lon','H500','H500 Grad X','H500 Grad Y','PMSL','PMSL Grad X','PMSL Grad Y','ERC']
    df_region = df.loc[ region_mask, cols_to_keep ]

    # Remove invalid ERC values if present
    df_region = df_region.loc[ df_region.ERC >= 0, : ]

    # Region data:
    print('df_region shape:', df_region.shape)
    print('Region:', region)
    print('df_region head:\n', df_region.head().to_string())
    print('df_region tail:\n', df_region.tail().to_string())

    unique_lat = np.unique(df_region.lat)
    unique_lon = np.unique(df_region.lon)
    n_lat = len(unique_lat)
    n_lon = len(unique_lon)
    print('Number of latitudes in region:', n_lat)
    print('Number of longitudes in region:', n_lon)
    print('Number of locations in region:', n_lat*n_lon)

    # Making unique (lat,lon) location pairs:
    locations = [(x,y) for x in unique_lat for y in unique_lon]
    print('Locations:', locations)
    df_reg_key = pd.DataFrame(index=range(0,len(locations))) # Make empty so that the locations list can be put into one column 'loc',
    df_reg_key['loc'] = locations # otherwise pd.DataFrame(data=locations, columns='loc') fails as it tries
    df_reg_key.index.set_names(names='key', inplace=True) # to split the list's (lat,lon) tuples into two columns.
    print('df_reg_key:\n', df_reg_key)

    mapper = {k:v for k,v in zip(locations,range(len(locations)))}
    print('Mapper:', mapper)

    # Merge lat and lon columns in df_region:
    df_region['lat_lon'] = list(zip(df_region.lat, df_region.lon))

    print('df_region:\n', df_region)

    # Map locations to integers:
    df_region['loc'] = df_region['lat_lon'].map(mapper)
    print('df_region:\n', df_region)
    # Pivoting:
    df_region_piv = df_region.pivot(index='time', columns='loc', values=['H500','PMSL','H500 Grad X','H500 Grad Y','PMSL Grad X','PMSL Grad Y','ERC'])
    print('Using pivot: df_region_piv:\n', df_region_piv)

    ''' pivot_table conversion isn't needed: '''
    # # USING pd.pivot_table(df_region):
    # df_region_pt = pd.pivot_table(df_region, index='time', columns='loc', values=['H500','PMSL','ERC'])
    # print('Using pivot_table: df_region_pt:\n', df_region_pt)

    # Manipulating the multi-level columns into a single level
    multi_level_cols = list(df_region_piv.columns)
    one_level_cols = [x[0]+'_'+str(x[1]) for x in list(multi_level_cols)]
    df_region_piv.columns = one_level_cols
    print('df_region_piv with one level columns:\n', df_region_piv)

    df_reg_key.to_pickle(NARR_gridMET_reg_out_dir + '/pickle/Training/' + 'df_NARR_gridMET_'+region_name+'_reg_key.pkl')
    df_reg_key.to_csv(NARR_gridMET_reg_out_dir + '/csv/Training/' + 'df_NARR_gridMET_'+region_name+'_reg_key.csv', index=True)

    df_region_piv.to_pickle(NARR_gridMET_reg_out_dir + '/pickle/Training/' + 'df_NARR_gridMET_'+region_name+'_reg_all_years.pkl')
    df_region_piv.to_csv(NARR_gridMET_reg_out_dir + '/csv/Training/' + 'df_NARR_gridMET_'+region_name+'_reg_all_years.csv', index=True)
    return


# ----------------------------------------------- EXPORTING TO R & JULIA: -----------------------------------------------


def export_NARR_ERC_categorical__R_Julia(month_num, cols, NARR_gridMET_csv_in_dir, NARR_gridMET_csv_out_dir):
    # ----------------------------------------------------------------------------
    # NOTE: After running build_monthly_NARR_gridMET(), this function imports
    #       df_NARR_ERC_categorical.csv from FWP/NARR_gridMET/ and processes it
    #       for export to csv format for SOM training in R and Julia.
    #       This needs to be run for the month of the year for which you want to
    #       train the SOM.
    #
    # PARAMETERS:
    #
    #       month_num:                  8 is August
    #
    #       cols:                       Columns you want in the final csv
    #                                   files, e.g. cols = ['H500', 'ERC']
    #
    #       NARR_gridMET_csv_in_dir:    Directory of NARR-gridMET csv data
    #                                   containing all years worth of data
    #       
    #       NARR_gridMET_csv_out_dir:   csv output directory
    # 
    # SCRIPT FUNCTION:
    # 
    # 1)    NARR-gridMET data exists in the directory /FWP/NARR_gridMET.
    #       This function imports df_NARR_ERC_categorical.csv
    #
    # 2)    It iterates through a for loop to use the same naming conventions
    #       for making all_years data and monthly data
    #
    # 3)    Exports as csv files to R and Julia specific folders
    #
    # 4)    Normalizes and scales the data into x_train, y_train, x_test,
    #       y_test, xy_train, xy_test, etc and exports to Julia specific
    #       folder for SOM training.
    # ----------------------------------------------------------------------------

    # Using this loop allows for one naming convention, thus half the code
    format_types = ['all_years','monthly']
    for ft in format_types: # Either 'all_years' or 'monthly'
        
        # categorical_import_file example:
        # '/home/dp/Documents/FWP/NARR_gridMET/csv/Training/df_NARR_gridMET_categorical_all_years.csv'
        # '/home/dp/Documents/FWP/NARR_gridMET/csv/Training/df_NARR_gridMET_categorical_8.csv'
        if ft == 'all_years':
            categorical_import_file   = NARR_gridMET_csv_in_dir + 'df_NARR_gridMET_categorical_' + ft + '.csv'
        else:
            categorical_import_file   = NARR_gridMET_csv_in_dir + 'df_NARR_gridMET_categorical_' + str(month_num) + '.csv'
        
        # Ouput directories for R Training and Julia Training:
        csv_out_R     = NARR_gridMET_csv_out_dir + 'R/Training/'
        csv_out_Julia = NARR_gridMET_csv_out_dir + 'Julia/Training/'

        # Importing either all years of data or one month's data across all years:
        df = pd.read_csv(categorical_import_file, header='infer')
        print('df:\n', df)

        # Grab columns of interest:
        # cols = ['time','H500 Grad X', 'H500 Grad Y', 'PMSL Grad X', 'PMSL Grad Y', 'ERC']
        df = df[cols]
        
        # Removing invalid ERC value rows:
        invalid_mask = (df['ERC'] != 'invalid') # monthly
        df = df.loc[ invalid_mask ]             # removing invalid ERC

        """
        Export unnormalized, unscaled data to R and Julia folder for training runs. NOTE that
        R does the normalization and scaling itself. Julia's SOM.jl does the normalization and
        scaling as well but I've written fire_weather_som.jl to use xy_train.csv and xy_test.csv,
        so it won't be expecting 'data_all_years.csv' or 'data_monthly.csv'
        """

        if ft == 'all_years':
            df.to_csv(csv_out_R + 'data_all_years.csv', index=False)
            df.to_csv(csv_out_Julia + 'data_all_years.csv', index=False)
        else:
            df.to_csv(csv_out_R + 'data_monthly.csv', index=False)
            df.to_csv(csv_out_Julia + 'data_monthly.csv', index=False)

        """
        Beyond this point, creating a scaled and normalized train-test split
        is done for export to Julia. The dataframes as they stand at this point
        are exported just before this text.
        """

        # Make normalized and scaled data for Julia
        if ft == 'monthly':
            # Creating class weights (the weights aren't used anywhere, just using ):
            from sklearn.utils import class_weight
            erc_class_weights = class_weight.compute_class_weight(class_weight='balanced', classes=np.unique(df['ERC']), y=df['ERC'])
            print('erc_class_weights:\n', erc_class_weights)

            # --------------------------------------
            # Determining class imablance:
            classes = np.unique(df['ERC']) # Identify unique classes
            # print('classes:\n', classes)
            num_samples = len(df.index)
            # Loop through each class, count occurrences, compute class frequency:
            for c in classes:
                class_count = df.ERC.str.count(c).sum()
                # print('class_count:\n', class_count)
                class_freq =  round(class_count / num_samples * 100)
                print('{} class frequency = {}'.format(c, class_freq))

            # Another way of calculating class frequencies using collections.Counter:
            from collections import Counter
            counter = Counter(df['ERC']) # creates a dictionary {'extreme': 167448, 'very high': 31114, 'high': 20670, 'moderate': 12749, 'low': 6099}
            # print('counter:\n', counter)
            class_sums = sum(counter.values()) # sums all values of the dictionary
            class_freqs = {key:round(val / class_sums * 100) for key, val in counter.items()} # calculates class frequencies from the dictionary's items
            print('Class frequencies using collections.Counter:\n', class_freqs)
            # --------------------------------------

            # Scaling:
            x = df[cols[0:-1]] # Getting first column H500, avoiding last column ERC
            y = df[['ERC']]
            x_train, x_test, y_train, y_test = train_test_split(x, y , train_size=0.7, random_state=90)
            print('x_train:\n', x_train)
            print('y_train:\n', y_train)

            # Normalize Training Data (std_scale is used on train and test data)
            std_scale = preprocessing.StandardScaler().fit(x_train)
            
            # What is transforming doing? x_train_norm is a numpy array
            x_train_norm = std_scale.transform(x_train)
            
            #Converting numpy array to dataframe
            x_train_norm = pd.DataFrame(x_train_norm, index=x_train.index, columns=x_train.columns)

            # Normalize Testing Data by using mean and SD of training set
            x_test_norm = std_scale.transform(x_test) # x_test_norm is a numpy array
            x_test_norm = pd.DataFrame(x_test_norm, index=x_test.index, columns=x_test.columns)

            # Joining auto-aligns on the left df's index:
            xy_train = x_train.join(y_train)
            xy_test = x_test.join(y_test)

            # Sorting but not necessary:
            # xy_train.sort_index(inplace=True)
            # xy_test.sort_index(inplace=True)
            # print('xy_train joined:\n', xy_train.head())
            # print('xy_test joined:\n', xy_test.head())

            # Normalizing:
            xy_train_norm = x_train_norm.join(y_train)
            xy_test_norm = x_test_norm.join(y_test)

    # --------------------------------------
    # Not using this at the moment because R is doing its own
    # normalization and scaling:
    # x_train.to_csv(csv_out_R + 'x_train.csv', index=False)
    # x_test.to_csv(csv_out_R + 'x_test.csv', index=False)
    # y_train.to_csv(csv_out_R + 'y_train.csv', index=False)
    # y_test.to_csv(csv_out_R + 'y_test.csv', index=False)
    # xy_train.to_csv(csv_out_R + 'xy_train.csv', index=False)
    # xy_test.to_csv(csv_out_R + 'xy_test.csv', index=False)
    # x_train_norm.to_csv(csv_out_R + 'x_train_norm.csv', index=False)
    # x_test_norm.to_csv(csv_out_R + 'x_test_norm.csv', index=False)
    # xy_train_norm.to_csv(csv_out_R + 'xy_train_norm.csv', index=False)
    # xy_test_norm.to_csv(csv_out_R + 'xy_test_norm.csv', index=False)

    # Export to Julia folder
    x_train.to_csv(csv_out_Julia + 'x_train.csv', index=False)
    x_test.to_csv(csv_out_Julia + 'x_test.csv', index=False)
    y_train.to_csv(csv_out_Julia + 'y_train.csv', index=False)
    y_test.to_csv(csv_out_Julia + 'y_test.csv', index=False)
    xy_train.to_csv(csv_out_Julia + 'xy_train.csv', index=False)
    xy_test.to_csv(csv_out_Julia + 'xy_test.csv', index=False)
    x_train_norm.to_csv(csv_out_Julia + 'x_train_norm.csv', index=False)
    x_test_norm.to_csv(csv_out_Julia + 'x_test_norm.csv', index=False)
    xy_train_norm.to_csv(csv_out_Julia + 'xy_train_norm.csv', index=False)
    xy_test_norm.to_csv(csv_out_Julia + 'xy_test_norm.csv', index=False)

    return


# export_NARR_ERC_one_year__R_Julia imports df_NARR_ERC_categorical.csv for a particular year 
# and processes it for export to csv format for SOM training in R and Julia:
def export_NARR_ERC_one_year__R_Julia():
    # Exporting gradients and ERC to csv:
    df = pd.read_csv('/home/dp/Documents/FWP/NARR_gridMET/csv/1979/df_NARR_ERC_categorical.csv', header='infer')
    print('df:\n', df)
    # cols = ['time','H500 Grad X', 'H500 Grad Y', 'PMSL Grad X', 'PMSL Grad Y', 'ERC']
    cols = ['time','H500', 'ERC'] # Selecting the desired columns
    start_date_str = '1979-06-01'
    # end_date_str =   '1979-08-31'
    end_date_str =   '1979-06-30'
    # export_filename = 'NARR_ERC_June_thru_August_1979_gradients_categorical.csv'
    export_filename = 'NARR_ERC_June_1979_h500_only_categorical.csv'

    df = df[cols]
    invalid_mask = (df['ERC'] != 'invalid')
    print('invalid_mask:\n', invalid_mask)
    df = df.loc[ invalid_mask ]   # removing invalid ERC
    df['time'] = pd.to_datetime(df['time'])                 # convert time column to datetime
    df.set_index('time', inplace=True)                                # set time column to index
    start_date = datetime.strptime(start_date_str, '%Y-%m-%d').date()             # specify start and end dates
    end_date = datetime.strptime(end_date_str, '%Y-%m-%d').date()
    df = df.loc[[start_date,end_date]]                 # get data from date range
    df.reset_index(drop=True, inplace=True) # Don't need time index anymore

    # Scaling:
    x = df[cols[1:-1]] # cols[1,-1] avoids first time column and last ERC column
    y = df[['ERC']]
    x_train, x_test, y_train, y_test = train_test_split(x, y , train_size=0.7, random_state=90)
    print('x_train:\n', x_train)
    print('y_train:\n', y_train)

    # Normalize Training Data (std_scale is used on train and test data)
    std_scale = preprocessing.StandardScaler().fit(x_train)
    
    x_train_norm = std_scale.transform(x_train) # x_train_norm is a numpy array
    #Converting numpy array to dataframe
    x_train_norm = pd.DataFrame(x_train_norm, index=x_train.index, columns=x_train.columns) 


    # Normalize Testing Data by using mean and SD of training set
    x_test_norm = std_scale.transform(x_test) # x_test_norm is a numpy array
    x_test_norm = pd.DataFrame(x_test_norm, index=x_test.index, columns=x_test.columns)

    # WARNING: NOT USING ANY AXIS ALIGNMENT. THIS WORKS BUT MAY NOT BE GOOD PRACTICE.
    xy_train = pd.concat((x_train, y_train), axis=1)
    xy_test = pd.concat((x_test, y_test), axis=1)
    xy_train_norm = pd.concat((x_train_norm, y_train), axis=1)
    xy_test_norm = pd.concat((x_test_norm, y_test), axis=1)

    # NOTE: THE EXPORT FILENAMES ARE GENERIC. SHOULD FIND A MORE SUITABLE NAME.
    # Export to R folder
    x_train.to_csv(csv_out_R + 'x_train.csv', index=False)
    x_test.to_csv(csv_out_R + 'x_test.csv', index=False)
    y_train.to_csv(csv_out_R + 'y_train.csv', index=False)
    y_test.to_csv(csv_out_R + 'y_test.csv', index=False)
    xy_train.to_csv(csv_out_R + 'xy_train.csv', index=False)
    xy_test.to_csv(csv_out_R + 'xy_test.csv', index=False)
    x_train_norm.to_csv(csv_out_R + 'x_train_norm.csv', index=False)
    x_test_norm.to_csv(csv_out_R + 'x_test_norm.csv', index=False)
    xy_train_norm.to_csv(csv_out_R + 'xy_train_norm.csv', index=False)
    xy_test_norm.to_csv(csv_out_R + 'xy_test_norm.csv', index=False)

    # Export to Julia folder
    x_train.to_csv(csv_out_Julia + 'x_train.csv', index=False)
    x_test.to_csv(csv_out_Julia + 'x_test.csv', index=False)
    y_train.to_csv(csv_out_Julia + 'y_train.csv', index=False)
    y_test.to_csv(csv_out_Julia + 'y_test.csv', index=False)
    xy_train.to_csv(csv_out_Julia + 'xy_train.csv', index=False)
    xy_test.to_csv(csv_out_Julia + 'xy_test.csv', index=False)
    x_train_norm.to_csv(csv_out_Julia + 'x_train_norm.csv', index=False)
    x_test_norm.to_csv(csv_out_Julia + 'x_test_norm.csv', index=False)
    xy_train_norm.to_csv(csv_out_Julia + 'xy_train_norm.csv', index=False)
    xy_test_norm.to_csv(csv_out_Julia + 'xy_test_norm.csv', index=False)

    return


def plot_SOM_results__R():
    changes100 = pd.read_csv('/home/dp/Documents/FWP/R/SOM_changes_100iters.csv', header=0, delim_whitespace=True)
    changes200 = pd.read_csv('/home/dp/Documents/FWP/R/SOM_changes_200iters.csv', header=0, delim_whitespace=True)
    changes400 = pd.read_csv('/home/dp/Documents/FWP/R/SOM_changes_400iters.csv', header=0, delim_whitespace=True)
    changes800 = pd.read_csv('/home/dp/Documents/FWP/R/SOM_changes_800iters.csv', header=0, delim_whitespace=True)
    changes1600 = pd.read_csv('/home/dp/Documents/FWP/R/SOM_changes_1600iters.csv', header=0, delim_whitespace=True)
    changes2200 = pd.read_csv('/home/dp/Documents/FWP/R/SOM_changes_2200iters.csv', header=0, delim_whitespace=True)
    changes4400 = pd.read_csv('/home/dp/Documents/FWP/R/SOM_changes_4400iters.csv', header=0, delim_whitespace=True)
    # changes8800 = pd.read_csv('/home/dp/Documents/FWP/R/SOM_changes_8800iters.csv', header=0, delim_whitespace=True)

    df = pd.DataFrame([])
    df = pd.concat((changes100, changes200, changes400, changes800, changes1600, changes2200, changes4400), axis=1, ignore_index=True)
    df.columns = ['100','200','400','800','1600','2200','4400']
    
    df.plot(y=['100','200','400','800','1600','2200','4400'], kind='line')
    plt.show()

    return


def synvar_pickle_to_csv(pickle_in_filename, csv_out_filename, cols_list):
    # ----------------------------------------------------------------------------
    # PARAMETERS:
    #
    #
    #
    # SCRIPT FUNCTION:
    #
    # Import pickle file, export as csv for SOM training in Julia:
    # ----------------------------------------------------------------------------

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



# ----------------------------------------
''' Writing CSV files to Postgres '''
# table = 'narr'        # narr, narr_erc, narr_erc_categorical
# fire_weather_db(table)
# ----------------------------------------



# ----------------------------------------
''' Import NARR data (rectilinear grid) '''
# lon_min = 235
# lat_min = 244
# lon_max = 41
# lat_max = 50
# import_interval = 2
# export_interval = 10
# multi = False
# NARR_csv_in_dir  = '/home/dp/Documents/FWP/NARR/csv/1983/'  #'/mnt/seagate/NARR/3D/temp/csv/'  # '/home/dp/Documents/FWP/NARR/csv_exp/rectilinear_grid/'
# NARR_csv_out_dir = '/home/dp/Documents/FWP/NARR/csv/1983/'     # Used in Postgres
# NARR_pkl_out_dir = '/home/dp/Documents/FWP/NARR/pickle/1983/'     #'/home/dp/Documents/FWP/NARR/pickle_exp/rectilinear_grid/'

# import_NARR_csv(lon_min, lon_max, lat_min, lat_max, import_interval, export_interval, multi, NARR_csv_in_dir, NARR_csv_out_dir, NARR_pkl_out_dir)
# ----------------------------------------



# ----------------------------------------
''' Import all gridMET CSVs '''
# gridMET_csv_in_dir  = '/home/dp/Documents/FWP/gridMET/csv/1981/'
# gridMET_pkl_out_dir = '/home/dp/Documents/FWP/gridMET/pickle/1981/'

# import_gridMET_csv(gridMET_csv_in_dir, gridMET_pkl_out_dir)
# ----------------------------------------



# ----------------------------------------
# ''' Merge NARR and gridMET '''
# start_date               = '1983,1,1'
# end_date                 = '1983,12,31'
# multi                    = False    # multi refers to extra variables pulled from NCL. 1980 and 1981 set to True, all other years set False.
# gridMET_pkl_in_dir       = '/home/dp/Documents/FWP/gridMET/pickle/1983/'
# NARR_pkl_in_dir          = '/home/dp/Documents/FWP/NARR/pickle/1983/'       # '/home/dp/Documents/FWP/NARR/pickle/'
# NARR_gridMET_pkl_out_dir = '/home/dp/Documents/FWP/NARR_gridMET/pickle/1983/'
# NARR_gridMET_csv_out_dir = '/home/dp/Documents/FWP/NARR_gridMET/csv/1983/'

# merge_NARR_gridMET(start_date, end_date, multi, gridMET_pkl_in_dir, NARR_pkl_in_dir, NARR_gridMET_pkl_out_dir, NARR_gridMET_csv_out_dir)
# ----------------------------------------



# ----------------------------------------
''' Stack NARR and gridMET '''
# NARR_gridMET_pkl_in_dir  = '/home/dp/Documents/FWP/NARR_gridMET/pickle/'
# NARR_gridMET_pkl_out_dir = '/home/dp/Documents/FWP/NARR_gridMET/pickle/Training/'
# NARR_gridMET_csv_out_dir = '/home/dp/Documents/FWP/NARR_gridMET/csv/Training/'

# stack_NARR_gridMET(NARR_gridMET_pkl_in_dir, NARR_gridMET_pkl_out_dir, NARR_gridMET_csv_out_dir)
# ----------------------------------------



# ----------------------------------------
''' Plot NARR and gridMET data '''
# plot_date               = '1983,08,10'
# multi                   = False
# plot_lat                = (42.0588,  42.0588,  44.6078,  42.3137,  47.4118,  47.4118,  48.4314,  48.9412) # First four: OR. Last four: WA
# plot_lon                = (236.0590, 238.6080, 241.1570, 237.0780, 236.5690, 238.6080, 242.4310, 238.6080) # Kalmiopsis, Medford, Deschutes, John Day, Olympics, Snoqualmie, Colville, N Cascades
# NARR_gridMET_pkl_in_dir = '/home/dp/Documents/FWP/NARR_gridMET/pickle/1983/'

# plot_NARR_gridMET(plot_date, multi, plot_lon, plot_lat, NARR_gridMET_pkl_in_dir)
# ----------------------------------------



# ----------------------------------------
# ''' Build monthly NARR-gridMET '''
# NARR_gridMET_pkl_in_dir = '/home/dp/Documents/FWP/NARR_gridMET/pickle/'
# NARR_gridMET_csv_out_dir = '/home/dp/Documents/FWP/NARR_gridMET/csv/Training/'

# build_monthly_NARR_gridMET(NARR_gridMET_pkl_in_dir, NARR_gridMET_csv_out_dir)
# ----------------------------------------



# ----------------------------------------
# ''' Export NARR-gridMET location to RNN '''
# NARR_gridMET_pkl_in_dir = '/home/dp/Documents/FWP/NARR_gridMET/pickle/Training/df_NARR_gridMET_all_years.pkl'
# NARR_gridMET_loc_out_dir = '/home/dp/Documents/FWP/NARR_gridMET/'
# location = (42.0588, 236.0590)
# export_NARR_gridMET_loc_to_rnn(NARR_gridMET_pkl_in_dir, NARR_gridMET_loc_out_dir, location)
# ----------------------------------------



# ----------------------------------------
# ''' Export NARR-gridMET region to RNN '''
NARR_gridMET_pkl_in_dir = '/home/dp/Documents/FWP/NARR_gridMET/pickle/Training/df_NARR_gridMET_all_years.pkl'
NARR_gridMET_reg_out_dir = '/home/dp/Documents/FWP/NARR_gridMET/'
region = ((39.2549, 235.294), (46.3, 244))
region_name = 'OR'
#((42.0588, 236.0590), (42.3137, 236.314)) # Kalmiopsis region
#((42, 236), (45, 238)) # From Oregon-California border south of Kalmiopsis up to Warm Springs reservation (96 grid points)
#
export_NARR_gridMET_reg_to_rnn(NARR_gridMET_pkl_in_dir, NARR_gridMET_reg_out_dir, region, region_name)
# ----------------------------------------



# ----------------------------------------
''' Run synvar_pickle_to_csv '''
# synvar_pickle_to_csv('df_all_synvar_grid_interp.pkl','df_all_synvar_grid_interp.csv',['H500 Grad X'])
# ----------------------------------------



# THIS FUNCTION TAKES IN df_MULTI.csv DATA AND PLOTS ITS VARIABLES INCLUDING H500, TEMP, SPFH, PVEL, CWTR. NOT
# VERY USEFUL BECAUSE THE MULTI DATA CAN NOW BE PROCESSED BY import_NARR_csv():
# ----------------------------------------
''' Import Multi 3D NARR data (rectilinear grid) '''
# lon_min = 235
# lat_min = 244
# lon_max = 41
# lat_max = 50
# import_interval = 2
# export_interval = 10
# multi = False
# NARR_csv_in_dir  = '/home/dp/Documents/FWP/NARR/csv/1983/'  #'/mnt/seagate/NARR/3D/temp/csv/'  # '/home/dp/Documents/FWP/NARR/csv_exp/rectilinear_grid/'
# NARR_csv_out_dir = '/home/dp/Documents/FWP/NARR/csv/1983/'     # Used in Postgres
# NARR_pkl_out_dir = '/home/dp/Documents/FWP/NARR/pickle/1983/'     #'/home/dp/Documents/FWP/NARR/pickle_exp/rectilinear_grid/'

# import_multi_NARR_csv(lon_min, lon_max, lat_min, lat_max, import_interval, export_interval, multi, NARR_csv_in_dir, NARR_csv_out_dir, NARR_pkl_out_dir)
# ----------------------------------------


# def plot_multi_NARR_csv():
#     # IMPORT NARR MULTI 3D:
#     # df = pd.read_csv('/home/dp/Documents/FWP/NARR/csv/1980/df_MULTI.csv', header='infer', index_col=None)
#     df = pd.read_pickle('/home/dp/Documents/FWP/NARR/pickle/1983/df_MULTI.pkl')
#     df.reset_index(inplace=True)
#     df.set_index(['lat','lon'], inplace=True)
#     print('df:\n', df)

#     df_loc = df.loc[[(43.0784,241.412)]]# 39.0, 233.000)]]
#     df.reset_index(inplace=True)
#     # df_loc['SPFH RM'] = df_loc['SPFH'].rolling(30).mean()
#     # print('df_loc[[SPFH, SPFH RM]]:\n', df_loc[['SPFH', 'SPFH RM']])

#     df_loc.reset_index(inplace=True)
#     df_loc.set_index('time', inplace=True)
#     # df_loc_fall = df_loc.loc['1981-09-15':'1981-11-15']
#     # print('df_loc_fall:\n', df_loc_fall.to_string())

#     df_loc.reset_index(inplace=True)
#     df.set_index(['lat','lon'], inplace=True)


#     # Plotting time to identify missing data
#     plt.close()
#     plt.figure()
#     plt.scatter(x=df_loc.index.values, y=df_loc['time'], c='k', edgecolor='k', facecolor='white', s=3)
#     plt.show()

#     # Plotting H500
#     plt.close()
#     f, (ax1, ax2, ax3) = plt.subplots(1,3, figsize=(10,4))
#     df_loc.plot(x='time', y='H500', ax=ax1, c='#673a8e')
#     df_loc.plot(x='time', y='H500 Grad X', ax=ax2, c='#adcfd8')
#     df_loc.plot(x='time', y='H500 Grad Y', ax=ax3, c='#50844b')
#     # df_erc_loc.plot(x='time', y='erc', ax=ax5, c='r')
#     plt.subplots_adjust(hspace=0.3)
#     plt.savefig('NARR H500', bbox_inches='tight')
#     plt.show()

#     # IMPORT ERC:
#     df_erc = pd.read_pickle('/home/dp/Documents/FWP/gridMET/pickle/1983/df_erc.pkl')
#     df_erc.reset_index(inplace=True)
#     df_erc['lon'] = df_erc['lon'] + 360
#     df_erc.set_index(['lat','lon'], inplace=True)
#     print('df_erc:\n', df_erc)
#     df_erc_loc = df_erc.loc[(43.0667, 241.4)]
#     print('df_erc_loc:\n', df_erc_loc)


#     # Plotting all multi 3D NARR:
#     plt.close()
#     f, (ax1, ax2, ax3, ax4, ax5) = plt.subplots(5,1, figsize=(4,10))
#     df_loc.plot(x='time', y='H500', ax=ax1, c='#673a8e')
#     df_loc.plot(x='time', y='TEMP', ax=ax2, c='#adcfd8')
#     df_loc.plot(x='time', y='SPFH', ax=ax3, c='#50844b')
#     df_loc.plot(x='time', y='SPFH RM', ax=ax3, c='#2a4c27')
#     df_loc.plot(x='time', y='CWTR', ax=ax4, c='#821b1b')
#     df_erc_loc.plot(x='time', y='erc', ax=ax5, c='r')
#     plt.subplots_adjust(hspace=0.3)
#     plt.savefig('NARR 3D Multi', bbox_inches='tight')
#     plt.show()

#     return

# # plot_multi_NARR_csv()



def plot_SOM_results__Julia(csv_import_dir):
    # ------------------------------------------------------------------------------------
    # PURPOSE: This function plots the following csv files from Julia training and testing SOM runs:
    #   
    #   Training:
    #       FWP_SOM_trained_specific_diff.csv
    #       FWP_SOM_trained_specific_codes.csv
    #       FWP_SOM_trained_specific_pop.csv
    #   
    #   Testing data:
    #       FWP_SOM_testing_specific_diff.csv
    #       FWP_SOM_testing_specific_codes.csv
    #       FWP_SOM_testing_specific_pop.csv
    #
    # The _diff files show the difference between the final trained or tested
    # codebook vectors and the initialized codebook vectors.
    #
    # Dozens of SOMs are trained and tested to cover a wide range of timepoints
    # in the development of a final SOM to show learning (SOM.jl doesn't keep track
    # of learning information on its own)
    #
    # IMPROVEMENTS: Use glob to get csv file list, iterate through each file, importing and
    #               plotting it, and saving the plot.
    # ------------------------------------------------------------------------------------

    # Trained SOM:
    df_som_learn = pd.read_csv(csv_import_dir+'FWP_SOM_trained_specific_diff.csv')
    num_cols = len(df_som_learn.columns)
    col_names = [str(x) for x in range(1,num_cols+1)]
    df_som_learn.columns = col_names
    # print('df_som_learn:\n', df_som_learn)
    df_som_learn_rolling = df_som_learn.rolling(10).mean()
    print('df_som_learn_rolling:\n', df_som_learn_rolling)
    plt.figure()
    plt.plot(df_som_learn)
    plt.plot(df_som_learn_rolling)
    plt.legend(['1','2','3','4'])
    plt.xlabel('Iteration')
    plt.ylabel('Final Trained Weight - Initial Train Weight')
    plt.title('Final Trained SOMs - Initalized Train SOM: 3rd neuron\'s weights')
    plt.savefig('SOM_Trained_Diffs.png', bbox_inches='tight')
    plt.show()

    # Tested SOM:
    df_som_learn = pd.read_csv(csv_import_dir+'FWP_SOM_tested_specific_diff.csv')
    num_cols = len(df_som_learn.columns)
    col_names = [str(x) for x in range(1,num_cols+1)]
    df_som_learn.columns = col_names
    df_som_learn_rolling = df_som_learn.rolling(10).mean()
    plt.figure()
    plt.plot(df_som_learn)
    plt.plot(df_som_learn_rolling)
    plt.legend(['1','2','3','4'])
    plt.xlabel('Iteration')
    plt.ylabel('Final Tested Weight - Initial Test Weight')
    plt.title('Final Tested SOMs - Initalized Test SOM: 15th neuron\'s weights')
    plt.savefig('SOM_Tested_Diffs.png', bbox_inches='tight')
    plt.show()
    # ------------------------------------------------------------------------------------
    df_som_code = pd.read_csv(csv_import_dir+'FWP_SOM_trained_specific_codes.csv')
    num_cols = len(df_som_code.columns)
    col_names = [str(x) for x in range(1,num_cols+1)]
    df_som_code.columns = col_names
    print('df_som_code:\n', df_som_code)
    df_som_code_rolling = df_som_code.rolling(10).mean()
    plt.figure()
    plt.plot(df_som_code)
    plt.plot(df_som_code_rolling)
    plt.legend(['1','2','3','4'])
    plt.xlabel('Iteration')
    plt.ylabel('Weight')
    plt.title('Final Trained SOMs: 15th neuron\'s weights')
    plt.savefig('SOM_Trained_Codes.png', bbox_inches='tight')
    plt.show()
   
    df_som_code = pd.read_csv(csv_import_dir+'FWP_SOM_tested_specific_codes.csv')
    num_cols = len(df_som_code.columns)
    col_names = [str(x) for x in range(1,num_cols+1)]
    df_som_code.columns = col_names
    df_som_code_rolling = df_som_code.rolling(10).mean()
    plt.figure()
    plt.plot(df_som_code)
    plt.plot(df_som_code_rolling)
    plt.legend(['1','2','3','4'])
    plt.xlabel('Iteration')
    plt.ylabel('Weight')
    plt.title('Final Tested SOMs: 15th neuron\'s weights')
    plt.savefig('SOM_Tested_Codes.png', bbox_inches='tight')
    plt.show()
    # ------------------------------------------------------------------------------------

    df_som_pop = pd.read_csv(csv_import_dir+'FWP_SOM_trained_specific_pop.csv')
    num_cols = len(df_som_pop.columns)
    col_names = [str(x) for x in range(1,num_cols+1)]
    df_som_pop.columns = col_names
    print('df_som_pop:\n', df_som_pop)
    df_som_pop_rolling = df_som_pop.rolling(50).mean()
    plt.figure()
    plt.plot(df_som_pop)
    plt.plot(df_som_pop_rolling)
    plt.xlabel('Iteration')
    plt.ylabel('Population')
    plt.title('Final Trained SOMs: Ten neuron\'s populations')
    plt.savefig('SOM_Trained_Populations.png', bbox_inches='tight')
    plt.show()

    df_som_pop = pd.read_csv(csv_import_dir+'FWP_SOM_tested_specific_pop.csv')
    num_cols = len(df_som_pop.columns)
    col_names = [str(x) for x in range(1,num_cols+1)]
    df_som_pop.columns = col_names
    df_som_pop_rolling = df_som_pop.rolling(50).mean()
    plt.figure()
    plt.plot(df_som_pop)
    plt.plot(df_som_pop_rolling)
    plt.xlabel('Iteration')
    plt.ylabel('Population')
    plt.title('Final Tested SOM Populations: Ten neuron\'s populations')
    plt.savefig('SOM_Tested_Populations.png', bbox_inches='tight')
    plt.show()

    return





# ----------------------------------------
''' Import & plot SOM learning from Julia '''
# csv_import_dir = '/home/dp/Documents/FWP/Julia/SOMs_H500_June_1979/'
# plot_SOM_results__Julia(csv_import_dir)
# ----------------------------------------

# ----------------------------------------
''' Export one year of data for SOM processing in R and Julia '''
# export_NARR_ERC_one_year__R_Julia()
# ----------------------------------------

# ----------------------------------------
''' Export monthly and all years data for SOM processing in R and Julia '''
# month_num                = 8
# cols                     = ['H500','ERC'] # Selecting the desired columns
# NARR_gridMET_csv_in_dir  = '/home/dp/Documents/FWP/NARR_gridMET/csv/Training/Categorical/' # Using categorical data
# NARR_gridMET_csv_out_dir = '/home/dp/Documents/FWP/'

# export_NARR_ERC_categorical__R_Julia(month_num, cols, NARR_gridMET_csv_in_dir, NARR_gridMET_csv_out_dir)
# ----------------------------------------

# ----------------------------------------
''' Import & plot SOM learning from R '''
# plot_SOM_results__R()
# ----------------------------------------