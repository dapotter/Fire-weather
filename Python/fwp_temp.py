import numpy as np
import pandas as pd
import math
from datetime import datetime
from datetime import timedelta
from random import randint
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
# import scipy.interpolate as interpolate
from scipy import spatial
from scipy import interpolate
from sklearn import neighbors

s = '72,558'
s = s.replace(',','')
v = int(float(s))
print('v:\n', v)


z = np.array([[1,3,5,6],[5,6,7,6],[8,10,11,9]], dtype=float)
print(z)
print('grad:\n', np.gradient(z))
print('grad axis = 0:\n', np.gradient(z, axis=0))
print('grad axis = 1:\n', np.gradient(z, axis=1))

print('grad x axis:', 102637.61850161 - 102645.99308614)
print('grad y axis:', 102656.02644742 - 102645.99308614)

clevs = list(range(5000, 5600, 10))
print('clevs:\n', clevs)

l1 = list(range(0,5))
l2 = list(range(0,5))
l1b = [8,3,4,6,1,4,2,6,9,1]
l2b = [3,6,7,1,3,9,3,6,7,3]
data = [[i,j] for i,j in zip(l1,l1b)]
data = [[i,j] for i,j in zip(l1,l1b)]

df1 = pd.DataFrame(data=data, columns=['a','b'])
#df1.set_index('a', inplace=True)
print('df1:\n', df1)
df2 = pd.DataFrame(data=data, columns=['c','d'])
#df2.set_index('c', inplace=True)
print('df2:\n', df2)

df = pd.concat([df1,df2], axis=1, join='outer', sort=False)
print('df:\n', df)
df.set_index(['a','c'], inplace=True)
print('df:\n', df)
print('df.loc[2,2]:\n', df.loc[0,0])
print('df.loc[2,2]:\n', df.loc[[0,0]])


a = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
m,n,r = a.shape
out_arr = np.column_stack((np.repeat(np.arange(m),n),a.reshape(m*n,-1)))
out_df = pd.DataFrame(out_arr)
out_df.columns = ['a','b','c']
print('a shape:', np.shape(a))
print('a:\n', a)
print('out_arr:\n', out_arr)
print('out_df:\n', out_df)

a = np.array([[1, 2], [3, 4], [5, 6], [7, 8]])
print('a shape:', np.shape(a))
b = np.array([10,18,12,15])
ab_col_stack = np.column_stack((a,b))
print('ab_col_stack:\n', ab_col_stack)

a = [0,0,0,0,1,1,1,1,4,4,4,4]
a_unique = np.unique(a)
print('a_unique:\n', a_unique)
a_unique_expanded = np.repeat(a_unique,5)

a = [0,1,4]
a_tiled = np.tile(a,3)
print('a_tiled:\n', a_tiled)

s = 'hello'
print(s)

SYNABBR_shortlist = ['H500', 'CAPE', 'PMSL']
for i, SYNVAR in enumerate(SYNABBR_shortlist):
	print('i:', i)
	print('SYNVAR:', SYNVAR)


l = [0]; l2 = [1,2,3]
print(l + l2)
print(l2)
new_list = [x+1 for x in l2]
print('new_list:', new_list)

dt = datetime(1979,1,1,0,0)
print(dt)

print('***************************** Guided list fill:')
# Using l as a guide to fill l2 with np.NaN where values are missing in l2
l = ['yang','biden','warren','buttigieg','harris']
l2 = ['yang','warren','harris']
l3 = []
for n in l:
	print('n:', n)
	if n in l2:
		print('n2 matches n!:', n)
		l3.append(n)
	else:
		print('no match:', n)
		l3.append(np.NaN)
print('l3:\n', l3)
l3 = (n if n in l2 else np.NaN for n in l)
print('l3 list comp:\n', list(l3))


print('***************************** Guided tuple fill:')
# Using l as a guide to fill l2 with np.NaN where values are missing in l2
old_data_list = [('yang',4.5),('biden',11.5),('warren',2.1),('buttigieg',8.3),('harris',20.8)]
new_data_list = [('yang',3.5),('harris',21.2)] # Note this list has new values next to names
l3 = []
for n in old_data_list:
	print('n:', n)
	if n in new_data_list:
		print('new_data_list matches n!:', n)
		l3.append(n)
	else:
		print('no match:', n)
		l3.append(np.NaN)
print('l3:\n', l3)
l3 = (n if n in new_data_list else (np.NaN,np.NaN) for n in old_data_list)
print('l3 list comp:\n', list(l3))


print('***************************** Guided dictionary fill:')
# Using l as a guide to fill l2 with np.NaN where values are missing in l2
old_data_dict = dict([('yang',4.5),('biden',11.5),('warren',2.1),('buttigieg',8.3),('harris',20.8)])
new_data_dict = dict([('yang',3.5),('harris',21.2)]) # Note this list has new values next to names
print('old_data_dict:\n', old_data_dict)
print('new_data_dict:\n', new_data_dict)

# Going through old data key-value pairs, if old name matches any in the new dictionary,
# return name:odds, otherwise return name:np.NaN. I'm not sure how it knows to return name:np.NaN.
# If I specify 'else k:np.NaN' it errors with invalid syntax
dict_matched = {k:new_data_dict[k] if k in new_data_dict else np.NaN for (k,v) in old_data_dict.items()}
print('dict_matched in dict comp:\n', dict_matched)

dt_datetime = datetime(2020,1,17,12,35,1)
print('dt_datetime:', dt_datetime)
dt_date = datetime.date(dt)
print('dt_date:', dt_date)


''' Two dataframes each with datetime indices. Reduce each index to dates,
	average data by the day, merge both df's on date indices. '''

# Make timedelta array:
td = timedelta(1)

# Making a range of datetimes:
dt_start = datetime(1979,1,1,0,0,0)
dt_objects = 10
dt_iterable = np.arange(0,dt_objects,0.5)
data_objects = 20
data_iterable = range(0, data_objects)

dt_list = [dt_start+timedelta(x) for x in dt_iterable]
print('dt_list:\n', dt_list)

data_list = [randint(0,10) for x in data_iterable]
df_a = pd.DataFrame(data={'Time':dt_list, 'B':data_list})
df_a.set_index('Time', inplace=True)
print('df_a:\n', df_a)

data_list = [randint(0,10) for x in data_iterable]
df_c = pd.DataFrame(data={'Time':dt_list, 'D':data_list})
df_c.set_index('Time', inplace=True)

df = pd.merge(df_a, df_c, left_index=True, right_index=True)
print('df merged:\n', df)
df.reset_index(inplace=True)
df['Date'] = df['Time'].dt.date
print('df:\n', df)
df_mean = df.groupby('Date')[['B','D']].mean()
print('df_mean:\n', df_mean)



''' Two arrays with slightly different x-y values, find where they nearly overlap,
	interpolate one to the other, then merge their data columns on their matched
	x-y values '''

# It's so nice
arr_rand = np.random.rand(5,5)/5 # Division by 5 reduces the added randomness
arr_range = np.arange(0,5)
arr_pivot = arr_range + arr_rand
arr_C_flat = arr_pivot.flatten('C')
# We flatten twice
arr_rand = np.random.rand(5,5)/5
arr_range = np.arange(0,5)
arr_pivot = arr_range + arr_rand
arr_F_flat = arr_pivot.flatten('F')
# Make x,y coordinates
arr_xy = np.vstack((arr_F_flat, arr_C_flat)).T
print('arr_xy:\n', arr_xy)

''' Data for narr is turned off, I only need the lon lat points
# Make the data for narr
arr_data = np.array([randint(5,10) for x in range(0,25)])
arr_data = np.reshape(arr_data, (25,1))
narr = np.hstack((arr_xy, arr_data))
'''
narr = arr_xy # Need to do this when data is turned off for narr

# Do for gridMET now:
arr_rand = np.random.rand(20,20)/5 # Division by 5 reduces the added randomness
arr_range = np.arange(0,5,0.25)
arr_pivot = arr_range + arr_rand
arr_C_flat = arr_pivot.flatten('C')

arr_rand = np.random.rand(20,20)/5
arr_range = np.arange(0,5,0.25)
arr_pivot = arr_range + arr_rand
arr_F_flat = arr_pivot.flatten('F')

arr_xy = np.vstack((arr_F_flat, arr_C_flat)).T
print('arr_xy:\n', arr_xy)

arr_data = np.array([randint(10,40) for x in range(0,400)])
arr_data = np.reshape(arr_data, (400,1))
gridmet = np.hstack((arr_xy, arr_data))

df_narr = pd.DataFrame(data=narr, columns=['lon','lat'])
df_gridmet = pd.DataFrame(data=gridmet, columns=['lon','lat','erc'])
print('df_narr:\n', df_narr)
print('df_gridmet:\n', df_gridmet)

np.random.seed(0)
X = np.random.random((10, 3))  # 10 points in 3 dimensions
print('X:\n', X)
print('X[:1]:\n', X[:1])
tree = neighbors.KDTree(X, leaf_size=2)
dist, ind = tree.query(X[:1], k=3)
print(ind)  # indices of 3 closest neighbors
print(dist)  # distances to 3 closest neighbors
would_you_be_my_neighbor = 5
tree = neighbors.KDTree(df_gridmet.values[:,0:2], leaf_size=2)
dist, ind = tree.query(df_narr.values[:], k=would_you_be_my_neighbor)
print('df_gridmet.values:\n', df_gridmet.values)
print('df_gridmet.values[:,0:2]:\n', df_gridmet.values[:,0:2])
print('df_narr:\n', df_narr)
print('df_narr.loc[6]:\n', df_narr.loc[6])
print('df_narr[5:6]:\n', df_narr[5:6])
print('df_narr[:1]:\n', df_narr[:1])

print('df_gridmet[ind]:\n', df_gridmet.iloc[0])
print('df_gridmet[ind]:\n', df_gridmet.iloc[20])
print('df_gridmet[ind]:\n', df_gridmet.iloc[1])

print('indices:', ind)  # indices of 3 closest neighbors
print('distances:', dist)  # distances to 3 closest neighbors

x = df_gridmet.lon.values
y = df_gridmet.lat.values
z = df_gridmet.erc.values
xi = df_narr.lon.values
yi = df_narr.lat.values
print('x:\n{}\n y:\n{}\n z:\n{}\n'.format(x, y, z))
print('xi:\n{}\n yi:\n{}\n'.format(xi,yi))
print('x shape:\n{}\n y shape:\n{}\n z shape:\n{}\n'.format(np.shape(x), np.shape(y), np.shape(z)))
print('xi shape:\n{}\n yi shape:\n{}\n'.format(np.shape(xi),np.shape(yi)))

zi = interpolate.griddata((x,y),z,(xi,yi),method='nearest')
print('zi:\n', zi)
print('zi shape:\n{}\n'.format(np.shape(zi)))

arr = np.array([1.1,2.9])
arr_rounded = np.round(arr)
print('arr rounded:\n', arr_rounded)

# # Plotting before and after interpolation:
# plt.close()
# plt.figure()

# plt.scatter(x=df_gridmet.lon, y=df_gridmet.lat, color='white', marker='o', edgecolors='g', s=df_gridmet.erc*1.5, label='gridMET')
# plt.scatter(x=df_gridmet.lon.iloc[np.ravel(ind)], y=df_gridmet.lat.iloc[np.ravel(ind)], color='r', marker='x', s=9, label='nearest gridMET')
# plt.scatter(x=df_narr.lon, y=df_narr.lat, color='k', marker='+', label='NARR')
# plt.xlabel('x-coordinates'); plt.ylabel('y-coordinates')
# plt.legend()
# plt.savefig('irregular_grid_before_interp.png', bbox_inches='tight')
# plt.show()

# plt.scatter(x=df_gridmet.lon, y=df_gridmet.lat, color='white', marker='o', edgecolors='g', s=df_gridmet.erc*1.5, label='gridMET')
# plt.scatter(x=df_gridmet.lon.iloc[np.ravel(ind)], y=df_gridmet.lat.iloc[np.ravel(ind)], color='r', marker='x', s=9, label='nearest gridMET')
# plt.scatter(x=xi, y=yi, color='y', edgecolors='y', alpha=0.6, marker='o', s=zi*1.5, label='interp NARR')
# plt.scatter(x=df_narr.lon, y=df_narr.lat, color='k', marker='+', label='NARR')
# plt.xlabel('x-coordinates'); plt.ylabel('y-coordinates')
# plt.legend()
# plt.savefig('irregular_grid_after_interp.png', bbox_inches='tight')
# plt.show()

l = [1,2,3,4]
print('l[1:]:\n', l[1:])

arr = np.array([1, 7, 4, 6, 2, 8])
arr_cut = pd.cut(arr, 3, labels=['low','medium','high'])
print('arr:\n', arr)
print('arr_cut:\n', arr_cut)


# Can you pull two items out of a sublist in a loop?
l = [(1,2),(3,4),(5,6)]
for a,b in l:
	print('a:', a)
	print('b:', b)

# You can also pull out the entire tuple on each loop:
for tup in l:
	print('tup:', tup)

print('\n')
print('*****************************************************')
print('Calculating gradient of arr with x and y coordinates:')
# Unevenly spaced gradient:
dx = 2. # np.gradient will assume spacing of 2
arr = np.array([[1, 2, 5, 10], [3, 4, 5, 6]], dtype=float)
print('arr:\n', arr)
x = 2.
y = [1., 1.5, 3.5, 1.]
grad = np.gradient(arr, x, y)
print('grad:\n', grad)


print('\n')
print('***************************************************************')
print('Calculating gradient of arr with irregular x and y coordinates:')
x = [2., 4., 6., 9., 12., 13., 15., 21., 2.5, 4.5, 6.5, 9.5, 12.5, 13.5, 15.5, 21.5]
x = np.array(x)
y = [1., 1.5, 3.5, 4.5, 5., 6.5, 8., 10.5, 1., 1.5, 3.5, 4.5, 5., 6.5, 8., 10.5]
y = np.array(y)
z = [10., 20., 30., 40., 8., 12., 15., 19., 11., 21., 31., 41., 9., 13., 16., 20.]
z = np.array(z)
print('z:\n', z)
# Calculate gradients
dx = np.gradient(z,x)
print('dx:\n', dx)
dy = np.gradient(z,y)
print('dy:\n', dy)
grad = dy/dx
print('grad:\n', grad)
spot_grad_x = (z[8]-z[6])/(x[8]-x[6])
spot_grad_x = (z[1:]-z[:-1])/(x[1:]-x[:-1])
real_grad_x = dx[:]
print('spot_grad_x:\n', spot_grad_x)
print('real_grad_x:\n', real_grad_x)

# diff = np.diff(arr)
# print('diff:\n', diff)


print('\n')
print('********************************************')
print('Calculating gradient of f with x coordinates')
f = np.array([1, 2, 4, 7, 11, 16], dtype=float)
print('f:\n', f)
x = np.arange(f.size)
print('x:\n', x)
grad = np.gradient(f,x)
print('gradient of f with x spacing:\n', grad)


print('\n')
print('******************************')
print('Sorting filenames in sublists:')
all_NARR_files = [['2_H500.csv', '1_H500.csv'], ['1_CAPE.csv', '2_CAPE.csv'], ['1_PMSL.csv', '2_PMSL.csv']]
all_NARR_files = [sorted(files) for files in all_NARR_files]
print('all_NARR_files sorted:\n', all_NARR_files)

SYNVAR = 'H500'
title_str = SYNVAR+' X Gradient'
print('title_str:', title_str)




print('\n')
print('******************************')
print('Playing with merge:')
df1 = pd.DataFrame({'lon': [1.1, 2.2, 3.3, 4.4], 'lat': [1.2, 2.3, 3.4, 4.5], 'value': [1, 2, 3, 5]}).set_index(['lon','lat'])
df2 = pd.DataFrame({'lon': [1.1, 3.3, 2.2, 4.4], 'lat': [1.2, 3.4, 2.3, 4.5], 'value': [5, 6, 7, 8]}).set_index(['lon','lat'])
df_index_merge = df1.merge(df2, how='left', left_index=True, right_index=True)
print('df1:\n', df1)
print('df2:\n', df2)
print('Get second index levels:\n', df1.index.levels[0])
print('df_index_merge:\n', df_index_merge)



print('\n')
print('******************************')
print('Playing with pd.cut:')
arr = np.arange(0,19)
print('arr:\n', arr)
erc_levels = {'low':np.arange(0,19),\
				'moderate':np.arange(19,27),\
				'high':np.arange(27,35),\
				'very high':np.arange(35,44),\
				'extreme':np.arange(44,100)}
print('erc_levels:\n', erc_levels)

df = pd.DataFrame(data={'index':[1,2,3,4,5], 'erc':[10,4,32,12,63]})
print('df to cut:\n', df)

df_cut = pd.cut(df['erc'], bins=[0,19,27,35,44,100], labels=['low','moderate','high','very high','extreme'])
print('df_cut:\n', df_cut)


print('\n')
print('******************************')
print('Using iloc with groupby:')
time_list = [datetime(1979,1,1,0,0,0), datetime(1979,1,1,3,0,0), datetime(1979,1,1,6,0,0), datetime(1979,1,1,9,0,0), datetime(1979,1,1,12,0,0)]
print('time_list:\n', time_list)
df1 = pd.DataFrame({'lon': [1.1, 1.1, 2.2, 3.3, 4.4], 'lat': [1.2, 1.8, 2.3, 3.4, 4.5], 'time':time_list, 'value': [1, 2, 3, 5, 9]}).set_index(['lon','lat','time'])
df2 = pd.DataFrame({'lon': [1.1, 1.1, 3.3, 2.2, 4.4], 'lat': [1.2, 1.8, 3.4, 2.3, 4.5], 'time':time_list, 'value': [5, 6, 7, 8, 12]}).set_index(['lon','lat','time'])
df_index_merge = df1.merge(df2, how='left', left_index=True, right_index=True)
print('df_index_merge:\n', df_index_merge)
# df_time_range = df_index_merge.between_time('1979-01-01 00:00:00', '1979-01-01 09:00:00')
df_time_range = df_index_merge.loc[(1.1,1.2,datetime(1979,1,1,0,0,0)):(4.4,4.5,datetime(1979,1,1,9,0,0))]


print('df_time_range:\n', df_time_range)
df_avg = df_time_range.groupby('lon').mean()
print('df_avg:\n', df_avg)


df_index_merge.reset_index(inplace=True)
df_index_merge.set_index('time', inplace=True)
first_time = df_index_merge.index[0]
print('first_time:\n', first_time)


print('\n')
print('******************************')
print('Iterating through a list of tuples:')
l = [(0,2), (2,4), (4,6)]
for i, (a,b) in enumerate(l):
	print('i:', i)
	print('a={0}, b={1}'.format(a,b))


print('\n')
print('******************************')
print('Importing CAPE csv:')
file = '/home/dp/Documents/FWP/NARR/csv_exp/rectilinear_grid/0_CAPE.csv'
df = pd.read_csv(file)
print('df:\n', df)

df2 = pd.read_csv(file, header='infer', index_col=['lon', 'lat', 'time'])
print('df2:\n', df2)

print('\n')
print('******************************')
print('Meshgrid csv:')
N = 5
x = np.linspace(0, 5, N)
y = np.linspace(0, 5, N)
xx, yy = np.meshgrid(x, y)
# d = np.sin(xx)*np.cos(yy)
N = N**2
d = np.linspace(0, 125, N)
print('np.shape(xx):', np.shape(xx))
print('xx:', xx)
print('np.shape(yy):', np.shape(yy))
print('yy:', yy)
print('np.shape(d):', np.shape(d))
print('d:', d)

mesh_shape = np.shape(xx)
print('mesh_shape:\n', mesh_shape)
d = d.reshape(mesh_shape)
print('d reshape to mesh:\n', d)


lat = [42, 42, 42, 43, 43, 43, 44, 44, 44]
lon = [236, 237, 238, 236, 237, 238, 236, 237, 238]
z = [-8, 4, 28, 1, -3, -9, 11, 14, 2]

d = [[lat,lon,z] for lat,lon,z in zip(lat,lon,z)]
df = pd.DataFrame(data=d, columns=['lat','lon','h500'])
print('df:\n', df)

lat = df['lat'].drop_duplicates('first').to_numpy()
lon = df['lon'].drop_duplicates('first').to_numpy()
# Above: lat is first df column, lon is second, but
# below, lon is specified as x, lat as y because that's
# how the contour plot needs to appear. If meshgrid(lat,lon)
# is used, then the values will not correctly map to
# the data. The fact that lat is the first column in
# the dataframe is an indexing convenience. For clarity,
# lon should be made the first column.
#########################################################
# RULE: the x array in meshgrid(x,y) will be created to
# grow from left to right in the plot and will be constant
# from bottom to top. The y array will grow from bottom
# to top and be constant from left to right.
#########################################################
lonlon, latlat = np.meshgrid(lon,lat)
mesh_shape = np.shape(lonlon)
d = df['h500'].to_numpy().reshape(mesh_shape)

print('lonlon:\n', lonlon)
print('latlat:\n', latlat)
print('d:\n', d)




l = [[1,2,3], [4,5,6]]
# new_list = []
# for sublist in l:
# 	new_sublist = []
# 	for el in sublist:
# 		el += 2
# 		new_sublist.append(el)
# 	new_list.append(new_sublist)
# print('l:\n', l)
# print('new_list:\n', new_list)

arr = np.array([np.array(sublist) for sublist in l])
arr += 2
print('arr:\n', arr)





