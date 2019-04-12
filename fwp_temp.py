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

# x = df_gridmet.lon.iloc[ind[0]]
# y = df_gridmet.lat.iloc[ind[0]]
# z = df_gridmet.erc.iloc[ind[0]]
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

plt.close()
plt.figure()

plt.scatter(x=df_gridmet.lon, y=df_gridmet.lat, color='white', marker='o', edgecolors='g', s=df_gridmet.erc*1.5, label='gridMET')
plt.scatter(x=df_gridmet.lon.iloc[np.ravel(ind)], y=df_gridmet.lat.iloc[np.ravel(ind)], color='r', marker='x', s=9, label='nearest gridMET')
plt.scatter(x=df_narr.lon, y=df_narr.lat, color='k', marker='+', label='NARR')
plt.xlabel('x-coordinates'); plt.ylabel('y-coordinates')
plt.legend()
plt.savefig('irregular_grid_before_interp.png')
plt.show()

plt.scatter(x=df_gridmet.lon, y=df_gridmet.lat, color='white', marker='o', edgecolors='g', s=df_gridmet.erc*1.5, label='gridMET')
plt.scatter(x=df_gridmet.lon.iloc[np.ravel(ind)], y=df_gridmet.lat.iloc[np.ravel(ind)], color='r', marker='x', s=9, label='nearest gridMET')
plt.scatter(x=xi, y=yi, color='y', edgecolors='y', alpha=0.6, marker='o', s=zi*1.5, label='interp NARR')
plt.scatter(x=df_narr.lon, y=df_narr.lat, color='k', marker='+', label='NARR')
plt.xlabel('x-coordinates'); plt.ylabel('y-coordinates')
plt.legend()
plt.savefig('irregular_grid_after_interp.png')
plt.show()

