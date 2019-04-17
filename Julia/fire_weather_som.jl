using SOM
using RDatasets
using CSV

# print("----------------------------------------------\n\n\n\n\n\n\n")
# """Import fire weather data from csv:"""
# H500 = CSV.read("//home//dp//Documents//FWP//NARR//All_H500_som.csv")
# print("H500[1:100,:] imported from csv:\n", H500[1:100,:])
#
# train = H500[:,2:4] # All columns: lon, lat, H500
# print("size of train:\n", size(train))
# print("typeof(train):\n", typeof(train))
#
# som = initSOM(train, 62, 62)
#
# # Initial training:
# som = trainSOM(som, train, 10000)
# # Finalise training with additional round with smaller radius:
# som = trainSOM(som, train, 10000, r = 3.0)
#
# # Make X,Y vector of SOM map locations for best matching units
# # and return their row numbers in the train data
# winners = mapToSOM(som, train[:, :])
# print("winners first 100 rows:\n", winners[1:100,:])
# # Why is the output different every time? Because each initialization
# # of the SOM is different.
#
# """ Plotting """
# # Density plot of number of traning samples at each neuron:
# plotDensity(som, fileName="som_density_H500.png")
# # Classes plot shows class labels of training samples (uses)
# # iris data which still has class labels in it:
# #print("som:\n", som)
# # freqs = classFrequencies(som, iris, :Species)
# # print("freqs:\n", freqs)
# # plotClasses(som, freqs, fileName="iris_som_classes.png")

"""Import data from csv:"""
#all_synvar = CSV.read("//home//dp//Documents//FWP//NARR//df_all_synvar_grid_interp.csv")
#H500_x_grad = all_synvar[:,:] # Columns lon, lat, H500 Grad X (skip time column)
#print("H500_x_grad[1:100,:] imported from csv:\n", H500_x_grad[1:100,:])

NARR_ERC = CSV.read("//home//dp//Documents//FWP//NARR//df_NARR_ERC.csv")
print("NARR_ERC[1:100,:] imported from csv:\n", NARR_ERC[1:100,:])
NARR_ERC = deletecols!(NARR_ERC, :time) # Removing time column
# train = H500_x_grad[1:500000,[1,2,4]] # All columns: lon, lat, H500 Grad X
train = NARR_ERC[:,1:5] # All columns except for the ERC column
print("train[1:20,:]:\n", train[1:20,:])
print("size of train:\n", size(train))

# Initialize:
hexgrid_x = 40
hexgrid_y = 40
som = initSOM(train, hexgrid_x, hexgrid_y)
# Initial training:
som = trainSOM(som, train, 10000)
# Finalise training with additional round with smaller radius:
som = trainSOM(som, train, 10000, r = 3.0)

# Make X,Y vector of SOM map locations for best matching units
# and return their row numbers in the train data
winners = mapToSOM(som, train[:, :])
print("winners first 100 rows:\n", winners[1:100,:])
# Why is the output different every time? Because each initialization
# of the SOM is different.

""" Plotting """
# Density plot of number of traning samples at each neuron:
#plotDensity(som, fileName="som_density_H500_x_grad.png")
f = "//home//dp//Documents//FWP//Julia//NARR_ERC_SOM_density_"*string(hexgrid_x)*"x"*string(hexgrid_y)*".png"
plotDensity(som, fileName=f)

# NARR ERC data class labels in it:
print("som:\n", som)
freqs = classFrequencies(som, NARR_ERC, :ERC)
print("freqs:\n", freqs)
f = "//home//dp//Documents//FWP//Julia//NARR_ERC_SOM_classes_"*string(hexgrid_x)*"x"*string(hexgrid_y)*".png"
plotClasses(som, freqs, fileName=f)
