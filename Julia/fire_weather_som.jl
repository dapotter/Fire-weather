using SOM
using RDatasets
using CSV
using Dates

function fire_weather_som(hexgrid_x, hexgrid_y, iterations, NARR_gridMET_csv_in)
    """Import data from csv:"""
    #all_synvar = CSV.read("//home//dp//Documents//FWP//NARR//df_all_synvar_grid_interp.csv")
    #H500_x_grad = all_synvar[:,:] # Columns lon, lat, H500 Grad X (skip time column)
    #print("H500_x_grad[1:100,:] imported from csv:\n", H500_x_grad[1:100,:])
    NARR_ERC = CSV.read(NARR_gridMET_csv_in)

    names!(NARR_ERC, Symbol.(["lat","lon","time","h500_grad_x","h500_grad_y","pmsl_grad_x","pmsl_grad_y","cape","ERC"]) )
    NARR_ERC = deletecols!(NARR_ERC, :lat)
    NARR_ERC = deletecols!(NARR_ERC, :lon)
    print("NARR_ERC[1:100,:] imported from csv:\n", NARR_ERC[1:100,:])
    # df = NARR_ERC[(NARR_ERC[:time].>=Date(1979,7,1))&(NARR_ERC[:time].<=Date(1979,7,31)),:]
    start_date =    Dates.Date("1979-10-01","y-m-d")
    end_date =      Dates.Date("1979-10-31","y-m-d")
    month_name =               "October"
    month_num =                "10"
    df_start = NARR_ERC[(NARR_ERC[:time].>=start_date), :]
    print("df_start[1:100,:]:\n", df_start[1:100,:])
    df_end = NARR_ERC[NARR_ERC[:time].>=end_date, :]
    print("df_end[1:100,:],:", df_end[1:100,:])
    NARR_ERC = NARR_ERC[.&(NARR_ERC[:time] .>= start_date, NARR_ERC[:time] .<= end_date), :]
    print("NARR_ERC[1:100,:] after time selection July 1 - 31, 1979:\n", NARR_ERC[1:100,:])

    # Renaming columns:
    NARR_ERC = deletecols!(NARR_ERC, :time)
    # NARR_ERC = deletecols!(NARR_ERC, :time) # Removing time column
    # NARR_ERC = deletecols!(NARR_ERC, :h500_grad_x)
    # NARR_ERC = deletecols!(NARR_ERC, :pmsl_grad_x)
    # NARR_ERC = deletecols!(NARR_ERC, :cape)
    print("NARR_ERC[1:100,:] after deleting columns:\n", NARR_ERC[1:100,:])

    # train = H500_x_grad[1:500000,[1,2,4]] # All columns: lon, lat, H500 Grad X
    train = NARR_ERC[:,1:5] # All columns except for the ERC column at the end
    print("train[1:20,:]:\n", train[1:20,:])
    print("size of train:\n", size(train))

    # Initialize (hexgrid_x, hexgrid_y specified in function parameters)
    som = initSOM(train, hexgrid_x, hexgrid_y)
    # Initial training:
    som = trainSOM(som, train, iterations)
    # Finalise training with additional round with smaller radius:
    som = trainSOM(som, train, iterations, r = 3.0)

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
    plotDensity(som, device=:png, fileName=f)

    # NARR ERC data class labels in it:
    print("som:\n", som)
    freqs = classFrequencies(som, NARR_ERC, :ERC)
    print("freqs:\n", freqs)

    plot_title = "SOM Class Frequencies - "*month_name*", 1979"
    color_dict = Dict("low"=>"green","moderate"=>"yellowgreen","high"=>"yellow","very high"=>"orange","extreme"=>"red")
    file_name = "//home//dp//Documents//FWP//Julia//"*month_num*"_NARR_ERC_SOM_classes_"*string(hexgrid_x)*"x"*string(hexgrid_y)*"_"*string(iterations)*"iters_"*month_name*".png"
    plotClasses(som, freqs, title=plot_title, device=:png, colors=color_dict, fileName=file_name)

end

# -----------------------------------------------------
# Run Fire Weather SOM on categorical ERC data:
hexgrid_x = 30
hexgrid_y = 30
iterations = 10000
NARR_gridMET_csv_in = "//home//dp//Documents//FWP//NARR_gridMET//csv//df_NARR_ERC_categorical.csv"
fire_weather_som(hexgrid_x, hexgrid_y, iterations, NARR_gridMET_csv_in) # hexgrid_x, hexgrid_y
