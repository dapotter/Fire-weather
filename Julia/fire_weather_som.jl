using SOM
using RDatasets
using CSV
using Dates
using DelimitedFiles

function fire_weather_som(hexgrid_x, hexgrid_y, step_size, iterations, NARR_gridMET_csv_in)
    """Import data from csv:"""
    #all_synvar = CSV.read("//home//dp//Documents//FWP//NARR//df_all_synvar_grid_interp.csv")
    #H500_x_grad = all_synvar[:,:] # Columns lon, lat, H500 Grad X (skip time column)
    #print("H500_x_grad[1:100,:] imported from csv:\n", H500_x_grad[1:100,:])
    NARR_ERC = CSV.read(NARR_gridMET_csv_in)

    names!(NARR_ERC, Symbol.(["lat","lon","time","h500","h500_grad_x","h500_grad_y","pmsl","pmsl_grad_x","pmsl_grad_y","cape","ERC"]) )
    # df_lat_ERC = NARR_ERC[:,[1,11]]
    # df_lon = NARR_ERC[:,[2,11]]
    # df_time = NARR_ERC[:,[3,11]]
    # df_h500 = NARR_ERC[:,[4,11]]
    # df_h500_grad_x = NARR_ERC[:,[5,11]]

    NARR_ERC = deletecols!(NARR_ERC, :lat)
    NARR_ERC = deletecols!(NARR_ERC, :lon)
    NARR_ERC = deletecols!(NARR_ERC, :h500)
    NARR_ERC = deletecols!(NARR_ERC, :pmsl)
    NARR_ERC = deletecols!(NARR_ERC, :cape)


    # # WARNING: SETTING NARR_ERC TO ONE VARIABLE FOR SOM TRAINING:
    # NARR_ERC = df_lat_ERC
    print("NARR_ERC[1:100,:] imported from csv:\n", NARR_ERC[1:50,:])
    # df = NARR_ERC[(NARR_ERC[:time].>=Date(1979,7,1))&(NARR_ERC[:time].<=Date(1979,7,31)),:]
    train_start_date =    Dates.Date("1979-06-01","y-m-d")
    train_end_date =      Dates.Date("1979-06-21","y-m-d")
    train_month_name =               "June"
    train_month_num =                "6"

    test_start_date =     Dates.Date("1979-06-22","y-m-d")
    test_end_date =       Dates.Date("1979-06-30","y-m-d")
    test_month_name =               "June"
    test_month_num =                "6"

    # Selecting time range
    # df_start = NARR_ERC[(NARR_ERC[:time].>=train_start_date), :]
    # print("df_start[1:100,:]:\n", df_start[1:100,:])
    # df_end = NARR_ERC[NARR_ERC[:time].>=end_date, :]
    # print("df_end[1:100,:],:", df_end[1:100,:])
    NARR_ERC_train = NARR_ERC[.&(NARR_ERC[:time] .>= train_start_date, NARR_ERC[:time] .<= train_end_date), :]
    print("NARR_ERC_train[1:100,:] after time selection "*train_month_name*":\n", NARR_ERC_train[1:50,:])
    NARR_ERC_test = NARR_ERC[.&(NARR_ERC[:time] .>= test_start_date, NARR_ERC[:time] .<= test_end_date), :]
    print("NARR_ERC_test[1:100,:] after time selection "*test_month_name*":\n", NARR_ERC_test[1:50,:])

    # Removing 'invalid' ERC values:
    # Example for removing all values in col x1 that don't equal 8:
    # data[data[:x1].!=8,:]
    NARR_ERC_train = NARR_ERC_train[NARR_ERC_train[:ERC].!="invalid",:]
    print("Invalid ERC values removed from NARR_ERC:\n", NARR_ERC_train[1:50,:])

    NARR_ERC_test = NARR_ERC_test[NARR_ERC_test[:ERC].!="invalid",:]
    print("Invalid ERC values removed from NARR_ERC:\n", NARR_ERC_test[1:50,:])

    # Renaming columns:
    NARR_ERC_train = deletecols!(NARR_ERC_train, :time)
    NARR_ERC_test = deletecols!(NARR_ERC_test, :time)

    # NARR_ERC = deletecols!(NARR_ERC, :time) # Removing time column
    # NARR_ERC = deletecols!(NARR_ERC, :h500_grad_x)
    # NARR_ERC = deletecols!(NARR_ERC, :pmsl_grad_x)
    # NARR_ERC = deletecols!(NARR_ERC, :cape)
    print("NARR_ERC_train[1:100,:] after deleting columns:\n", NARR_ERC_train[1:50,:])
    print("NARR_ERC_test[1:100,:] after deleting columns:\n", NARR_ERC_test[1:50,:])

    # train = H500_x_grad[1:500000,[1,2,4]] # All columns: lon, lat, H500 Grad X
    train = NARR_ERC_train[:,1:4] # All columns except for the ERC column at the end
    print("train[1:20,:]:\n", train[1:20,:])
    print("size of train:\n", size(train))

    test = NARR_ERC_test[:,1:4] # All columns except for the ERC column at the end
    print("test[1:20,:]:\n", test[1:20,:])
    print("size of train:\n", size(test))

    # Initialize (hexgrid_x, hexgrid_y specified in function parameters)
    global som_init_train = initSOM(train, hexgrid_x, hexgrid_y)
    global som_init_test = initSOM(test, hexgrid_x, hexgrid_y)
    # # Initial training:
    # som = trainSOM(som, train, iterations)
    # # Finalise training with additional round with smaller radius:
    # som = trainSOM(som, train, iterations, r = 3.0)

    # som_trained_all_diff = Array{Float64, 2}[]
    som_trained_specific_diff = Array{Float64, 1}[]
    som_trained_specific_codes = Array{Int64, 1}[]
    som_trained_specific_pop = Array{Int64, 1}[]

    # som_tested_all_diff = Array{Float64, 2}[]
    som_tested_specific_diff = Array{Float64, 1}[]
    som_tested_specific_codes = Array{Int64, 1}[]
    som_tested_specific_pop = Array{Int64, 1}[]

    iter_list = Int64[]
    # global som_init = initSOM(train, 10, 8)
    for i = 0:step_size:iterations
        # print("\nsom initialized - codes:\n", som_init.codes)
        # print("\nsom initialized - population:\n", som_init.population)
        global rough_iters = i+1
        global fine_iters = i+1

        global som_trained = trainSOM(som_init_train, train, rough_iters)
        global som_trained = trainSOM(som_trained, train, fine_iters, r = 3)

        global som_tested = trainSOM(som_init_test, test, rough_iters)
        global som_tested = trainSOM(som_tested, test, fine_iters, r = 3)

        # print("\nsom\n", som)
        # print("\nsom trained - codes:\n", som.codes)
        # print("\nsom trained - population:\n", som.population)

        som_trained_diff = som_trained.codes - som_init_train.codes
        som_trained_codes = som_trained.codes
        som_trained_pop = som_trained.population

        som_tested_diff = som_tested.codes - som_init_test.codes
        som_tested_codes = som_tested.codes
        som_tested_pop = som_tested.population

        # push!(som_trained_all_diff, som_trained_diff)
        push!(som_trained_specific_diff, som_trained_diff[15,:])
        push!(som_trained_specific_codes, som_trained_codes[15,:])
        push!(som_trained_specific_pop, som_trained_pop[15:15])

        # push!(som_tested_all_diff, som_tested_diff)
        push!(som_tested_specific_diff, som_tested_diff[15,:])
        push!(som_tested_specific_codes, som_tested_codes[15,:])
        push!(som_tested_specific_pop, som_tested_pop[15:15])

        push!(iter_list, i)

        print("\ni = ", i)
        print("\nProgress, %: ", i/iterations*100)
    end

    # Make X,Y vector of SOM map locations for best matching units
    # and return their row numbers in the train data
    winners_trained = mapToSOM(som_trained, train[:, :])
    print("Trained winners first 100 rows:\n", winners_trained[1:100,:])
    winners_tested = mapToSOM(som_tested, test[:, :])
    print("Tested winners first 100 rows:\n", winners_tested[1:100,:])
    # Why is the output different every time? Because each initialization
    # of the SOM is different.

    """ Plotting """
    # Density plot of number of traning samples at each neuron:
    #plotDensity(som, fileName="som_density_H500_x_grad.png")
    f_trained = "//home//dp//Documents//FWP//Julia//NARR_ERC_SOM_density_trained_"*string(hexgrid_x)*"x"*string(hexgrid_y)*".png"
    plotDensity(som_trained, device=:png, fileName=f_trained)
    f_tested = "//home//dp//Documents//FWP//Julia//NARR_ERC_SOM_density_tested_"*string(hexgrid_x)*"x"*string(hexgrid_y)*".png"
    plotDensity(som_tested, device=:png, fileName=f_tested)

    # NARR ERC data class labels:
    print("som_trained:\n", som_trained)
    freqs_trained = classFrequencies(som_trained, NARR_ERC_train, :ERC)
    print("freqs_trained:\n", freqs_trained)

    print("som_tested:\n", som_tested)
    freqs_tested = classFrequencies(som_tested, NARR_ERC_test, :ERC)
    print("freqs_tested:\n", freqs_tested)

    # Create frequency plots
    plot_title = "SOM Class Frequencies - Trained - "*train_month_name*", 1979"
    color_dict = Dict("low"=>"green","moderate"=>"yellowgreen","high"=>"yellow","very high"=>"orange","extreme"=>"red")
    file_name = "//home//dp//Documents//FWP//Julia//"*train_month_num*"_NARR_ERC_SOM_classes_trained_"*string(hexgrid_x)*"x"*string(hexgrid_y)*"_"*string(iterations)*"iters_"*train_month_name*".png"
    plotClasses(som_trained, freqs_trained, title=plot_title, device=:png, colors=color_dict, fileName=file_name)

    plot_title = "SOM Class Frequencies - Tested - "*test_month_name*", 1979"
    color_dict = Dict("low"=>"green","moderate"=>"yellowgreen","high"=>"yellow","very high"=>"orange","extreme"=>"red")
    file_name = "//home//dp//Documents//FWP//Julia//"*test_month_num*"_NARR_ERC_SOM_classes_tested_"*string(hexgrid_x)*"x"*string(hexgrid_y)*"_"*string(iterations)*"iters_"*test_month_name*".png"
    plotClasses(som_tested, freqs_tested, title=plot_title, device=:png, colors=color_dict, fileName=file_name)

    # Write to csv:
    writedlm( "FWP_SOM_trained_specific_diff.csv",  som_trained_specific_diff, ',')
    writedlm( "FWP_SOM_trained_specific_codes.csv",   som_trained_specific_codes, ',')
    writedlm( "FWP_SOM_trained_specific_pop.csv",   som_trained_specific_pop, ',')

    writedlm( "FWP_SOM_tested_specific_diff.csv",   som_tested_specific_diff, ',')
    writedlm( "FWP_SOM_tested_specific_codes.csv",   som_tested_specific_codes, ',')
    writedlm( "FWP_SOM_tested_specific_pop.csv",    som_tested_specific_pop, ',')
end

# -----------------------------------------------------
# Run Fire Weather SOM on categorical ERC data:
hexgrid_x = 30
hexgrid_y = 30
step_size = 1#20
iterations = 3#2000
NARR_gridMET_csv_in = "//home//dp//Documents//FWP//NARR_gridMET//csv//df_NARR_ERC_categorical.csv"
fire_weather_som(hexgrid_x, hexgrid_y, step_size, iterations, NARR_gridMET_csv_in) # hexgrid_x, hexgrid_y
