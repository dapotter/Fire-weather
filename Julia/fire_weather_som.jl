using SOM
using RDatasets
using CSV
using Dates
using DelimitedFiles

function fire_weather_som(write_to_csv, plot_every_iter, grid_x, grid_y, step_size, iterations,
                train_month_name, train_month_num, test_month_name, test_month_num,
                x_train_in, x_test_in, y_train_in, y_test_in, xy_train_in, xy_test_in,
                SOM_out)
    """ Import data from csv: """
    x_train = CSV.read(x_train_in)
    x_test = CSV.read(x_test_in)
    y_train = CSV.read(y_train_in)
    y_test = CSV.read(y_test_in)
    xy_train = CSV.read(xy_train_in)
    xy_test = CSV.read(xy_test_in)

    """ Name columns """
    names!(x_train, Symbol.(["h500"]))
    names!(x_test, Symbol.(["h500"]))
    names!(y_train, Symbol.(["ERC"]))
    names!(y_test, Symbol.(["ERC"]))

    train_col_count = 1
    print("x_train:\n", x_train)
    print("y_train:\n", y_train)

    # Initialize (grid_x, grid_y specified in function parameters)
    global som_init_train = initSOM(x_train, grid_x, grid_y)
    global som_init_test = initSOM(x_test, grid_x, grid_y)

    # som_trained_all_diff = Array{Float64, 2}[]
    som_trained_specific_diff = Array{Float64, 1}[]
    som_trained_specific_codes = Array{Float64, 1}[]
    som_trained_specific_pop = Array{Int64, 1}[]

    # som_tested_all_diff = Array{Float64, 2}[]
    som_tested_specific_diff = Array{Float64, 1}[]
    som_tested_specific_codes = Array{Float64, 1}[]
    som_tested_specific_pop = Array{Int64, 1}[]

    iter_list = Int64[]
    i = 0
    j = 0
    # global som_init = initSOM(train, 10, 8)
    for i = 0:step_size:iterations
        # print("\nsom initialized - codes:\n", som_init.codes)
        # print("\nsom initialized - population:\n", som_init.population)
        global rough_iters = i+1
        global fine_iters = i+1

        global som_trained = trainSOM(som_init_train, x_train, rough_iters)
        global som_trained = trainSOM(som_trained, x_train, fine_iters, r = 3)

        # global som_tested = trainSOM(som_init_test, test, rough_iters)
        global som_tested = trainSOM(som_trained, x_test, rough_iters)
        global som_tested = trainSOM(som_tested, x_test, fine_iters, r = 3)

        # print("\nsom\n", som)
        # print("\nsom trained - codes:\n", som.codes)
        # print("\nsom trained - population:\n", som.population)

        som_trained_diff = som_trained.codes - som_init_train.codes
        som_trained_codes = som_trained.codes
        som_trained_pop = som_trained.population

        som_tested_diff = som_tested.codes - som_init_test.codes
        som_tested_codes = som_tested.codes
        som_tested_pop = som_tested.population

        num_neurons = grid_x*grid_y
        pop_list = [
            1,
            trunc(Int, num_neurons/10), trunc(Int, num_neurons/5), trunc(Int, num_neurons/2), trunc(Int, num_neurons/1.5),
            num_neurons
            ]
        print(pop_list)

        # push!(som_trained_all_diff, som_trained_diff)
        push!(som_trained_specific_diff, som_trained_diff[15,:]) # [15,:] is 15th neuron, all difference
        push!(som_trained_specific_codes, som_trained_codes[15,:]) # [15,:] is 15th neuron, all codes
        push!(som_trained_specific_pop, som_trained_pop[pop_list]) # [15:17] is 15th through 17th neuron populations

        # push!(som_tested_all_diff, som_tested_diff)
        push!(som_tested_specific_diff, som_tested_diff[15,:])
        push!(som_tested_specific_codes, som_tested_codes[15,:])
        push!(som_tested_specific_pop, som_tested_pop[pop_list])

        if plot_every_iter == true
            """ Calculating Class Frequencies every iteration """
            freqs_trained = classFrequencies(som_trained, xy_train, :ERC)
            freqs_tested = classFrequencies(som_tested, xy_test, :ERC)

            """ Plotting Class Frequencies every iteration """
            # Train
            plot_title = "SOM Class Frequencies - Trained - "*train_month_name*", 1979"
            color_dict = Dict("low"=>"green","moderate"=>"yellowgreen","high"=>"yellow","very high"=>"orange","extreme"=>"red")
            file_name = SOM_out*"//Timelapse_Trained//"*string(j)*"_"*train_month_num*"_NARR_ERC_SOM_classes_trained_"*string(grid_x)*"x"*string(grid_y)*"_"*string(iterations)*"iters_"*string(step_size)*"step_"*train_month_name*".png"
            plotClasses(som_trained, freqs_trained, title=plot_title, device=:png, colors=color_dict, fileName=file_name)
            # Test
            plot_title = "SOM Class Frequencies - Tested - "*test_month_name*", 1979"
            color_dict = Dict("low"=>"green","moderate"=>"yellowgreen","high"=>"yellow","very high"=>"orange","extreme"=>"red")
            file_name = SOM_out*"//Timelapse_Tested//"*string(j)*"_"*test_month_num*"_NARR_ERC_SOM_classes_tested_"*string(grid_x)*"x"*string(grid_y)*"_"*string(iterations)*"iters_"*string(step_size)*"step_"*test_month_name*".png"
            plotClasses(som_tested, freqs_tested, title=plot_title, device=:png, colors=color_dict, fileName=file_name)

            j += 1
        end

        push!(iter_list, i)
        print("\ni = ", i)
        print("\nProgress, %: ", i/iterations*100)
    end

    """ Winners """
    # Make X,Y vector of SOM map locations for best matching units
    # and return their row numbers in the train data
    winners_trained = mapToSOM(som_trained, x_train)
    print("Trained winners first 100 rows:\n", winners_trained[1:100,:])
    winners_tested = mapToSOM(som_tested, x_test)
    print("Tested winners first 100 rows:\n", winners_tested[1:100,:])

    """ Plotting Densities """
    f_trained = SOM_out*train_month_num*"_NARR_ERC_SOM_density_trained_"*string(grid_x)*"x"*string(grid_y)*"_"*string(iterations)*"iters_"*string(step_size)*"step_"*train_month_name*".png"
    plotDensity(som_trained, device=:png, fileName=f_trained)
    f_tested = SOM_out*string(test_month_num)*"_NARR_ERC_SOM_density_tested_"*string(grid_x)*"x"*string(grid_y)*"_"*string(iterations)*"iters_"*string(step_size)*"step_"*train_month_name*".png"
    plotDensity(som_tested, device=:png, fileName=f_tested)

    """ Calculating Class Frequencies """
    print("som_trained:\n", som_trained)
    freqs_trained = classFrequencies(som_trained, xy_train, :ERC)
    print("freqs_trained:\n", freqs_trained)
    print("som_tested:\n", som_tested)
    freqs_tested = classFrequencies(som_tested, xy_test, :ERC)
    print("freqs_tested:\n", freqs_tested)

    """ Plotting Class Frequencies """
    # Train
    plot_title = "SOM Class Frequencies - Trained - "*train_month_name*", 1979"
    color_dict = Dict("low"=>"green","moderate"=>"yellowgreen","high"=>"yellow","very high"=>"orange","extreme"=>"red")
    file_name = SOM_out*train_month_num*"_NARR_ERC_SOM_classes_trained_"*string(grid_x)*"x"*string(grid_y)*"_"*string(iterations)*"iters_"*string(step_size)*"step_"*train_month_name*".png"
    plotClasses(som_trained, freqs_trained, title=plot_title, device=:png, colors=color_dict, fileName=file_name)
    # Test
    plot_title = "SOM Class Frequencies - Tested - "*test_month_name*", 1979"
    color_dict = Dict("low"=>"green","moderate"=>"yellowgreen","high"=>"yellow","very high"=>"orange","extreme"=>"red")
    file_name = SOM_out*test_month_num*"_NARR_ERC_SOM_classes_tested_"*string(grid_x)*"x"*string(grid_y)*"_"*string(iterations)*"iters_"*string(step_size)*"step_"*test_month_name*".png"
    plotClasses(som_tested, freqs_tested, title=plot_title, device=:png, colors=color_dict, fileName=file_name)

    # Write to csv:
    if write_to_csv == true
        writedlm( SOM_out*"FWP_SOM_trained_specific_diff.csv",  som_trained_specific_diff, ',')
        writedlm( SOM_out*"FWP_SOM_trained_specific_codes.csv",   som_trained_specific_codes, ',')
        writedlm( SOM_out*"FWP_SOM_trained_specific_pop.csv",   som_trained_specific_pop, ',')

        writedlm( SOM_out*"FWP_SOM_tested_specific_diff.csv",   som_tested_specific_diff, ',')
        writedlm( SOM_out*"FWP_SOM_tested_specific_codes.csv",   som_tested_specific_codes, ',')
        writedlm( SOM_out*"FWP_SOM_tested_specific_pop.csv",    som_tested_specific_pop, ',')
    end
end

# -----------------------------------------------------
# Run Fire Weather SOM on categorical ERC data:
write_to_csv = true
plot_every_iter = true
grid_x = 9
grid_y = 9
step_size = 20
iterations = 40
train_month_name = "June"
train_month_num =  "6"
test_month_name =  "June"
test_month_num =   "6"
x_train_in =  "//home//dp//Documents//FWP//Julia/x_train.csv"
x_test_in =   "//home//dp//Documents//FWP//Julia/x_test.csv"
y_train_in =  "//home//dp//Documents//FWP//Julia/y_train.csv"
y_test_in =   "//home//dp//Documents//FWP//Julia/y_test.csv"
xy_train_in = "//home//dp//Documents//FWP//Julia/xy_train.csv"
xy_test_in =  "//home//dp//Documents//FWP//Julia/xy_test.csv"
SOM_out =     "//home//dp//Documents//FWP//Julia//SOMs_H500_June_1979//"

fire_weather_som(write_to_csv, plot_every_iter, grid_x, grid_y, step_size, iterations,
                train_month_name, train_month_num, test_month_name, test_month_num,
                x_train_in, x_test_in, y_train_in, y_test_in, xy_train_in, xy_test_in,
                SOM_out)
