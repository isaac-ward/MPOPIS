# julia ./src/script.jl

using Pkg
Pkg.status()
#Pkg.add("DataFrames")

# 'include' includes the content of the named file into your current program,
# we need to do this before attempting to use a local module
include("MPOPIS.jl")
# 'using' makes all the exports available, 'import' would just bring the module name 
# into scope. The . here is because we want to look in the current module's namespace,
# which makes sense because the include pulled the contents into the current module
using .MPOPIS

# Whereas this file does not define a module, or export anything, so
# we can just include it and get the 'get_policy' function
include("examples/example_utils.jl")

# And these pull in non-local modules that have been installed
# with Pkg.add
using Printf 
using Random
using Plots
using ProgressMeter
using Dates
using LinearAlgebra
using Distributions
using CSV
using DataFrames

# Similarly
include("examples/car_example.jl")

# And then generate data
# for track_name in ["curve", "curve1", "curve2", "curve3", "curve4", "curve5", "cubic", "cubic1", "cubic2", "cubic3", "cubic4", "cubic5"]
for track_name in ["curve"]
    simulate_car_racing(
        num_steps=600,
        num_samples=150,
        save_gif=true,
        # This plots the rollouts on the gif
        plot_traj=false,
        num_trials=1,
        track="C:/Users/moose/Desktop/dev/MPOPIS/src/envs/car_racing_tracks/$track_name.csv",
        #log_folder="C:/Users/moose/Desktop/dev/MPOPIS/dump/$track_name/",
    )
end