using Pkg
# Don't need to build every time, only when the python
# environment changes
build = false 
if build
    ENV["PYTHON"] = "C:/Users/moose/miniconda3/envs/aa290b/python"
    Pkg.build("PyCall")
end
using PyCall

# This is useful for formatting input data
@pyimport torch

function get_python_trained_model()
    # Use my python module which holds model definitions
    file_directory = "C:/Users/moose/Desktop/dev/MPOPIS/src/examples/py/"
    pushfirst!(PyVector(pyimport("sys")["path"]), file_directory)
    model_module = pyimport("model_module")
    println("Python learning module imported")

    # Want it in eval mode
    model = model_module.load_model()
    model.eval()
    println("Python learning model loaded and ready for evaluation")

    return model
end

function get_inference(model, window)
    # This is the mean and std of the training data
    means = [ 0.47824945  0.7140721   0.86563648  1. -0.02893655 -0.01286843 -0.0065491   0.81765136]
    stds  = [1.00000001 1.00000001 1.00000001 0.72686883 1.00000001 0.301784 0.14164948 0.4792477 ]

    # What is the window and state size? Gets input as (window_size, state_size)
    window_size, state_size = size(window)

    # Denormalise the window
    window = window .* stds .+ means
    # Reshape to (1, window_size, state_size)
    window = reshape(window, (1, window_size, state_size))
    # Tensorise 
    window = torch.FloatTensor(window)
    # Perform inference
    return model(window)
end