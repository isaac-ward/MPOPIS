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
@pyimport numpy as np

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

function get_inference_batch(model, batch_window)
    # This is the mean and std of the training data
    means = [ 0.47824945  0.7140721   0.86563648  1. -0.02893655 -0.01286843 -0.0065491   0.81765136]
    stds  = [1.00000001 1.00000001 1.00000001 0.72686883 1.00000001 0.301784 0.14164948 0.4792477 ]

    # What is the window and state size? Gets input as (batch_size, window_size, state_size)
    batch_size, window_size, state_size = size(batch_window)
    # println("Batch size: ", batch_size)
    # println("Window size: ", window_size)
    # println("State size: ", state_size)
    # println("means size", size(means))
    # println("stds size", size(stds))

    # Normalise the batch_window by subtracting the mean and dividing by the std
    # on every last dimension
    #batch_window = (batch_window .- means) ./ stds
    #println("Batch window shape: ", size(batch_window))
    # Reshape to (batch_size, window_size, state_size)
    #batch_window = reshape(batch_window, (batch_size, window_size, state_size))
    #println("Batch window shape: ", size(batch_window))
    # Tensorise
    batch_window = torch.FloatTensor(batch_window)
    #println("Batch window shape: ", batch_window.size())

    # Perform inference
    inference, _ = model(batch_window)

    # Detach from the graph and numpify
    inference = inference.detach().numpy()
    #println("Inference shape: ", size(inference))
    # Resize so that we're size (batch_size, state_size), using numpy
    inference = np.resize(inference, (batch_size, state_size))
    #println("Inference shape: ", size(inference))
    # Convert back to a julia array that is (batch_size, state_size)
    inference = convert(Array{Float64, 2}, inference)
    #println("Inference shape: ", size(inference))
    # Re normalise the inference, in an elementwise way
    #inference = (inference .* stds) .+ means
    #println("Inference shape: ", size(inference))

    return inference

end

function get_inference(model, window)
    # This is the mean and std of the training data
    means = [ 0.47824945  0.7140721   0.86563648  1. -0.02893655 -0.01286843 -0.0065491   0.81765136]
    stds  = [1.00000001 1.00000001 1.00000001 0.72686883 1.00000001 0.301784 0.14164948 0.4792477 ]

    # What is the window and state size? Gets input as (window_size, state_size)
    window_size, state_size = size(window)

    # Normalise the window
    window = (window .- means) ./ stds
    # Reshape to (1, window_size, state_size)
    window = reshape(window, (1, window_size, state_size))
    # Tensorise 
    window = torch.FloatTensor(window)

    # Perform inference
    inference, _ = model(window)

    # Detach from the graph and numpify
    inference = inference.detach().numpy()
    # Resize so that we're size (state_size,), using numpy
    inference = np.resize(inference, (state_size,))
    # Convert back to a julia array that is 1,state_size
    inference = convert(Array{Float64,2}, inference')
    # Re normalise the inference, in an elementwise way
    inference = (inference .* stds) .+ means

    return inference

end