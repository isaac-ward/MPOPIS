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
    model_module = pyimport("model_module2")
    println("Python learning module imported")

    # Want it in eval mode
    model = model_module.load_model()
    model.eval()
    println("Python learning model loaded and ready for evaluation")

    return model
end

function normalize_tensor(tensor, means, stds)
    num_samples, window_size, state_size = size(tensor)
    # Iterate over every num_sample and window 
    # and normalise each state measurement (there will
    # be num_samples*window_size of these)
    for i in 1:num_samples
        for j in 1:window_size
            for k in 1:state_size
                tensor[i, j, k] = (tensor[i, j, k] - means[k]) / stds[k]
            end
        end
    end
    return tensor
end
function denormalize_tensor(tensor, means, stds)
    num_samples, state_size = size(tensor)
    # Iterate over every sample 
    # and normalise each state measurement (there will
    # be num_samples of these)
    for i in 1:num_samples
        for j in 1:state_size
            tensor[i, j] = (tensor[i, j] * stds[j]) + means[j]
        end
    end
    return tensor
end

function get_inference_batch(model, batch_window)
    # This is the mean and std of the training data
    means = [ 0.47824945  0.7140721   0.86563648  1. -0.02893655 -0.01286843 -0.0065491   0.81765136]
    stds  = [1.00000001 1.00000001 1.00000001 0.72686883 1.00000001 0.301784 0.14164948 0.4792477 ]

    # What is the window and state size? Gets input as (batch_size, window_size, state_size)
    batch_size, window_size, state_size = size(batch_window)
    # println("")
    # println("Batch size: ", batch_size)
    # println("Window size: ", window_size)
    # println("State size: ", state_size)
    # println("Input size: ", size(batch_window))
    # println("means size", size(means))
    # println("stds size", size(stds))

    # Normalise the batch_window
    batch_window_original = batch_window
    batch_window = normalize_tensor(batch_window, means, stds)
    # Reshape to (batch_size, window_size*state_size) - the model deals in 
    # inputs of size window_size*state_size
    batch_window = reshape(batch_window, (batch_size, window_size*state_size))
    # Tensorise
    batch_window = torch.FloatTensor(batch_window)

    # Perform inference
    output = model(batch_window)
    inference, _ = output

    # Detach from the graph and numpify
    inference = inference.detach().numpy()
    # Resize so that we're size (batch_size, state_size), using numpy
    inference = np.resize(inference, (batch_size, state_size))
    # Convert back to a julia array that is (batch_size, state_size)
    inference = convert(Array{Float64, 2}, inference)
    # Re normalise the inference, in an elementwise way
    inference = denormalize_tensor(inference, means, stds)

    # The result is a (batch_size, state_size) array
    # Create a zeros array for debugging
    debug_inference = zeros(batch_size, state_size)
    # Go through every batch
    for i in 1:batch_size
        # Enter the x and y (first and second) elements
        last1 = batch_window_original[i, end, :]
        last2 = batch_window_original[i, end-1, :]
        scale_velocity = +1
        xguess = last1[1] + (last1[1] - last2[1])*scale_velocity
        yguess = last1[2] + (last1[2] - last2[2])*scale_velocity
        # Add a random perturbation
        scale_factor = 0.5
        xguess = xguess + randn() * scale_factor * (i/batch_size) + 0.5
        yguess = yguess + randn() * scale_factor * (i/batch_size) + 0.5
        # Normalise xy guessses as a vector and scale 
        xnorm = xguess / sqrt(xguess^2 + yguess^2)
        ynorm = yguess / sqrt(xguess^2 + yguess^2)
        # Add to the debug inference
        debug_inference[i, 1] = xguess
        debug_inference[i, 2] = yguess
    end

    #return inference
    return debug_inference

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