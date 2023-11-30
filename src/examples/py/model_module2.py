from glob import glob
import sys
import torch

# Before custom imports
sys.path.append('C:/Users/moose/Desktop/dev/aa290/')
import src.main

# ----------------------------------------

def load_model():
    """
    Load a model from a checkpoint
    """

    # Create the data module used throughout here. In eval mode
    # this is really just used to determine shapes
    data_module = src.main.make_data_module()

    # Create the model
    model = src.main.make_model("nn", data_module)
    
    # Load the weights
    folder_checkpoint_glob = "C:/Users/moose/Desktop/dev/aa290/autodrift15/4kbau6q9/checkpoints/*.ckpt"
    available_checkpoints = glob(folder_checkpoint_glob)
    print(f"Found {len(available_checkpoints)} checkpoints:")
    for i,c in enumerate(available_checkpoints):
        # Indicate which one is being used
        if i == len(available_checkpoints) - 1:
            print(f"\t{c} <-- using")
        else:
            print(f"\t{c}")

    # Get the latest checkpoint
    checkpoint_path = available_checkpoints[-1]
    model.load_state_dict(torch.load(checkpoint_path)["state_dict"])

    # Print a summary of the model including the shapes of weights
    print("Using model:")
    print(model)
    
    return model