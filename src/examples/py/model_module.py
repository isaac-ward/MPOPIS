import csv
import os
import ast
import math
import numpy as np
from tqdm import tqdm
from glob import glob
import random
import tempfile
import time
from datetime import datetime

# conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
import torch
import torchvision.transforms as transforms
import torch.nn as nn
import torch.utils.data
from torch.utils.data import DataLoader

# conda install pytorch-lightning -c conda-forge
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import TQDMProgressBar

import torchmetrics
from torchmetrics.regression import MeanAbsoluteError
from torchmetrics.regression import MeanSquaredError
from torchmetrics.regression import R2Score

# conda install -c conda-forge matplotlib
import matplotlib.pyplot as plt

# pip install wandb
import wandb

# pip install transformers
import transformers
from transformers import AutoModel, AutoConfig
from transformers import AutoformerConfig, AutoformerModel

# File location
folder_checkpoint_glob = "C:/Users/moose/Desktop/dev/aa290/src/pynb/autodrift9/qzk0tw36/checkpoints/*.ckpt"
window_size = 1

# ----------------------------------------

class SequenceLearner(pl.LightningModule):
    """
    This is a PyTorch Lightning module that does sequence learning 
    using Adam optimisation 
    """
    def __init__(self, model, lr=0.005):
        super().__init__()
        self.model = model
        self.lr = lr
        
        # Convienent for determining device
        self.dummy_param = nn.Parameter(torch.empty(0))
        
    def get_device(self):
        return self.dummy_param.device
    
    def forward(self, x):
        return self.model(x)

    def _train_val_helper(self, batch, batch_idx, name):
        """
        Training and validation are near identical so we'll
        use this function to reuse code
        """
        # Extract inputs and prediction targets from batch
        x, y = batch
        y_hat, _ = self.model.forward(x)
        y_hat.view_as(y)
        
        def simple_log(quantity_name, quantity):
            self.log(
                f"{name}_{quantity_name}", 
                quantity, 
                prog_bar=True,
                logger=True, 
                #on_step=True, 
                on_epoch=True,
            )
        
        # Use MSE as loss metric (this is a regression task)
        loss = nn.MSELoss()(y_hat, y)
        simple_log("loss (mse)", loss)
        
        # And the rest
        metrics_of_interest = [
            MeanAbsoluteError,
            MeanSquaredError,
            R2Score
        ]
        for m in metrics_of_interest:
            # Should really reuse computers
            computer = m().to(self.get_device())
            # These are all linear, so we can flatten from 
            # (batch_size, num_outputs)
            value = computer(
                torch.flatten(y_hat), 
                torch.flatten(y)
            )
            simple_log(m.__name__.lower(), value)
        
        # TODO should the other quantities be in the dict too?
        return {
            "loss": loss
        }

    def training_step(self, batch, batch_idx):
        return self._train_val_helper(
            batch, batch_idx, "train"
        )

    def validation_step(self, batch, batch_idx):
        return self._train_val_helper(
            batch, batch_idx, "val"
        )

    def test_step(self, batch, batch_idx):
        """
        When we test we're actually going to try to predict
        an entire trial, starting from the start state
        """
        
        # What type are the model params?
        model_parameters = next(self.model.parameters())
        
        # In this case, the batch is 1 single trial. The trial
        # has been issued by the datamodule so it's a tensor
        true_trial = batch[0]
        trial_length = len(true_trial)
        state_size = len(true_trial[0])
        
        # Rather than see W ground truth states at the start and 
        # predict the whole trial, we want to see W ground truth states
        # at every time step. Since the first W states don't have W
        # states before them, we assume them to be all zero, meaning
        # that we can always inference a full trial
        zero_states = torch.zeros(window_size, state_size).to(true_trial.get_device())
        true_trial_padded = torch.cat((zero_states, true_trial), dim=0)
        
        # Predict every state in the trial                       
        pred_trial = []
        for i in range(trial_length):
            # Get the first 'window_size' elements
            window_tensor = true_trial_padded[i:i+window_size]
        
            # Each window is (W, |S|) shaped. We want it to be 
            # a (W|S|,) shaped pytorch tensor
            window_tensor = window_tensor.view(-1)
                               
            # We're in a test callback so no_grad should be
            # active. TODO: check
            
            # Needs to be 'batched' to go into model
            input_tensor = torch.unsqueeze(window_tensor, 0)
            # Cast this input to be the same type as the 
            # model weights
            input_tensor = input_tensor.to(model_parameters.dtype)
            # And then unbatched
            pred_state = self.model.forward(input_tensor)[0][0]
            
            # Note the predicted state (we'll plot this later)
            pred_trial.append(pred_state)
            
        # Numpify for convienient indexing, and
        # detach for statistics/plotting/viz
        true_trial = np.array([_.cpu() for _ in true_trial])
        pred_trial = np.array([_.cpu() for _ in pred_trial])
        
        # TODO label logs with number of training epochs
        
        # Log the plots to weights and biases, do this
        # by saving a figure in a temporary location and
        # then sending it up to the cloud
        def log_fig(title, fig):
            """
            Use tempfiles to make this easy
            """
            with tempfile.TemporaryDirectory() as temp_dir:
                temp_file = os.path.join(temp_dir, f"tmp.png")
                fig.savefig(temp_file)

                # Log the figure to Weights and Biases
                self.logger.experiment.log({
                    title: wandb.Image(temp_file)
                })
        # log_fig(
        #     "test/plots/states", 
        #     figure_pred_vs_true_trial(true_trial, pred_trial)
        # )
        
        # **Anything that comes from our data module is normalised,
        # so plots will be in normalised space
        
        # We also want to compute a numerical score (loss) for each 
        # state element / feature
        features = [
            "x position",
            "y position",
            "heading",
#             "longitudinal velocity",
#             "lateral velocity",
#             "heading rate",
#             "commanded turn",
#             "commanded acceleration",
        ]
        results_dict = {}
        for i, f in enumerate(features):
            # Compute and log, using torchmetrics as no
            # backprop is required
            s_y = np.array(true_trial[:,i])
            s_yhat = np.array(pred_trial[:,i])
            results_dict[f] = ((s_y - s_yhat)**2).mean(axis=0)
            self.log(f"test/{f}_loss", results_dict[f], prog_bar=False)

        # Calculate and log the average loss
        average_loss = np.mean(list(results_dict.values()))
        self.log("test/combined_loss (averaged)", average_loss, prog_bar=False)
        
        # Make a racetrack plot of this - remember to denormalise the trial
        dn_trial = dm.denormalise_trial(pred_trial)
        # log_fig(
        #     "test/plots/racetrack",
        #     plot_trial(
        #         dn_trial,
        #         # Want to map from the predicted trial back to the 
        #         # index of the track so that the plot can be overlaid
        #         gt_positions=[],
        #         plot_each_n=1, 
        #         return_fig=True
        #     )
        # )
        
        return results_dict

    def configure_optimizers(self):
        return torch.optim.Adam(self.model.parameters(), lr=self.lr)


class RegressionNN(nn.Module):
    def __init__(
        self, 
        input_size, 
        hidden_layers,
        output_size
    ):
        super(RegressionNN, self).__init__()

        # Create a list to store the layers
        layers = []

        # Add the input layer
        layers.append(nn.Linear(input_size, hidden_layers[0]))
        layers.append(nn.ReLU(inplace=True))

        # Add hidden layers
        for i in range(1, len(hidden_layers)):
            layers.append(nn.Linear(hidden_layers[i - 1], hidden_layers[i]))
            layers.append(nn.ReLU(inplace=True))

        # Add the output layer
        layers.append(nn.Linear(hidden_layers[-1], output_size))

        # Combine all layers into a Sequential model
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x), None # To match output of LTCN

def make_nn(
    input_size,
    output_size,
    hidden_layers,
    lr=0.01
):
    """
    Use transformers for the sequence to sequence prediction task
    """
    
    # Make the network
    nn = RegressionNN(
        input_size,
        hidden_layers,
        output_size,
    )
    
    # Wrap in a sequence learner so that we
    # can use PyTorch Lightning utilities
    return SequenceLearner(
        nn,
        lr=lr
    )

def load_model():
    """
    Load a model from a checkpoint
    """
    
    # Make the network
    nn = RegressionNN(
        input_size=8*window_size,
        hidden_layers=[24,24],
        output_size=8,
    )
    
    # Wrap in a sequence learner so that we
    # can use PyTorch Lightning utilities
    model = SequenceLearner(nn)
    
    # Load the weights
    folder_checkpoint_glob = "C:/Users/moose/Desktop/dev/aa290/src/pynb/autodrift10/m0q2wffn/checkpoints/*.ckpt"
    # Get the latest checkpoint
    checkpoint_path = sorted(glob(folder_checkpoint_glob))[-1]
    model.load_state_dict(torch.load(checkpoint_path)["state_dict"])
    
    return model