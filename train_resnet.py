import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "7"
from sys import path
path.append('/home/fodl/shiratomer/dloptics/utils')
path.append('/home/fodl/shiratomer/dloptics/preprocessing')
from spots_to_images_gaussian_hdf5 import load_and_stack_arrays
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import tqdm
import torch.nn.functional as F

MODEL_PATH = '/home/fodl/shiratomer/dloptics/panasonic_residuals_5_layers_4096_noDropOut.pt'

DATASET_PATH = "/home/fodl/deanoren/DLens/dloptics/panasonic_June_20"

HIDDEN_LAYER_SIZE = 4096

NUM_OF_EXAMPLES = 500000
BATCH_SIZE = 64*4
LR = 0.00001
LR_GAMMA = 0.1
LR_MILESTONES = [30, 80, 120, 200, 280, 350]
CHECKPOINT_INTERVAL = 10
N_EPOCHS = 150000000
TRAIN_SIZE_PERCENT = 0.8
NULL_BY_COLUMNS = False
np.random.seed(0)

# Network structure
class ResidualBlock(nn.Module):
    """ A residual block that applies a linear transformation followed by ReLU and dropout. """
    def __init__(self, input_dim, output_dim):
        super(ResidualBlock, self).__init__()
        self.linear = nn.Linear(input_dim, output_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        identity = x
        out = self.dropout(self.relu(self.linear(x)))
        out += identity  # Add the input x to the output of linear layer
        return out

class ModelWithResiduals(nn.Module):
    """ A neural network model with an initial layer, five residual blocks, and a final output layer. """
    def __init__(self, input_dim, hidden_layer_size, output_dim):
        super(ModelWithResiduals, self).__init__()
        self.initial_layer = nn.Linear(input_dim, hidden_layer_size)
        self.relu = nn.ReLU()
        self.initial_dropout = nn.Dropout(0.0001)
        self.residual_blocks = nn.Sequential(
            ResidualBlock(hidden_layer_size, hidden_layer_size),
            ResidualBlock(hidden_layer_size, hidden_layer_size),
            ResidualBlock(hidden_layer_size, hidden_layer_size),
            ResidualBlock(hidden_layer_size, hidden_layer_size),
            ResidualBlock(hidden_layer_size, hidden_layer_size),
            ResidualBlock(hidden_layer_size, hidden_layer_size),
            ResidualBlock(hidden_layer_size, hidden_layer_size),
            ResidualBlock(hidden_layer_size, hidden_layer_size),
            ResidualBlock(hidden_layer_size, hidden_layer_size)
        )
        self.final_layer = nn.Linear(hidden_layer_size, output_dim)

    def forward(self, x):
        x = self.initial_dropout(self.relu(self.initial_layer(x)))
        x = self.residual_blocks(x)
        x = self.final_layer(x)
        return x
    

def main(model_path, dataset_path, hidden_layer_size, num_of_examples, batch_size, lr, lr_gamma, lr_milestones, \
    checkpoint_interval, n_epochs, null_by_columns): 

    # load data
    data = load_and_stack_arrays(dataset_path, num_of_examples, save_results = True,  prepare_anyway = True)
    spots = data["outputs_united"].reshape(data["outputs_united"].shape[0], -1)
    translations = data["inputs_united"]

    print("spots shape before filtering: ", spots.shape)
    if null_by_columns:
        nan_columns_indices = np.where(np.isnan(spots))[1].tolist() #Columns that have at least one nan value
        all_columns = np.arange(spots.shape[1])
    else:
        nan_columns_indices = np.where(np.isnan(spots))[0].tolist()
        all_columns = np.arange(spots.shape[0])

    nan_columns_indices = set(nan_columns_indices)
    columns_to_keep = np.setdiff1d(all_columns, list(nan_columns_indices))

    if null_by_columns:
        spots = spots[:, columns_to_keep]
    else:
        spots = spots[columns_to_keep, :]
        translations = translations[columns_to_keep, :]

    has_nan = np.any(np.isnan(spots))

    print(has_nan)

    print("spots shape after filtering: ", spots.shape)

    # Split the data into training and testing sets
    indices = np.arange(spots.shape[0])
    np.random.shuffle(indices)
    train_size = int(TRAIN_SIZE_PERCENT*len(indices))
    train_indices = indices[:train_size] 
    test_indices = indices[train_size:]

    spots_tensor = torch.tensor(spots, dtype=torch.float32)
    trans_tensor = torch.tensor(translations, dtype=torch.float32)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using {device} device")

    X_train = spots_tensor[train_indices].to(device)
    y_train = trans_tensor[train_indices].to(device)
    X_test = spots_tensor[test_indices].to(device)
    y_test  = trans_tensor[test_indices].to(device)

    #Normalizing
    X_train_mean = X_train.mean(axis=0)
    X_train_std = X_train.std(axis=0)
    X_train = (X_train - X_train_mean)/(X_train_std)

    X_test_mean = X_test.mean(axis=0)
    X_test_std = X_test.std(axis=0)
    X_test = (X_test - X_test_mean)/(X_test_std)

    y_train_mean = y_train.mean(axis=0)
    y_train_std = y_train.std(axis=0)
    y_train = (y_train - y_train_mean)/(y_train_std)

    y_test_mean = y_test.mean(axis=0)
    y_test_std = y_test.std(axis=0)
    y_test = (y_test - y_test_mean)/(y_test_std)

    model = ModelWithResiduals(X_train.shape[1], hidden_layer_size, y_train.shape[1]).to(device)

    if torch.cuda.device_count() > 1:
        print("Using", torch.cuda.device_count(), "GPUs")
    model = nn.DataParallel(model)

    loss_fn = nn.MSELoss()  # mean square error
    optimizer = optim.AdamW(model.parameters(), lr=lr)

    batch_start = torch.arange(0, len(X_train), batch_size)

    train_losses = []
    test_losses = []


    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5, verbose=True)

    for epoch in range(n_epochs):
        # evaluate accuracy at beginning of each epoch
        with torch.no_grad():
            model.eval()
            test_pred = model(X_test)
            test_mse = loss_fn(test_pred, y_test)
            test_mse = float(test_mse)
            test_losses.append(test_mse)
            train_pred = model(X_train)
            train_mse = loss_fn(train_pred, y_train)
            train_mse = float(train_mse)
            train_losses.append(train_mse)
            print("epoch ", epoch, "train loss ", train_mse, "test loss ", test_mse)
            print("spots shape after filtering: ", spots.shape)
        model.train()

        # shuffle batches
        indices = torch.randperm(len(X_train)) 
        X_train = X_train[indices] 
        y_train = y_train[indices] 

        with tqdm.tqdm(batch_start, unit="batch", mininterval=0, disable=True) as bar:
            bar.set_description(f"Epoch {epoch}")
            for start in bar:
                # take a batch
                X_batch = X_train[start:start+batch_size]
                y_batch = y_train[start:start+batch_size]
                # forward pass
                y_pred = model(X_batch)
                loss = loss_fn(y_pred, y_batch)
                # backward pass
                optimizer.zero_grad()
                loss.backward()
                # update weights
                optimizer.step()
                # print progress
                bar.set_postfix(mse=float(loss))

        # evaluate accuracy at end of each epoch
        scheduler.step(train_mse)
        # scheduler.step()
        if epoch%checkpoint_interval == 0:
            torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': loss_fn,
                'train_loss': train_losses,
                'test_loss':test_losses,
                'test_pred':test_pred,
                'train_pred':train_pred,
                'X_test':X_test,
                'X_train':X_train,
                'X_train_mean':X_train_mean,
                'X_train_std':X_train_std,
                'X_test_mean':X_test_mean,
                'X_test_std':X_test_std,
                'y_train_mean':y_train_mean,
                'y_train_std':y_train_std,
                'y_test_mean':y_test_mean,
                'y_test_std':y_test_std, 
                'train_indices':train_indices,
                'test_indices':test_indices
                }, model_path)
            if test_mse < 0.00001:
                break

if __name__ == '__main__':
    main(MODEL_PATH, DATASET_PATH, HIDDEN_LAYER_SIZE, NUM_OF_EXAMPLES, BATCH_SIZE, LR, LR_GAMMA, LR_MILESTONES, \
        CHECKPOINT_INTERVAL, N_EPOCHS, NULL_BY_COLUMNS)
