
import numpy as np
import pandas as pd
import os
import json
import pickle
np.random.seed(0)

def load_and_stack_arrays(data_dir_path, num_of_files, num_of_subsets = 1, subset_num = 0, prepare_anyway = False, save_results = False):
    """
    Load and stack arrays from NPZ files. If the processed data is already saved on the disk, it can be loaded directly.
    """
    
    # 8-l dean load
    """
    df = pd.read_pickle(data_dir_path)
    data_dict = {'inputs_united':df['inputs_united'], 'outputs_united':df['outputs_united']}
    return data_dict
    """
    """
    #June 20 Panasonic load (Same comments as 4 -lens)
    # Get list of files in the directory.
    files = [f for f in os.listdir(data_dir_path) if f.endswith('.npz')]
    files.sort(key=lambda x: int(x.split('.')[0]))
    num_of_files = min([len(files), num_of_files])

    # Initialize an empty array
    stacked_input = np.array([])
    stacked_output = np.array([])
    temp_input = []
    temp_output = []
    finish_index = min( num_of_files, num_of_files//num_of_subsets * (subset_num + 1))

    for i in range(num_of_files//num_of_subsets * subset_num, finish_index):
        if i >= num_of_files:
            break
        # Retrieve the file path of the current file in the loop from the sorted files list.
        file_path = files[i]
        try:
            # Load the data from the .npz file at the specified path.
            file = np.load(os.path.join(data_dir_path, file_path))
        except:
            print(os.path.join(data_dir_path, file_path))
        output = file['output']
        input = file['input']


        # Append the 'output' array from the loaded file to the temp_output list. 
        # A new axis is added to the 'output' to ensure it has the correct shape for later stacking.
        output = output[np.newaxis, :]
        temp_output.append(output)  # replace 'arr_0' with the specific key you have used while saving
        temp_input.append(input)  # replace 'arr_0' with the specific key you have used while saving
        # Merge data periodically to avoid large memory consumption.
        if i%10000 == 0 or i >= finish_index:
            stacked_input = np.vstack(temp_input) 
            stacked_output = np.vstack(temp_output) 
            temp_input = [stacked_input]
            temp_output = [stacked_output]
    data_dict = {'inputs_united':stacked_input, 'outputs_united':stacked_output}
    return data_dict
    """
    """
    # may 18 8-l load
    df = pd.read_pickle(data_dir_path)
    # Removes z angles
    indices_to_remove = [7,10,13,16]
    stacked_input = df['inputs_united']

    # Create a mask that is True for indices to keep
    mask = np.ones(stacked_input.shape[1], dtype=bool)
    mask[indices_to_remove] = False

    # Use the mask to filter out the specified indices
    stacked_input = stacked_input[:, mask]

    data_dict = {'inputs_united':stacked_input, 'outputs_united':df['outputs_united']}
    return data_dict

    stacked_data_path = '/home/fodl/shiratomer/dloptics/stacked_raw_data/' + data_dir_path.split('/')[-1] + '_subset_ ' + str(subset_num) + 'num_of_subsets' + str(num_of_subsets) +'.npz'
    # Check if processed data is already available.
    if os.path.isfile(stacked_data_path) and not prepare_anyway:
        with np.load(stacked_data_path) as data:
            stacked_input = data['inputs_united']
            stacked_output = data['outputs_united']
        data_dict = {'inputs_united':stacked_input, 'outputs_united':stacked_output}
        return data_dict
    
    # new format parsing 
    
    stacked_input = np.load(data_dir_path + "/inputs_united.npz")["arr_0"]

    # Removes z angles
    indices_to_remove = [7,10,13,16]

    # Create a mask that is True for indices to keep
    mask = np.ones(stacked_input.shape[1], dtype=bool)
    mask[indices_to_remove] = False

    # Use the mask to filter out the specified indices
    stacked_input = stacked_input[:, mask]

    stacked_output = np.load(data_dir_path + "/outputs_united.npz")["arr_0"]
    data_dict = {'inputs_united':stacked_input, 'outputs_united':stacked_output}
    return data_dict
    """

    # 4-l without screens load
    # Get list of files in the directory.
    files = [f for f in os.listdir(data_dir_path) if f.endswith('.npz')]
    files.sort(key=lambda x: int(x.split('.')[0]))
    num_of_files = min([len(files), num_of_files])

    # Initialize an empty array
    stacked_input = np.array([])
    stacked_output = np.array([])
    temp_input = []
    temp_output = []
    # Calculate the index where the loading process should stop. 
    # The logic is to divide the files into `num_of_subsets` equal portions and 
    # load only the portion specified by `subset_num`.
    # If the total number of files is not exactly divisible by `num_of_subsets`, 
    # this ensures we don't exceed the actual number of files available.
    finish_index = min( num_of_files, num_of_files//num_of_subsets * (subset_num + 1))

    # Start iterating from the beginning of the current subset, as defined by subset_num, 
    # and continue up to the previously calculated finish_index.
    # The purpose of this loop is to load a specific subset of files from the directory.
    for i in range(num_of_files//num_of_subsets * subset_num, finish_index):
    #for i in range(400001, 499999):
    #for i in range(0, 400000):
        if i >= num_of_files:
            break
        # Retrieve the file path of the current file in the loop from the sorted files list.
        file_path = files[i]
        try:
            # Load the data from the .npz file at the specified path.
            file = np.load(os.path.join(data_dir_path, file_path))
        except:
            print(os.path.join(data_dir_path, file_path))
        output = file['output']

        input = file['input']

        # Removes inside screens
        indices_to_remove = [2, 3, 6, 7]

        # Create a boolean mask to select indices to keep
        mask = np.ones(output.shape[0], dtype=bool)
        mask[indices_to_remove] = False

        # Apply the mask to select only the desired indices
        output = output[mask]

        # Append the 'output' array from the loaded file to the temp_output list. 
        # A new axis is added to the 'output' to ensure it has the correct shape for later stacking.
        output = output[np.newaxis, :]
        temp_output.append(output)  # replace 'arr_0' with the specific key you have used while saving
        temp_input.append(input)  # replace 'arr_0' with the specific key you have used while saving
        # Merge data periodically to avoid large memory consumption.
        if i%10000 == 0 or i >= finish_index:
            stacked_input = np.vstack(temp_input) 
            stacked_output = np.vstack(temp_output) 
            temp_input = [stacked_input]
            temp_output = [stacked_output]
    data_dict = {'inputs_united':stacked_input, 'outputs_united':stacked_output}
    return data_dict
