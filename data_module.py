"""
Module containing functions used to parse the WISDM dataset, and create CSVs that will ultimately be fed into classifiers.

Inspo: https://github.com/bartkowiaktomasz/har-wisdm-lstm-rnns/blob/master/HAR_Recognition.py

NOTE: load_raw_dataset may be deprecated or largely re-written

Author: Nate Burley
"""

import glob
import string
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from typing import Tuple
from sklearn.model_selection import train_test_split



def load_raw_dataset(data_file: str='data/wisdm_master_raw_dataset.csv',
                     device_type='watch',
                     sensor_type='accel',
                     segment_time_size: int=180, 
                     time_step: int=100,
                     ) -> Tuple[np.array, np.array]:
    """
    Function that loads and segments raw data into arrays of data and labels
    """
    # Load raw data from file
    data = pd.read_csv(data_file)

    # Add activity label (integer for each class)
    data['activity_label'] = data['activity_code'].apply(lambda x: ord(x)-65)  # TODO: Move this to build_datasets.py
    
    # Select data for given device and sensor
    data = data.loc[(data['device_type'] == device_type) & (data['sensor_type'] == sensor_type)]
    
    # Data preprocessing
    X_data = []
    y_data = []

    # Slide a "segment_time_size" wide window with a step size of "time_step"
    for i in range(0, len(data) - segment_time_size, time_step):
        x = data['x'].values[i: i + segment_time_size]
        y = data['y'].values[i: i + segment_time_size]
        z = data['z'].values[i: i + segment_time_size]
        X_data.append([x, y, z])

        # Label for a data window is the label that appears most commonly
        label = stats.mode(data['activity_label'][i: i + segment_time_size])[0][0]
        y_data.append(label)

    # Convert to numpy
    X_data = np.asarray(X_data, dtype=np.float32).transpose(0, 2, 1)

    # One-hot encoding
    y_data = np.asarray(pd.get_dummies(y_data), dtype=np.float32)

    ### DEBUGGING
    print("Convoluted data shape: ", X_data.shape)
    print("Labels shape:", y_data.shape)

    # Return data
    return X_data, y_data


def load_preprocessed_dataset(data_df_filename: str='data/wisdm_preprocessed_watch_accel.csv') -> Tuple[np.array, np.array]:
    """
    Function that loads one of the pre-processed datasets.
    """
    # Open data
    data_df = pd.read_csv(data_df_filename)
    
    # Get y_data, remove from data frame
    y = data_df.ACTIVITY
    X = data_df.drop('ACTIVITY', axis=1)

    # Fix issue with activity drop above
    X.drop('Unnamed: 0', inplace=True, axis=1)

    return X, y