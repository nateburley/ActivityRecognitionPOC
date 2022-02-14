"""
Script that loads, preprocesses, and cleans the WISDM dataset, and then saves the resulting Pandas DataFrame.

Link to data: https://archive.ics.uci.edu/ml/datasets/WISDM+Smartphone+and+Smartwatch+Activity+and+Biometrics+Dataset+
Helpful reading: https://www.kaggle.com/kgmeirm/human-activity-detection-by-smartphone-sensors-pa

Author: Nate Burley
"""

import os, glob
import pandas as pd

# Define mapping dict between activity name and code
activity_codes_mapping = {'A': 'walking',
                          'B': 'jogging',
                          'C': 'stairs',
                          'D': 'sitting',
                          'E': 'standing',
                          'F': 'typing',
                          'G': 'brushing teeth',
                          'H': 'eating soup',
                          'I': 'eating chips',
                          'J': 'eating pasta',
                          'K': 'drinking from cup',
                          'L': 'eating sandwich',
                          'M': 'kicking soccer ball',
                          'O': 'playing catch tennis ball',
                          'P': 'dribbling basket ball',
                          'Q': 'writing',
                          'R': 'clapping',
                          'S': 'folding clothes'}

# Define features for the pre-processed data
PROCESSED_FEATURE_NAMES = ['ACTIVITY',
            'X0', # 1st bin fraction of x axis acceleration distribution
            'X1', # 2nd bin fraction ...
            'X2',
            'X3',
            'X4',
            'X5',
            'X6',
            'X7',
            'X8',
            'X9',
            'Y0', # 1st bin fraction of y axis acceleration distribution
            'Y1', # 2nd bin fraction ...
            'Y2',
            'Y3',
            'Y4',
            'Y5',
            'Y6',
            'Y7',
            'Y8',
            'Y9',
            'Z0', # 1st bin fraction of z axis acceleration distribution
            'Z1', # 2nd bin fraction ...
            'Z2',
            'Z3',
            'Z4',
            'Z5',
            'Z6',
            'Z7',
            'Z8',
            'Z9',
            'XAVG', # average sensor value over the window (per axis)
            'YAVG',
            'ZAVG',
            'XPEAK', # Time in milliseconds between the peaks in the wave associated with most activities. heuristically determined (per axis)
            'YPEAK',
            'ZPEAK',
            'XABSOLDEV', # Average absolute difference between the each of the 200 readings and the mean of those values (per axis)
            'YABSOLDEV',
            'ZABSOLDEV',
            'XSTANDDEV', # Standard deviation of the 200 window's values (per axis)  ***BUG!***
            'YSTANDDEV',
            'ZSTANDDEV',
            'XVAR', # Variance of the 200 window's values (per axis)   ***BUG!***
            'YVAR',
            'ZVAR',
            'XMFCC0', # short-term power spectrum of a wave, based on a linear cosine transform of a log power spectrum on a non-linear mel scale of frequency (13 values per axis)
            'XMFCC1',
            'XMFCC2',
            'XMFCC3',
            'XMFCC4',
            'XMFCC5',
            'XMFCC6',
            'XMFCC7',
            'XMFCC8',
            'XMFCC9',
            'XMFCC10',
            'XMFCC11',
            'XMFCC12',
            'YMFCC0', # short-term power spectrum of a wave, based on a linear cosine transform of a log power spectrum on a non-linear mel scale of frequency (13 values per axis)
            'YMFCC1',
            'YMFCC2',
            'YMFCC3',
            'YMFCC4',
            'YMFCC5',
            'YMFCC6',
            'YMFCC7',
            'YMFCC8',
            'YMFCC9',
            'YMFCC10',
            'YMFCC11',
            'YMFCC12',
            'ZMFCC0', # short-term power spectrum of a wave, based on a linear cosine transform of a log power spectrum on a non-linear mel scale of frequency (13 values per axis)
            'ZMFCC1',
            'ZMFCC2',
            'ZMFCC3',
            'ZMFCC4',
            'ZMFCC5',
            'ZMFCC6',
            'ZMFCC7',
            'ZMFCC8',
            'ZMFCC9',
            'ZMFCC10',
            'ZMFCC11',
            'ZMFCC12',
            'XYCOS', # The cosine distances between sensor values for pairs of axes (three pairs of axes)
            'XZCOS',
            'YZCOS',
            'XYCOR', # The correlation between sensor values for pairs of axes (three pairs of axes)
            'XZCOR',
            'YZCOR',
            'RESULTANT', # Average resultant value, computed by squaring each matching x, y, and z value, summing them, taking the square root, and then averaging these values over the 200 readings
            'PARTICIPANT'] # Categirical: 1600 -1650


def read_data(dir_name: str) -> pd.DataFrame:
    """
    Helper function that reads a subset of thee WISDM dataset raw files into a Pandas DataFrame
    
    Args:
        dir_name: String. Directory containing raw data text files
    Returns:
        pd.DataFrame
    """

    raw_data_df_list = []

    for file in os.listdir(dir_name):
        # Format filename with absolute path
        data_file = dir_name + '/' + file
        
        # Read the data in
        raw_data = pd.read_csv(f'{data_file}', names=['participant_id' , 'activity_code' , 'timestamp', 'x', 'y', 'z'], index_col=None, header=None)

        # Remove formatting error from z-axis
        raw_data.z = raw_data.z.str.strip(';')
        raw_data.z = pd.to_numeric(raw_data.z)

        # Add column with activity name
        raw_data['activity'] = raw_data['activity_code'].map(activity_codes_mapping)

        # Add column with device type (watch vs. phone)
        if 'phone' in dir_name:
            raw_data['device_type'] = 'phone'
        elif 'watch' in dir_name:
            raw_data['device_type'] = 'watch'
        else:
            raw_data['device_type'] = None

        # Add column with sensor type (accel vs. gyro)
        if 'accel' in dir_name:
            raw_data['sensor_type'] = 'accel'
        elif 'gyro' in dir_name:
            raw_data['sensor_type'] = 'gyro'
        else:
            raw_data['sensor_type'] = None

        # Format final columns
        raw_data = raw_data[['participant_id', 'device_type', 'sensor_type', 'activity_code', 'activity', 'timestamp', 'x', 'y', 'z']]

        raw_data_df_list.append(raw_data)

    # Join the dataframes, return result
    raw_data_agg_df = pd.concat(raw_data_df_list)
    return raw_data_agg_df


def build_raw_data(data_root_path: str=os.getcwd()) -> pd.DataFrame:
    """
    Function that reads the raw WISDM sensor data into a "master" Pandas DataFrame.

    Args:
        data_root_path: String. Path to wisdm-dataset. Defaults to current directory.
    Returns:
        master_data_df: Pandas DataFrame. Raw WISDM dataset.
    """

    # Format directory strings for watch data
    watch_accel_data_dir = data_root_path + 'wisdm-dataset/raw/watch/accel'
    watch_gyro_data_dir = data_root_path + 'wisdm-dataset/raw/watch/gyro'

    # Format directory strings for phone data
    phone_accel_data_dir = data_root_path + 'wisdm-dataset/raw/phone/accel'
    phone_gyro_data_dir = data_root_path + 'wisdm-dataset/raw/phone/gyro'

    # Read in watch data
    watch_accel_df = read_data(watch_accel_data_dir)
    watch_gyro_df = read_data(watch_gyro_data_dir)
    watch_data_df = pd.concat([watch_accel_df, watch_gyro_df])

    # Read in phone data
    phone_accel_df = read_data(phone_accel_data_dir)
    phone_gyro_df = read_data(phone_gyro_data_dir)
    phone_data_df = pd.concat([phone_accel_df, phone_gyro_df])

    # Join watch and phone datasets into one, return results
    master_data_df = pd.concat([watch_data_df, phone_data_df])
    
    return master_data_df



def build_preprocessed_dataset(processed_data_root: str='wisdm-dataset/arff_files/watch/accel') -> pd.DataFrame:
    """
    Function that loads one of the pre-processed datasets.
    """
    #the duplicate files to be ignored; all identical to 1600
    duplicate_files = [str(i) for i in range(1611, 1618)] # '1611',...'1617'

    all_files = glob.glob(processed_data_root + "/*.arff")

    watch_accel_df_list = []

    for filename in all_files:
        # Ignore duplicate files
        if any(dup_fn in filename for dup_fn in duplicate_files):
            continue
        # Process non-duplicate files
        else:
            df = pd.read_csv(filename, names = PROCESSED_FEATURE_NAMES, skiprows = 96, index_col=None, header=0)
            watch_accel_df_list.append(df)

    # Join the individual watch accelerometer dataframes
    watch_accel_df = pd.concat(watch_accel_df_list, axis=0, ignore_index=True, sort=False)

    # The columns below are suspected to contain erroneous data; therefore, they will be removed
    watch_accel_df.drop(['XSTANDDEV','YSTANDDEV','ZSTANDDEV','XVAR','YVAR','ZVAR'], axis=1, inplace=True)

    # Remove further unnecessary column
    watch_accel_df.drop('PARTICIPANT', axis=1, inplace=True)

    # Return the "pre-processed" data
    return watch_accel_df



if __name__ == '__main__':

    # Read all the raw data in
    master_raw_data_df = build_raw_data(data_root_path='/Users/nathanielburley/ActivityRecognitionPOC/data/')

    # Read the pre-processed watch accelerometer data in
    watch_accel_df = build_preprocessed_dataset(processed_data_root='data/wisdm-dataset/arff_files/watch/accel')

    # Save results to csv
    master_raw_data_df.to_csv('data/wisdm_master_dataset.csv')
    watch_accel_df.to_csv('data/wisdm_preprocessed_watch_accel.csv')

    ### DEBUGGING
    print(f"There are {len(master_raw_data_df)} rows in the raw dataset")
    print(f"There are {len(set(master_raw_data_df['participant_id']))} users")
    print(f"There are {len(set(master_raw_data_df['activity']))} activities")
    print(f"There are {len(set(master_raw_data_df['sensor_type']))} sensors")