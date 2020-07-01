import gc
import os
import sys

import pandas as pd

sys.path.append(os.path.abspath(os.path.join(os.getcwd(), os.pardir)))

from configuration.ConfigChecker import ConfigChecker
from configuration.Configuration import Configuration


def clean_up_dataframe(df: pd.DataFrame, config: Configuration):
    print('\nCleaning up dataframe with shape: ', df.shape, '...')
    # get list of attributes by type and remove those that aren't in the dataframe
    used = set(config.features_used)
    cats = list(set(config.categoricalValues).intersection(used))
    combined = list(set(config.categoricalValues + config.zeroOne + config.intNumbers).intersection(used))
    real_values = list(set(config.realValues).intersection(used))

    print('\tFill NA for categorical and integer columns with values')
    df[combined] = df[combined].fillna(method='ffill')

    print('\tChange categorical columns to numeric representation')
    for col in cats:
        df[col] = pd.Categorical(df[col])
        df[col] = df[col].cat.codes
        df[col] = pd.to_numeric(df[col])

    print('\tInterpolate NA values for real valued streams')
    df[real_values] = df[real_values].apply(pd.Series.interpolate, args=('linear',))

    print('\tDrop first/last rows that contain NA for any of the streams')
    df = df.dropna(axis=0)

    print('\tResampling (depending on freq: Downsampling) the data at a constant frequency'
          ' using nearest neighbor to forward fill NAN values')
    # print(df.head(10))
    df = df.resample(config.resample_frequency).pad()  # .nearest
    # print(df.head(10))
    print('\nShape after cleaning up the dataframe: ', df.shape)

    return df


def main():
    config = Configuration()  # Get config for data directory

    checker = ConfigChecker(config, None, 'preprocessing', training=None)
    checker.pre_init_checks()

    number_data_sets = len(config.datasets)
    for i in range(number_data_sets):
        print('\n\nImporting dataframe ' + str(i) + '/' + str(number_data_sets - 1) + ' from file')

        # read the imported dataframe from the saved file
        path_to_file = config.datasets[i][0] + config.filename_pkl
        df: pd.DataFrame = pd.read_pickle(path_to_file)

        df = clean_up_dataframe(df, config)

        print('\nSaving datafrane as pickle file in', config.datasets[i][0])
        path_to_file = config.datasets[i][0] + config.filename_pkl_cleaned
        df.to_pickle(path_to_file)
        print('Saving finished')

        del df
        gc.collect()


if __name__ == '__main__':
    main()
