import datetime
import os
import sys

import numpy as np
import pandas as pd
from tsfresh import extract_features

sys.path.append(os.path.abspath(os.path.join(os.getcwd(), os.pardir)))

from configuration.Configuration import Configuration


def main():
    config = Configuration()
    print("TS Fresh Feature Extraction Script started at: ", datetime.datetime.now())
    # y_train = np.load(config.training_data_folder + 'train_labels.npy')  # labels of the training data
    # x_train = np.load(config.training_data_folder + 'train_features.npy')  # labels of the training data
    # feature_names = np.load(config.training_data_folder + 'feature_names.npy')
    # failureTimes_train = np.load(config.training_data_folder + 'train_failure_times.npy')
    # windowTimes_train = np.load(config.training_data_folder + 'train_window_times.npy')
    # y_test = np.load(config.training_data_folder + 'test_labels.npy')  # labels of the training data
    # x_test = np.load(config.training_data_folder + 'test_features.npy')  # labels of the training data
    x_train = np.load(config.case_base_folder + 'train_features.npy')  # data training
    y_train_strings = np.expand_dims(np.load(config.case_base_folder + 'train_labels.npy'), axis=-1)
    feature_names = np.load(config.training_data_folder + 'feature_names.npy')
    columns = np.concatenate((['id', 'time'], feature_names))
    print(columns.shape)

    print(x_train.shape)
    examples = x_train.shape[0]
    time_series_length = x_train.shape[1]
    attributes = x_train.shape[2]

    # tsfresh_input_x_test = np.zeros([examples * time_series_length, attributes+2])
    tsfresh_input_x_test = np.zeros([1, 63])
    # add 2 columns for id and timestamp

    for example in range(examples):
        print("example: ", example)
        id_vec = np.ones(x_train.shape[1]) * example
        time_vec = np.arange(x_train.shape[1])

        # stack id and time and example matrix together
        id_time_matrix = np.dstack((id_vec, time_vec)).squeeze()  # (1500,2)
        # print("id_time_matrix: ", id_time_matrix.shape)
        # print("x_test[example]: ", x_train[example,:,:].shape)
        curr_ex = np.concatenate((id_time_matrix, x_train[example, :, :]), axis=1)  # (1500, 63)
        print(example, " shape: ", curr_ex.shape)
        if example == 0:
            tsfresh_input_x_test = curr_ex
        else:
            tsfresh_input_x_test = np.concatenate((tsfresh_input_x_test, curr_ex), axis=0)
        # print("dummy: ", tsfresh_input_x_test.shape)

        # Append to overall array
        # tsfresh_input_x_test[2:,:]
        # tsfresh_input_x_test[]

    # get unique classes

    df_timeSeries_container = pd.DataFrame(data=tsfresh_input_x_test, columns=columns)
    df_labels = pd.DataFrame(data=y_train_strings)
    print("TS Fresh Feature Extraction started at: ", datetime.datetime.now())
    extracted_features = extract_features(df_timeSeries_container, column_id="id", column_sort="time")
    extracted_features.to_pickle(config.case_base_folder + 'extractedFeatures_X_caseBase_unfiltered_4ms4sec.pkl')

    # extracted_features.to_csv('extractedFeatures_X_caseBase_unfiltered.csv', sep=',', encoding='WINDOWS-1252')
    print('extracted features size unfiltered: ', extracted_features.shape)

    from tsfresh.utilities.dataframe_functions import impute
    # Remove NANs
    extracted_features = impute(extracted_features)
    print('extracted features size after impute: ', extracted_features.shape)

    from tsfresh import select_features
    X_filtered = select_features(extracted_features, y_train_strings)
    print('filtered features size: ', X_filtered.shape)
    print('filtered features: ', X_filtered)
    X_filtered.to_pickle(config.case_base_folder + 'extractedFeatures_X_filtered_4ms4sec.pkl')

    y_train_strings = np.squeeze(y_train_strings)
    print("y_train_strings: ", y_train_strings.shape)
    X = pd.read_pickle(config.case_base_folder + 'extractedFeatures_X_caseBase_unfiltered_4ms4sec.pkl')

    print("X shape: ", X.shape)

    print(X.head())
    # Remove NANs
    # X = impute(X)
    print('extracted features size after impute: ', X.shape)
    # print(np.unique(y_train_strings))


if __name__ == '__main__':
    main()
