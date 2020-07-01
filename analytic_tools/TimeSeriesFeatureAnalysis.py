import datetime
import os
import sys

import numpy as np
import pandas as pd
from sklearn.feature_selection import mutual_info_classif

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
    x_train = np.load(config.case_base_folder + '2ms_3sec/train_features.npy')  # data training
    y_train_strings = np.expand_dims(np.load(config.case_base_folder + '2ms_3sec/train_labels.npy'), axis=-1)
    feature_names = np.load(config.training_data_folder + '2ms_3sec/feature_names.npy')
    columns = np.concatenate((['id', 'time'], feature_names))
    print(columns.shape)

    print(x_train.shape)
    examples = x_train.shape[0]
    time_series_length = x_train.shape[1]
    attributes = x_train.shape[2]

    # tsfresh_input_x_test = np.zeros([examples * time_series_length, attributes+2])
    tsfresh_input_x_test = np.zeros([1, 63])
    # add 2 columns for id and timestamp
    '''
    for example in range(examples):
        print("example: ", example)
        id_vec = np.ones(x_train.shape[1]) * example
        time_vec = np.arange(x_train.shape[1])

        # stack id and time and example matrix together
        id_time_matrix = np.dstack((id_vec, time_vec)).squeeze() #(1500,2)
        #print("id_time_matrix: ", id_time_matrix.shape)
        #print("x_test[example]: ", x_train[example,:,:].shape)
        curr_ex = np.concatenate((id_time_matrix, x_train[example,:,:]), axis=1) #(1500, 63)
        print(example, " shape: ", curr_ex.shape)
        if example == 0:
            tsfresh_input_x_test = curr_ex
        else:
            tsfresh_input_x_test = np.concatenate((tsfresh_input_x_test, curr_ex), axis=0)
        #print("dummy: ", tsfresh_input_x_test.shape)

        #Append to overall array
        #tsfresh_input_x_test[2:,:]
        #tsfresh_input_x_test[]

    # get unique classes
    
    df_timeSeries_container = pd.DataFrame(data=tsfresh_input_x_test, columns=columns)
    df_labels = pd.DataFrame(data=y_train_strings)
    print("TS Fresh Feature Extraction started at: ", datetime.datetime.now())
    extracted_features = extract_features(df_timeSeries_container, column_id="id", column_sort="time")
    extracted_features.to_pickle(config.case_base_folder +'extractedFeatures_X_caseBase_unfiltered.pkl')

    #extracted_features.to_csv('extractedFeatures_X_caseBase_unfiltered.csv', sep=',', encoding='WINDOWS-1252')
    print('extracted features size unfiltered: ', extracted_features.shape)

    from tsfresh.utilities.dataframe_functions import impute
    # Remove NANs
    impute(extracted_features)
    print('extracted features size after impute: ', extracted_features.shape)

    from tsfresh import select_features
    X_filtered = select_features(extracted_features, y_train_strings)
    print('filtered features size: ', X_filtered.shape)
    print('filtered features: ', X_filtered)
    X_filtered.to_pickle(config.case_base_folder +'extractedFeatures_X_filtered.pkl')

    y_train_strings = np.squeeze(y_train_strings)
    print("y_train_strings: ", y_train_strings.shape)
    X = pd.read_pickle(config.case_base_folder +'extractedFeatures_X_caseBase_unfiltered.pkl')

    print("X shape: ", X.shape)

    from tsfresh.utilities.dataframe_functions import impute
    print(X.head())
    # Remove NANs
    #X = impute(X)
    print('extracted features size after impute: ', X.shape)
    #print(np.unique(y_train_strings))

    from tsfresh import select_features
    X_filtered = select_features(X, y_train_strings)
    print('filtered features size: ', X_filtered.shape)
    print('filtered features: ', X_filtered)
    X_filtered.to_pickle(config.case_base_folder +'extractedFeatures_X_filtered.pkl')
    print("TS Fresh Feature Extraction finished at: ", datetime.datetime.now())
    '''

    X = pd.read_pickle(config.case_base_folder + '2ms_3sec/extractedFeatures_X_caseBase_unfiltered.pkl')

    df_information_gain_of_feature = pd.DataFrame(data=feature_names)
    # print(df_information_gain_of_feature)
    df_information_gain_of_feature.columns = ['Attribut']
    df_information_gain_of_feature["InfoGainSum"] = 0
    df_information_gain_of_feature = df_information_gain_of_feature.set_index('Attribut')
    # df_information_gain_of_feature.loc['a_16_3_x', "InfoGainSum"] = 2
    # df_information_gain_of_feature.loc['a_16_3_x', "InfoGainSum"] += 2
    # df_information_gain_of_feature.loc['a_16_3_x', "InfoGainSum"] += 1
    # print("df_information_gain_of_feature.loc[a_16_3_x, InfoGainSum]_",
    # df_information_gain_of_feature.loc['a_16_3_x', "InfoGainSum"])
    # print(dsd)

    from tsfresh.utilities.dataframe_functions import impute
    # Remove NANs
    X = impute(X)

    # Select Labels to analyze
    # labels_to_analyze = ['txt15_m1_t1_high_wear','txt15_m1_t1_low_wear','txt15_m1_t2_wear',
    #                           "no_failure", "txt16_m3_t1_high_wear", "txt16_m3_t1_low_wear"]

    labels_to_analyze = ["no_failure", 'txt16_i3_switch_failure_mode_2']

    for i, failureLabel in enumerate(labels_to_analyze):
        # Get indexes to delete:
        if i == 0:
            indexes_to_use = np.where(y_train_strings == failureLabel)
        else:
            indexes_to_use = np.append(indexes_to_use, np.where(y_train_strings == failureLabel))

    print("Indexes to use: ", indexes_to_use, "used: ", len(indexes_to_use))
    X_npy = X.values[indexes_to_use, :]  # np.delete(features, indexesToExtract, 0)
    y_train_strings = y_train_strings[indexes_to_use]  # np.delete(labels, indexesToExtract, 0)

    print("X shape: ", X_npy.shape, " Labels shape: ", y_train_strings.shape)
    headers = X.dtypes.index
    headers = headers.values
    X = pd.DataFrame(X_npy, columns=headers)
    X['Label'] = y_train_strings
    # print(X[['2__kurtosis', 'Label']].to_string())
    # print(X[['3__kurtosis','Label']].to_string())
    # print(X[['1__kurtosis','Label']].to_string())
    # test = X[['2__kurtosis','4__kurtosis', '6__kurtosis','Label']]
    # print(test.groupby('Label').median())
    feature_scores = mutual_info_classif(X_npy, y_train_strings)
    # for score, fname in sorted(zip(feature_scores, dv.get_feature_names()), reverse=True)[:10]:
    print('Feature Scores: ', feature_scores)

    result_list = zip(feature_scores, headers)
    file = open(config.case_base_folder + 'score_feature_file_2.txt', 'w')
    for score, featureName in sorted(zip(feature_scores, headers), key=lambda x: x[0], reverse=True):
        print(score, featureName)
        file.write(str(score) + ' - ' + featureName + '\n')
        df_information_gain_of_feature.loc[featureName.split('__')[0], "InfoGainSum"] += score
        # print('result_list unsorted: ', set(result_list))
    # sorted(result_list)
    # print('sorted: ', sorted(result_list, key=lambda x: x[0],reverse=True))
    # print(df_information_gain_of_feature.to_string())
    print(df_information_gain_of_feature.sort_values(by=['InfoGainSum']).to_string())
    file.close()

    # print('Number of exaples in training data set:', x_train.shape[0])
    # print('Reducing to', config.examples_per_class, 'examples per class with', len(classes), 'classes')

    # transform list of values back into an array and safe to file
    # casebase_labels = np.stack(casebase_labels_list, axis=0)
    # casebase_features = np.stack(casebase_features_list, axis=0)
    # casebase_failures =  np.stack(casebase_failures_list, axis=0)
    # casebase_windowtimes =  np.stack(casebase_windowtimes_list, axis=0)

    # print('Number of exaples in training data set:', casebase_features.shape[0])
    '''
    np.save(config.case_base_folder + 'train_features.npy', casebase_features.astype('float32'))
    np.save(config.case_base_folder + 'train_labels.npy', casebase_labels)
    np.save(config.case_base_folder + 'train_failure_times.npy', casebase_failures)
    np.save(config.case_base_folder + 'train_window_times.npy', casebase_windowtimes)

    # in order for the dataset object to be created, there must also be files for test data in the folder,
    # even if these are not used for live processing.
    y_test = np.load(config.training_data_folder + 'test_labels.npy')  # labels of the training data
    x_test = np.load(config.training_data_folder + 'test_features.npy')  # labels of the training data
    np.save(config.case_base_folder + 'test_features.npy', x_test.astype('float32'))
    np.save(config.case_base_folder + 'test_labels.npy', y_test)

    # also copy the file that stores the names of the features
    np.save(config.case_base_folder + 'feature_names.npy', feature_names)
    '''


if __name__ == '__main__':
    main()
