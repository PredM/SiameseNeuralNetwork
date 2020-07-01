import os
import sys

import numpy as np
import pandas as pd

sys.path.append(os.path.abspath(os.path.join(os.getcwd(), os.pardir)))

from configuration.Configuration import Configuration


def main():
    config = Configuration()

    # import datasets
    y_test_features = np.load(config.training_data_folder + "test_features.npy")
    y_train_features = np.load(config.training_data_folder + "train_features.npy")
    y_train = np.load(config.training_data_folder + "train_labels.npy")
    y_test = np.load(config.training_data_folder + "test_labels.npy")
    y_train_window_times = np.load(config.training_data_folder + "train_window_times.npy")
    y_test_window_times = np.load(config.training_data_folder + "test_window_times.npy")
    y_train_failure_times = np.load(config.training_data_folder + "train_failure_times.npy")
    y_test_failure_times = np.load(config.training_data_folder + "test_failure_times.npy")
    feature_names = np.load(config.training_data_folder + "feature_names.npy")

    # print("y_train_window_times: ",y_train_window_times.shape, "| y_test_window_times: ",y_test_window_times.shape)
    # print("y_train_window_times[0:]", y_train_window_times[0,:])
    # print("y_train_window_times[1:]", y_train_window_times[1,:])

    # print("y_train_features shape: ", y_train_features.shape)
    # print("y_train shape: ", y_train.shape)
    # indexesToDelete = np.where(y_train_failure_times == "2019-06-08 13:31:51")
    # print("np.argwhere(x>1): ", np.where(y_train_failure_times == "2019-06-08 13:31:51"))
    # print("y_train y_train_failure_times: ", y_train_failure_times.shape)
    #
    # example_to_delete = 1
    # y_train_features = np.delete(y_train_features, indexesToDelete, 0)
    # print("y_train_features shape: ", y_train_features.shape)

    concat = np.stack((y_train, y_train_failure_times), axis=-1)
    concat = pd.DataFrame.from_records(concat)
    print("Failure cases from Train: ")
    print(concat.groupby([0])[1].nunique())
    concat = np.stack((y_test, y_test_failure_times), axis=-1)
    concat = pd.DataFrame.from_records(concat)
    print("Failure cases from Test: ")
    print(concat.groupby([0])[1].nunique())

    failure_labels_to_remove = ['txt16_failuremode_conveyorbelt_operation_without_a_workpiece',
                                'txt16_failuremode_oven_door_blocked',
                                'txt16_failuremode_turntable_broken_bigh_gear_tooth_2',
                                'txt18_WorkpieceDroppedCrosswiseIntoOven', 'txt19_hrs_position_switch_not_reached',
                                'txt19_failuremode_tansport_without_bucket_box', 'txt19_failure_mode_crashed_into_HRS',
                                'txt18_failuremode1_Workpiecelost_white_1', 'txt18_failuremode1_Workpiecelost_white_2',
                                'txt18_WorkpieceLostDuringTransport_diagnosis', 'txt16_i4_lightbarrier_failuremode2',
                                'txt18_WorkpieceLostDuringTransport_diagnosis',
                                'txt19_failuremode_bucket_box_fallen_down', 'txt16_i7_lightbarrier_failuremode1',
                                'txt17_pneumatic_leakage_failure_mode_1_1']

    failure_label_to_remove_2 = ['txt15_i3_lightbarrier_failure_mode_1',
                                 'txt16_pneumatic_leakage_failure_mode_1',
                                 'txt18_pneumatic_leakage_failure_mode_3_faulty']

    failure_to_extract_from_train_to_test = ["2019-05-28 08:49:58", "2019-05-28 08:27:24", "2019-05-28 08:19:18",
                                             "2019-05-23 10:17:54", "2019-05-28 19:35:08",
                                             "2019-06-08 11:46:28", "2019-06-08 11:57:40", "2019-05-23 13:39:55",
                                             "2019-05-28 19:12:09", "2019-06-07 20:20:52",
                                             "2019-05-28 20:26:03", "2019-06-07 20:13:24", "2019-06-08 13:30:11",
                                             "2019-05-28 15:31:22"]
    features_to_remove = ['a_15_c_x', 'a_15_c_y', 'a_15_c_z', 'a_18_1_x', 'a_18_1_y', 'a_18_1_z', 'hrs_gyr_x',
                          'hrs_gyr_y',
                          'hrs_gyr_z', 'hrs_mag_x', 'hrs_mag_y', 'hrs_mag_z', 'vsg_acc_x', 'vsg_acc_y', 'vsg_acc_z',
                          'vsg_gyr_x', 'vsg_gyr_y', 'vsg_gyr_z', 'vsg_mag_x', 'vsg_mag_y', 'vsg_mag_z']

    # removeExamplesBasedOnFailureTimes(features=y_train_features, labels= y_train, window_times= y_train_failure_times, failure_times = y_train_failure_times, failuresToRemove = ["2019-06-08 13:25:05", "2019-06-08 13:31:51"])

    # print(" Before y_train_features shape: ", y_train_features.shape, "y_train shape: ", y_train.shape)
    y_train_features, feature_names_ = remove_attributes(features=y_train_features, feature_list_names=feature_names,
                                                         features_to_remove_list=features_to_remove)
    y_test_features, feature_names = remove_attributes(features=y_test_features, feature_list_names=feature_names,
                                                       features_to_remove_list=features_to_remove)
    # print("After: y_train_features shape: ", y_train_features.shape, "y_train shape: ", y_train.shape)

    # Remove specific types of failure from the test and training data ###
    print(" Before y_train_features shape: ", y_train_features.shape, "y_train_window_times shape: ",
          y_train_window_times.shape)
    y_train_features, y_train, y_train_window_times, y_train_failure_times = remove_examples_based_on_failure_labels(
        features=y_train_features, labels=y_train, window_times=y_train_window_times,
        failure_times=y_train_failure_times, failureLabelsToRemove=failure_labels_to_remove)
    print(" After: y_train_features shape: ", y_train_features.shape, "y_train_window_times shape: ",
          y_train_window_times.shape)

    print(" Before y_test_features shape: ", y_test_features.shape, "y_test shape: ", y_test.shape)
    y_test_features, y_test, y_test_window_times, y_test_failure_times = remove_examples_based_on_failure_labels(
        features=y_test_features, labels=y_test, window_times=y_test_window_times,
        failure_times=y_test_failure_times, failureLabelsToRemove=failure_labels_to_remove)
    print(" After y_test_features shape: ", y_test_features.shape, "y_test shape: ", y_test.shape)

    ### Extract failure cases from the training data and move it to the test data ###
    features_ext, labels_ext, window_times_ext, failure_times_ext = extract_examples_based_on_failure_times(
        features=y_train_features, labels=y_train, window_times=y_train_window_times,
        failure_times=y_train_failure_times, failuresToExtract=failure_to_extract_from_train_to_test)

    ### Append previous extracted failure cases from the training data and move it to the test data ###
    y_test_features, y_test, y_test_window_times, y_test_failure_times = append_extracted_examples(
        features=y_test_features, labels=y_test, window_times=y_test_window_times,
        failure_times=y_test_failure_times, features_ext=features_ext, labels_ext=labels_ext,
        window_times_ext=window_times_ext, failure_times_ext=failure_times_ext)

    ### Remove extracted failure cases from the training data ###
    y_train_features, y_train, y_train_window_times, y_train_failure_times = remove_examples_based_on_failure_times(
        features=y_train_features, labels=y_train, window_times=y_train_window_times,
        failure_times=y_train_failure_times, failuresToRemove=failure_to_extract_from_train_to_test)

    #

    ### Rename Labels Train ###
    y_train = np.char.replace(y_train, "txt15_failuremode_driveshaft_slippage_2",
                              "txt15_conveyor_failuremode_driveshaft_slippage_failure")
    y_train = np.char.replace(y_train, "txt15_i1_lightbarrier_failuremode3_1", "txt15_i1_lightbarrier_failuremode1")
    y_train = np.char.replace(y_train, "txt15_i1_lightbarrier_failuremode3_2", "txt15_i1_lightbarrier_failuremode2")
    y_train = np.char.replace(y_train, "txt16_failuremode_conveyorbelt_broken_big_gear_tooth_diagnosis",
                              "txt16_conveyorbelt_big_gear_tooth_broken_failure")
    y_train = np.char.replace(y_train, "txt16_failuremode_conveyorbelt_broken_big_gear_tooth_diagnosis_2",
                              "txt16_conveyorbelt_big_gear_tooth_broken_failure")
    y_train = np.char.replace(y_train, "txt16_failuremode_conveyorbelt_broken_small_gear_tooth_diagnosis",
                              "txt16_conveyorbelt_small_gear_tooth_broken_failure")
    y_train = np.char.replace(y_train, "txt16_failuremode_driveshaft_slippage",
                              "txt16_conveyor_failuremode_driveshaft_slippage_failure")
    y_train = np.char.replace(y_train, "txt16_failuremode_turntable_broken_bigh_gear_tooth_2",
                              "txt16_turntable_big_gear_tooth_broken_failure")
    y_train = np.char.replace(y_train, "txt17_i1_switch_failuremode3_1", "txt17_i1_switch_failuremode1")
    y_train = np.char.replace(y_train, "txt17_i1_switch_failuremode3_2", "txt17_i1_switch_failuremode2")
    y_train = np.char.replace(y_train, "txt16_conveyorbelt_big_gear_tooth_broken_failure_2",
                              "txt16_conveyorbelt_big_gear_tooth_broken_failure")
    y_train = np.char.replace(y_train, "txt18_pneumatic_leakage_failure_mode_1_2",
                              "txt18_pneumatic_leakage_failure_mode_1")
    y_train = np.char.replace(y_train, "txt18_pneumatic_leakage_failure_mode_2_2",
                              "txt18_pneumatic_leakage_failure_mode_2")
    y_train = np.char.replace(y_train, "txt18_pneumatic_leakage_failure_mode_2_1",
                              "txt18_pneumatic_leakage_failure_mode_2_faulty")
    y_train = np.char.replace(y_train, "txt18_pneumatic_leakage_failure_mode_3_1",
                              "txt18_pneumatic_leakage_failure_mode_3_faulty")
    y_train = np.char.replace(y_train, "txt17_pneumatic_leakage_failure_mode_1_1",
                              "txt17_pneumatic_leakage_failure_mode_1_faulty")
    y_train = np.char.replace(y_train, "txt17_pneumatic_leakage_failure_mode_1_2",
                              "txt17_pneumatic_leakage_failure_mode_1")
    y_train = np.char.replace(y_train, "txt17_failuremode_workingstation_transport_without_Workpiece",
                              "txt17_workingstation_transport_failuremode_wout_workpiece")
    y_train = np.char.replace(y_train, "txt18_TransportWithoutWorkpiece_diagnosis",
                              "txt18_transport_failuremode_wout_workpiece")
    # General
    y_train = np.char.replace(y_train, "failuremode", "failure_mode")
    y_train = np.char.replace(y_train, "mode1", "mode_1")
    y_train = np.char.replace(y_train, "mode2", "mode_2")
    y_train = np.char.replace(y_train, "mode3", "mode_3")

    ### Rename Labels Test ###
    y_test = np.char.replace(y_test, "txt15_failuremode_driveshaft_slippage_2",
                             "txt15_conveyor_failuremode_driveshaft_slippage_failure")
    y_test = np.char.replace(y_test, "txt15_i1_lightbarrier_failuremode3_1", "txt15_i1_lightbarrier_failuremode1")
    y_test = np.char.replace(y_test, "txt15_i1_lightbarrier_failuremode3_2", "txt15_i1_lightbarrier_failuremode2")
    y_test = np.char.replace(y_test, "txt16_failuremode_conveyorbelt_broken_big_gear_tooth_diagnosis",
                             "txt16_conveyorbelt_big_gear_tooth_broken_failure")
    y_test = np.char.replace(y_test, "txt16_failuremode_conveyorbelt_broken_big_gear_tooth_diagnosis_2",
                             "txt16_conveyorbelt_big_gear_tooth_broken_failure")
    y_test = np.char.replace(y_test, "txt16_failuremode_conveyorbelt_broken_small_gear_tooth_diagnosis",
                             "txt16_conveyorbelt_small_gear_tooth_broken_failure")
    y_test = np.char.replace(y_test, "txt16_failuremode_driveshaft_slippage",
                             "txt16_conveyor_failuremode_driveshaft_slippage_failure")
    y_test = np.char.replace(y_test, "txt16_failuremode_turntable_broken_bigh_gear_tooth_2",
                             "txt16_turntable_big_gear_tooth_broken_failure")
    y_test = np.char.replace(y_test, "txt17_i1_switch_failuremode3_1", "txt17_i1_switch_failuremode1")
    y_test = np.char.replace(y_test, "txt17_i1_switch_failuremode3_2", "txt17_i1_switch_failuremode2")
    y_test = np.char.replace(y_test, "txt16_conveyorbelt_big_gear_tooth_broken_failure_2",
                             "txt16_conveyorbelt_big_gear_tooth_broken_failure")
    y_test = np.char.replace(y_test, "txt18_pneumatic_leakage_failure_mode_1_2",
                             "txt18_pneumatic_leakage_failure_mode_1")
    y_test = np.char.replace(y_test, "txt18_pneumatic_leakage_failure_mode_2_2",
                             "txt18_pneumatic_leakage_failure_mode_2")
    y_test = np.char.replace(y_test, "txt18_pneumatic_leakage_failure_mode_2_1",
                             "txt18_pneumatic_leakage_failure_mode_2_faulty")
    y_test = np.char.replace(y_test, "txt18_pneumatic_leakage_failure_mode_3_1",
                             "txt18_pneumatic_leakage_failure_mode_3_faulty")
    y_test = np.char.replace(y_test, "txt17_pneumatic_leakage_failure_mode_1_1",
                             "txt17_pneumatic_leakage_failure_mode_1_faulty")
    y_test = np.char.replace(y_test, "txt17_pneumatic_leakage_failure_mode_1_2",
                             "txt17_pneumatic_leakage_failure_mode_1")
    y_test = np.char.replace(y_test, "txt17_failuremode_workingstation_transport_without_Workpiece",
                             "txt17_workingstation_transport_failuremode_wout_workpiece")
    y_test = np.char.replace(y_test, "txt18_TransportWithoutWorkpiece_diagnosis",
                             "txt18_transport_failuremode_wout_workpiece")
    y_test = np.char.replace(y_test, "txt18_TransportWithoutWorkpiece_diagnosis",
                             "txt18_transport_failuremode_wout_workpiece")
    # General
    y_test = np.char.replace(y_test, "failuremode", "failure_mode")
    y_test = np.char.replace(y_test, "mode1", "mode_1")
    y_test = np.char.replace(y_test, "mode2", "mode_2")
    y_test = np.char.replace(y_test, "mode3", "mode_3")

    # Remove specific types of failure from the test and training data ###
    print(" Before y_train_features shape: ", y_train_features.shape, "y_train_window_times shape: ",
          y_train_window_times.shape)
    y_train_features, y_train, y_train_window_times, y_train_failure_times = remove_examples_based_on_failure_labels(
        features=y_train_features, labels=y_train, window_times=y_train_window_times,
        failure_times=y_train_failure_times, failureLabelsToRemove=failure_label_to_remove_2)
    print(" After: y_train_features shape: ", y_train_features.shape, "y_train_window_times shape: ",
          y_train_window_times.shape)

    print(" Before y_test_features shape: ", y_test_features.shape, "y_test shape: ", y_test.shape)
    y_test_features, y_test, y_test_window_times, y_test_failure_times = remove_examples_based_on_failure_labels(
        features=y_test_features, labels=y_test, window_times=y_test_window_times,
        failure_times=y_test_failure_times, failureLabelsToRemove=failure_label_to_remove_2)
    print(" After y_test_features shape: ", y_test_features.shape, "y_test shape: ", y_test.shape)

    # get unqiue classes and the number of examples in each
    y_train_single, y_train_counts = np.unique(y_train, return_counts=True)
    y_test_single, y_test_counts = np.unique(y_test, return_counts=True)

    # create a dataframe
    x = np.stack((y_train_single, y_train_counts)).transpose()
    x = pd.DataFrame.from_records(x)

    y = np.stack((y_test_single, y_test_counts)).transpose()
    y = pd.DataFrame.from_records(y)

    x = x.merge(y, how='outer', on=0)
    # x = x.merge(z, how='outer', on=0)
    x = x.rename(index=str, columns={0: 'Class', '1_x': 'Train', '1_y': 'Test'})

    # convert column types in order to be able to sum the values
    x['Train'] = pd.to_numeric(x['Train']).fillna(value=0).astype(int)
    x['Test'] = pd.to_numeric(x['Test']).fillna(value=0).astype(int)
    x['Total'] = x[['Test', 'Train']].sum(axis=1)
    x = x.set_index('Class')

    # print the information to console
    print('----------------------------------------------')
    print('Train and test data sets:')
    print('----------------------------------------------')
    print(x)
    print('\nTotal sum examples:', x['Total'].sum(axis=0))
    print('----------------------------------------------')

    print('\n\n')

    print("y_train_window_times[1:]", y_train_window_times[1, :])

    # save the modified data
    print('\nSave  to np arrays in ' + config.training_data_folder)
    print('Step 1/9')
    np.save(config.training_data_folder + 'train_features.npy', y_train_features)
    print('Step 2/9')
    np.save(config.training_data_folder + 'test_features.npy', y_test_features)
    print('Step 3/9')
    np.save(config.training_data_folder + 'train_labels.npy', y_train)
    print('Step 4/9')
    np.save(config.training_data_folder + 'test_labels.npy', y_test)
    print('Step 5/9')
    np.save(config.training_data_folder + 'feature_names.npy', feature_names)
    print('Step 6/9')
    np.save(config.training_data_folder + 'train_failure_times.npy',
            y_train_failure_times)  # Contains the associated time of a failure (if not no failure) for each example
    print('Step 7/9')
    np.save(config.training_data_folder + 'test_failure_times.npy',
            y_test_failure_times)
    print('Step 8/9')
    np.save(config.training_data_folder + 'train_window_times.npy',
            y_train_window_times)  # Contains the start and end time stamp for each training example
    print('Step 9/9')
    np.save(config.training_data_folder + 'test_window_times.npy',
            y_test_window_times)

    '''
    # repeat the process for the case base
    y_train = np.load(config.case_base_folder + "train_labels.npy")  # labels of the case base
    y_train_single, y_train_counts = np.unique(y_train, return_counts=True)

    x = np.stack((y_train_single, y_train_counts)).transpose()
    x = pd.DataFrame.from_records(x)
    x = x.rename(index=str, columns={0: 'Class', 1: 'Number of cases'})
    x['Number of cases'] = pd.to_numeric(x['Number of cases']).fillna(value=0).astype(int)
    x = x.set_index('Class')

    print('----------------------------------------------')
    print('Case base:')
    print('----------------------------------------------')
    print(x)
    print('\nTotal sum examples:', x['Number of cases'].sum(axis=0))
    print('----------------------------------------------\n')
    '''


def remove_examples_based_on_failure_times(features, labels, window_times, failure_times, failuresToRemove):
    # delete examples based on the failure time from training or test data

    # features: the matrix (examples,timestamp,channels) of the train_features.npy or test_features.npy
    # labels: 1-d array of labels associated to each example
    # window times: 1-d array associated to each example
    # failure_times: 1-d array associated failure to each failure example

    # print("features shape before deleting: ", features.shape)
    for i, failureTime in enumerate(failuresToRemove):
        # Get indexes to delete:
        if i == 0:
            indexesToDelete = np.where(failure_times == failureTime)
        else:
            indexesToDelete = np.append(indexesToDelete, np.where(failure_times == failureTime))

    # print("Indexes to delete: ", indexesToDelete, "deleted: ",len(indexesToDelete))
    features_del = np.delete(features, indexesToDelete, 0)
    labels_del = np.delete(labels, indexesToDelete, 0)
    window_times_del = np.delete(window_times, indexesToDelete, 0)
    failure_times_del = np.delete(failure_times, indexesToDelete, 0)
    # print("features shape after deleting: ", features_del.shape)
    return features_del, labels_del, window_times_del, failure_times_del


def remove_examples_based_on_failure_labels(features, labels, window_times, failure_times, failureLabelsToRemove):
    # delete examples based on the failure label (e.g., txt17_i1_switch_failuremode3_1) from training or test data

    # features: the matrix (examples,timestamp,channels) of the train_features.npy or test_features.npy
    # labels: 1-d array of labels associated to each example
    # window times: 1-d array associated to each example
    # failure_times: 1-d array associated failure to each failure example

    # print("features shape before deleting: ", features.shape)
    for i, failureLabel in enumerate(failureLabelsToRemove):
        # Get indexes to delete:
        if i == 0:
            indexesToDelete = np.where(labels == failureLabel)
        else:
            indexesToDelete = np.append(indexesToDelete, np.where(labels == failureLabel))

    # print("Indexes to delete: ", indexesToDelete, "deleted: ", len(indexesToDelete))
    features_del = np.delete(features, indexesToDelete, 0)
    labels_del = np.delete(labels, indexesToDelete, 0)
    window_times_del = np.delete(window_times, indexesToDelete, 0)
    failure_times_del = np.delete(failure_times, indexesToDelete, 0)
    # print("features shape after deleting: ", features_del.shape)
    return features_del, labels_del, window_times_del, failure_times_del


def remove_attributes(features, feature_list_names, features_to_remove_list):
    # delete examples based on the failure label (e.g., txt17_i1_switch_failuremode3_1) from training or test data

    # features: the matrix (examples,timestamp,channels) of the train_features.npy or test_features.npy
    # labels: 1-d array of labels associated to each example
    # window times: 1-d array associated to each example
    # failure_times: 1-d array associated failure to each failure example

    # print("features shape before deleting: ", features.shape)
    for i, featureToRemove in enumerate(features_to_remove_list):
        # Get indexes to delete:
        if i == 0:
            indexesToDelete = np.where(featureToRemove == feature_list_names)
        else:
            indexesToDelete = np.append(indexesToDelete, np.where(featureToRemove == feature_list_names))

    # print("Indexes to delete: ", indexesToDelete, "deleted: ", len(indexesToDelete))
    features_del = np.delete(features, indexesToDelete, 2)
    feature_list_names_del = np.delete(feature_list_names, indexesToDelete, 0)
    return features_del, feature_list_names_del


def extract_examples_based_on_failure_times(features, labels, window_times, failure_times, failuresToExtract):
    # extract examples based on the failure time from training or test data

    # features: the matrix (examples,timestamp,channels) of the train_features.npy or test_features.npy
    # labels: 1-d array of labels associated to each example
    # window times: 1-d array associated to each example
    # failure_times: 1-d array associated failure to each failure example
    # failuresToExtract: list of time stamps of failures that should be extracted
    # print("features shape before extraction: ", features.shape)
    for i, failureTime in enumerate(failuresToExtract):
        # Get indexes to delete:
        if i == 0:
            indexesToExtract = np.where(failure_times == failureTime)
        else:
            indexesToExtract = np.append(indexesToExtract, np.where(failure_times == failureTime))
        if ((np.where(failure_times == failureTime))[0].size == 0):
            print("For ", failureTime, " no failure could be extracted from train data. Probably already in test!")
        # print("extract: ", np.where(failure_times == failureTime))
    # print("Indexes to extract: ", indexesToExtract, "extracted: ", len(indexesToExtract))
    features_ext = features[indexesToExtract, :, :]  # np.delete(features, indexesToExtract, 0)
    labels_ext = labels[indexesToExtract]  # np.delete(labels, indexesToExtract, 0)
    # print("window_times: ", window_times.shape)
    window_times_ext = window_times[indexesToExtract, :]  # np.delete(window_times, indexesToExtract, 0)
    failure_times_ext = failure_times[indexesToExtract]  # np.delete(failure_times, indexesToExtract, 0)
    # print("features shape after extraction: ", features.shape, " and size of extracted features: ", features_ext.shape)
    return features_ext, labels_ext, window_times_ext, failure_times_ext


def append_extracted_examples(features, labels, window_times, failure_times, features_ext, labels_ext, window_times_ext,
                              failure_times_ext):
    # print("features shape before appending extraction: ", features.shape)
    features = np.concatenate((features, features_ext))
    labels = np.concatenate((labels, labels_ext))
    print("window_times: ", window_times.shape, "window_times_ext: ", window_times_ext.shape)
    window_times = np.concatenate((window_times, window_times_ext))
    failure_times = np.concatenate((failure_times, failure_times_ext))
    # print("features shape after extraction: ", features.shape, " and size of extracted features: ", features_ext.shape)
    return features, labels, window_times, failure_times


if __name__ == '__main__':
    main()
