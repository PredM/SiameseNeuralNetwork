import os
import sys
from shutil import copyfile

import numpy as np
from numpy.random.mtrand import RandomState

sys.path.append(os.path.abspath(os.path.join(os.getcwd(), os.pardir)))

from configuration.Configuration import Configuration


def main():
    config = Configuration()

    y_train = np.load(config.training_data_folder + 'train_labels.npy')  # labels of the training data
    x_train = np.load(config.training_data_folder + 'train_features.npy')  # labels of the training data
    feature_names = np.load(config.training_data_folder + 'feature_names.npy')
    failure_times_train = np.load(config.training_data_folder + 'train_failure_times.npy')
    window_times_train = np.load(config.training_data_folder + 'train_window_times.npy')

    # get unique classes
    classes = np.unique(y_train)

    print('Number of examples in training data set:', x_train.shape[0])
    print('Reducing to', config.examples_per_class, 'examples per class with', len(classes), 'classes')

    # for each class get the indices of all examples with this class
    indices_of_classes = []
    for c in classes:
        indices_of_classes.append(np.where(y_train == c)[0])

    # reduce classes to equal many examples
    new_indices = []
    ran = RandomState(config.random_seed_index_selection)
    for i in range(len(indices_of_classes)):
        length = len(indices_of_classes[i])

        # if there are less examples than there should be for each class only those can be used
        epc = config.examples_per_class if config.examples_per_class < length else length

        temp = ran.choice(indices_of_classes[i], epc, replace=False)
        # print(len(indices_of_classes[i]), len(temp))

        new_indices.append(temp)

    casebase_features_list = []
    casebase_labels_list = []
    casebase_failures_list = []
    casebase_window_times_list = []

    # extract the values at the selected indices and add to list
    for i in range(len(classes)):
        casebase_labels_list.extend(y_train[new_indices[i]])
        casebase_features_list.extend(x_train[new_indices[i]])
        casebase_failures_list.extend(failure_times_train[new_indices[i]])
        casebase_window_times_list.extend(window_times_train[new_indices[i]])

    # transform list of values back into an array and safe to file
    casebase_labels = np.stack(casebase_labels_list, axis=0)
    casebase_features = np.stack(casebase_features_list, axis=0)
    casebase_failures = np.stack(casebase_failures_list, axis=0)
    casebase_window_times = np.stack(casebase_window_times_list, axis=0)

    print('Number of exaples in training data set:', casebase_features.shape[0])

    np.save(config.case_base_folder + 'train_features.npy', casebase_features.astype('float32'))
    np.save(config.case_base_folder + 'train_labels.npy', casebase_labels)
    np.save(config.case_base_folder + 'train_failure_times.npy', casebase_failures)
    np.save(config.case_base_folder + 'train_window_times.npy', casebase_window_times)

    files_to_copy = ['feature_names.npy', 'test_labels.npy', 'test_features.npy', 'test_window_times.npy',
                     'test_failure_times.npy', 'FailureMode_Sim_Matrix.csv', 'Lokalization_Sim_Matrix.csv',
                     'Condition_Sim_Matrix.csv']

    for file in files_to_copy:
        copyfile(config.training_data_folder + file, config.case_base_folder + file)


# this script is used to reduce the training data to a specific amount of examples per class
# to use during live classification because using all examples is not efficient enough
if __name__ == '__main__':
    main()
