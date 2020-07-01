from time import perf_counter

import numpy as np
import pandas as pd
from sklearn import preprocessing

from configuration.Configuration import Configuration


class Dataset:

    def __init__(self, dataset_folder, config: Configuration):
        self.dataset_folder = dataset_folder
        self.config: Configuration = config

        self.x_train = None  # training data (examples,time,channels)
        self.y_train = None  # One hot encoded class labels (numExamples,numClasses)
        self.y_train_strings = None  # class labels as strings (numExamples,1)
        self.one_hot_encoder_labels = None  # one hot label encoder
        self.classes_Unique_oneHotEnc = None
        self.num_train_instances = None
        self.num_instances = None

        # Class names as string
        self.classes_total = None

        self.time_series_length = None
        self.time_series_depth = None

        # the names of all features of the dataset loaded from files
        self.feature_names_all = None

        self.x_train_TSFresh_features = None
        self.x_test_TSFresh_features = None
        self.relevant_features_by_TSFresh = None

    def load(self):
        raise NotImplemented('Not implemented for abstract class')

    @staticmethod
    def draw_from_ds(self, dataset, num_instances, is_positive, class_idx=None):
        # dataset: vector with one-hot encoded label of the data set

        # draw as long as is_positive criterion is not satisfied

        # draw two random examples index
        if class_idx is None:
            while True:
                first_idx = np.random.randint(0, num_instances, size=1)[0]
                second_idx = np.random.randint(0, num_instances, size=1)[0]
                # return the two indexes if they match the is_positive criterion
                if is_positive:
                    if np.array_equal(dataset[first_idx], dataset[second_idx]):
                        return first_idx, second_idx
                else:
                    if not np.array_equal(dataset[first_idx], dataset[second_idx]):
                        return first_idx, second_idx
        else:
            # examples are drawn by a given class index
            # contains idx values of examples from the given class
            class_idx_arr = self.class_idx_to_ex_idxs_train[class_idx]

            # print("class_idx:", class_idx, " class_idx_arr: ", class_idx_arr, "self.class_idx_to_class_string: ",
            #      self.class_idx_to_class_string[class_idx])

            # Get a random idx of an example that is part of this class
            first_rand_idx = np.random.randint(0, len(class_idx_arr), size=1)[0]
            first_idx = class_idx_arr[first_rand_idx]

            if is_positive:
                while True:
                    second_rand_idx = np.random.randint(0, len(class_idx_arr), size=1)[0]
                    second_idx = class_idx_arr[second_rand_idx]
                    if first_idx != second_idx:
                        return first_idx[0], second_idx[0]
            else:
                while True:
                    uniform_sampled_class = np.random.randint(low=0,
                                                              high=len(self.y_train_strings_unique),
                                                              size=1)
                    class_idx_arr_neg = self.class_idx_to_ex_idxs_train[uniform_sampled_class[0]]
                    second_rand_idx_neg = np.random.randint(0, len(class_idx_arr_neg), size=1)[0]
                    # print("uniform_sampled_class: ", uniform_sampled_class, "class_idx_arr_neg: ", class_idx_arr_neg,
                    #       "second_rand_idx_neg: ", second_rand_idx_neg)

                    second_idx = class_idx_arr_neg[second_rand_idx_neg]
                    # second_idx = np.random.randint(0, num_instances, size=1)[0]

                    if second_idx not in class_idx_arr[:, 0]:
                        # print("class_idx_arr: ", class_idx_arr, " - uniform_sampled_class: ",
                        # uniform_sampled_class[0])
                        return first_idx[0], second_idx[0]


class FullDataset(Dataset):

    def __init__(self, dataset_folder, config: Configuration, training):
        super().__init__(dataset_folder, config)

        self.x_test = None
        self.y_test = None
        self.y_test_strings = None
        self.num_test_instances = None
        self.training = training

        self.x_train_TSFresh_features = None
        self.x_test_TSFresh_features = None
        self.relevant_features_by_TSFresh = None

        # total number of classes
        self.num_classes = None

        # dictionary with key: class as integer and value: array with index positions
        self.class_idx_to_ex_idxs_train = {}
        self.class_idx_to_ex_idxs_test = {}

        # np array that contains the number of instances for each classLabel in the training data
        self.num_instances_by_class_train = None

        # np array that contains the number of instances for each classLabel in the test data
        self.num_instances_by_class_test = None

        # np array that contains a list classes that occur in training OR test data set
        self.classes_total = None

        # np array that contains a list classes that occur in training AND test data set
        self.classes_in_both = None

        # dictionary, key: class label, value: np array which contains 0s or 1s depending on whether the attribute
        # at this index is relevant for the class described with the label key
        self.class_label_to_masking_vector = {}

        self.group_id_to_masking_vector = {}

        #
        # new
        #

        self.y_train_strings_unique = None
        self.y_test_strings_unique = None

        # additional information for each example about their window time frame and failure occurrence time
        self.window_times_train = None
        self.window_times_test = None
        self.failure_times_train = None
        self.failure_times_test = None

        # numpy array (x,2) that contains each unique permutation between failure occurrence time and assigned label
        self.unique_failure_times_label = None
        self.failure_times_count = None

        # pandas df ( = matrix) with pair-wise similarities between labels in respect to a metric
        self.df_label_sim_localization = None
        self.df_label_sim_failuremode = None
        self.df_label_sim_condition = None

    def load_files(self):

        self.x_train = np.load(self.dataset_folder + 'train_features.npy')  # data training
        self.y_train_strings = np.expand_dims(np.load(self.dataset_folder + 'train_labels.npy'), axis=-1)
        self.window_times_train = np.expand_dims(np.load(self.dataset_folder + 'train_window_times.npy'), axis=-1)
        self.failure_times_train = np.expand_dims(np.load(self.dataset_folder + 'train_failure_times.npy'), axis=-1)

        self.x_test = np.load(self.dataset_folder + 'test_features.npy')  # data testing
        self.y_test_strings = np.expand_dims(np.load(self.dataset_folder + 'test_labels.npy'), axis=-1)
        self.window_times_test = np.expand_dims(np.load(self.dataset_folder + 'test_window_times.npy'), axis=-1)
        self.failure_times_test = np.expand_dims(np.load(self.dataset_folder + 'test_failure_times.npy'), axis=-1)
        self.feature_names_all = np.load(self.dataset_folder + 'feature_names.npy')  # names of the features (3. dim)

    def load(self, print_info=True):
        self.load_files()

        # create a encoder, sparse output must be disabled to get the intended output format
        # added categories='auto' to use future behavior
        one_hot_encoder = preprocessing.OneHotEncoder(sparse=False, categories='auto')

        # prepare the encoder with training and test labels to ensure all are present
        # the fit-function 'learns' the encoding but does not jet transform the data
        # the axis argument specifies on which the two arrays are joined
        one_hot_encoder = one_hot_encoder.fit(np.concatenate((self.y_train_strings, self.y_test_strings), axis=0))

        # transforms the vector of labels into a one hot matrix
        self.y_train = one_hot_encoder.transform(self.y_train_strings)
        self.y_test = one_hot_encoder.transform(self.y_test_strings)

        # reduce to 1d array
        self.y_train_strings = np.squeeze(self.y_train_strings)
        self.y_test_strings = np.squeeze(self.y_test_strings)

        ##
        # safe information about the dataset
        ##

        # length of the first array dimension is the number of examples
        self.num_train_instances = self.x_train.shape[0]
        self.num_test_instances = self.x_test.shape[0]

        # the total sum of examples
        self.num_instances = self.num_train_instances + self.num_test_instances

        # length of the second array dimension is the length of the time series
        self.time_series_length = self.x_train.shape[1]

        # length of the third array dimension is the number of channels = (independent) readings at this point of time
        self.time_series_depth = self.x_train.shape[2]

        # get the unique classes and the corresponding number
        self.classes_total = np.unique(np.concatenate((self.y_train_strings, self.y_test_strings), axis=0))
        self.classes_Unique_oneHotEnc = one_hot_encoder.transform(np.expand_dims(self.classes_total, axis=1))
        self.num_classes = self.classes_total.size

        # Create two dictionaries to link/associate each class with all its training examples
        for i in range(self.num_classes):
            self.class_idx_to_ex_idxs_train[i] = np.argwhere(self.y_train[:, i] > 0)
            self.class_idx_to_ex_idxs_test[i] = np.argwhere(self.y_test[:, i] > 0)

        # collect number of instances for each class in training and test
        self.y_train_strings_unique, counts = np.unique(self.y_train_strings, return_counts=True)
        self.num_instances_by_class_train = np.asarray((self.y_train_strings_unique, counts)).T
        self.y_test_strings_unique, counts = np.unique(self.y_test_strings, return_counts=True)
        self.num_instances_by_class_test = np.asarray((self.y_test_strings_unique, counts)).T

        # calculate the number of classes that are the same in test and train
        self.classes_in_both = np.intersect1d(self.num_instances_by_class_test[:, 0],
                                              self.num_instances_by_class_train[:, 0])

        # required for inference metric calculation
        # get all failures and labels as unique entry
        failure_times_label = np.stack((self.y_test_strings, np.squeeze(self.failure_times_test))).T
        # extract unique permutations between failure occurrence time and labeled entry
        unique_failure_times_label, failure_times_count = np.unique(failure_times_label, axis=0, return_counts=True)
        # remove noFailure entries
        idx = np.where(np.char.find(unique_failure_times_label, 'noFailure') >= 0)
        self.unique_failure_times_label = np.delete(unique_failure_times_label, idx, 0)
        self.failure_times_count = np.delete(failure_times_count, idx, 0)

        self.calculate_maskings()
        self.load_sim_matrices()

        # data
        # 1. dimension: example
        # 2. dimension: time index
        # 3. dimension: array of all channels

        if print_info:
            print()
            print('Dataset loaded:')
            print('Shape of training set (example, time, channels):', self.x_train.shape)
            print('Shape of test set (example, time, channels):', self.x_test.shape)
            print('Num of classes in train and test together:', self.num_classes)
            # print('Classes used in training: ', len(self.y_train_strings_unique)," :",self.y_train_strings_unique)
            # print('Classes used in test: ', len(self.y_test_strings_unique)," :", self.y_test_strings_unique)
            # print('Classes in total: ', self.classes_total)
            print()

    def load_feature_based_representation(self):

        # Load TS-Fresh generated features
        filtered_file = 'extractedFeatures_X_caseBase_filtered_4ms4sec.pkl'
        unfiltered_file = 'extractedFeatures_X_test_unfiltered_imputed_4ms4sec.pkl'
        filtered_cb_df = (pd.read_pickle(self.dataset_folder + filtered_file))
        unfiltered_test_examples_df = (pd.read_pickle(self.dataset_folder + unfiltered_file))

        # Attributes selected after TSFresh significance test on case base
        self.relevant_features_by_TSFresh = filtered_cb_df.columns
        filtered_test_examples_df = unfiltered_test_examples_df[self.relevant_features_by_TSFresh]

        # print("unfiltered_test_examples_df: ", unfiltered_test_examples_df.shape)
        # print("filtered_cb_df: ", filtered_cb_df.shape)
        # merged_df = pd.concat([filtered_cb_df, unfiltered_test_examples_df])

        # Preprocessing
        # min_max_scaler = preprocessing.MinMaxScaler()
        # filteredCB_np_scaled = min_max_scaler.fit_transform(filtered_cb_df)
        # filteredTestExamples_np_scaled = min_max_scaler.transform(filtered_test_examples_df)

        self.x_test_TSFresh_features = filtered_test_examples_df.values  # filteredTestExamples_np_scaled
        self.x_train_TSFresh_features = filtered_cb_df.values  # filteredCB_np_scaled

    def load_sim_matrices(self):
        # load a matrix with pair-wise similarities between labels in respect
        # to different metrics
        self.df_label_sim_failuremode = pd.read_csv(self.dataset_folder + 'FailureMode_Sim_Matrix.csv', sep=';',
                                                    index_col=0)
        self.df_label_sim_failuremode.index = self.df_label_sim_failuremode.index.str.replace('\'', '')
        self.df_label_sim_localization = pd.read_csv(self.dataset_folder + 'Lokalization_Sim_Matrix.csv', sep=';',
                                                     index_col=0)
        self.df_label_sim_localization.index = self.df_label_sim_localization.index.str.replace('\'', '')
        self.df_label_sim_condition = pd.read_csv(self.dataset_folder + 'Condition_Sim_Matrix.csv', sep=';',
                                                  index_col=0)
        self.df_label_sim_condition.index = self.df_label_sim_condition.index.str.replace('\'', '')

    def calculate_maskings(self):
        for case in self.classes_total:

            if self.config.individual_relevant_feature_selection:
                relevant_features_for_case = self.config.get_relevant_features_case(case)
            else:
                relevant_features_for_case = self.config.get_relevant_features_group(case)

            if self.config.use_additional_strict_masking_for_attribute_sim:
                masking1 = np.isin(self.feature_names_all, relevant_features_for_case[0])
                masking2 = np.isin(self.feature_names_all, relevant_features_for_case[1])
                self.class_label_to_masking_vector[case] = [masking1, masking2]
            else:
                masking = np.isin(self.feature_names_all, relevant_features_for_case)
                self.class_label_to_masking_vector[case] = masking

        for group_id, features in self.config.group_id_to_features.items():
            masking = np.isin(self.feature_names_all, features)
            self.group_id_to_masking_vector[group_id] = masking

    # returns a boolean array with values depending on whether the attribute at this index is relevant
    # for the class of the passed label
    def get_masking(self, class_label):

        if class_label not in self.class_label_to_masking_vector:
            raise ValueError('Passed class label', class_label, 'was not found in masking dictionary')
        else:
            if self.config.use_additional_strict_masking_for_attribute_sim:
                masking = self.class_label_to_masking_vector.get(class_label)
                masking_vec = np.concatenate((masking[0], masking[1]))
                return masking_vec
            else:
                return self.class_label_to_masking_vector.get(class_label)

    def get_masked_example_group(self, test_example, group_id):

        if group_id not in self.group_id_to_masking_vector:
            raise ValueError('Passed group id', group_id, 'was not found in masking dictionary')
        else:
            mask = self.group_id_to_masking_vector.get(group_id)
            return test_example[:, mask]

    def get_masking_float(self, class_label):
        return self.get_masking(class_label).astype(float)

    # Will return the test example and the train example (of the passed index) reduced to the
    # relevant attributes of the case of the train_example
    def reduce_to_relevant(self, test_example, train_example_index):
        class_label_train_example = self.y_train_strings[train_example_index]
        mask = self.get_masking(class_label_train_example)
        return test_example[:, mask], self.x_train[train_example_index][:, mask]

    def get_ts_fresh_masking(self, train_example_index):
        class_label_train_example = self.y_train_strings[train_example_index]
        relevant_features_for_case = self.config.get_relevant_features_case(class_label_train_example)
        masking = np.zeros(len(self.relevant_features_by_TSFresh))

        idx = [i for i, x in enumerate(self.relevant_features_by_TSFresh) if
               x.split('__')[0] in relevant_features_for_case]
        masking[idx] = 1

        return masking

    def get_time_window_str(self, index, dataset_type):
        if dataset_type == 'test':
            dataset = self.window_times_test
        elif dataset_type == 'train':
            dataset = self.window_times_train
        else:
            raise ValueError('Unknown dataset type')

        rep = lambda x: str(x).replace("['YYYYMMDD HH:mm:ss (", "").replace(")']", "")

        t1 = rep(dataset[index][0])
        t2 = rep(dataset[index][2])
        return " - ".join([t1, t2])

    def get_indices_failures_only_test(self):
        return np.where(self.y_test_strings != 'no_failure')[0]

    def encode(self, snn, encode_test_data=False):

        start_time_encoding = perf_counter()
        print('Encoding of dataset started')

        x_train_unencoded = self.x_train
        '''
        x_train_masking = np.zeros((self.x_train.shape[0],self.x_train.shape[2]))
        for i,label in enumerate(self.y_train_strings):
            #print(i,label)
            x_train_masking[i,:] = self.get_masking(label)
        print("x_train_masking: ", x_train_masking.shape)
        '''
        self.x_train = None
        batchsize = self.config.sim_calculation_batch_size
        '''
        for example in range(x_train_unencoded.shape[0]//batchsize):
            start = example * batchsize
            end = (example +1) * batchsize
            #print("x_train_unencoded: ", x_train_unencoded.shape)
            print("batch: ", example, "start: ", start, "end: ", end)
            batch = x_train_unencoded[:132,:,:]
            batch = snn.reshape(batch)
            x_train_unencoded = snn.reshape_and_add_aux_input(x_train_unencoded, batch_size=66)
        '''

        x_train_unencoded_reshaped = snn.reshape_and_add_aux_input(x_train_unencoded,
                                                                   batch_size=(x_train_unencoded.shape[0] // 2))
        encoded = snn.encode_in_batches(x_train_unencoded_reshaped)

        if snn.hyper.encoder_variant == 'cnn2dwithaddinput':
            # encoded output is a list with each entry has an encoded batchjob with the number of outputs
            x_train_encoded_0 = encoded[0][0]
            x_train_encoded_1 = encoded[0][1]
            x_train_encoded_2 = encoded[0][2]
            x_train_encoded_3 = encoded[0][3]
            for encoded_batch in encoded:
                x_train_encoded_0 = np.append(x_train_encoded_0, encoded_batch[0], axis=0)
                x_train_encoded_1 = np.append(x_train_encoded_1, encoded_batch[1], axis=0)
                x_train_encoded_2 = np.append(x_train_encoded_2, encoded_batch[2], axis=0)
                x_train_encoded_3 = np.append(x_train_encoded_3, encoded_batch[3], axis=0)
            '''
            print("x_train_encoded_0: ", x_train_encoded_0.shape)
            print("x_train_encoded_1: ", x_train_encoded_1.shape)
            print("x_train_encoded_2: ", x_train_encoded_2.shape)
            print("x_train_encoded_3: ", x_train_encoded_3.shape)
            '''
            x_train_encoded_0 = x_train_encoded_0[batchsize:, :, :]
            x_train_encoded_1 = x_train_encoded_1[batchsize:, :]
            x_train_encoded_2 = x_train_encoded_2[batchsize:, :, :]
            x_train_encoded_3 = x_train_encoded_3[batchsize:, :]
            '''
            print("x_train_encoded_0: ", x_train_encoded_0.shape)
            print("x_train_encoded_1: ", x_train_encoded_1.shape)
            print("x_train_encoded_2: ", x_train_encoded_2.shape)
            print("x_train_encoded_3: ", x_train_encoded_3.shape)
            '''
            '''
            x_train_unencoded = snn.reshape(x_train_unencoded[:132,:,:])
            print("x_train_unencoded: ", x_train_unencoded.shape)
            x_train_unencoded = snn.reshape_and_add_aux_input(x_train_unencoded, batch_size=66)
            print("x_train_unencoded: ", x_train_unencoded[0].shape, x_train_unencoded[1].shape)
            x_train_encoded = snn.encoder.model(x_train_unencoded, training=False)
            x_train_encoded = np.asarray(x_train_encoded)
            self.x_train = x_train_encoded
            '''
            self.x_train = [x_train_encoded_0, x_train_encoded_1, x_train_encoded_2, x_train_encoded_3]
        else:
            print("shape. ", encoded[0].shape)
            x_train_encoded_0 = encoded[0]
            for encoded_batch in encoded:
                x_train_encoded_0 = np.append(x_train_encoded_0, encoded_batch, axis=0)
            x_train_encoded_0 = x_train_encoded_0[batchsize:, :, :]
            self.x_train = x_train_encoded_0
        # x_test will not be encoded by default because examples should simulate "new data" --> encoded at runtime
        # but can be done for visualisation purposes
        if encode_test_data:
            x_test_unencoded = self.x_test
            self.x_test = None
            x_test_unencoded = snn.reshape(x_test_unencoded)
            x_test_encoded = snn.encoder.model(x_test_unencoded, training=False)
            x_test_encoded = np.asarray(x_test_encoded)
            self.x_test = x_test_encoded

        encoding_duration = perf_counter() - start_time_encoding
        print('Encoding of dataset finished. Duration:', encoding_duration)
        # return [x_train_encoded_0, x_train_encoded_1,x_train_encoded_2,x_train_encoded_3]

    # Returns a pairwise similarity matrix (NumTrainExamples,NumTrainExamples) of all training examples
    def get_similarity_matrix(self, snn, encode_test_data=False):

        print('Producing similarity matrix of dataset started')

        x_train_unencoded = self.x_train

        sim_matrix = np.zeros((x_train_unencoded.shape[0], x_train_unencoded.shape[0]))
        for example_id in range(x_train_unencoded.shape[0]):
            example = x_train_unencoded[example_id, :, :]
            sims_4_example = snn.get_sims(example)
            # print("example_id:", example_id)
            sim_matrix[example_id, :] = sims_4_example[0]

        return sim_matrix

    # Draw a random pair of instances
    def draw_pair(self, is_positive, from_test):

        # Select dataset depending on parameter
        ds_y = self.y_test if from_test else self.y_train
        num_instances = self.num_test_instances if from_test else self.num_train_instances

        return Dataset.draw_from_ds(self, ds_y, num_instances, is_positive)

    def draw_pair_by_class_idx(self, is_positive, from_test, class_idx):

        # select dataset depending on parameter
        ds_y = self.y_test if from_test else self.y_train
        num_instances = self.num_test_instances if from_test else self.num_train_instances

        return Dataset.draw_from_ds(self, ds_y, num_instances, is_positive, class_idx)

    def get_sim_label_pair_for_notion(self, label_1: str, label_2: str, notion_of_sim: str):
        # Output similarity value under consideration of the metric

        if notion_of_sim == 'failuremode':
            pair_label_sim = self.df_label_sim_failuremode.loc[label_1, label_2]
        elif notion_of_sim == 'localization':
            pair_label_sim = self.df_label_sim_localization.loc[label_1, label_2]
        elif notion_of_sim == 'condition':
            pair_label_sim = self.df_label_sim_condition.loc[label_1, label_2]
        else:
            print("Similarity notion: ", notion_of_sim, " unknown! Results in sim 0")
            pair_label_sim = 0

        return float(pair_label_sim)

    # used to calculate similarity value based on the local similarities of the tree characteristics
    def get_sim_label_pair(self, index_1, index_2, dataset_type):
        if dataset_type == 'test':
            dataset = self.y_test_strings
        elif dataset_type == 'train':
            dataset = self.y_train_strings
        else:
            raise ValueError('Unkown dataset type. dataset_type: ', dataset_type)

        class_label_1 = dataset[index_1]
        class_label_2 = dataset[index_2]
        sim = (self.get_sim_label_pair_for_notion(class_label_1, class_label_2, "condition")
               + self.get_sim_label_pair_for_notion(class_label_1, class_label_2, "localization")
               + self.get_sim_label_pair_for_notion(class_label_1, class_label_2, "failuremode")) / 3
        return sim


class CBSDataset(FullDataset):

    def __init__(self, dataset_folder, config: Configuration, training):
        super().__init__(dataset_folder, config, training)
        self.group_to_indices_train = {}
        self.group_to_indices_test = {}
        self.group_to_negative_indices_train = {}

    def load(self, print_info=True):
        super().load(print_info)

        for group, cases in self.config.group_id_to_cases.items():
            self.group_to_indices_train[group] = [i for i, case in enumerate(self.y_train_strings) if case in cases]

        for group, cases in self.config.group_id_to_cases.items():
            self.group_to_indices_test[group] = [i for i, case in enumerate(self.y_test_strings) if case in cases]

        all_indices = [i for i in range(self.x_train.shape[0])]
        for group, pos_indices in self.group_to_indices_train.items():
            negative_indices = [x for x in all_indices if x not in pos_indices]
            self.group_to_negative_indices_train[group] = negative_indices

    def encode(self, encoder, encode_test_data=False):
        raise NotImplementedError('')

    def draw_pair_cbs(self, is_positive, group_id):
        pos_indices = self.group_to_indices_train.get(group_id)
        neg_indices = self.group_to_negative_indices_train.get(group_id)

        if is_positive:
            i1, i2 = np.random.choice(pos_indices, 2, replace=True)
            return i1, i2
        else:
            i1 = np.random.choice(pos_indices, 1, replace=True)[0]
            i2 = np.random.choice(neg_indices, 1, replace=True)[0]
            return i1, i2

    def get_masking_group(self, group_id):

        if group_id not in self.group_id_to_masking_vector:
            raise ValueError('Passed group id', group_id, 'was not found in masking dictionary')
        else:
            mask = self.group_id_to_masking_vector.get(group_id)
            return mask

    def create_group_dataset(self, group_id):
        dataset = GroupDataset(self.dataset_folder, self.config, self.training, self, group_id)
        dataset.load()
        return dataset

    def get_masked_example_group(self, test_example, group_id):
        mask = self.get_masking_group(group_id)
        return test_example[:, mask]


class GroupDataset(FullDataset):

    def __init__(self, dataset_folder, config: Configuration, training, main_dataset: CBSDataset, group_id):
        super().__init__(dataset_folder, config, training)
        self.main_dataset = main_dataset
        self.group_id = group_id

    def load_files(self):
        self.x_train = self.main_dataset.x_train.copy()  # data training
        self.y_train_strings = np.expand_dims(self.main_dataset.y_train_strings.copy(), axis=-1)
        self.window_times_train = self.main_dataset.window_times_train.copy()
        self.failure_times_train = self.main_dataset.failure_times_train.copy()

        # Temp solution, x_test only in here so load of full dataset works
        self.x_test = self.main_dataset.x_test.copy()
        self.y_test_strings = np.expand_dims(self.main_dataset.y_test_strings.copy(), axis=-1)
        self.window_times_test = self.main_dataset.window_times_test
        self.failure_times_test = self.main_dataset.failure_times_test
        self.feature_names_all = self.main_dataset.feature_names_all

        # Reduce training data to the cases of this group
        indices = self.main_dataset.group_to_indices_train.get(self.group_id)
        self.x_train = self.x_train[indices, :, :]
        self.y_train_strings = self.y_train_strings[indices, :]

        # Reduce x_train and feature_names_all to the features of this group
        mask = self.main_dataset.group_id_to_masking_vector.get(self.group_id)
        self.x_train = self.x_train[:, :, mask]
        self.feature_names_all = self.feature_names_all[mask]

        # Reduce metadata to relevant indices, too
        # (currently not used by SNN, so it wouldn't be necessary but done ensure future correctness, wrong index call,
        # may not be noticed)
        self.window_times_train = self.window_times_train[indices, :]
        self.failure_times_train = self.failure_times_train[indices, :]

    def load(self, print_info=False):
        super().load(print_info)
