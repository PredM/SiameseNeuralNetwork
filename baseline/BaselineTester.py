import multiprocessing
import os
import sys
from time import perf_counter

import numpy as np
from fastdtw import fastdtw
from scipy.spatial.distance import euclidean, minkowski
from sklearn import preprocessing

sys.path.append(os.path.abspath(os.path.join(os.getcwd(), os.pardir)))

from neural_network.Evaluator import Evaluator
from neural_network.Dataset import FullDataset
from configuration.Configuration import Configuration


class Counter:
    def __init__(self, total, temp_output_interval):
        self.total = total
        self.temp_output_interval = temp_output_interval
        self.val = multiprocessing.Value('i')

    def increment(self):
        with self.val.get_lock():
            self.val.value += 1
            if self.temp_output_interval > 0 and self.val.value % self.temp_output_interval == 0:
                print("Training examples compares for current test:", self.val.value, '/', self.total)

    def get_value(self):
        with self.val.get_lock():
            return self.val.value


def run(proc_id, return_dict, counter, dataset, test_index, indices_train_examples, algorithm, relevant_only):
    try:

        results = np.zeros(len(indices_train_examples))

        for array_index, example_index in enumerate(indices_train_examples):

            ###
            # Prepare examples
            ###

            if algorithm == 'feature_based':
                # feature based data is 2d-structured (examples,features)
                test_example = dataset.x_test_TSFresh_features[test_index, :]
                train_example = dataset.x_train_TSFresh_features[example_index, :]
            elif relevant_only:
                test_example = dataset.x_test[test_index]
                test_example, train_example = dataset.reduce_to_relevant(test_example, example_index)
            else:
                test_example = dataset.x_test[test_index]
                train_example = dataset.x_train[example_index]

            ##
            # Execute algorithm
            ##
            if algorithm == 'dtw':
                distance, _ = fastdtw(test_example, train_example, dist=euclidean)

            elif algorithm == 'dtw_weighting_nbr_features':
                distance, _ = fastdtw(test_example, train_example, dist=euclidean)
                distance = distance / test_example.shape[1]

            elif algorithm == 'feature_based':
                if relevant_only:
                    masking = dataset.get_ts_fresh_masking(example_index)
                    weights = masking / (np.sum(masking))
                    distance = minkowski(test_example, train_example, 2, weights)
                    # Adjustment based on feature amount (improved performance)
                    small_num_of_attributes_penalty = (1 / (np.sum(masking)))
                    # if small_num_of_attributes_penalty > 1:
                    #    small_num_of_attributes_penalty = 1
                    distance = distance * small_num_of_attributes_penalty
                else:
                    distance = minkowski(test_example, train_example, 2)

            else:
                raise ValueError('Unkown algorithm:', algorithm)

            results[array_index] = distance
            counter.increment()
        return_dict[proc_id] = results

    except KeyboardInterrupt:
        pass


def execute_baseline_test(dataset: FullDataset, start_index, end_index, nbr_threads, algorithm, k_of_knn,
                          temp_output_interval, use_relevant_only=False, conversion_method=None):
    evaluator = Evaluator(dataset, end_index - start_index, k_of_knn)

    if algorithm == 'feature_based':
        # Load features from TSFresh
        dataset.load_feature_based_representation()

    start_time = perf_counter()

    for test_index in range(start_index, end_index):
        results = np.zeros(dataset.num_train_instances)

        # Split the training examples into n chunks, where n == the number of threads that should be used.
        chunks = np.array_split(range(dataset.num_train_instances), nbr_threads)

        threads = []
        counter = Counter(dataset.num_train_instances, temp_output_interval)
        manager = multiprocessing.Manager()
        return_dict = manager.dict()

        for i, chunk in enumerate(chunks):
            # Important: Passing object instances as args to multiprocessing.Process may lead to problems
            # Use carefully and ensure correct passing
            # proc_id, return_dict, counter, dataset, test_index, indices_train_examples, algorithm, relevant_only
            # chunk, dataset, test_index, use_relevant_only, counter, algorithm_used, i, return_dict
            args = (i, return_dict, counter, dataset, test_index, chunk, algorithm, use_relevant_only)
            t = multiprocessing.Process(target=run, args=args)
            t.start()
            threads.append(t)

        for i, chunk in enumerate(chunks):
            threads[i].join()
            results[chunk] = return_dict.get(i)

        # If algorithm returns distance instead of similarity
        if algorithm in ['dtw', 'dtw_weighting_nbr_features', 'feature_based']:
            results = distance_to_sim(results, conversion_method)

        # Add additional empty line if temp. outputs are enabled
        if temp_output_interval > 0:
            print('')
        evaluator.add_single_example_results(results, test_index)

    elapsed_time = perf_counter() - start_time
    evaluator.calculate_results()
    evaluator.print_results(elapsed_time)


def distance_to_sim(distances, conversion_method):
    if conversion_method == 'min_max_scaling':
        return 1 - preprocessing.minmax_scale(distances, feature_range=(0, 1))
    elif conversion_method == '1/(1+d)':
        func = np.vectorize(lambda x: 1 / (1 + x))
        return func(distances)
    elif conversion_method == 'div_max':
        max_dist = np.amax(distances)
        func = np.vectorize(lambda x: 1 - x / max_dist)
        return func(distances)
    else:
        raise ValueError('Selected algorithm is a distance measure but no conversation method has been provided. '
                         'Necessary for correct evaluation.')


def main():
    config = Configuration()

    if config.case_base_for_inference:
        dataset: FullDataset = FullDataset(config.case_base_folder, config, training=False)
    else:
        dataset: FullDataset = FullDataset(config.training_data_folder, config, training=False)

    dataset.load()

    # select which part of the test dataset to test
    start_index = 0  # dataset.num_test_instances
    end_index = dataset.num_test_instances  # dataset.num_test_instances

    # Output interval of how many examples have been compared so far. < 0 for no output
    temp_output_interval = -1
    use_relevant_only = True
    implemented_algorithms = ['dtw', 'dtw_weighting_nbr_features', 'feature_based']
    algorithm_used = implemented_algorithms[0]
    distance_to_sim_methods = ['1/(1+d)', 'div_max', 'min_max_scaling']
    distance_to_sim_method = distance_to_sim_methods[0]

    relevant_type = 'Individual' if config.individual_relevant_feature_selection else 'Group based'
    print('Algorithm used:', algorithm_used)
    print('Used relevant only:', use_relevant_only)
    print('Type of feature selection:', relevant_type)
    print('Start index:', start_index)
    print('End index:', end_index)
    print('Number of parallel threads:', config.max_parallel_cores)
    print('Case Based used for inference:', config.case_base_for_inference)
    print()

    execute_baseline_test(dataset, start_index, end_index, config.max_parallel_cores, algorithm_used, config.k_of_knn,
                          temp_output_interval, use_relevant_only, distance_to_sim_method)


# this script is used to execute the dtw and other baseline methods for comparision with the neural network
if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(0)
