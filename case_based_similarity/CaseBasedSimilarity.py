import os
import sys
import threading

import numpy as np
import tensorflow as tf

from neural_network.OptimizerHelper import CBSOptimizerHelper

sys.path.append(os.path.abspath(os.path.join(os.getcwd(), os.pardir)))

from multiprocessing import Queue
from configuration.Configuration import Configuration
from neural_network.Dataset import CBSDataset
from neural_network.SNN import SimpleSNN, AbstractSimilarityMeasure, initialise_snn


class CBS(AbstractSimilarityMeasure):

    def __init__(self, config: Configuration, training, dataset):
        super().__init__(training)

        self.config: Configuration = config
        self.dataset = dataset
        self.group_handlers: [CBSGroupHandler] = []
        self.number_of_groups = 0

        self.gpus = tf.config.experimental.list_logical_devices('GPU')
        self.nbr_gpus_used = config.max_gpus_used if 1 <= config.max_gpus_used < len(self.gpus) else len(self.gpus)
        self.gpus = self.gpus[0:self.nbr_gpus_used]
        self.gpus = [gpu.name for gpu in self.gpus]

        # with contextlib.redirect_stdout(None):
        self.initialise_group_handlers()

    def initialise_group_handlers(self):

        # Limit used groups for debugging purposes
        if self.config.cbs_groups_used is None or len(self.config.cbs_groups_used) == 0:
            id_to_cases = self.config.group_id_to_cases
        else:
            id_to_cases = dict((k, self.config.group_id_to_cases[k]) for k in self.config.cbs_groups_used if
                               k in self.config.group_id_to_cases)

        self.number_of_groups = len(id_to_cases.keys())

        # Counts the number of groups already processed, used for gpu distribution
        for group, cases in id_to_cases.items():

            # Check if there is at least one training example of this group,
            # otherwise a group handler can't be created
            nbr_examples_for_group = len(self.dataset.group_to_indices_train.get(group))
            if nbr_examples_for_group <= 0:
                print('-------------------------------------------------------')
                print('WARNING: No case handler could be created for', group, cases)
                print('Reason: No training example of this case in training dataset')
                print('-------------------------------------------------------')
                print()
                continue

            else:
                gh: CBSGroupHandler = CBSGroupHandler(group, self.config, self.dataset, self.training)
                gh.start()

                print('Creating group handler', group, ':')
                print('GPU:', 'Dynamically assigned')
                print('Cases:')
                for case in cases:
                    print('   ' + case)

                # Wait until initialisation of run finished and the group handler is ready to process input
                # the group handler will send a message for which we are waiting here
                _ = gh.output_queue.get()
                self.group_handlers.append(gh)
                print()

    # Is called when a keyboard interruption stops the main thread
    # Will send a message to each group handler, which leads to the termination of the run function
    def kill_threads(self):
        for group_handler in self.group_handlers:
            group_handler.input_queue.put('stop')

    def print_info(self):
        print()
        for group_handler in self.group_handlers:
            group_handler.print_group_handler_info()

    def get_sims(self, example):
        # Used to combine the results of all group handlers
        # Using a numpy array instead of a simple list to ensure index_sims == index_labels
        sims_groups = np.empty(self.number_of_groups, dtype='object_')
        labels_groups = np.empty(self.number_of_groups, dtype='object_')

        if self.config.batch_wise_handler_execution:
            num_gpus = len(self.gpus)

            for i in range(0, self.number_of_groups, num_gpus):

                # Creation of a batch of group handlers that are called up simultaneously,
                # so that only one is running at a time on a GPU
                ghs_batch = [(i + j, j) for j in range(num_gpus) if
                             i + j < self.number_of_groups]

                # Pass the example to each group handler using it's input queue
                for gh_index, gpu_index in ghs_batch:
                    self.group_handlers[gh_index].input_queue.put((example, self.gpus[gpu_index]))

                # Get the results via the output queue, will wait until it's available
                for gh_index, gpu_index in ghs_batch:
                    sims_groups[gh_index], labels_groups[gh_index] = self.group_handlers[gh_index].output_queue.get()

        else:
            # Pass the example to each group handler using it's input queue
            for index, group_handler in enumerate(self.group_handlers):
                # Distribute the group handlers equally to the available gpus
                gpu = self.gpus[index % len(self.gpus)]
                group_handler.input_queue.put((example, gpu))

            # Get the results via the output queue, will wait until it's available
            for gh_index, group_handler in enumerate(self.group_handlers):
                sims_groups[gh_index], labels_groups[gh_index] = group_handler.output_queue.get()

        return np.concatenate(sims_groups), np.concatenate(labels_groups)

    def get_sims_for_batch(self, batch):
        raise NotImplementedError(
            'Not implemented for this architecture. '
            'The models function will be called directly by each group handler.')


# Currently threading.Thread is used instead of multiprocessing because of unfixed incompatibility with tf/cuda
class CBSGroupHandler(threading.Thread):

    def __init__(self, group_id, config, dataset, training):
        self.group_id = group_id
        self.config = config
        self.dataset: CBSDataset = dataset
        self.training = training

        # The transfer and return of data to this process class must take place via these queues
        # to ensure correct functionality when using the multiprocessing library.
        # (Alternatively, additional data structures such as Multiprocessing.Manager.dict() would have to be used)
        self.input_queue = Queue()
        self.output_queue = Queue()

        # noinspection PyTypeChecker
        self.model = None
        self.optimizer_helper = None

        # self.print_group_handler_info()

        # Must be last entry in __init__
        super(CBSGroupHandler, self).__init__()

    def run(self):
        group_ds = self.dataset.create_group_dataset(self.group_id)
        self.model: SimpleSNN = initialise_snn(self.config, group_ds, self.training, True, self.group_id)

        if self.training:
            self.optimizer_helper = CBSOptimizerHelper(self.model, self.config, self.dataset, self.group_id)

        # Change the execution of the process depending on
        # whether the model is trained or applied
        # as additional variable so it can't be changed during execution
        is_training = self.training

        # Send message so that the initiator knows that the preparations are complete.
        self.output_queue.put(str(self.group_id) + ' init finished. ')

        while True:
            elem = self.input_queue.get(block=True)

            # Stop the process execution if a stop message was send via the queue
            if isinstance(elem, str) and elem == 'stop':
                break

            elem, gpu = elem

            with tf.device(gpu):
                if is_training:

                    # Train method must be called by the process itself so that the advantage of parallel execution
                    # of the training of the individual groups can be exploited.
                    # Feedback contains loss and additional information using a single string
                    feedback = self.train(elem)
                    self.output_queue.put(feedback)
                else:

                    # Reduce the input example to the features required for this group
                    # and pass it to the model to calculate the similarities
                    elem = self.dataset.get_masked_example_group(elem, self.group_id)
                    output = self.model.get_sims(elem)
                    self.output_queue.put(output)

    # Debugging method, can be removed when implementation is finished
    def print_group_handler_info(self):
        np.set_printoptions(threshold=np.inf)
        print('CBSGroupHandler with ID', self.group_id)
        print('Cases of group:')
        print(self.config.group_id_to_cases.get(self.group_id))
        print('Relevant features:')
        print(self.config.group_id_to_features.get(self.group_id))
        print('Indices of cases in case base with case:')
        print(self.dataset.group_to_indices_train.get(self.group_id))
        print()
        print()

    # Will be executed by  a group handler process so the training can be executed in parallel for each group
    def train(self, training_interval):
        group_id = self.group_id
        losses = []

        # print('started', self.group_id, 'on', self.gpu)
        for epoch in range(training_interval):
            # print(group_id, epoch)

            epoch_loss_avg = tf.keras.metrics.Mean()

            batch_pairs_indices, true_similarities = self.optimizer_helper.compose_batch()

            # Get the example pairs by the selected indices
            model_input = np.take(a=self.dataset.x_train, indices=batch_pairs_indices, axis=0)

            # Reduce to the features used by this case handler
            model_input = model_input[:, :, self.dataset.get_masking_group(group_id)]

            # Untested
            # model_input = self.model.reshape(model_input)

            batch_loss = self.optimizer_helper.update_single_model(model_input, true_similarities)

            # Track progress
            epoch_loss_avg.update_state(batch_loss)
            current_loss = epoch_loss_avg.result()

            losses.append(current_loss)

            if self.optimizer_helper.execute_early_stop(current_loss):
                return losses, 'early_stopping'

        # print('finished', self.group_id, 'on', self.gpu)

        # Return the losses for all epochs during this training interval
        return losses, ''
