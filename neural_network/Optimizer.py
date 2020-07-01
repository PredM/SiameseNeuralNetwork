import os
import shutil
from datetime import datetime
from os import listdir
from time import perf_counter

import numpy as np
import tensorflow as tf

from case_based_similarity.CaseBasedSimilarity import CBS, CBSGroupHandler
from configuration.Configuration import Configuration
from configuration.Hyperparameter import Hyperparameters
from neural_network.Dataset import FullDataset, CBSDataset
from neural_network.Inference import Inference
from neural_network.OptimizerHelper import OptimizerHelper
from neural_network.SNN import initialise_snn


class Optimizer:

    def __init__(self, architecture, dataset, config):
        self.architecture = architecture
        self.dataset: FullDataset = dataset
        self.config: Configuration = config
        self.last_output_time = None

    def optimize(self):
        raise NotImplementedError('Not implemented for abstract class')

    def single_epoch(self, epoch):
        raise NotImplementedError('Not implemented for abstract class')

    def delete_old_checkpoints(self, current_epoch):
        if current_epoch <= 0:
            return

        # For each directory in the folder check which epoch was safed
        for dir_name in listdir(self.config.models_folder):
            if 'temp' in dir_name:

                if type(self) == SNNOptimizer and 'snn' in dir_name or type(self) == CBSOptimizer and 'cbs' in dir_name:

                    try:
                        epoch = int(dir_name.split('-')[-1])

                        # Delete the directory if the stored epoch is smaller than the ones should be kept
                        # in with respect to the configured amount of models that should be kept
                        if epoch <= current_epoch - self.config.model_files_stored * self.config.output_interval:
                            # Maybe needs to be set to true
                            shutil.rmtree(self.config.models_folder + dir_name, ignore_errors=False)

                    except ValueError:
                        pass


class SNNOptimizer(Optimizer):

    def __init__(self, architecture, dataset, config):

        super().__init__(architecture, dataset, config)
        self.dir_name_last_model_saved = None

        # early stopping
        self.train_loss_results = []
        self.best_loss = 1000
        self.stopping_step_counter = 0

        self.optimizer_helper = OptimizerHelper(self.architecture, self.config, self.dataset)

    def optimize(self):
        current_epoch = 0

        if self.config.continue_training:
            self.architecture.load_model(cont=True)
            current_epoch = self.architecture.hyper.epochs_current

            if current_epoch >= self.architecture.hyper.epochs:
                print('Training already finished. If training should be continued'
                      ' increase the number of epochs in the hyperparameter file of the safed model')
            else:
                print('Continuing the training at epoch', current_epoch)

        self.last_output_time = perf_counter()

        for epoch in range(current_epoch, self.architecture.hyper.epochs):
            self.single_epoch(epoch)

            if self.execute_early_stop():
                print("Early stopping: Training stopped at epoch ", epoch, " because loss did not decrease since ",
                      self.stopping_step_counter, "epochs.")

                break

            self.inference_during_training(epoch)

    def single_epoch(self, epoch):
        """
        Compute the loss of one epoch based on a batch that is generated randomly from the training data
        Generation of batch in a separate method
          Args:
            epoch: int. current epoch
        """
        epoch_loss_avg = tf.keras.metrics.Mean()

        batch_pairs_indices, true_similarities = self.optimizer_helper.compose_batch()

        # Get the example pairs by the selected indices
        model_input = np.take(a=self.dataset.x_train, indices=batch_pairs_indices, axis=0)

        # Create auxiliary inputs if necessary for encoder variant
        model_input_class_strings = np.take(a=self.dataset.y_train_strings, indices=batch_pairs_indices, axis=0)
        model_aux_input = None
        if self.architecture.hyper.encoder_variant in ['cnn2dwithaddinput']:
            model_aux_input = np.array([self.dataset.get_masking_float(label) for label in model_input_class_strings],
                                       dtype='float32')

        # Reshape (and integrate model_aux_input) if necessary for encoder variant
        # batch_size and index are irrelevant because not used if aux_input is passed
        model_input = self.architecture.reshape_and_add_aux_input(model_input, 0, aux_input=model_aux_input)

        batch_loss = self.optimizer_helper.update_single_model(model_input, true_similarities,
                                                               query_classes=model_input_class_strings)

        # Track progress
        epoch_loss_avg.update_state(batch_loss)  # Add current batch loss
        self.train_loss_results.append(epoch_loss_avg.result())

        if epoch % self.config.output_interval == 0:
            print("Timestamp: {} ({:.2f} Seconds since last output) - Epoch: {} - Loss: {:.5f}".format(
                datetime.now().strftime('%d.%m %H:%M:%S'),
                (perf_counter() - self.last_output_time),
                epoch,
                epoch_loss_avg.result()
            ))

            self.delete_old_checkpoints(epoch)
            self.save_models(epoch)
            self.last_output_time = perf_counter()

    def execute_early_stop(self):
        if self.config.use_early_stopping:

            # Check if the loss of the last epoch is better than the best loss
            # If so reset the early stopping progress else continue approaching the limit
            if self.train_loss_results[-1] < self.best_loss:
                self.stopping_step_counter = 0
                self.best_loss = self.train_loss_results[-1]
            else:
                self.stopping_step_counter += 1

            # Check if the limit was reached
            if self.stopping_step_counter >= self.config.early_stopping_epochs_limit \
                    or self.train_loss_results[-1] <= self.config.early_stopping_loss_minimum:
                return True
            else:
                return False
        else:
            # Always continue if early stopping should not be used
            return False

    def inference_during_training(self, epoch):
        # TODO Maybe add this to cbs
        if self.config.use_inference_test_during_training and epoch != 0:
            if epoch % self.config.inference_during_training_epoch_interval == 0:
                print("Inference at epoch: ", epoch)
                dataset2: FullDataset = FullDataset(self.config.training_data_folder, self.config, training=False)
                dataset2.load()
                self.config.directory_model_to_use = self.dir_name_last_model_saved
                print("self.dir_name_last_model_saved: ", self.dir_name_last_model_saved)
                print("self.config.filename_model_to_use: ", self.config.directory_model_to_use)
                architecture2 = initialise_snn(self.config, dataset2, False)
                inference = Inference(self.config, architecture2, dataset2)
                inference.infer_test_dataset()

    def save_models(self, current_epoch):
        if current_epoch <= 0:
            return

        # generate a name and create the directory, where the model files of this epoch should be stored
        epoch_string = 'epoch-' + str(current_epoch)
        dt_string = datetime.now().strftime("%m-%d_%H-%M-%S")
        dir_name = self.config.models_folder + '_'.join(['temp', 'snn', 'model', dt_string, epoch_string]) + '/'
        os.mkdir(dir_name)
        self.dir_name_last_model_saved = dir_name

        # generate the file names and save the model files in the directory created before
        subnet_file_name = '_'.join(['encoder', self.architecture.hyper.encoder_variant, epoch_string]) + '.h5'

        # write model configuration to file
        self.architecture.hyper.epochs_current = current_epoch
        self.architecture.hyper.write_to_file(dir_name + 'hyperparameters_used.json')

        self.architecture.encoder.model.save_weights(dir_name + subnet_file_name)

        if self.config.architecture_variant in ['standard_ffnn', 'fast_ffnn']:
            ffnn_file_name = '_'.join(['ffnn', epoch_string]) + '.h5'
            self.architecture.ffnn.model.save_weights(dir_name + ffnn_file_name)

        loss = str(self.train_loss_results[-1].numpy())

        with open(dir_name + 'loss.txt', 'w') as f:
            f.write(loss)


class CBSOptimizer(Optimizer):

    def __init__(self, architecture, dataset, config):
        super().__init__(architecture, dataset, config)
        self.architecture: CBS = architecture
        self.dataset: CBSDataset = self.dataset
        self.handlers_still_training = self.architecture.group_handlers.copy()

        self.gpus = tf.config.experimental.list_logical_devices('GPU')
        self.nbr_gpus_used = config.max_gpus_used if 1 <= config.max_gpus_used < len(self.gpus) else len(self.gpus)
        self.gpus = self.gpus[0:self.nbr_gpus_used]
        self.gpus = [gpu.name for gpu in self.gpus]

        self.losses = dict()
        self.goal_epochs = dict()

        for group_handler in self.architecture.group_handlers:
            group_handler: CBSGroupHandler = group_handler
            group_id = group_handler.group_id
            group_hyper: Hyperparameters = group_handler.model.hyper

            self.losses[group_id] = []
            self.goal_epochs[group_id] = group_hyper.epochs

        self.max_epoch = max(self.goal_epochs.values())

    def optimize(self):

        current_epoch = 0

        if self.config.continue_training:
            raise NotImplementedError()

        self.last_output_time = perf_counter()

        while len(self.handlers_still_training) > 0:

            if self.config.batch_wise_handler_execution:
                num_gpus = len(self.architecture.gpus)

                for i in range(0, len(self.handlers_still_training), num_gpus):
                    ghs_batch = [(self.handlers_still_training[i + j], j) for j in range(num_gpus) if
                                 i + j < len(self.handlers_still_training)]

                    for group_handler, gpu_index in ghs_batch:
                        training_interval = self.config.output_interval

                        # Goal epoch for this case handler will be reached during this training step
                        if self.goal_epochs.get(group_handler.group_id) <= current_epoch + training_interval:
                            training_interval = self.goal_epochs.get(group_handler.group_id) - current_epoch

                        # When training, the only input is the number of epochs that should be trained for
                        # before next output/save
                        group_handler.input_queue.put((training_interval, self.architecture.gpus[gpu_index]))

                    for group_handler, _ in ghs_batch:
                        # Wait for the group handlers to finish the training interval
                        losses_of_training_interval, info = group_handler.output_queue.get()

                        # Append losses to list with full history
                        loss_list = self.losses.get(group_handler.group_id)
                        loss_list += losses_of_training_interval

                        # Evaluate the information provided in addition to the losses
                        if info == 'early_stopping':
                            self.handlers_still_training.remove(group_handler)
                            print('Early stopping group handler', group_handler.group_id)
            else:

                for index, group_handler in enumerate(self.handlers_still_training):
                    gpu = self.gpus[index % len(self.gpus)]

                    training_interval = self.config.output_interval

                    # Goal epoch for this case handler will be reached during this training step
                    if self.goal_epochs.get(group_handler.group_id) <= current_epoch + training_interval:
                        training_interval = self.goal_epochs.get(group_handler.group_id) - current_epoch

                    # When training, the only input is the number of epochs that should be trained for
                    # before next output/save
                    group_handler.input_queue.put((training_interval, gpu))

                for group_handler in self.handlers_still_training:
                    # Wait for the group handlers to finish the training interval
                    losses_of_training_interval, info = group_handler.output_queue.get()

                    # Append losses to list with full history
                    loss_list = self.losses.get(group_handler.group_id)
                    loss_list += losses_of_training_interval

                    # Evaluate the information provided in addition to the losses
                    if info == 'early_stopping':
                        self.handlers_still_training.remove(group_handler)
                        print('Early stopping group handler', group_handler.group_id)

            self.output(current_epoch)
            current_epoch += self.config.output_interval

        self.architecture.kill_threads()

    def output(self, current_epoch):
        print("Timestamp: {} ({:.2f} Seconds since last output) - Epoch: {}".format(
            datetime.now().strftime('%d.%m %H:%M:%S'),
            perf_counter() - self.last_output_time, current_epoch))

        for group_handler in self.architecture.group_handlers:
            group_handler: CBSGroupHandler = group_handler
            group_id = group_handler.group_id

            loss_of_case = self.losses.get(group_id)[-1].numpy()

            # Dont continue training if goal epoch was reached for this case
            if group_handler in self.handlers_still_training \
                    and self.goal_epochs.get(group_id) < current_epoch + self.config.output_interval:
                self.handlers_still_training.remove(group_handler)

            status = 'Yes' if group_handler in self.handlers_still_training else 'No'
            print("   GroupID: {: <28} Still training: {: <15} Loss: {:.5}"
                  .format(group_id, status, loss_of_case))

        print()
        self.delete_old_checkpoints(current_epoch)
        self.save_models(current_epoch)
        self.last_output_time = perf_counter()

    def save_models(self, current_epoch):
        if current_epoch <= 0:
            return

        # Generate a name and create the directory, where the model files of this epoch should be stored
        epoch_string = 'epoch-' + str(current_epoch)
        dt_string = datetime.now().strftime("%m-%d_%H-%M-%S")
        dir_name = self.config.models_folder + '_'.join(['temp', 'cbs', 'model', dt_string, epoch_string]) + '/'
        os.mkdir(dir_name)

        for group_handler in self.architecture.group_handlers:
            group_handler: CBSGroupHandler = group_handler
            group_id = group_handler.group_id
            group_hyper = group_handler.model.hyper

            # Create a subdirectory for the model files of this case handler
            subdirectory = group_id + '_model'
            full_path = os.path.join(dir_name, subdirectory)
            os.mkdir(full_path)

            # Write model configuration to file
            group_hyper.epochs_current = current_epoch if current_epoch <= group_hyper.epochs \
                else group_hyper.epochs
            group_hyper.write_to_file(full_path + '/' + group_id + '.json')

            # generate the file names and save the model files in the directory created before
            encoder_file_name = '_'.join(['encoder', group_hyper.encoder_variant, epoch_string]) + '.h5'
            group_handler.model.encoder.model.save_weights(os.path.join(full_path, encoder_file_name))

            if self.config.architecture_variant in ['standard_ffnn', 'fast_ffnn']:
                ffnn_file_name = '_'.join(['ffnn', epoch_string]) + '.h5'
                group_handler.model.ffnn.model.save_weights(os.path.join(full_path, ffnn_file_name))
