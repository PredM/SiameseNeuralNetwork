import numpy as np
import tensorflow as tf
import tensorflow.keras.backend as K


class OptimizerHelper:

    def __init__(self, model, config, dataset):
        self.model = model
        self.config = config
        self.hyper = self.model.hyper
        self.dataset = dataset
        self.optimizer = None

        if self.hyper.gradient_cap >= 0:
            self.adam_optimizer = tf.keras.optimizers.Adam(learning_rate=self.hyper.learning_rate,
                                                           clipnorm=self.hyper.gradient_cap)
        else:
            self.adam_optimizer = tf.keras.optimizers.Adam(learning_rate=self.hyper.learning_rate)

    def update_single_model(self, model_input, true_similarities, query_classes=None):
        with tf.GradientTape() as tape:
            pred_similarities = self.model.get_sims_for_batch(model_input)

            # Get parameters of subnet and ffnn (if complex sim measure)
            if self.config.architecture_variant in ['standard_ffnn', 'fast_ffnn']:
                trainable_params = self.model.ffnn.model.trainable_variables + \
                                   self.model.encoder.model.trainable_variables
            else:
                trainable_params = self.model.encoder.model.trainable_variables

            # Calculate the loss based on configuration
            if self.config.type_of_loss_function == "binary_cross_entropy":

                if self.config.use_margin_reduction_based_on_label_sim:
                    sim = self.get_similarity_between_two_label_string(query_classes, neg_pair_wbce=True)
                    loss = self.weighted_binary_crossentropy(y_true=true_similarities, y_pred=pred_similarities,
                                                             weight=sim)
                else:
                    loss = tf.keras.losses.binary_crossentropy(y_true=true_similarities, y_pred=pred_similarities)

            elif self.config.type_of_loss_function == "constrative_loss":

                if self.config.use_margin_reduction_based_on_label_sim:
                    sim = self.get_similarity_between_two_label_string(query_classes)
                    loss = self.contrastive_loss(y_true=true_similarities, y_pred=pred_similarities,
                                                 classes=sim)
                else:
                    loss = self.contrastive_loss(y_true=true_similarities, y_pred=pred_similarities)

            elif self.config.type_of_loss_function == "mean_squared_error":
                loss = tf.keras.losses.MSE(true_similarities, pred_similarities)

            elif self.config.type_of_loss_function == "huber_loss":
                huber = tf.keras.losses.Huber(delta=0.1)
                loss = huber(true_similarities, pred_similarities)
            else:
                raise AttributeError(
                    'Unknown loss function name. Use: "binary_cross_entropy" or "constrative_loss": ',
                    self.config.type_of_loss_function)

            grads = tape.gradient(loss, trainable_params)

            # Apply the gradients to the trainable parameters
            self.adam_optimizer.apply_gradients(zip(grads, trainable_params))

            return loss

    def contrastive_loss(self, y_true, y_pred, classes=None):
        """
        Contrastive loss from Hadsell-et-al.'06
        http://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf
        """
        margin = self.config.margin_of_loss_function
        if self.config.use_margin_reduction_based_on_label_sim:
            # label adapted margin, classes contains the
            margin = (1 - classes) * margin
        # print("margin: ", margin)
        square_pred = tf.square(y_pred)
        margin_square = tf.square(tf.maximum(margin - y_pred, 0))
        return tf.keras.backend.mean(y_true * square_pred + (1 - y_true) * margin_square)

    # noinspection PyMethodMayBeStatic
    def weighted_binary_crossentropy(self, y_true, y_pred, weight=None):
        """
        Weighted BCE that smoothes only the wrong example according to interclass similarities
        """
        weight = 1.0 if weight is None else weight

        y_true = K.clip(tf.convert_to_tensor(y_true, dtype=tf.float32), K.epsilon(), 1 - K.epsilon())
        y_pred = K.clip(y_pred, K.epsilon(), 1 - K.epsilon())
        # org: logloss = -(y_true * K.log(y_pred) * weight + (1 - y_true) * K.log(1 - y_pred))
        logloss = -(y_true * K.log(y_pred) + (1 - y_true + (weight / 2)) * K.log(1 - y_pred))
        return K.mean(logloss, axis=-1)

    # wbce = weighted_binary_cross_entropy
    def get_similarity_between_two_label_string(self, classes, neg_pair_wbce=False):
        # Returns the similarity between 2 failures (labels) in respect to the location of occurrence,
        # the type of failure (failure mode) and the condition of the data sample.
        # Input: 1d npy array with pairwise class labels as strings [2*batchsize]
        # Output: 1d npy array [batchsize]
        pairwise_class_label_sim = np.zeros([len(classes) // 2])
        for pair_index in range(len(classes) // 2):
            a = classes[2 * pair_index]
            b = classes[2 * pair_index + 1]

            sim = (self.dataset.get_sim_label_pair_for_notion(a, b, "condition")
                   + self.dataset.get_sim_label_pair_for_notion(a, b, "localization")
                   + self.dataset.get_sim_label_pair_for_notion(a, b, "failuremode")) / 3

            if neg_pair_wbce and sim < 1:
                sim = 1 - sim

            pairwise_class_label_sim[pair_index] = sim

        return pairwise_class_label_sim

    def compose_batch(self):
        batch_true_similarities = []  # similarity label for each pair
        batch_pairs_indices = []  # index number of each example used in the training

        # Generate a random vector that contains the number of classes that should be considered in the current batch
        # 4 means approx. half of the batch contains no-failure, 1 and 2 uniform
        equal_class_part = self.config.upsampling_factor
        failure_classes_considered = np.random.randint(low=0, high=len(self.dataset.y_train_strings_unique),
                                                       size=self.hyper.batch_size // equal_class_part)
        # print("failure_classes_considered: ", failure_classes_considered)

        # Compose batch
        # // 2 because each iteration one similar and one dissimilar pair is added
        for i in range(self.hyper.batch_size // 2):
            # print("i: ", i)
            #
            # pos pair
            #
            if self.config.equalClassConsideration:
                if i < self.hyper.batch_size // equal_class_part:
                    # print(i,": ", failure_classes_considered[i-self.architecture.hyper.batch_size // 4])

                    idx = (failure_classes_considered[i - self.hyper.batch_size // equal_class_part])
                    # print(i, "idx: ", idx)
                    pos_pair = self.dataset.draw_pair_by_class_idx(True, from_test=False, class_idx=idx)
                    # class_idx=(i % self.dataset.num_classes))
                else:
                    pos_pair = self.dataset.draw_pair(True, from_test=False)
            else:
                pos_pair = self.dataset.draw_pair(True, from_test=False)
            batch_pairs_indices.append(pos_pair[0])
            batch_pairs_indices.append(pos_pair[1])
            # print("PosPair: ", self.dataset.y_train_strings[pos_pair[0]]," - ", #
            # self.dataset.y_train_strings[pos_pair[1]])
            batch_true_similarities.append(1.0)

            #
            # neg pair here
            #

            # Find a negative pair
            if self.config.equalClassConsideration:
                if i < self.hyper.batch_size // equal_class_part:

                    idx = (failure_classes_considered[i - self.hyper.batch_size // equal_class_part])
                    neg_pair = self.dataset.draw_pair_by_class_idx(False, from_test=False, class_idx=idx)
                else:
                    neg_pair = self.dataset.draw_pair(False, from_test=False)
            else:
                neg_pair = self.dataset.draw_pair(False, from_test=False)
            batch_pairs_indices.append(neg_pair[0])
            batch_pairs_indices.append(neg_pair[1])
            # print("NegPair: ", self.dataset.y_train_strings[neg_pair[0]], " - ",
            # self.dataset.y_train_strings[neg_pair[1]])

            # If configured a similarity value is used for the negative pair instead of full dissimilarity
            if self.config.use_sim_value_for_neg_pair:
                sim = self.dataset.get_sim_label_pair(neg_pair[0], neg_pair[1], 'train')
                batch_true_similarities.append(sim)
            else:
                batch_true_similarities.append(0.0)

        # Change the list of ground truth similarities to an array
        true_similarities = np.asarray(batch_true_similarities)

        return batch_pairs_indices, true_similarities


class CBSOptimizerHelper(OptimizerHelper):

    def __init__(self, model, config, dataset, group_id):
        super().__init__(model, config, dataset)
        self.group_id = group_id

        self.losses = []
        self.best_loss = 1000
        self.stopping_step_counter = 0

    # Overwrites the standard implementation because some features are not compatible with the cbs currently
    def compose_batch(self):
        batch_true_similarities = []  # similarity label for each pair
        batch_pairs_indices = []  # index number of each example used in the training
        group_hyper = self.model.hyper

        # Compose batch
        # // 2 because each iteration one similar and one dissimilar pair is added

        for i in range(group_hyper.batch_size // 2):

            i1, i2 = self.dataset.draw_pair_cbs(True, self.group_id)
            batch_pairs_indices.append(i1)
            batch_pairs_indices.append(i2)
            batch_true_similarities.append(1.0)

            i1, i2 = self.dataset.draw_pair_cbs(False, self.group_id)
            batch_pairs_indices.append(i1)
            batch_pairs_indices.append(i2)

            # If configured a similarity value is used for the negative pair instead of full dissimilarity
            if self.config.use_sim_value_for_neg_pair:
                sim = self.dataset.get_sim_label_pair(i1, i2, 'train')
                batch_true_similarities.append(sim)
            else:
                batch_true_similarities.append(0.0)

        # Change the list of ground truth similarities to an array
        true_similarities = np.asarray(batch_true_similarities)

        return batch_pairs_indices, true_similarities

    def execute_early_stop(self, last_loss):
        if self.config.use_early_stopping:
            self.losses.append(last_loss)

            # Check if the loss of the last epoch is better than the best loss
            # If so reset the early stopping progress else continue approaching the limit
            if last_loss < self.best_loss:
                self.stopping_step_counter = 0
                self.best_loss = last_loss
            else:
                self.stopping_step_counter += 1

            # Check if the limit was reached
            if self.stopping_step_counter >= self.config.early_stopping_epochs_limit:
                return True
            else:
                return False
        else:
            # Always continue if early stopping should not be used
            return False
