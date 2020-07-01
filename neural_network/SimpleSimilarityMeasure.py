import tensorflow as tf


# noinspection PyMethodMayBeStatic
class SimpleSimilarityMeasure:

    def __init__(self, sim_type):
        self.sim_type = sim_type

        self.a_weights = None
        self.b_weights = None
        self.a_context = None
        self.b_context = None
        self.w = None

        self.implemented = ['abs_mean', 'euclidean_sim', 'euclidean_dis', 'dot_product', 'cosine']
        assert sim_type in self.implemented

    @tf.function
    def get_sim(self, a, b, a_weights=None, b_weights=None, a_context=None, b_context=None, w=None):

        # assign to class variables so only common parameters must be passed below
        self.a_weights = a_weights
        self.b_weights = b_weights
        self.a_context = a_context
        self.b_context = b_context
        self.w = w

        switcher = {
            'abs_mean': self.abs_mean,
            'euclidean_sim': self.euclidean_sim,
            'euclidean_dis': self.euclidean_dis,
            'dot_product': self.dot_product,
            'cosine': self.cosine
        }

        # Get the function from switcher dictionary
        func = switcher.get(self.sim_type)
        return func(a, b)

    def get_weight_matrix(self, a):
        weight_matrix = tf.reshape(tf.tile(self.a_weights, [a.shape[0]]), [a.shape[0], a.shape[1]])
        a_weights_sum = tf.reduce_sum(weight_matrix)
        a_weights_sum = tf.add(a_weights_sum, tf.keras.backend.epsilon())
        weight_matrix = weight_matrix / a_weights_sum

        return weight_matrix

    # Siamese Deep Similarity as defined in NeuralWarp
    # Mean absolute difference of all time stamp combinations
    @tf.function
    def abs_mean(self, a, b):

        use_weighted_sim = self.a_weights is not None and self.b_weights is not None
        use_additional_sim = self.a_context is not None and self.b_context is not None

        if use_weighted_sim:
            # Note: only one weight vector is used (a_weights) to simulate a retrieval situation
            # where only weights of the case are known
            # tf.print(self.a_weights)
            # tf.print(self.w, output_stream=sys.stdout)
            weight_matrix = self.get_weight_matrix(a)

            diff = tf.abs(a - b)
            # feature weighted distance:
            distance = tf.reduce_mean(weight_matrix * diff)
            # tf. print("self.a_weights: ", tf.reduce_sum(self.a_weights))

            if use_additional_sim:
                # calculate context distance
                diff_con = tf.abs(self.a_context - self.b_context)
                distance_con = tf.reduce_mean(diff_con)
                if self.w is None:
                    self.w = 0.3
                    distance = self.w * distance + (1 - self.w) * distance_con
                    distance = tf.squeeze(distance)
                else:
                    # weight both distances
                    # tf.print("w: ",self.w)
                    distance = self.w * distance + (1 - self.w) * distance_con
                    distance = tf.squeeze(distance)
        else:
            diff = tf.abs(a - b)
            distance = tf.reduce_mean(diff)
        sim = tf.exp(-distance)

        return sim

    # Euclidean distance (required in contrastive loss function and converted sim)
    @tf.function
    def euclidean_dis(self, a, b):
        # cnn2d, [T,C]
        use_weighted_sim = self.a_weights is not None and self.b_weights is not None
        use_additional_sim = self.a_context is not None and self.b_context is not None

        if use_weighted_sim:
            # Note: only one weight vector is used (a_weights) to simulate a retrieval situation
            # where only weights of the case are known
            weight_matrix = self.get_weight_matrix(a)
            q = a - b
            weighted_dist = tf.sqrt(tf.reduce_sum(weight_matrix * q * q))
            diff = weighted_dist
            if use_additional_sim:
                # calculate context distance
                diff_con = tf.norm(self.a_context - self.b_context, ord='euclidean')
                distance_con = tf.reduce_mean(diff_con)
                # weight both distances
                distance = self.w * diff + (1 - self.w) * distance_con
                diff = tf.squeeze(distance)
        else:
            # tf.print("a: ", a)
            # tf.print("b: ", b)
            diff = tf.norm(a - b, ord='euclidean')
            # tf.print("diff: ", diff)

        return diff

    # Euclidean distance converted to a similarity
    @tf.function
    def euclidean_sim(self, a, b):

        diff = self.euclidean_dis(a, b)
        sim = 1 / (1 + tf.reduce_sum(diff))
        return sim

    # TODO Doesn't work with binary cross entropy loss, always leads to same loss
    #  Reason might be that this doesn't return a sim in [0,1]
    @tf.function
    def dot_product(self, a, b):
        sim = tf.matmul(a, b, transpose_b=True)
        return tf.reduce_mean(sim)

    # TODO Doesn't work with binary cross entropy loss, always leads to same loss
    #  Reason might be that this doesn't return a sim in [0,1]
    #  possibly this could be used: https://www.tensorflow.org/api_docs/python/tf/keras/losses/CosineSimilarity
    # source: https://bit.ly/390bDPQ
    @tf.function
    def cosine(self, a, b):
        use_weighted_sim = self.a_weights is not None and self.b_weights is not None
        if use_weighted_sim:
            # source: https://stats.stackexchange.com/questions/384419/weighted-cosine-similarity
            weight_vec = self.a_weights / tf.reduce_sum(self.a_weights)
            normalize_a = tf.nn.l2_normalize(a, 0) * weight_vec
            normalize_b = tf.nn.l2_normalize(b, 0) * weight_vec
            cos_similarity = tf.reduce_sum(tf.multiply(normalize_a, normalize_b) * weight_vec)
            # cos_similarity = 1-distance.cosine(a.numpy(),b.numpy(),self.a_weights)
        else:
            normalize_a = tf.nn.l2_normalize(a, 0)
            normalize_b = tf.nn.l2_normalize(b, 0)
            cos_similarity = tf.reduce_sum(tf.multiply(normalize_a, normalize_b))
            # tf.print(cos_similarity)

        return cos_similarity
