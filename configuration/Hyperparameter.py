import json


class Hyperparameters:

    def __init__(self):

        ##
        # Important: Variable names must match json file entries
        ##

        self.encoder_variants = ['cnn', 'rnn', 'cnn2dwithaddinput', 'cnn2d']
        self.encoder_variant = None

        # Need to be changed after dataset was loaded
        self.time_series_length = None
        self.time_series_depth = None

        self.batch_size = None
        self.epochs = None

        # will store the current epoch when saving a model to continue at this epoch
        self.epochs_current = None

        self.learning_rate = None

        # if gradient capping shouldn't be used this variable must be < 0
        self.gradient_cap = None

        self.dropout_rate = None

        self.ffnn_layers = None

        self.cnn_layers = None
        self.cnn_kernel_length = None
        self.cnn_strides = None

        self.cnn2d_layers = None
        self.cnn2d_kernel_length = None
        self.cnn2d_strides = None

        self.fc_after_cnn1d_layers = None

        self.lstm_layers = None

        self.tcn_layers = None
        self.tcn_kernel_length = None

        self.abcnn1 = None
        self.use1dContext = None
        self.useAttributeWiseAggregation = None
        self.cnn2d_AttributeWiseAggregation = None
        self.useAddContextForSim = None
        self.useAddContextForSim_LearnOrFixWeightVale = None
        self.cnn2d_contextModule = None
        self.cnn2d_learnWeightForContextUsedInSim = None
        self.learnFeatureWeights = None
        self.use_weighted_distance_as_standard_ffnn_hyper = None
        self.use_additional_strict_masking = None

    def set_time_series_properties(self, dataset):
        self.time_series_length = dataset.time_series_length
        self.time_series_depth = dataset.time_series_depth

    # Allows the import of a hyper parameter configuration from a json file
    def load_from_file(self, file_path, use_hyper_file=True):

        if not use_hyper_file:
            return

        file_path = file_path + '.json' if not file_path.endswith('.json') else file_path

        with open(file_path, 'r') as f:
            data = json.load(f)

        self.batch_size = data['batch_size']
        self.epochs = data['epochs']
        self.epochs_current = data['epochs_current']

        self.dropout_rate = data['dropout_rate']
        self.learning_rate = data['learning_rate']

        self.gradient_cap = data['gradient_cap']
        if self.gradient_cap is None:
            self.gradient_cap = -1

        # Load assign layers for ffnn as similarity measure if declared
        if data.get('ffnn_layers') is not None:
            self.ffnn_layers = data['ffnn_layers']

        self.encoder_variant = data['encoder_variant'].lower()

        if self.encoder_variant not in self.encoder_variants:
            raise ValueError('Unknown encoder variant:', self.encoder_variants)

        # Load encoder details depending on which encoder variant is specified
        if self.encoder_variant == 'rnn':
            self.lstm_layers = data['lstm_layers']

        if self.encoder_variant == 'tcn':
            self.tcn_layers = data['tcn_layers']
            self.tcn_kernel_length = data['tcn_kernel_length']

        if 'cnn' in self.encoder_variant:
            self.cnn_layers = data['cnn_layers']
            self.cnn_kernel_length = data['cnn_kernel_length']
            self.cnn_strides = data['cnn_strides']

            fc_layer = data.get('fc_after_cnn1d_layers')
            if fc_layer is not None and len(fc_layer) > 0:
                self.fc_after_cnn1d_layers = fc_layer

        if self.encoder_variant in ["cnn2d", "cnn2dwithaddinput"]:
            self.cnn2d_layers = data['cnn2d_layers']
            self.cnn2d_kernel_length = data['cnn2d_kernel_length']
            self.cnn2d_strides = data['cnn2d_strides']

        if self.encoder_variant in ["cnn2dwithaddinput"]:
            self.useAttributeWiseAggregation = data['useAttributeWiseAggregation']
            self.cnn2d_AttributeWiseAggregation = data['cnn2d_AttributeWiseAggregation']
            self.cnn2d_contextModule = data['cnn2d_contextModule']
            self.cnn2d_learnWeightForContextUsedInSim = data['cnn2d_learnWeightForContextUsedInSim']
            self.use_additional_strict_masking = data['use_additional_strict_masking']

            if data['useAddContextForSim'] == 'True':
                self.useAddContextForSim = 'True'
            else:
                self.useAddContextForSim = 'False'
            self.useAddContextForSim_LearnOrFixWeightVale = data['useAddContextForSim_LearnOrFixWeightVale']

    def write_to_file(self, path_to_file):

        # Creates a dictionary of all class variables and their values
        dict_of_vars = {key: value for key, value in self.__dict__.items() if
                        not key.startswith('__') and not callable(key)}

        with open(path_to_file, 'w') as outfile:
            json.dump(dict_of_vars, outfile, indent=4)

    def print_hyperparameters(self):
        dict_of_vars = {key: value for key, value in self.__dict__.items() if
                        not key.startswith('__') and not callable(key)}

        print(dict_of_vars)
