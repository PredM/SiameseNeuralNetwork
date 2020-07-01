from case_based_similarity.CaseBasedSimilarity import CBS
from configuration.Configuration import Configuration
from neural_network.SNN import SimpleSNN


class ConfigChecker:

    def __init__(self, config: Configuration, dataset, architecture, training):
        self.config: Configuration = config
        self.dataset = dataset
        self.architecture_type = architecture
        self.training = training
        self.list_of_warnings = []

    @staticmethod
    def implication(p, q, error):
        # p --> q == !p or q
        assert not p or q, error

    # Can be used to define forbidden / impossible parameter configurations
    # and to output corresponding error messages if they are set in this way.
    def pre_init_checks(self):
        assert self.architecture_type in ['snn', 'cbs', 'preprocessing'], 'invalid architecture passed to configChecker'

        ##
        # SNN
        ##

        ConfigChecker.implication(self.config.simple_measure == 'euclidean_dis',
                                  self.config.type_of_loss_function == 'constrative_loss',
                                  'euclidean_dis should only be used for training with constrative loss.')

        ##
        # CBS
        ##

        ConfigChecker.implication(self.architecture_type == 'cbs',
                                  not self.config.individual_relevant_feature_selection,
                                  'For the CBS the group based feature selection must be used. '
                                  'Set individual_relevant_feature_selection to False.')

        ConfigChecker.implication(self.architecture_type == 'cbs', self.config.feature_variant == 'cbs_features',
                                  'Please use feature_variant == \'cbs_features\' for CBS models.')

        ##
        # Preprocessing
        ##
        ConfigChecker.implication(self.architecture_type == 'preprocessing',
                                  self.config.feature_variant == 'all_features',
                                  'For preprocessing data and dataset generation feature_variant == \'all_features\' '
                                  'should be used. Should contain a superset of the cbs features.')

        if self.architecture_type == 'preprocessing':
            self.warnings()

    @staticmethod
    def print_warnings(warnings):
        print('##########################################')
        print('WARNINGS:')
        for warning in warnings:
            if type(warning) == str:
                print('-  ' + warning)
            elif type(warning) == list:
                print('-  ' + warning.pop(0))
                for string in warning:
                    print('   ' + string)
        print('##########################################')
        print()

    # Add entries for which the configuration is valid but may lead to errors or unexpected behaviour
    def warnings(self):

        # Add new entries below this line

        if self.training and 'fast' in self.config.architecture_variant:
            self.list_of_warnings.append([
                'The fast version can only be used for inference.',
                'The training routine will use the standard version, otherwise the encoding',
                'would have to be recalculated after each iteration anyway.'
            ])

        if not self.config.use_hyper_file:
            self.list_of_warnings.append('Hyperparameters shouldn\'t be read from file. '
                                         'Ensure entries in Hyperparameters.py are correct.')

        if not self.config.split_sim_calculation and not self.training:
            self.list_of_warnings.append('Batchwise similarity calculation is disabled. '
                                         'If any errors occur, the first step should be to try '
                                         'and activate split_sim_calculation or lower sim_calculation_batch_size.')

        ignored_by_ffnn = [self.config.normalize_snn_encoder_output,
                           self.config.use_time_step_wise_simple_similarity, ]

        if 'ffnn' in self.config.architecture_variant and any(ignored_by_ffnn):
            self.list_of_warnings.append([
                'FFNN architecture ignores the following configurations:',
                'normalize_snn_encoder_output, use_time_step_wise_simple_similarity, use_time_step_matching_simple_similarity',
                'At least one is set to true.'])

        # Add new entries before this line

        if len(self.list_of_warnings) > 0:
            self.print_warnings(self.list_of_warnings)

    def post_init_checks(self, architecture):
        if self.architecture_type == 'snn':
            architecture: SimpleSNN = architecture

            self.implication('ffnn' in self.config.architecture_variant,
                             architecture.hyper.fc_after_cnn1d_layers is None,
                             'Additional fully connected layers shouldn\'t be used with FFNN. '
                             'fc_after_cnn1d_layers list should be empty.')

        elif self.architecture_type == 'cbs':
            architecture: CBS = architecture

            one_not_none = False
            for gh in architecture.group_handlers:
                self.implication('ffnn' in self.config.architecture_variant,
                                 gh.model.hyper.fc_after_cnn1d_layers is None,
                                 'Additional fully connected layers shouldn\'t be used with FFNN. '
                                 'fc_after_cnn1d_layers list should be empty.')

        self.warnings()
