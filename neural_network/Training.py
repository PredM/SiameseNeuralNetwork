import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.getcwd(), os.pardir)))

from configuration.ConfigChecker import ConfigChecker
from configuration.Configuration import Configuration
from neural_network.Dataset import FullDataset
from neural_network.Optimizer import SNNOptimizer
from neural_network.SNN import initialise_snn


def main():
    # suppress debugging messages of TensorFlow
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

    config = Configuration()

    dataset = FullDataset(config.training_data_folder, config, training=True)
    dataset.load()

    checker = ConfigChecker(config, dataset, 'snn', training=True)
    checker.pre_init_checks()

    snn = initialise_snn(config, dataset, True)
    snn.print_detailed_model_info()

    checker.post_init_checks(snn)

    print('Training:')
    optimizer = SNNOptimizer(snn, dataset, config)
    optimizer.optimize()


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        pass
