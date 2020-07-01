import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.getcwd(), os.pardir)))

from case_based_similarity.CaseBasedSimilarity import CBS
from configuration.Configuration import Configuration
from configuration.ConfigChecker import ConfigChecker
from neural_network.Dataset import CBSDataset
from neural_network.Optimizer import CBSOptimizer


def main():
    try:
        # suppress debugging messages of TensorFlow
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
        # os.environ['report_tensor_allocations_upon_oom'] = '1'
        config = Configuration()

        dataset = CBSDataset(config.training_data_folder, config, training=True)
        dataset.load()

        checker = ConfigChecker(config, dataset, 'cbs', training=True)
        checker.pre_init_checks()

        print('Initializing case based similarity measure ...\n')
        cbs = CBS(config, True, dataset)

        checker.post_init_checks(cbs)

        print('\nTraining:\n')
        optimizer = CBSOptimizer(cbs, dataset, config)
        optimizer.optimize()

    except KeyboardInterrupt:
        try:
            cbs.kill_threads()
        except:
            pass


if __name__ == '__main__':
    main()
