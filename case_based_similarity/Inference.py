import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.getcwd(), os.pardir)))

from case_based_similarity.CaseBasedSimilarity import CBS
from configuration.ConfigChecker import ConfigChecker
from configuration.Configuration import Configuration
from neural_network.Dataset import CBSDataset
from neural_network.Inference import Inference


# only initialisation must be changed, inference process of the snn can be used
def main():
    try:
        # suppress debugging messages of TensorFlow
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

        config = Configuration()

        if config.case_base_for_inference:
            dataset: CBSDataset = CBSDataset(config.case_base_folder, config, training=False)
        else:
            dataset: CBSDataset = CBSDataset(config.training_data_folder, config, training=False)

        dataset.load()

        checker = ConfigChecker(config, dataset, 'cbs', training=False)
        checker.pre_init_checks()

        cbs = CBS(config, False, dataset)
        inference = Inference(config, cbs, dataset)

        checker.post_init_checks(cbs)

        print('Ensure right model file is used:')
        print(config.directory_model_to_use, '\n')

        inference.infer_test_dataset()
        cbs.kill_threads()

    except KeyboardInterrupt:
        try:
            cbs.kill_threads()
        except:
            pass


if __name__ == '__main__':
    main()
