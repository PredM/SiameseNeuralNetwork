import fnmatch
import sys
import os
import contextlib
import json
import threading
import time
import multiprocessing
import joblib
import numpy as np
import pandas as pd

# noinspection PyProtectedMember
from kafka import KafkaConsumer, TopicPartition, KafkaProducer, errors
from sklearn.preprocessing import MinMaxScaler

from case_based_similarity.CaseBasedSimilarity import CBS
from neural_network.SNN import initialise_snn

sys.path.append(os.path.abspath(os.path.join(os.getcwd(), os.pardir)))

from configuration.Configuration import Configuration
from data_processing.DataframeCleaning import clean_up_dataframe
from neural_network.Dataset import FullDataset


class Importer(threading.Thread):

    def __init__(self, config: Configuration, consumer: KafkaConsumer, topic: str, end_timestamp: pd.Timestamp):
        super().__init__()

        self.config = config
        self.consumer = consumer
        self.topic = topic  # the name of the topic the consumer is subscribed to
        self.end_ts = end_timestamp  # timestamp until which messages should be read from kafka
        self.result = []  # list of extracted messages

        # set topic type, which determines the import method that needs to be used
        self.topic_type = -1

        if topic in self.config.txt_topics:
            self.topic_type = 0
        elif topic in self.config.acc_topics:
            self.topic_type = 1
        elif topic in self.config.bmx_acc_topics:
            self.topic_type = 2
        elif topic in self.config.pressure_topics:
            self.topic_type = 3

    def run(self):
        extracted_msgs = []
        end_ts_reached: bool = False

        # read and process new messages until the messages reach the end timestamp
        for msg in self.consumer:
            value = msg.value
            topic_in_msg = msg.topic

            # depending on the topic the message was received from the message must be split up
            # in different ways to single timestamps
            if self.topic_type == 0:
                end_ts_reached = Importer.extract_txt(value, extracted_msgs, topic_in_msg, self.end_ts, self.config)
            elif self.topic_type == 1:
                end_ts_reached = Importer.extract_acc(value, extracted_msgs, topic_in_msg, self.end_ts, self.config)
            elif self.topic_type == 2:
                end_ts_reached = Importer.extract_bmx_acc(value, extracted_msgs, topic_in_msg, self.end_ts, self.config)
            elif self.topic_type == 3:
                end_ts_reached = Importer.extract_pres_sensor(value, extracted_msgs, self.end_ts, self.config)
            if end_ts_reached:
                break

        self.result = extracted_msgs

    @staticmethod
    def extract_acc(value, extracted_messages, topic, end_timestamp: pd.Timestamp, config: Configuration):
        # get the prefix for this topic
        prefix = config.prefixes[topic]

        # extract the single messages from the kafka message value
        # and rename the single streams with the right prefix
        for msg in value:
            msg[prefix + '_x'] = msg.pop('x')
            msg[prefix + '_y'] = msg.pop('y')
            msg[prefix + '_z'] = msg.pop('z')

            # partly different naming
            if 'timestamp' not in msg.keys():
                msg['timestamp'] = msg.pop('time')

            # Append to the list of all messages
            extracted_messages.append(msg)

        # check if the last single message exceeds the end_timestamp
        msg = extracted_messages[-1]
        return pd.to_datetime(msg['timestamp']) > end_timestamp

    @staticmethod
    def extract_bmx_acc(value, extracted_messages, topic, end_timestamp: pd.Timestamp, config: Configuration):
        # get the prefix for this topic
        prefix = config.prefixes[topic]

        # extract the single messages from the kafka message value
        # and rename the single streams with the right prefix
        for msg in value:
            msg[prefix + '_x'] = msg.pop('x')
            msg[prefix + '_y'] = msg.pop('y')
            msg[prefix + '_z'] = msg.pop('z')
            msg[prefix + '_t'] = msg.pop('t')

            # partly different naming
            if 'timestamp' not in msg.keys():
                msg['timestamp'] = msg.pop('time')

            # Append to the list of all messages
            extracted_messages.append(msg)

        # check if the last single message exceeds the end_timestamp
        msg = extracted_messages[-1]
        return pd.to_datetime(msg['timestamp']) > end_timestamp

    @staticmethod
    def extract_pres_sensor(value, extracted_messages, end_timestamp: pd.Timestamp, config):
        sensor_names = config.pressure_sensor_names

        if len(sensor_names) < 1:
            return

        # each message value contains three submessages that are tagged with the right prefix
        # these dont have to be extracted into single messages because they share the same timestamp
        msg = value[sensor_names[0]]
        suffix = config.prefixes[sensor_names[0]]
        msg['hPa_' + suffix] = msg.pop('hPa')
        msg.pop('tC')  # msg['tC_' + suffix] = msg.pop('tC') # Not used, removed for performance optimisation
        msg['timestamp'] = value['meta']['time']

        for i in range(1, len(sensor_names)):
            c = value[sensor_names[i]]
            suffix = config.prefixes[sensor_names[i]]
            msg['hPa_' + suffix] = c.pop('hPa')
            # msg['tC_' + suffix] = c.pop('tC') # Not used, removed for performance optimisation

        extracted_messages.append(msg)

        # check if the message exceeds the end_timestamp
        return pd.to_datetime(msg['timestamp']) > end_timestamp

    @staticmethod
    def extract_txt(msg: dict, extracted_messages, topic, end_timestamp: pd.Timestamp, config: Configuration):
        # get the prefix for this topic
        prefix_topic = config.prefixes[topic]

        # special case for txt controller 18 which has a sub message containing the position of the crane
        if 'currentPos' in msg.keys():
            # split position column into 3 columns containing the x,y,z position
            pos = dict(eval(msg.pop('currentPos')))
            msg['vsg_x'] = pos['x']
            msg['vsg_y'] = pos['y']
            msg['vsg_z'] = pos['z']

        # timestamp should not be tagged with the prefix
        keys = list(msg.keys())
        keys.remove('timestamp')

        # tag each stream with the right prefix
        for key in keys:
            new_key = prefix_topic + '_' + key
            msg[new_key] = msg.pop(key)

        extracted_messages.append(msg)

        # check if the message exceeds the end_timestamp
        return pd.to_datetime(msg['timestamp']) > end_timestamp


def read_single_example(consumers: [KafkaConsumer], limiting_consumer: KafkaConsumer, config: Configuration):
    # get the next message and its timestamp from the limiting topic, which determines the time interval
    # of messages to read from kafka
    start_message = next(limiting_consumer)
    start_timestamp = pd.to_datetime(start_message.value['timestamp'])
    end_timestamp = start_timestamp + pd.Timedelta(seconds=config.interval_in_seconds)

    results = []
    interval_consumers = []

    # process start message, currently only works for txt topics as limiting topic
    temp_for_first = []
    Importer.extract_txt(start_message.value, temp_for_first, config.limiting_topic, end_timestamp, config)

    # create a interval consumer for each topic that imports the messages of the corresponding topic
    for i in range(len(consumers)):
        interval_consumer = Importer(config, consumers[i], config.topic_list[i], end_timestamp)
        interval_consumer.start()
        interval_consumers.append(interval_consumer)

    # wait until all interval consumers finished extracting messages from their topic
    for interval_consumer in interval_consumers:
        interval_consumer.join()

    # add the results of all interval consumers to a list
    for interval_consumer in interval_consumers:

        # add the first message to the others of the same topic
        if interval_consumer.topic == config.limiting_topic:
            interval_consumer.result.extend(temp_for_first)

        results.append(interval_consumer.result)

    return results


def list_to_dataframe(results: [[object]], config: Configuration):
    # for each topic create a data frame from the result list containing the  extracted messages
    dfs = []

    for m_list in results:
        # combine list of json objects into a single json object
        json_dump = json.dumps(m_list)

        # create a dataframe from the json object and convert the timestamps into a datetime format
        df_temp: pd.DataFrame = pd.read_json(json_dump)
        df_temp['timestamp'] = pd.to_datetime(df_temp['timestamp'], format='%Y-%m-%d %H:%M:%S.%f')

        # remove unnecessary columns
        # errors for non existing columns are ignored because not all datasets have the same
        try:
            df_temp = df_temp[config.feature_variant]
        except:
            raise AttributeError('Relevant feature not found in current dataset.')

        # remove duplicated timestamps, first will be kept
        df_temp = df_temp.loc[~df_temp['timestamp'].duplicated(keep='first')]

        dfs.append(df_temp)

    # merge the dataframes of all topics
    df = dfs.pop(0)
    for df_temp in dfs:
        df = df.merge(df_temp, on='timestamp', how='outer')

    # set timestamp as index and sort rows by it
    # df['timestamp'] = pd.to_datetime(df['timestamp'], format='%Y-%m-%d %H:%M:%S.%f') # unnecessary
    df = df.set_index('timestamp', drop=True)
    df = df.sort_index()

    # sort columns by name to ensure the same order as in the training data
    df = df.reindex(sorted(df.columns), axis=1)

    # clean up dataframe,
    # with ... is used to suppress output that would occur
    # because the data preprocessing method for training data is used
    with contextlib.redirect_stdout(None):
        df = clean_up_dataframe(df, config)

    # reduce dataframe to time_series_length many values
    start_timestamp = df.index[0]
    end_timestamp = df.index[0] + pd.Timedelta(seconds=config.interval_in_seconds)
    df = reduce_dataframe(df, start_timestamp, end_timestamp, config.time_series_length)

    return df


def reduce_dataframe(df: pd.DataFrame, start_timestamp: pd.Timestamp, end_timestamp: pd.Timestamp, time_series_length):
    # get the corresponding numerical index in the dataframe, if not found get the nearest one
    start = df.index.get_loc(start_timestamp)
    stop = df.index.get_loc(end_timestamp, method='nearest')

    # get time_series_length many indices with near equal distance in the interval
    samples = np.linspace(start, stop, time_series_length, dtype=int).tolist()

    # reduce the dataframe to the calculated indices
    return df.iloc[samples]


def normalise_dataframe(example: np.ndarray, scalers: [MinMaxScaler]):
    length = example.shape[1]

    if len(example[0]) != len(scalers):
        raise ValueError(
            'number of scalers =/= number of features in example. check if feature configuration is correct')

    for i in range(length):
        # scaler = MinMaxScaler(feature_range=(0, 1))
        scaler = scalers[i]
        # reshape column vector, normalize and assign values in example
        temp = example[:, i].reshape(-1, 1)
        temp = scaler.transform(temp)
        example[:, i] = temp.reshape(1, -1)

    return example


def load_scalers(config):
    scalers = []

    # calculate the number of columns
    number_of_scalers = len(fnmatch.filter(os.listdir(config.scaler_folder), '*.save'))

    # load scaler for each attribute and store in list
    for i in range(number_of_scalers):
        scaler_filename = config.scaler_folder + 'scaler_' + str(i) + '.save'
        scaler = joblib.load(scaler_filename)
        scalers.append(scaler)

    return scalers


class Classifier(threading.Thread):

    def __init__(self, config: Configuration, selection):
        super().__init__()
        self.examples_to_classify = multiprocessing.Manager().Queue(10)
        self.config = config
        self.stop = False

        self.nbr_classified = 0
        self.total_time_classification = 0
        self.total_time_all = 0
        self.total_diff = pd.Timedelta(0)

        if config.export_results_to_kafka:
            self.result_producer = KafkaProducer(bootstrap_servers=config.get_connection(),
                                                 value_serializer=lambda m: json.dumps(m).encode('utf-8'))

        self.architecture = None
        self.init_architecture(selection)

    def init_architecture(self, selection):

        if selection == 'snn':
            dataset: FullDataset = FullDataset(self.config.training_data_folder, self.config, training=False)
            dataset.load()
            self.architecture = initialise_snn(self.config, dataset, False)
        elif selection == 'cbs':
            self.architecture = CBS(self.config, False)
        else:
            raise ValueError('Unknown architecture variant')

    def run(self):

        try:
            # classify element of the queue as long as the stop flag is not set by a interrupt
            while not self.stop:
                classification_start = time.perf_counter()

                # get the next element in queue, wait if empty
                element = self.examples_to_classify.get(block=True)

                # extract information from the queue element
                example = element[0]
                time_start = element[1].strftime('%H:%M:%S')
                time_end = element[2].strftime('%H:%M:%S')
                start_time = element[3]

                # classify the example using knn and the neural network
                label, mean_sim = self.knn(example)

                error_description = self.config.get_error_description(label)

                if self.config.export_results_to_kafka and not self.stop:
                    results = {'class_label': label,
                               'description': error_description,
                               'mean_similarity_of_k_best': mean_sim,
                               'time_interval_start': time_start,
                               'time_interval_end': time_end}
                    self.result_producer.send(self.config.export_topic, results)

                classification_time = time.perf_counter() - classification_start
                total_time_for_example = time.perf_counter() - start_time
                time_span = (pd.Timestamp.now() - element[2])

                self.nbr_classified += 1
                self.total_time_classification += classification_time
                self.total_time_all += total_time_for_example
                self.total_diff += time_span

                if not self.stop:
                    print('Classification result for the time interval from', time_start, 'to', time_end + ':')

                    table_data = [
                        ['\tLabel:', label],
                        ['\tDescription:', error_description],
                        ['\tMean similarity:', '{0:.4f}'.format(mean_sim)],
                        ['\tClassification time:', '{0:.4f}'.format(classification_time)],
                        ['\tTotal time:', '{0:.4f}'.format(total_time_for_example)],
                        ['\tTime span:', '{0:.4f}'.format(time_span.total_seconds())],
                        ['\tMean classification time:',
                         '{0:.4f}'.format(self.total_time_classification / self.nbr_classified)],
                        ['\tMean total time:', '{0:.4f}'.format(self.total_time_all / self.nbr_classified)],
                        ['\tMean time span:',
                         '{0:.4f}'.format(self.total_diff.total_seconds() / self.nbr_classified)],
                        ['\tExamples left in queue:', self.examples_to_classify.qsize()]
                    ]

                    for row in table_data:
                        print("{: <28} {: <28}".format(*row))
                    print('')

        except BrokenPipeError or EOFError:
            self.stop = True

    # k nearest neighbor implementation to select the class based on the k most similar training examples
    def knn(self, example: np.ndarray):

        # calculate the similarities to all examples of the case base using the the similarity measure
        # example will be encoded by snn variant if necessary
        sims, labels = self.architecture.get_sims(example)

        # argsort is ascending only, but descending is needed
        # so 'invert' the array content first with - (element wise operation)
        # argsort returns the indices that would sort an array
        arg_sort = np.argsort(-sims)

        # get the indices of the k most similar examples
        k = self.config.k_of_knn if self.config.k_of_knn <= len(arg_sort) else len(arg_sort)
        arg_sort = arg_sort[0:k]

        # get the classes and similarities for those examples
        classes = labels[arg_sort]
        sims = sims[arg_sort]

        # calculate the mean similarity
        mean_sim_of_k = np.mean(sims)

        # select most common class
        unique, counts = np.unique(classes, return_counts=True)
        count_sort_ind = np.argsort(-counts)
        unique_sorted_by_count = unique[count_sort_ind]

        return unique_sorted_by_count[0], mean_sim_of_k


def main():
    config = Configuration()

    # suppress debugging messages of tensorflow
    # os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

    # load the scalers of the training data for the normalisation
    scalers = load_scalers(config)

    consumers = []
    limiting_consumer = None

    selection = ''
    while selection not in ['cbs', 'snn']:
        print('Please select architecture that should be used. Type "snn" or "cbs"')
        selection = input()
    print()

    print('Creating consumers ...\n')

    # if using the fabric simulation start at the start of the topics
    # for live classification start at newest messages possible
    offset = 'earliest' if config.testing_using_fabric_sim else 'latest'

    try:
        # create consumers for all topics
        for topic in config.topic_list:
            c = KafkaConsumer(topic, bootstrap_servers=config.get_connection(),
                              value_deserializer=lambda m: json.loads(m.decode('utf-8')),
                              auto_offset_reset=offset)

            # based on the topic select one of the consumers for time interval determination
            if topic == config.limiting_topic:
                limiting_consumer = c

            consumers.append(c)
    except errors.NoBrokersAvailable:
        print('Configured kafka server is not available. Please check the connection or change the configuration.')
        sys.exit(0)

    # create and start a classifier thread that handles the classification of processed examples
    print('\nCreating classifier ...')
    print('\nUsed model file:')
    print(config.directory_model_to_use, '\n')

    print('The classifier will use k=' + str(config.k_of_knn) + ' for the k-NN algorithm')
    print('The mean similarity output is calculated on the basis of the k most similar cases')
    print('The time span is the time between the end timestamp of the')
    print('interval and the current time right before the output.')
    print('The total time is the time needed for the completely processing the example,')
    print('including the time in the queue.\n')
    classifier = Classifier(config, selection)
    classifier.start()

    print('Waiting for data to classify ...\n')
    try:

        # classify as until interrupted
        while 1:
            start_time = time.perf_counter()
            # read data for a single example from kafka, results contains lists of single messages
            results = read_single_example(consumers, limiting_consumer, config)

            # combine into a single dataframe
            df = list_to_dataframe(results, config)

            # transform dataframe into a array that can be used as neural network input
            example = df.to_numpy()

            # normalize the data of the example
            example = normalise_dataframe(example, scalers)

            # create a queue element containing
            element = (example, df.index[0], df.index[-1], start_time)

            # add element to the queue of examples to classify
            classifier.examples_to_classify.put(element)

            # reset all consumer offsets by two messages to reduce the time intervals that are left out
            for i in range(len(consumers)):
                partition = TopicPartition(config.topic_list[i], 0)
                last_offset = consumers[i].position(partition)
                new_offset = last_offset - 2 if last_offset - 2 >= 0 else 0
                consumers[i].seek(partition, new_offset)

    except KeyboardInterrupt:
        # interrupt the classifier thread
        print('Exiting ...\n')
        classifier.stop = True


# python script that handles the live classification
if __name__ == '__main__':
    main()
