import json
import os
import sys
import threading
import time

sys.path.append(os.path.abspath(os.path.join(os.getcwd(), os.pardir)))

from kafka import KafkaProducer

from configuration.Configuration import Configuration


class ProducerSimulator(threading.Thread):

    def __init__(self, name, topic_name: str, file_name, config: Configuration):
        super().__init__(name=name)
        self.topic_name = topic_name

        file_path = config.pathPrefix + 'raw_data/' + file_name
        with open(file_path) as f:
            content = json.load(f)
            self.content = content

        self.length = len(self.content)
        self.index = 0
        self.producer = KafkaProducer(bootstrap_servers='localhost:9092',
                                      value_serializer=lambda v: json.dumps(v).encode('utf-8'))

    def run(self):
        while self.index < self.length:
            self.send_message(self.content[self.index])
            self.index += 1
            # timer = random.random() * 2
            # print(self.name + ' sleeping for ', timer)
            # time.sleep(timer)
        print('Thread', self.name, 'for', self.topic_name, 'done.')

    def send_message(self, msg):
        self.producer.send(self.topic_name, value=msg)


def main():
    config = Configuration(9)

    topic_list = config.topic_list

    topic_to_file = {
        'txt15': 'txt15.txt',
        'txt16': 'txt16.txt',
        'txt17': 'txt17.txt',
        'txt18': 'txt18.txt',
        'txt19': 'txt19.txt',
        'adxl1': 'TXT15_m1_acc.txt',
        'adxl0': 'TXT15_o8Compressor_acc.txt',
        'adxl3': 'TXT16_m3_acc.txt',
        'adxl2': 'TXT18_m1_acc.txt',
        'pressureSensors': 'pressureSensors.txt'
    }

    timer_start = time.perf_counter()

    for i in range(len(topic_list)):
        sim = ProducerSimulator(name='t' + str(i), topic_name=topic_list[i], file_name=topic_to_file[topic_list[i]],
                                config=config)
        sim.start()

    timer_end = time.perf_counter()

    print(timer_end - timer_start)


# script to simulate a running production simulation inserting messages to the kafka topics
if __name__ == '__main__':
    main()
