import json
import os
import sys

import pandas as pd

sys.path.append(os.path.abspath(os.path.join(os.getcwd(), os.pardir)))

from configuration.Configuration import Configuration
from data_processing.DataImport import import_single_pressure_sensor
from matplotlib import pyplot as plt


# transform a list of date strings to datetime timestamps
def transform(dates_as_string):
    dates_as_timestamps = []

    for date in dates_as_string:
        dates_as_timestamps.append(pd.to_datetime(date, format='%Y-%m-%d %H:%M:%S.%f'))

    return dates_as_timestamps


def plot_export_txt(df: pd.DataFrame, file_name: str, config, timestamps):
    print('Creating plot for', file_name)

    df = df.query(config.query)

    axes = df.plot(subplots=True, sharex=True, figsize=(100, 20), title=file_name)

    # specify date format and tick interval
    date_format = '%H:%M:%S'
    tick_intervall = 1000

    # select ticks from index
    ticks_to_use = df.index[::tick_intervall]

    # create labels by reformatting the selected indices
    labels = [i.strftime(date_format) for i in ticks_to_use]
    ax = axes[0]  # first subplot, alternative = ax = plt.gca()

    # apply indices and matching labels to the plot
    ax.set_xticks(ticks_to_use)
    ax.set_xticklabels(labels, rotation=45)

    # rename x axis
    plt.xlabel('Zeitpunkt')

    # disable minor ticks that would interfere with new ticks
    plt.minorticks_off()

    for i in range(len(timestamps)):
        c = 'g' if i % 2 == 0 else 'r'
        plt.axvline(x=timestamps[i], linewidth=1, color=c, ls='--')
        # plt.axvline(x=row['ts_end'], linewidth=0.5, color='r', ls='--')

    plt.show()


def main():
    config = Configuration(14)

    with open(config.topicPressureSensorsFile) as f:
        content = json.load(f)

    selection = 2

    if selection == 0:
        post = '15'
        df_sensor_data = import_single_pressure_sensor(content, 'Sorter', post)
    elif selection == 1:
        post = '17'
        df_sensor_data = import_single_pressure_sensor(content, 'Oven', post)
    elif selection == 2:
        post = '18'
        df_sensor_data = import_single_pressure_sensor(content, 'VSG', post)
    else:
        raise ValueError()

    # change format of timestamp, set it as index and reduce the time interval
    df_sensor_data['timestamp'] = pd.to_datetime(df_sensor_data['timestamp'])
    df_sensor_data = df_sensor_data.set_index(df_sensor_data['timestamp'])
    df_sensor_data.query(config.query, inplace=True)
    df_sensor_data.drop('timestamp', 1, inplace=True)
    df_sensor_data.drop('tC_' + post, 1, inplace=True)

    # get timestamps of labels
    # df_red = df_sensor_data.loc[df_sensor_data['txt16_label'] == 'TXT16_i4_failuremode1']
    # print(df_red)

    # add timestamps of cases to display in the plot
    if selection == 2:

        timestamps = [
            '2019-06-08 11:42:00.080158', '2019-06-08 11:49:00.082672',
            # '2019-06-08 11:50:45.008662', '2019-06-08 11:56:19.046031',
            '2019-06-08 11:56:5.046031', '2019-06-08 11:58:55.925578',
        ]

    elif selection == 1:
        timestamps = [
            # '2019-06-08 11:59:58.004812', '2019-06-08 12:01:00.974244', no error visible irl and in data
            '2019-06-08 12:13:07.064066', '2019-06-08 12:13:38.012230',
            '2019-06-08 12:16:57.061182', '2019-06-08 12:18:36.992235',
            '2019-06-08 12:20:15.060076', '2019-06-08 12:21:13.975116'
        ]

    elif selection == 0:
        timestamps = [

        ]
    else:
        raise ValueError()

    timestamps = transform(timestamps)

    plot_export_txt(df_sensor_data, 'pressure failure', config, timestamps)


# used for case determination of a pressure sensor failure dataset
if __name__ == '__main__':
    main()
