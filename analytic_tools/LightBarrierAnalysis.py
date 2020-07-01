import os
import sys

import matplotlib.pyplot as plt
import pandas as pd

sys.path.append(os.path.abspath(os.path.join(os.getcwd(), os.pardir)))

from configuration.Configuration import Configuration
from data_processing.DataImport import import_txt


# transform a list of date strings to datetime timestamps
def transform(dates_as_string):
    dates_as_timestamps = []

    for date in dates_as_string:
        dates_as_timestamps.append(pd.to_datetime(date, format='%Y-%m-%d %H:%M:%S.%f'))

    return dates_as_timestamps


def plot_export_txt(df: pd.DataFrame, file_name: str, config, timestamps):
    print('Creating plot for', file_name)

    df = df.query(config.query)

    axes = df.plot(subplots=True, sharex=True, figsize=(60, 15), title=file_name)

    # specify date format and tick interval
    date_format = '%H:%M:%S'
    tick_interval = 1500

    # select ticks from index
    ticks_to_use = df.index[::tick_interval]

    # create labels by reformatting the selected indices
    labels = [i.strftime(date_format) for i in ticks_to_use]
    ax = axes[0]  # First subplot, alternative = ax = plt.gca()

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

    # plt.savefig(config.pathPrefix + 'plots/' + file_name, dpi=200)


# used for case determination of a light barrier failure dataset
def main():
    config = Configuration(13)

    df = import_txt(config.txt16, 'txt16')
    df = df[['txt16_i4', 'txt16_label']]

    # get timestamps of labels
    df_red = df.loc[df['txt16_label'] == 'TXT16_i4_failuremode1']
    print(df_red)

    # add timestamps of cases to display in the plot
    timestamps = [
        '2019-05-28 15:20:23.14', '2019-05-28 15:20:37.010',
        '2019-05-28 15:22:45.21', '2019-05-28 15:23:08.320',
        '2019-05-28 15:25:29.20', '2019-05-28 15:25:54.720',
        '2019-05-28 15:31:21.04', '2019-05-28 15:31:40.320',
        '2019-05-28 15:34:13.76', '2019-05-28 15:34:23.910',
        '2019-05-28 15:39:30.77', '2019-05-28 15:40:59.820',
    ]

    timestamps = transform(timestamps)

    plot_export_txt(df, 'txt16_i4_only', config, timestamps)


if __name__ == '__main__':
    main()
