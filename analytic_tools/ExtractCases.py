import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.append(os.path.abspath(os.path.join(os.getcwd(), os.pardir)))

from configuration.Configuration import Configuration
from data_processing.DataImport import import_txt


def export_cases_all(dfs: [(int, pd.DataFrame)], config: Configuration):
    with open(config.cases_folder + 'cases.csv', 'w') as file:

        # exporting in a format the can directly be pasted into the configuration file
        for element in dfs:
            dataset = element[0]
            df = element[1]
            for index, row in df.iterrows():
                failure_type = row['failure']
                start = row['ts_start'].strftime('%Y-%m-%d %H:%M:%S.%f')
                end = row['ts_end'].strftime('%Y-%m-%d %H:%M:%S.%f')
                case = ', '.join([df, failure_type, start, end])
                print(case)
                file.write(case + '\n')
            # old version
            # file.write('cases_dataset_' + str(dataset) + ' = [\n')
            #
            # for index, row in df.iterrows():
            #     failure_type = row['failure']
            #     start = row['ts_start'].strftime('%Y-%m-%d %H:%M:%S.%f')
            #     end = row['ts_end'].strftime('%Y-%m-%d %H:%M:%S.%f')
            #
            #     case = '\t(gen_timestamp(\'' + failure_type + '\',\'' + start + '\',\'' + end + '\')),'
            #     # print(case)
            #     file.write(case + '\n')
            # file.write(']\n\n')


# export cases to a text file in the right format
def export_cases_single_dataset(df: pd.DataFrame, config, file_name, data_set_number):
    with open(config.cases_folder + str(data_set_number) + '_' + file_name + '.txt', 'w') as file:
        for index, row in df.iterrows():
            failure_type = row['failure']
            start = row['ts_start'].strftime('%Y-%m-%d %H:%M:%S.%f')
            end = row['ts_end'].strftime('%Y-%m-%d %H:%M:%S.%f')

            case = '(gen_timestamp(\'' + failure_type + '\',\'' + start + '\',\'' + end + '\')),'
            # print(case)
            file.write(case + '\n')
    print(len(df), 'cases written to file.')


# returns the failure type of the passed row.
# if the motor is not active no vibrations occur and there is no failure to detect
# if it is active is classified as a type of wear depending on the rul value
def set_error_value(row, activity, error_type, config, start_value=np.inf):
    if 't2' in error_type:

        if row[activity] == 0:
            return 'no_failure'
        elif row['rul_start'] >= config.type2_start_rul + 1:
            return 'no_failure'
        else:
            return error_type + '_wear'

    elif 't1' in error_type:
        wear_type = '_low_wear' if config.split_t1_high_low else '_wear'

        if row[activity] == 0:
            return 'no_failure'
        elif start_value * config.type1_start_percentage <= row['rul_end']:
            return 'no_failure'
        elif config.split_t1_high_low and row['rul_end'] <= config.type1_high_wear_rul:
            return error_type + '_high_wear'
        else:
            return error_type + wear_type


def plot_export_txt(df: pd.DataFrame, file_name: str, config, df_cases: pd.DataFrame, data_set_number, error_type,
                    start_values):
    print('Creating plot for', file_name)

    axes = df.plot(subplots=True, sharex=True, figsize=(20, 10), title=file_name)

    # specify date format and tick interval
    date_format = '%H:%M:%S'
    tick_interval = 1500

    # select ticks from index
    ticks_to_use = df.index[::tick_interval]

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

    if error_type == 't1':

        # print horizontal lines where the vibrations of one of the simulations start
        # for value in start_values:
        #     plt.axhline(y=value * config.type1_start_percentage, linewidth=0.5, color='k', ls='--')

        for i in range(len(df_cases)):
            row = df_cases.iloc[i]

            e_type = row['failure']

            # display active intervals differently depending on the class (no_failure, high or normal wear)
            if e_type == 'no_failure':
                plt.axvline(x=row['ts_start'], linewidth=1, color='g')
                plt.axvline(x=row['ts_end'], linewidth=1, color='g')
            elif 'high' in e_type:
                plt.axvline(x=row['ts_start'], linewidth=1, color='b')
                plt.axvline(x=row['ts_end'], linewidth=1, color='b')
            else:
                plt.axvline(x=row['ts_start'], linewidth=1, color='r')
                plt.axvline(x=row['ts_end'], linewidth=1, color='r')

    elif error_type == 't2':

        # horizontal line to display the rul value from which the vibrations start
        plt.axhline(y=config.type2_start_rul, linewidth=1, color='k', ls='--')

        for i in range(len(df_cases)):
            row = df_cases.iloc[i]

            e_type = row['failure']

            # display active intervals differently depending on the class
            if e_type == 'no_failure':
                plt.axvline(x=row['ts_start'], linewidth=1.5, color='g', ls='--')
                plt.axvline(x=row['ts_end'], linewidth=1.5, color='g', ls='--')
            else:
                plt.axvline(x=row['ts_start'], linewidth=1.5, color='r')
                plt.axvline(x=row['ts_end'], linewidth=1.5, color='r')

    plt.savefig(config.datasets[data_set_number][0] + 'plots/' + file_name, dpi=300, bbox_inches="tight")

    plt.show()


def extract_single_sim(df, info):
    # extract the dataset information from the passed info tuple
    dataset_to_import = info[0]
    sensor = info[1]
    motor = info[2]
    error_type = info[3]

    config = Configuration(dataset_to_import)

    # compose column names and error based on dataset information
    activity = sensor + '_' + motor + '.speed'
    rul = sensor + '_' + motor + 'RUL'
    error = sensor + '_' + motor + '_' + error_type

    # transform the dataframe into a one only containing the intervals by the change of values of the motor speed
    # df_comb contains the start and stop timestamps and rul values
    df_start = df.loc[df[activity].shift(1) != df[activity]]
    df_start = df_start.reset_index(level=df_start.index.names)
    df_start = df_start.rename(index=str, columns={"timestamp": "ts_start", rul: 'rul_start'})

    df_end = df.loc[df[activity].shift(-1) != df[activity]]
    df_end = df_end.reset_index(level=df_end.index.names)
    df_end = df_end.rename(index=str, columns={"timestamp": "ts_end", rul: 'rul_end'})
    df_end = df_end.drop([activity], axis=1)

    df_comb = df_start.merge(df_end, on=df_start.index, how='outer')
    df_comb = df_comb.drop(['key_0'], axis=1)
    df_comb = df_comb.reindex(['ts_start', 'ts_end', activity, 'rul_start', 'rul_end'], axis=1)

    rul_start_value = df_comb['rul_start'][0]

    # set failure value depending on motor activation and rul value
    df_comb['failure'] = df_comb.apply(
        lambda row: set_error_value(row, activity, error, config, rul_start_value), axis=1)

    # print(df_comb)

    # calculate length of time intervals
    df_comb['delta'] = df_comb.apply(lambda row: (row['ts_end'] - row['ts_start']).total_seconds(), axis=1)

    return df_comb, rul_start_value


def extract_cases(info, plot):
    # extract the dataset information from the passed info tuple
    dataset_to_import = info[0]
    sensor = info[1]
    motor = info[2]
    error_type = info[3]
    file_name = info[4]

    config = Configuration(dataset_to_import)

    # compose column names and error based on dataset information
    activity = sensor + '_' + motor + '.speed'
    rul = sensor + '_' + motor + 'RUL'
    cols = [activity, rul]

    file = None
    if sensor == 'txt15':
        file = config.txt15
    elif sensor == 'txt16':
        file = config.txt16

    # import using the method for importing datasets
    df: pd.DataFrame = import_txt(file, sensor)
    df = df.query(config.query)
    df = df[cols]

    # split into single simulations
    simulations = [g for _, g in df.groupby((df[rul].diff() > 0).cumsum())]
    simulations_tranformed = []

    # if failure type 2 the datasets must further be split depending on the configured value when the vibrations start
    if error_type == 't2':
        simulations_split = []
        for sim in simulations:
            mask = sim[rul] >= config.type2_start_rul + 1
            above = sim[mask]
            below = sim[~mask]

            if len(above) > 0:
                simulations_split.append(above)
            if len(below) > 0:
                simulations_split.append(below)

        simulations = simulations_split

    start_values = []

    # split each simulations into the intervals it contains
    for sim in simulations:
        df_temp, start = extract_single_sim(sim, info)
        # safe start rul values for error type 1 where this is needed to determine when the vibrations start
        start_values.append(start)
        simulations_tranformed.append(df_temp)

    # combine the single dataframes for each simulation back into a single one
    df_comb = pd.concat(simulations_tranformed, ignore_index=True)

    print(df_comb)
    print('')
    # export_cases_single_dataset(df_comb, config, file_name, dataset_to_import)

    if plot:
        # only display the time intervals where the motor is active
        df_comb_temp = df_comb.loc[df_comb[activity] != 0]
        plot_export_txt(df, file_name, config, df_comb_temp, dataset_to_import, error_type, start_values)

    return dataset_to_import, df_comb


def main():
    # change options who a pandas dataframe is printed to console, the full dataframe should be printed
    pd.set_option('display.max_rows', 500)
    pd.set_option('display.max_columns', 500)
    pd.set_option('display.width', 1000)

    # enable plotting for each dataset
    plot = False

    config = Configuration()
    datasets_info = []

    # generate dataset information based on folder name
    for i in range(len(config.datasets)):
        file_name = config.datasets[i][0].split('/')[-2]
        file_name_parts = file_name.split('_')
        if file_name_parts[0] in ['txt15', 'txt16']:
            info = (i, file_name_parts[0], file_name_parts[1], file_name_parts[2], file_name)
            datasets_info.append(info)

    dfs = []
    for info in datasets_info:
        # extract the cases from a single dataset into a dataframe
        dfs.append(extract_cases(info, plot))

    # export to a single text file
    export_cases_all(dfs, config)


# script to automatically extract the case intervals from motor failure datasets
# folders of datasets must match the pattern:
# txtMODULE_mMOTORNUMBER_tERRORTYPE_pPART (currently only type 1 and 2 supported)
# for example: txt16_m3_t1_p1
if __name__ == '__main__':
    main()
