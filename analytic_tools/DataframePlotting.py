import gc
import os
import sys

import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd
from pandas.plotting import register_matplotlib_converters

sys.path.append(os.path.abspath(os.path.join(os.getcwd(), os.pardir)))

from configuration.Configuration import Configuration


def plot_txts(config, df: pd.DataFrame):
    if not (config.plot_txts or config.export_plots):
        return

    txts = ['txt15', 'txt16', 'txt17', 'txt18', 'txt19']

    print('Creating TXT Plots:')
    for txt in txts:
        columns_current_txt = [col for col in config.features_used if txt in col]
        df_current_txt = df[columns_current_txt]
        plot_single_txt(df_current_txt, txt, config)
    print()


def plot_single_txt(df: pd.DataFrame, file_name: str, config: Configuration):
    print('\tCreating plot for:', file_name)

    axs = df.plot(subplots=True, sharex=True, figsize=(20, 20), title=file_name)
    for ax in axs:
        ax.legend(loc='lower left')

    xmarks = df.index.values[::10000]
    plt.xticks(xmarks)

    if config.export_plots:
        plt.savefig(config.pathPrefix + 'plots/' + file_name, dpi=300, bbox_inches='tight')
    if config.plot_txts:
        plt.show()


def plot_accs(config: Configuration, df: pd.DataFrame):
    if not (config.plot_acc_sensors or config.export_plots):
        return

    accs = [
        # (file_name, cols)
        ('acc_txt15_m1', ["a_15_1_x", "a_15_1_y", "a_15_1_z"]),
        ('acc_txt15_comp', ["a_15_c_x", "a_15_c_y", "a_15_c_z"]),
        ('acc_txt16_m3', ["a_16_3_x", "a_16_3_y", "a_16_3_z", ]),
        ('acc_txt18_m1', ["a_18_1_x", "a_18_1_y", "a_18_1_z", ])
    ]

    print('Creating ACC Plots:')
    for file_name, columns in accs:
        df_current_acc = df[columns]
        plot_single_acc(df_current_acc, file_name, config)
    print()


def plot_single_acc(df: pd.DataFrame, file_name: str, config: Configuration):
    print('\tCreating plot for:', file_name)

    axs = df.plot(subplots=True, sharex=True, figsize=(20, 20), title=file_name)
    for ax in axs:
        ax.legend(loc='lower left')

    if config.export_plots:
        plt.savefig(config.pathPrefix + 'plots/' + file_name, dpi=300, bbox_inches='tight')
    if config.plot_acc_sensors:
        plt.show()


def plot_pressure_sensors(config: Configuration, df: pd.DataFrame, ):
    if not (config.plot_pressure_sensors or config.export_plots):
        return

    print('Creating pressure sensor plot\n')

    df_rel = df[["hPa_15", "hPa_17", "hPa_18"]]

    axs = df_rel.plot(subplots=True, sharex=True, figsize=(20, 20), title="Pressure Sensors")
    for ax in axs:
        ax.legend(loc='lower left')

    xmarks = df_rel.index.values[::7500]
    plt.xticks(xmarks)

    if config.export_plots:
        plt.savefig(config.pathPrefix + 'plots/pressure_sensors.png', dpi=300, bbox_inches='tight')
    if config.plot_pressure_sensors:
        plt.show()


def plot_bmx(config: Configuration, df: pd.DataFrame):
    if not (config.plot_bmx_sensors or config.export_plots):
        return

    print('Creating bmx plots\n')

    cols_vsg = ["vsg_gyr_x", "vsg_gyr_y", "vsg_gyr_z", "vsg_mag_x", "vsg_mag_y",
                "vsg_mag_z", "vsg_acc_x", "vsg_acc_y", "vsg_acc_z"]

    cols_hrs = ["hrs_gyr_x", "hrs_gyr_y", "hrs_gyr_z",
                "hrs_mag_x", "hrs_mag_y", "hrs_mag_z"]

    df_vsg = df[cols_vsg]
    df_hrs = df[cols_hrs]

    axs = df_vsg.plot(subplots=True, sharex=True, figsize=(20, 20), title="BMX VSG")

    for ax in axs:
        ax.legend(loc='lower left')

    if config.export_plots:
        plt.savefig(config.pathPrefix + 'plots/bmx_vsg.png', dpi=300, bbox_inches='tight')
    if config.plot_acc_sensors:
        plt.show()

    axs = df_hrs.plot(subplots=True, sharex=True, figsize=(20, 20), title="BMX HRS")
    for ax in axs:
        ax.legend(loc='lower left')

    if config.export_plots:
        plt.savefig(config.pathPrefix + 'plots/bmx_hrs.png', dpi=300, bbox_inches='tight')
    if config.plot_acc_sensors:
        plt.show()


def plot_all_combined(config: Configuration, df: pd.DataFrame):
    if not (config.plot_all_sensors or config.export_plots):
        return

    # Interpolate and add missing data
    print('Interpolate data for plotting')
    df_combined = df.apply(pd.Series.interpolate, args=('linear',))
    df_combined = df_combined.fillna(method='backfill')

    print('Creating full plot')
    df_combined.plot(subplots=True, sharex=True, figsize=(40, 40),
                     title="All sensors")
    if config.export_plots:
        plt.savefig(config.pathPrefix + 'plots/all_sensors.png', dpi=200)
    if config.plot_all_sensors:
        plt.show()


def plot_df(config: Configuration, df):
    df = df.query(config.query)

    plot_txts(config, df)
    plot_accs(config, df)
    plot_pressure_sensors(config, df)
    plot_bmx(config, df)
    # plot_all_combined(config, df)


def main():
    register_matplotlib_converters()
    mpl.rcParams['agg.path.chunksize'] = 10000

    conf = Configuration()  # Get config for data directory
    number_data_sets = len(conf.datasets)

    for i in range(number_data_sets):
        print('\n\nPlotting dataframe ' + str(i) + '/' + str(number_data_sets - 1) + ' from file')
        print()

        # read the imported dataframe from the saved file
        config = Configuration(i)  # Get config for data directory
        path_to_file = config.datasets[i][0] + config.filename_pkl_cleaned
        df: pd.DataFrame = pd.read_pickle(path_to_file)
        plot_df(config, df)

        del df
        gc.collect()


if __name__ == '__main__':
    main()
