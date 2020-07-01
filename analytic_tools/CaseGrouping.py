import json

import pandas as pd

from configuration.Configuration import Configuration


def flatten(lis):
    return [item for sublist in lis for item in sublist]


def get_cols(df, row_name):
    return list(df.columns[df.loc[row_name] == 1])


def load():
    config = Configuration()
    relevant_features = config.case_to_individual_features

    for key in relevant_features:
        relevant_features[key] = sorted(relevant_features[key])

    return sorted(relevant_features.keys()), relevant_features


def export_excel_for_grouping(export_to_excel):
    # dataframes should be printed completely
    pd.set_option('display.max_rows', None)
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', None)
    pd.set_option('display.max_colwidth', None)

    cases, relevant_features = load()
    features_as_list = sorted(list(set(flatten(relevant_features.values()))))

    # create a dataframe which display which feature is relevant for which case
    df = pd.DataFrame(0, columns=features_as_list, index=cases)

    for case in cases:
        relevant_for_case = relevant_features.get(case)
        df.loc[case, relevant_for_case] = 1

    # export this dataframe as an excel sheet
    if export_to_excel:
        df.to_excel("../data/feature_selection/features.xlsx")

    return df


def main():
    enable_output = False
    export_to_excel = False

    # needs a dict with class -> relevant features in features.json
    df = export_excel_for_grouping(export_to_excel)

    if export_to_excel:
        return

    # lists of cases that should be grouped together,
    # must be filled manually based on available knowledge or the exported excel sheet
    groups = [
        [
            'no_failure'
        ],
        [
            'txt15_conveyor_failure_mode_driveshaft_slippage_failure',
            'txt15_i1_lightbarrier_failure_mode_1',
            'txt15_i1_lightbarrier_failure_mode_2',
            'txt15_i3_lightbarrier_failure_mode_1',
            'txt15_i3_lightbarrier_failure_mode_2',
            'txt15_m1_t1_high_wear',
            'txt15_m1_t1_low_wear',
            'txt15_m1_t2_wear'
        ],
        [
            'txt15_pneumatic_leakage_failure_mode_1',
            'txt15_pneumatic_leakage_failure_mode_2',
            'txt15_pneumatic_leakage_failure_mode_3'
        ],
        [
            'txt16_conveyor_failure_mode_driveshaft_slippage_failure',
            'txt16_conveyorbelt_big_gear_tooth_broken_failure',
            'txt16_conveyorbelt_small_gear_tooth_broken_failure',
            'txt16_m3_t1_high_wear',
            'txt16_m3_t1_low_wear',
            'txt16_m3_t2_wear',
        ],
        [
            'txt16_i3_switch_failure_mode_2',
            'txt16_i4_lightbarrier_failure_mode_1'
        ],
        [
            'txt16_pneumatic_leakage_failure_mode_1',
            'txt17_i1_switch_failure_mode_1',
            'txt17_i1_switch_failure_mode_2',
            'txt17_pneumatic_leakage_failure_mode_1',
            'txt17_workingstation_transport_failure_mode_wout_workpiece'
        ],
        [
            'txt18_pneumatic_leakage_failure_mode_1',
            'txt18_pneumatic_leakage_failure_mode_2',
            'txt18_pneumatic_leakage_failure_mode_2_faulty',
            'txt18_pneumatic_leakage_failure_mode_3_faulty',
            'txt18_transport_failure_mode_wout_workpiece'
        ],
        [
            'txt19_i4_lightbarrier_failure_mode_1',
            'txt19_i4_lightbarrier_failure_mode_2'
        ]
    ]

    # generate output tuples (grouped cases, relevant attributes for the group)
    group_id_to_cases = {}
    group_id_to_features = {}
    case_to_group_id = {}

    for index, group in enumerate(groups):
        all_features = []
        group_id = 'g' + str(index)

        for case in group:
            features_for_case = get_cols(df, case)
            all_features += features_for_case
            case_to_group_id[case] = group_id

        all_features = sorted(list(set(all_features)))

        group_id_to_cases[group_id] = group
        group_id_to_features[group_id] = all_features

    if enable_output:
        for i in range(len(groups)):
            group_id = 'g' + str(i)
            print('id:', group_id)
            print('cases:', group_id_to_cases[group_id])
            print('features:', group_id_to_features[group_id])
            print('--------------------------------------------------------------')

    with open('../data/feature_selection/group_id_to_cases.json', 'w', encoding='utf-8') as f:
        json.dump(group_id_to_cases, f, ensure_ascii=False, indent=2)

    with open('../data/feature_selection/group_id_to_features.json', 'w', encoding='utf-8') as f:
        json.dump(group_id_to_features, f, ensure_ascii=False, indent=2)

    with open('../data/feature_selection/case_to_group_id.json', 'w', encoding='utf-8') as f:
        json.dump(case_to_group_id, f, ensure_ascii=False, indent=2)


if __name__ == '__main__':
    main()
