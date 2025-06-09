import pandas as pd
from ucimlrepo import fetch_ucirepo

uciml_summary = None

def build_uciml_summary(uci_index_path = "./uci_datasets_index.csv", path = "untreated_uci_datasets"):
    uci_datasets_index = pd.read_csv(uci_index_path)
    dts_names = uci_datasets_index['name'].to_list()

    i = -1

    info_columns = ['uci_id', 'name', 'repository_url', 'data_url', 'num_instances', 'num_features', 'has_missing_values']
    additional_info_columns = ['preprocessing_description', 'recommended_data_splits']
    columns = info_columns + additional_info_columns + ['is_binary']
    uciml_summary = pd.DataFrame(columns=columns)

    for name in dts_names:
        i += 1
        try:
            dataset = fetch_ucirepo(name)
            df = dataset.data.features       # pandas.DataFrame
            metadata = dataset.get("metadata", {})
            additional_info = metadata.get("additional_info", {})        
            row = [metadata[key] for key in info_columns] + [additional_info[key] for key in additional_info_columns]
            row.append(True if len(metadata.target_col) == 1 else False)
            uciml_summary.loc[len(uciml_summary)] = row
        except Exception as e:
            print(f"Error with {name} => {e}")

    uciml_summary.to_csv("./uciml_summary.csv", index=False)



def fetch_uciml_datasets(uci_index_path = "./uci_datasets/uci_datasets_index.csv"):
    uci_datasets_index = pd.read_csv(uci_index_path)

    dts_names = uci_datasets_index['name'].to_list()

    info_columns = ['uci_id', 'name', 'repository_url', 'data_url', 'num_instances', 'num_features', 'has_missing_values']
    additional_info_columns = ['preprocessing_description', 'recommended_data_splits']
    columns = info_columns + additional_info_columns + ['is_binary']
    uciml_summary = pd.DataFrame(columns=columns)

    for name in dts_names:
        try:
            dataset = fetch_ucirepo(name)
            df = dataset.data.features       # pandas.DataFrame

            metadata = dataset.get("metadata", {})
            additional_info = metadata.get("additional_info", {})        
            row = [metadata[key] for key in info_columns] + [additional_info[key] for key in additional_info_columns]
            row.append(True if len(metadata.target_col) == 1 else False)
            uciml_summary.loc[len(uciml_summary)] = row

            df.to_csv(f"./uci_datasets/datasets/{row[0]}_{row[1]}.csv", index=False)

        except Exception as e:
            print(f"Error with {name} => {e}")

# build_uciml_summary()

# fetch_uciml_datasets()