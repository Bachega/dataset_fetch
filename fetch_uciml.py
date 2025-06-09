import pandas as pd
from ucimlrepo import fetch_ucirepo

uciml_summary = None

def build_uciml_summary(
    uci_index_path: str = "./uci_datasets_index.csv", 
    path: str = "untreated_uci_datasets"
):
    uci_datasets_index = pd.read_csv(uci_index_path)
    dts_names = uci_datasets_index['name'].to_list()

    info_columns = [
        'uci_id', 'name', 'repository_url', 'data_url',
        'num_instances', 'num_features', 'has_missing_values'
    ]
    additional_info_columns = [
        'preprocessing_description', 'recommended_data_splits'
    ]
    # Now include the target column name
    columns = info_columns + additional_info_columns + ['is_binary', 'target_col']
    uciml_summary = pd.DataFrame(columns=columns)

    for name in dts_names:
        try:
            dataset = fetch_ucirepo(name)
            df = dataset.data.features  # pandas.DataFrame
            metadata = dataset.get("metadata", {})
            additional_info = metadata.get("additional_info", {})

            # Extract core metadata
            row = [metadata.get(key) for key in info_columns]
            # Extract additional metadata
            row += [additional_info.get(key) for key in additional_info_columns]
            # Is binary target?
            is_binary = True if len(metadata.get('target_col', [])) == 1 else False
            row.append(is_binary)
            # Capture target column name(s)
            target_cols = metadata.get('target_col', [])
            if isinstance(target_cols, (list, tuple)):
                row.append(
                    ";".join(str(c) for c in target_cols)
                )
            else:
                row.append(str(target_cols))

            uciml_summary.loc[len(uciml_summary)] = row

        except Exception as e:
            print(f"Error with {name} => {e}")

    uciml_summary.to_csv("./uciml_summary.csv", index=False)


def fetch_uciml_datasets(
    uci_index_path: str = "./uci_datasets/uci_datasets_index.csv",
    output_dir: str = "./uci_datasets/datasets"
):
    uci_datasets_index = pd.read_csv(uci_index_path)
    dts_names = uci_datasets_index['name'].to_list()

    info_columns = [
        'uci_id', 'name', 'repository_url', 'data_url',
        'num_instances', 'num_features', 'has_missing_values'
    ]
    additional_info_columns = [
        'preprocessing_description', 'recommended_data_splits'
    ]
    columns = info_columns + additional_info_columns + ['is_binary', 'target_col']
    uciml_summary = pd.DataFrame(columns=columns)

    for name in dts_names:
        try:
            dataset = fetch_ucirepo(name)
            df = dataset.data.features.copy()
            target = dataset.data.target  
            metadata = dataset.get("metadata", {})
            additional_info = metadata.get("additional_info", {})

            # Determine target column names
            raw_target_cols = metadata.get('target_col', [])
            if isinstance(raw_target_cols, (list, tuple)):
                target_cols = [str(tc) for tc in raw_target_cols]
            else:
                target_cols = [str(raw_target_cols)]

            # Append target data to df
            if target.ndim == 1:
                # single target column
                df[target_cols[0]] = target
            else:
                # multi-output target: assume alignment of names
                for idx, col_name in enumerate(target_cols):
                    df[col_name] = target[:, idx]

            row = [metadata.get(key) for key in info_columns]
            row += [additional_info.get(key) for key in additional_info_columns]
            is_binary = True if len(metadata.get('target_col', [])) == 1 else False
            row.append(is_binary)
            target_cols = metadata.get('target_col', [])
            if isinstance(target_cols, (list, tuple)):
                row.append(
                    ";".join(str(c) for c in target_cols)
                )
            else:
                row.append(str(target_cols))

            uciml_summary.loc[len(uciml_summary)] = row

            # Save the dataset with uci_id and name
            uci_id = metadata.get('uci_id')
            name = metadata.get('name')
            filename = f"{uci_id}_{name}.csv"
            df.to_csv(f"{output_dir}/{filename}", index=False)

        except Exception as e:
            print(f"Error with {name} => {e}")

    # Optionally save summary of fetched datasets
    uciml_summary.to_csv("./uciml_summary.csv", index=False)

build_uciml_summary()
fetch_uciml_datasets()