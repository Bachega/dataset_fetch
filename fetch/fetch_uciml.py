import pandas as pd
from ucimlrepo import fetch_ucirepo

uciml_summary = None

def fetch_treat_uciml_datasets(uci_index_path = "./uci_datasets_index.csv", path = "untreated_uci_datasets"):
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
        
        # if i > 10: break
    uciml_summary.to_csv("./fetch/uciml_summary.csv", index=False)

fetch_treat_uciml_datasets()





# # 1) Fetch by the exact UCI name
# dataset = fetch_ucirepo("Abalone")

# # 2) Inspect the return object
# print(type(dataset))            # usually a dict or custom Bunch
# print(dataset.keys())           # what top‑level entries are present

# # 3) Pull out the main pieces
# df   = dataset["data"]         # a pandas.DataFrame
# metadata = dataset.get("metadata", {})  # metadata dict

# target_col = metadata.target_col
# num_instances = metadata.num_instances
# num_features = metadata.num_features


# df = df.features
# df
# # # 4) Quick look
# # print(df.head())
# # print("Target attribute:", meta.get("target_attribute"))
# # print("Other metadata:", {k: meta[k] for k in ("description", "n_instances") if k in meta})
