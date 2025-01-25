import pandas as pd
import numpy as np
from ucimlrepo import fetch_ucirepo
from openml import datasets
import os
from math import ceil
from run_tests import run_tests
from utils import run_with_timeout

def run():
  if os.path.exists('dataset_processing_log.csv'):
      preprocessing_log = pd.read_csv('dataset_processing_log.csv')
  else:
      preprocessing_log = pd.DataFrame(columns=['did', 'Dataset Name', 'Class Column' , 'No. of Rows', 'No. of Columns', 'No. of Extracted Meta-Features', 'One-Hot Encoded?', 'Label Encoded?'])

  openml_datasets_index = pd.read_csv('openml_datasets_index.csv')
  uci_datasets_index = pd.read_csv('uci_datasets_index.csv')

  openml_index = openml_datasets_index
  uci_index = uci_datasets_index


  openml_index = openml_datasets_index
  uci_index = uci_datasets_index

  openml_index = openml_index[openml_index['NumberOfClasses'] == 2]
  openml_index = openml_index[openml_index['NumberOfMissingValues'] == 0]
  openml_index = openml_index[openml_index['NumberOfInstancesWithMissingValues'] == 0]

  passing_dids = []
  for _row in openml_index.iterrows():
      row = _row[1]
      
      dataset_size = row['NumberOfInstances']
      pos_prop = row['MajorityClassSize'] / dataset_size
      neg_prop = row['MinorityClassSize'] / dataset_size

      sample_size = 100
      test_size = 0.5

      pos_number_sample_test = ceil(pos_prop * dataset_size * test_size)
      neg_number_sample_test = ceil(neg_prop * dataset_size * test_size)

      if pos_number_sample_test > sample_size and neg_number_sample_test > sample_size:
          passing_dids.append(row['did'])

  openml_index = openml_index[openml_index['did'].isin(passing_dids)]

  did_list = openml_index['did'].tolist()

  if not os.path.exists('current_did.txt'):
      with open('current_did.txt', 'w') as f:
          f.write(str(did_list[0]))

  with open('current_did.txt', 'r') as f:
      current_did = int(f.read().strip())

  _start = did_list.index(current_did)
  for i in range(_start, len(did_list)):
      print(f"ITER. {i} - DID. {did_list[i]}")

      try:
        did = did_list[i]
        with open('current_did.txt', 'w') as f:
            f.write(str(did))

        dataset = datasets.get_dataset(int(did))
        dataset_name = dataset.name
        X, y, categorical_indicator, attribute_names = dataset.get_data(
            dataset_format="dataframe", target=dataset.default_target_attribute
        )
      except:
        continue
      
      result = run_with_timeout(600, run_tests, X, y, categorical_indicator)
      if result is None:
          continue
      
      X_processed, y_processed, dataset_shape, num_meta_features_extracted, one_hot_encoded, label_encoded = result
      data_processed = np.column_stack((X_processed, y_processed))
      
      # X_processed, y_processed, dataset_shape, num_meta_features_extracted, one_hot_encoded, label_encoded = run_tests(X, y, categorical_indicator)
      
      processed_df = pd.DataFrame(columns=[_i for _i in range(1, data_processed.shape[1] + 1)], data=data_processed)

      preprocessing_log.loc[len(preprocessing_log.index)] = [did, dataset_name, y.name, dataset_shape[0], dataset_shape[1], num_meta_features_extracted, one_hot_encoded, label_encoded]
      processed_df.to_csv(f"./treated_datasets/{dataset_name}.csv", index=False)
      preprocessing_log.to_csv("dataset_processing_log.csv", index=False)

if __name__ == "__main__":
  run()