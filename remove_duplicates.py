import os

old_folder = "./OLD_DATASETS"
new_folder = "./OLD_DATASETS/NEW_DATASETS"

old_filenames = {f for f in os.listdir(old_folder) if f.endswith(".csv")}

for filename in os.listdir(new_folder):
    if filename.endswith(".csv") and filename in old_filenames:
        file_path = os.path.join(new_folder, filename)
        print(f"Removing duplicate: {file_path}")
        os.remove(file_path)
