import tensorflow as tf
from tensorflow import keras
import os 
import numpy as np
#%%
X=tf.range(10)
dataset=tf.data.Dataset.from_tensor_slices(X)
dataset

for item in dataset:
    print(item)
#%%
dataset=dataset.repeat(3).batch(7)
for item in dataset:
    print(item)
#%%
dataset=dataset.map(lambda x: x*2)
for item in dataset:
    print(item)

#%%
dataset=dataset.apply(tf.data.experimental.unbatch())
dataset=dataset.filter(lambda x:x<10)

for item in dataset.take(3):
    print(item)

#%%
dataset=tf.data.Dataset.range(10).repeat(3)
dataset=dataset.shuffle(buffer_size=5,seed=42).batch(7)
for item in dataset:
    print(item)
#%%
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

housing = fetch_california_housing()
X_train_full, X_test, y_train_full, y_test = train_test_split(
    housing.data, housing.target.reshape(-1, 1), random_state=42)
X_train, X_valid, y_train, y_valid = train_test_split(
    X_train_full, y_train_full, random_state=42)

scaler = StandardScaler()
scaler.fit(X_train)
X_mean = scaler.mean_
X_std = scaler.scale_
#%%
data=X_train
def save_to_multiple_csv_files(data, name_prefix='train', header=None, n_parts=10):
    housing_dir = os.path.join("datasets", "housing")
    os.makedirs(housing_dir, exist_ok=True)
    path_format = os.path.join(housing_dir, "my_{}_{:02d}.csv")

    filepaths = []
    m = len(data)
    for file_idx, row_indices in enumerate(np.array_split(np.arange(m), n_parts)):
        print(file_idx)
        print(row_indices)
        part_csv = path_format.format(name_prefix, file_idx)
        filepaths.append(part_csv)
        with open(part_csv, "wt", encoding="utf-8") as f:
            if header is not None:
                f.write(header)
                f.write("\n")
            for row_idx in row_indices:
                f.write(",".join([repr(col) for col in data[row_idx]]))
                f.write("\n")
    return filepaths


#%%
train_data = np.c_[X_train, y_train]
valid_data = np.c_[X_valid, y_valid]
test_data = np.c_[X_test, y_test]

header_cols = housing.feature_names + ["MedianHouseValue"]
header = ",".join(header_cols)
#%%
train_filepaths = save_to_multiple_csv_files(train_data, "train", header, n_parts=20)
valid_filepaths = save_to_multiple_csv_files(valid_data, "valid", header, n_parts=10)
test_filepaths = save_to_multiple_csv_files(test_data, "test", header, n_parts=10)

import pandas as pd

pd.read_csv(train_filepaths[0]).head()

#%%
filepath_dataset=tf.data.Dataset.list_files(train_filepaths,seed=42)
for item in filepath_dataset:
    print(item)

#%%
