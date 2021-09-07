import pandas as pd
import numpy as np
from tensorflow.keras.utils import to_categorical


def data_preparation():
    col_names = ['time', 'phone', 'version', 'ct', 'cv',
                 'age', 'sex',
                 'cid1', 'cid2', 'cid3', 'brand_id', 'merchant_id']


    train = pd.read_csv(
        'data/trainA.csv',
        delimiter=',',
        engine="c",
        header=None,
        index_col=None,
        names=col_names,
        nrows=20000000
    )
    other = pd.read_csv(
        'data/trainA.csv',
        delimiter=',',
        engine="c",
        header=None,
        index_col=None,
        names=col_names,
        skiprows=range(20000000),
        nrows=2000000
    )

    label_columns = ['ct', 'cv']

    categorical_columns = ['phone', 'version', 'sex', 'cid1', 'cid2', 'cid3', 'brand_id', 'merchant_id']

    train_raw_labels = train[label_columns]
    other_raw_labels = other[label_columns]
    transformed_train = pd.get_dummies(train.drop(label_columns, axis=1), columns=categorical_columns)
    transformed_other = pd.get_dummies(other.drop(label_columns, axis=1), columns=categorical_columns)

    # One-hot encoding categorical labels
    train_ct = to_categorical((train_raw_labels.ct == '1').astype(int), num_classes=2)
    train_cv = to_categorical((train_raw_labels.cv == '1').astype(int), num_classes=2)
    other_ct = to_categorical((other_raw_labels.ct == '1').astype(int), num_classes=2)
    other_cv = to_categorical((other_raw_labels.cv == '1').astype(int), num_classes=2)

    validation_indices = transformed_other.sample(frac=0.5, replace=False, random_state=1).index
    test_indices = list(set(transformed_other.index) - set(validation_indices))
    validation_data = transformed_other.iloc[validation_indices].to_numpy(dtype="float32")
    validation_label = np.concatenate([other_ct[validation_indices], other_cv[validation_indices]], axis=-1)
    # validation_label = other_ctr[validation_indices]
    test_data = transformed_other.iloc[test_indices].to_numpy(dtype="float32")
    test_label = np.concatenate([other_ct[test_indices], other_cv[test_indices]], axis=-1)
    # test_label = other_ctr[test_indices]
    train_data = transformed_train.to_numpy(dtype="float32")
    train_label = np.concatenate([train_ct, train_cv], axis=-1)
    # train_label = train_ctr

    return train_data, train_label, validation_data, validation_label, test_data, test_label
