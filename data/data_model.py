import tensorflow as tf
keras = tf.keras
import numpy as np

def create_tf_data(
    X,
    Y,
    sequence_stride,
    sequence_length_x,
    sequence_length_y,
    sampling_rate,
    batch_size,
    shuffle=True
    ):
    """
    Create dataset tensorflow for model
    Args:
        X: Data inputs
        Y: Data outputs
        sequence_stride: example sequence_stride =2, [1,2,3,4] --> [3,4,5,6]
        sequence_length_x: len of sequence inputs
        sequence_length_y: len of sequence outputs
        sampling_rate: split squence to sampling, example: sequence days to 24 sequence hours
    Return:
        tf dataset 
    """
    inputs = keras.preprocessing.timeseries_dataset_from_array(
        X,
        None,
        sequence_stride = sequence_stride,
        sequence_length= sequence_length_x,
        sampling_rate= sampling_rate,
        batch_size=batch_size,
    )

    outputs = keras.preprocessing.timeseries_dataset_from_array(
        Y,
        None,
        sequence_stride = sequence_stride,
        sequence_length= sequence_length_y,
        sampling_rate=sampling_rate,
        batch_size=batch_size,
    )

    dataset = tf.data.Dataset.zip((inputs, outputs))
    if shuffle:
        dataset = dataset.shuffle(100)

    return dataset.prefetch(16).cache()