import os
import random
import numpy as np
import tensorflow as tf

import warnings

from training.model_training import modelTrain
from data.format_cea_data import FormatData

seed = 1234
os.environ['PYTHONHASHSEED'] = str(seed)
random.seed(seed)
tf.random.set_seed(seed)
np.random.seed(seed)


warnings.filterwarnings("ignore")

if __name__ == "__main__":
    DATA_PATH = "../data/raw/lr_transformed_cea_data.csv"

    model_store = "../models"
    data_store = "../data/outputs"

    window_size_list = [5, 6, 7, 10, 15]
    formatted_frames = None
    for window_size in window_size_list:
        skip_size = window_size
        data_formattor = FormatData(DATA_PATH)
        formatted_frames = data_formattor(window_size=window_size, skip_size=skip_size, transform_type='all')
        for transform_type in [None, 'fft', 'dct']:
            print('#'*70)
            print('WINDOW SIZE: {} - TRANSFORM TYPE: {}'.format(window_size, transform_type))
            print('#'*70)
            model_trainer = modelTrain(formatted_frames, window_size, transform_type, model_store, data_store)
            model_trainer()
