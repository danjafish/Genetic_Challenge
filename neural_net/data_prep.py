import numpy as np
from keras.preprocessing.sequence import pad_sequences
from utils.support_func import letter2index
from torch import LongTensor, FloatTensor


class DataPreparation:
    def transform(self, max_len, data_csv, y=None):
        X_one_hot = np.array(data_csv.drop(['sequence', 'sequence_id'], axis=1).values)
        encoded_sequence = data_csv.sequence.apply(lambda x: letter2index(x))
        padded_sequence = pad_sequences(encoded_sequence, maxlen=max_len)
        X = LongTensor(padded_sequence)
        X_one_hot = FloatTensor(X_one_hot)
        if y is not None:
            y = LongTensor(y)
            return X, X_one_hot, y
        else: return X, X_one_hot
