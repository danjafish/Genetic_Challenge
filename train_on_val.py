import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from neural_net.data_prep import DataPreparation
from neural_net.train_loop import TrainLoop
from neural_net.nn import MyCNN
import torch

config = {
    'bs': 32,
    'maxlen': 5000,
    'n_channels': 100,
    'epochs': 25,
    'best_val_loss': 1e7,
    'lr_reduce_parametr': 0.1,
    'change_lr_treshhold': 2,
    'early_stopping_criteria': 5,
    'tolerance': 1e-3
}
if __name__ == '__main__':
    train_data = pd.read_csv('../train_values.csv')
    train_labels = pd.read_csv('../train_labels.csv')
    y = np.where(train_labels.drop('sequence_id', axis=1).values == 1.0)[1]
    X_train, X_test, y_train, y_test = train_test_split(train_data, y, random_state=2020, test_size=0.2)

    preparator_train = DataPreparation()
    X_train, X_train_one_hot, y_train = preparator_train.transform(config['maxlen'], X_train, y_train)
    X_test, X_test_one_hot, y_test = preparator_train.transform(config['maxlen'], X_test, y_test)

    model = MyCNN(100)
    model.cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    train_looper = TrainLoop(config['bs'], config['epochs'], config['lr_reduce_parametr'],
                             config['change_lr_treshhold'],
                             config['early_stopping_criteria'], model, optimizer, X_train, y_train, X_test, y_test,
                             X_train_one_hot,
                             X_test_one_hot)

    train_looper.train_loop()
