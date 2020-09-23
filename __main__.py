import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from neural_net.data_prep import DataPreparation
from neural_net.train_loop import TrainLoop
from neural_net.nn import MyCNN, Cnn_seq
from config import config
from utils.support_func import seed_everything
import torch
import os

if __name__ == '__main__':
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = '0'

    seed_everything(2020)
    train_data = pd.read_csv(config['path_to_data_train'])
    train_labels = pd.read_csv(config['path_to_labels_train'])
    y = np.where(train_labels.drop('sequence_id', axis=1).values==1.0)[1]
    X_train, X_test, y_train, y_test = train_test_split(train_data, y, random_state=2020, test_size=0.1)

    preparator_train = DataPreparation()
    X_train, X_train_one_hot, y_train = preparator_train.transform(config['maxlen'], X_train, y_train)
    X_test, X_test_one_hot, y_test = preparator_train.transform(config['maxlen'], X_test, y_test)

    #  model = Cnn_seq(100)
    kernel_sizes = [i for i in range(5, 26, 10)] + [i for i in range(40, 120, 30)]
    model = MyCNN(n_channels=config.n_channels, kernel_sizes=kernel_sizes)
    model.cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    train_looper = TrainLoop(config['bs'], config['epochs'], config['lr_reduce_parameter'],
                             config['change_lr_threshold'],
                             config['early_stopping_criteria'], model, optimizer, X_train, y_train, X_test, y_test,
                             X_train_one_hot,
                             X_test_one_hot)
    train_looper.train_loop()

    #model.load_state_dict(torch.load('model_dict.h5'))
    #model.cuda()
    #optimizer.load_state_dict(torch.load('opt_dict.dict'))
    # train_looper = TrainLoop(config['bs'], config['epochs'], config['lr_reduce_parametr'],
    #                          config['change_lr_treshhold'],
    #                          config['early_stopping_criteria'], model, optimizer, X_train, y_train, X_test, y_test,
    #                          X_train_one_hot,
    #                          X_test_one_hot)

    train_looper.train_on_val(3)